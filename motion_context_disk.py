"""
MiniMax H3 Motion Context - disk-backed sequential chain (v13 clean).

Two-node design:
  * MiniMax H3 Motion Context Disk Join
  * MiniMax H3 Motion Context Disk Final Decode

Goals:
  * full_batch and clip_by_clip workflows with the SAME graph
  * validated clips never require their sampler branch again
  * one .h3cache + one .json per chain, stored in this custom-node folder/cache
  * rerendering truncates/reuses the cache tail instead of creating files forever
  * cached LATENT output is memory-mapped/lazy; validated prefix does not fill RAM
  * final export decodes one seam pair at a time and streams to ffmpeg

The v10 RAM Motion Context conditioning and its validated seam corrections remain
unchanged. This module only changes persistence/execution and final streaming.
"""

import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import uuid

import numpy as np
import torch
import comfy.nested_tensor
import comfy.utils

try:
    import folder_paths
except Exception:  # static tests outside ComfyUI
    folder_paths = None

from .motion_context_ram import (
    FPS,
    _audio_exact_frames,
    _audio_t_for_frames,
    _auto_early_seam_shift,
    _frames_from_video_t,
    _photometric_match_segment,
    _pixel_frames,
    _steps_for_frames,
    _streams_from_latent,
)

BUILD = "motion-context-disk-v14.17-clip-by-clip-autosave"
CACHE_VERSION = 12

class _FinalDecodeNativeProgress:
    """Native ComfyUI progress bound to the *currently executing* Final Decode node.

    We deliberately drive both paths used by recent ComfyUI versions:
      1. comfy.utils.ProgressBar -> normal global progress hook / legacy progress event
      2. comfy_execution.progress registry -> native per-node progress_state

    The registry path is optional so the custom node remains import-compatible with
    older ComfyUI versions.  No custom JS or websocket event is involved.
    """

    def __init__(self, unique_id, total):
        self.total = max(1, int(total))
        self.value = 0
        self.registry = None

        # The execution context is the authoritative runtime node id.  UNIQUE_ID is
        # retained as a fallback for older executors / direct tests.
        context_node_id = None
        try:
            from comfy_execution.utils import get_executing_context

            ctx = get_executing_context()
            context_node_id = getattr(ctx, "node_id", None) if ctx is not None else None
        except Exception:
            context_node_id = None

        raw_node_id = context_node_id if context_node_id is not None else unique_id
        self.node_id = None if raw_node_id is None else str(raw_node_id)

        try:
            self.pbar = comfy.utils.ProgressBar(self.total, node_id=self.node_id)
        except TypeError:
            # Compatibility with older ComfyUI ProgressBar signatures.
            self.pbar = comfy.utils.ProgressBar(self.total)

        try:
            from comfy_execution.progress import get_progress_state

            self.registry = get_progress_state()
            if self.node_id is not None:
                self.registry.start_progress(self.node_id)
        except Exception:
            self.registry = None

        # Send a non-zero first state immediately, before the first long VAE decode.
        self.update_absolute(1)

    def update_absolute(self, value):
        value = max(0, min(int(value), self.total))
        if value < self.value:
            return
        self.value = value

        # This is the exact native ProgressBar API sampler callbacks use.
        self.pbar.update_absolute(self.value, self.total)

        # On current ComfyUI, the inline node bar is sourced from progress_state.
        # Drive it explicitly as well; this is harmless if the global hook already
        # updated the same registry entry.
        if self.registry is not None and self.node_id is not None:
            try:
                self.registry.update_progress(
                    self.node_id, self.value, self.total, None
                )
            except Exception:
                pass

    def advance(self, units=1):
        self.update_absolute(self.value + max(0, int(units)))

    def finish(self):
        self.update_absolute(self.total)
        if self.registry is not None and self.node_id is not None:
            try:
                self.registry.finish_progress(self.node_id)
            except Exception:
                pass


CACHE_TYPE = "H3_MOTION_DISK_CACHE"
_LOG = logging.getLogger("minimax_h3_tail_from_latent.motion_context_disk")

_NODE_DIR = Path(__file__).resolve().parent
_CACHE_ROOT = _NODE_DIR / "cache"
_DATA_MAGIC = b"H3MCACHE12\x00"
_DATA_START = len(_DATA_MAGIC)

_DTYPE_MAP = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}
# Float8 names exist only on recent torch builds.
for _name in (
    "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"
):
    _dt = getattr(torch, _name, None)
    if _dt is not None:
        _DTYPE_MAP[_name] = _dt


def _safe_name(value):
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value or "h3_chain"


def _ensure_cache_root():
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return _CACHE_ROOT


def _chain_paths(owner_id):
    root = _ensure_cache_root()
    stem = "chain_" + _safe_name(owner_id)
    return root / f"{stem}.h3cache", root / f"{stem}.json"


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _dtype_name(tensor):
    return str(tensor.dtype).replace("torch.", "")


def _dtype_from_name(name):
    dt = _DTYPE_MAP.get(str(name))
    if dt is None:
        raise ValueError(f"H3 Disk Cache: unsupported tensor dtype '{name}'.")
    return dt


def _new_data_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_DATA_MAGIC)
        f.flush()
        os.fsync(f.fileno())


def _ensure_data_file(path):
    path = Path(path)
    if not path.exists():
        _new_data_file(path)
        return
    with open(path, "rb") as f:
        magic = f.read(_DATA_START)
    if magic != _DATA_MAGIC:
        raise ValueError(f"H3 Disk Cache: invalid cache data file: {path}")


def _geometry(video, audio):
    return {
        "video_batch": int(video.shape[0]),
        "video_channels": int(video.shape[1]),
        "video_h": int(video.shape[3]),
        "video_w": int(video.shape[4]),
        "audio_batch": int(audio.shape[0]),
        "audio_channels": int(audio.shape[1]),
        "audio_planes": int(audio.shape[2]),
    }


def _validate_batch_one(video, audio):
    if int(video.shape[0]) != 1 or int(audio.shape[0]) != 1:
        raise ValueError("MiniMax H3 Disk Join supports batch size 1 only.")


def _validate_geometry(manifest, video, audio):
    current = _geometry(video, audio)
    expected = manifest.get("geometry")
    if expected is not None and current != expected:
        raise ValueError(
            "MiniMax H3 Disk Join: latent geometry changed between clips. "
            f"Expected {expected}, got {current}."
        )


def _final_frame_count(segments):
    if not segments:
        return 0
    total = int(segments[0]["frames"])
    for desc in segments[1:]:
        total += int(desc["frames"]) - int(desc["trim_frames"])
    return int(total)


def _segment_end(desc):
    return int(desc["segment_end"])


def _segment_start(desc):
    return int(desc["video"]["offset"])


def _recover_manifest(data_path, manifest_path, manifest):
    """Recover a safe prefix after an interrupted tail rewrite."""
    data_path = Path(data_path)
    manifest_path = Path(manifest_path)
    _ensure_data_file(data_path)
    size = int(data_path.stat().st_size)
    good = []
    for desc in manifest.get("segments", []):
        try:
            start = _segment_start(desc)
            end = _segment_end(desc)
            if start < _DATA_START or end < start or end > size:
                break
            good.append(desc)
        except Exception:
            break

    if len(good) != len(manifest.get("segments", [])):
        fixed = dict(manifest)
        fixed["segments"] = [dict(x) for x in good]
        fixed["final_frame_count"] = _final_frame_count(good)
        fixed["updated_at"] = time.time()
        _write_json_atomic(manifest_path, fixed)
        manifest = fixed
        _LOG.warning(
            "H3 Disk Cache recovered %d valid clip(s) after incomplete tail write.",
            len(good),
        )
    return manifest


def _load_manifest_from_paths(data_path, manifest_path):
    data_path = Path(data_path)
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("version", -1)) != CACHE_VERSION:
        raise ValueError(
            f"H3 Disk Cache version {manifest.get('version')} is incompatible; "
            f"expected {CACHE_VERSION}."
        )
    return _recover_manifest(data_path, manifest_path, manifest)


def _load_manifest(cache):
    if not isinstance(cache, dict):
        raise ValueError("MiniMax H3 Disk Cache: invalid cache handle.")
    data_path = Path(cache["data_path"])
    manifest_path = Path(cache["manifest_path"])
    manifest = _load_manifest_from_paths(data_path, manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"H3 Disk Cache manifest not found: {manifest_path}")
    return data_path, manifest_path, manifest


def _make_handle(
    data_path, manifest_path, manifest, run_mode, stop=False, status="", next_index=None
):
    if next_index is None:
        next_index = len(manifest.get("segments", []))
    return {
        "version": CACHE_VERSION,
        "data_path": str(Path(data_path).resolve()),
        "manifest_path": str(Path(manifest_path).resolve()),
        "run_mode": str(run_mode),
        "stop": bool(stop),
        "next_index": int(next_index),
        "status": str(status),
    }


def _cache_size_mb(data_path, manifest_path):
    total = 0
    data_path = Path(data_path)
    preview_path = data_path.with_suffix(".preview.mp4")
    for p in (data_path, Path(manifest_path), preview_path):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return float(total / (1024.0 * 1024.0))


def _write_tensor_raw(file_obj, tensor):
    """Write one contiguous tensor without a giant serialization bytes object."""
    x = tensor.detach()
    if x.device.type != "cpu":
        x = x.to(device="cpu")
    if not x.is_contiguous():
        x = x.contiguous()

    offset = int(file_obj.tell())
    shape = [int(v) for v in x.shape]
    dtype_name = _dtype_name(x)
    nbytes = int(x.numel() * x.element_size())

    # Viewing as uint8 makes numpy() work for BF16/FP8 too; memoryview avoids
    # another full-size bytes copy in Python.
    raw = x.view(torch.uint8).numpy()
    written = int(file_obj.write(memoryview(raw)))
    if written != nbytes:
        raise IOError(f"H3 Disk Cache short write: {written}/{nbytes} bytes.")

    return {
        "offset": offset,
        "nbytes": nbytes,
        "shape": shape,
        "dtype": dtype_name,
    }


def _map_tensor(data_path, spec):
    """Memory-map one tensor. Pages are faulted in only when actually touched."""
    data_path = Path(data_path)
    offset = int(spec["offset"])
    nbytes = int(spec["nbytes"])
    shape = tuple(int(v) for v in spec["shape"])
    dtype = _dtype_from_name(spec["dtype"])

    if nbytes <= 0:
        raise ValueError("H3 Disk Cache: invalid zero-sized tensor.")
    mm = np.memmap(
        str(data_path), mode="c", dtype=np.uint8, offset=offset, shape=(nbytes,)
    )
    raw = torch.from_numpy(mm)
    tensor = raw.view(dtype).reshape(shape)
    return tensor


def _load_segment_video(data_path, desc):
    video = _map_tensor(data_path, desc["video"])
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5 or int(video.shape[0]) != 1:
        raise ValueError(f"Invalid cached H3 video shape: {tuple(video.shape)}")
    return video


def _load_segment_audio(data_path, desc):
    audio = _map_tensor(data_path, desc["audio"])
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4 or int(audio.shape[0]) != 1:
        raise ValueError(f"Invalid cached H3 audio shape: {tuple(audio.shape)}")
    return audio


def _append_segment(data_path, latent, index, trim_frames, validated, manifest):
    video, audio = _streams_from_latent(latent, "samples")
    _validate_batch_one(video, audio)
    if manifest.get("geometry") is not None:
        _validate_geometry(manifest, video, audio)

    trim = int(trim_frames)
    frames = _frames_from_video_t(int(video.shape[2]))
    if index == 0:
        trim = 0
    else:
        if trim < 0:
            raise ValueError("MiniMax H3 Disk Join: trim_frames cannot be negative.")
        if trim > 0 and _steps_for_frames(trim) is None:
            raise ValueError(
                f"MiniMax H3 Disk Join: trim_frames={trim} is not on H3 grid."
            )
        if trim >= frames:
            raise ValueError("MiniMax H3 Disk Join: trim removes the entire clip.")

    _ensure_data_file(data_path)
    with open(data_path, "ab", buffering=0) as f:
        video_spec = _write_tensor_raw(f, video)
        audio_spec = _write_tensor_raw(f, audio)
        segment_end = int(f.tell())
        f.flush()
        os.fsync(f.fileno())

    return {
        "index": int(index),
        "validated": bool(validated),
        "frames": int(frames),
        "trim_frames": int(trim),
        "video": video_spec,
        "audio": audio_spec,
        "segment_end": segment_end,
    }, _geometry(video, audio)


def _truncate_chain(data_path, manifest_path, manifest, index):
    """
    Keep clips [0:index), discard index and everything after it.
    Manifest prefix is committed first, so an interruption can only roll back.
    """
    index = max(0, int(index))
    old = [dict(x) for x in manifest.get("segments", [])]
    prefix = old[:index]
    truncate_at = _DATA_START if not prefix else _segment_end(prefix[-1])

    reduced = dict(manifest)
    reduced["segments"] = prefix
    if index == 0:
        reduced["geometry"] = None
    reduced["final_frame_count"] = _final_frame_count(prefix)
    reduced["updated_at"] = time.time()
    _write_json_atomic(manifest_path, reduced)

    _ensure_data_file(data_path)
    with open(data_path, "r+b") as f:
        f.truncate(truncate_at)
        f.flush()
        os.fsync(f.fileno())
    return reduced


class _LazyDiskLatent(dict):
    """Small LATENT-compatible proxy; tensors stay mmap-backed until consumed."""

    def __init__(self, data_path, desc):
        super().__init__()
        self.data_path = str(data_path)
        self.desc = dict(desc)

    def _samples(self):
        video = _load_segment_video(self.data_path, self.desc)
        audio = _load_segment_audio(self.data_path, self.desc)
        return comfy.nested_tensor.NestedTensor((video, audio))

    def get(self, key, default=None):
        if key == "samples":
            return self._samples()
        if key in ("noise_mask", "batch_index"):
            return default
        return default

    def __getitem__(self, key):
        if key == "samples":
            return self._samples()
        raise KeyError(key)

    def __contains__(self, key):
        return key == "samples"

    def keys(self):
        return ("samples",)

    def copy(self):
        # Compatibility with nodes that copy LATENT dictionaries.
        return {"samples": self._samples()}


def _proxy_at(data_path, manifest, index):
    segments = manifest.get("segments", [])
    index = int(index)
    if index < 0 or index >= len(segments):
        raise ValueError(
            f"MiniMax H3 Disk Join: cached clip {index + 1} is unavailable."
        )
    return _LazyDiskLatent(data_path, segments[index])


def _last_proxy(data_path, manifest):
    segments = manifest.get("segments", [])
    if not segments:
        raise ValueError("MiniMax H3 Disk Join: cache contains no clip.")
    return _proxy_at(data_path, manifest, len(segments) - 1)


def _manifest_for_first(owner_id, fps):
    data_path, manifest_path = _chain_paths(owner_id)
    manifest = _load_manifest_from_paths(data_path, manifest_path)
    if manifest is None:
        _new_data_file(data_path)
        manifest = {
            "version": CACHE_VERSION,
            "build": BUILD,
            "owner_id": str(owner_id),
            "fps": float(fps),
            "geometry": None,
            "segments": [],
            "final_frame_count": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        _write_json_atomic(manifest_path, manifest)
    return data_path, manifest_path, manifest


def _effective_state(previous_cache, run_mode, fps, unique_id):
    if previous_cache is not None:
        data_path, manifest_path, manifest = _load_manifest(previous_cache)
        mode = str(previous_cache.get("run_mode", run_mode))
        stop = bool(previous_cache.get("stop", False))
        index = int(previous_cache.get("next_index", len(manifest.get("segments", []))))
    else:
        owner = unique_id if unique_id is not None else "first_join"
        data_path, manifest_path, manifest = _manifest_for_first(owner, fps)
        mode = str(run_mode)
        stop = False
        index = 0

    if mode not in ("full_batch", "clip_by_clip"):
        mode = "full_batch"
    if abs(float(manifest.get("fps", fps)) - float(fps)) > 1e-6:
        raise ValueError(
            f"MiniMax H3 Disk Join: chain fps={manifest.get('fps')} but node fps={fps}."
        )
    return data_path, manifest_path, manifest, mode, stop, index


class MiniMaxH3MotionContextDiskJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"lazy": True}),
                "validated": ("BOOLEAN", {"default": False}),
                "run_mode": (
                    ["full_batch", "clip_by_clip"],
                    {"default": "full_batch"},
                ),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001},
                ),
            },
            "optional": {
                "previous_cache": (CACHE_TYPE,),
                "trim_frames": ("INT", {"forceInput": True, "lazy": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (CACHE_TYPE, "LATENT", "INT", "INT", "STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = (
        "cache",
        "cached_samples",
        "clip_count",
        "frame_count",
        "status",
        "cache_size_mb",
        "cache_path",
        "build",
    )
    FUNCTION = "join"
    CATEGORY = "MiniMax H3"

    def check_lazy_status(
        self,
        samples=None,
        trim_frames=None,
        validated=False,
        run_mode="full_batch",
        fps=24.0,
        previous_cache=None,
        unique_id=None,
    ):
        try:
            data_path, manifest_path, manifest, mode, stop, index = _effective_state(
                previous_cache, run_mode, fps, unique_id
            )
        except Exception:
            # Let execute surface the real error; request the minimum normal path.
            needed = []
            if samples is None:
                needed.append("samples")
            if previous_cache is not None and trim_frames is None:
                needed.append("trim_frames")
            return needed

        # In clip-by-clip mode, once the first candidate has been generated,
        # every later Disk Join becomes a metadata-only pass-through. Its
        # sampler/RAM branch is therefore never requested in this execution.
        if mode == "clip_by_clip" and stop:
            return []

        segments = manifest.get("segments", [])
        existing = index < len(segments)

        # A validated cached clip is immutable and needs no sampler branch.
        if bool(validated) and existing:
            return []

        needed = []
        if samples is None:
            needed.append("samples")
        if index > 0 and trim_frames is None:
            needed.append("trim_frames")
        return needed

    def join(
        self,
        samples=None,
        trim_frames=None,
        validated=False,
        run_mode="full_batch",
        fps=24.0,
        previous_cache=None,
        unique_id=None,
    ):
        data_path, manifest_path, manifest, mode, stop, index = _effective_state(
            previous_cache, run_mode, fps, unique_id
        )
        segments = [dict(x) for x in manifest.get("segments", [])]

        # Downstream of the first unvalidated clip in incremental mode:
        # preserve the same cache handle and do not touch any sampler input.
        if mode == "clip_by_clip" and stop:
            status = f"skipped after clip {len(segments)} (clip_by_clip)"
            handle = _make_handle(
                data_path, manifest_path, manifest, mode, stop=True, status=status
            )
            size = _cache_size_mb(data_path, manifest_path)
            return (
                handle,
                _last_proxy(data_path, manifest),
                len(segments),
                int(manifest.get("final_frame_count", 0)),
                status,
                size,
                str(data_path.parent),
                BUILD,
            )

        if index < 0 or index > len(segments):
            raise RuntimeError(
                f"MiniMax H3 Disk Join: invalid chain index {index}/{len(segments)}."
            )

        existing = index < len(segments)

        if bool(validated) and index > 0:
            if not all(bool(x.get("validated", False)) for x in segments[:index]):
                raise ValueError(
                    f"MiniMax H3 Disk Join: clip {index + 1} cannot be validated "
                    "before every previous clip is validated."
                )

        if bool(validated) and existing:
            # Commit/freeze the existing candidate without evaluating samples.
            if not bool(segments[index].get("validated", False)):
                segments[index]["validated"] = True
                manifest = dict(manifest)
                manifest["segments"] = segments
                manifest["build"] = BUILD
                manifest["updated_at"] = time.time()
                _write_json_atomic(manifest_path, manifest)
                status = f"clip {index + 1} validated from disk"
            else:
                status = f"clip {index + 1} validated (disk)"

        else:
            # OFF means candidate: every active execution rewrites this clip and
            # invalidates/truncates the entire downstream tail in one operation.
            if samples is None:
                raise RuntimeError("MiniMax H3 Disk Join: active clip needs samples.")
            trim = 0 if index == 0 else int(trim_frames if trim_frames is not None else 22)

            if existing or len(segments) > index:
                manifest = _truncate_chain(data_path, manifest_path, manifest, index)
                segments = [dict(x) for x in manifest.get("segments", [])]

            desc, geom = _append_segment(
                data_path,
                samples,
                index=index,
                trim_frames=trim,
                validated=bool(validated),
                manifest=manifest,
            )
            segments = [dict(x) for x in manifest.get("segments", [])] + [desc]
            manifest = dict(manifest)
            manifest["geometry"] = geom if manifest.get("geometry") is None else manifest["geometry"]
            manifest["segments"] = segments
            manifest["final_frame_count"] = _final_frame_count(segments)
            manifest["build"] = BUILD
            manifest["updated_at"] = time.time()
            _write_json_atomic(manifest_path, manifest)
            status = (
                f"clip {index + 1} validated + cached"
                if bool(validated)
                else f"clip {index + 1} candidate cached"
            )

        # The first OFF clip terminates only the current incremental execution.
        stop_out = bool(mode == "clip_by_clip" and not bool(validated))
        handle = _make_handle(
            data_path, manifest_path, manifest, mode, stop=stop_out, status=status,
            next_index=index + 1,
        )
        size = _cache_size_mb(data_path, manifest_path)

        _LOG.info(
            "H3 Disk Join: mode=%s clip=%d validated=%s stop=%s clips=%d frames=%d cache=%.1f MB",
            mode,
            index + 1,
            bool(validated),
            stop_out,
            len(manifest.get("segments", [])),
            int(manifest.get("final_frame_count", 0)),
            size,
        )

        return (
            handle,
            _proxy_at(data_path, manifest, index),
            len(manifest.get("segments", [])),
            int(manifest.get("final_frame_count", 0)),
            status,
            size,
            str(data_path.parent),
            BUILD,
        )


# -----------------------------------------------------------------------------
# Final streaming decode - minimum RAM path: one seam pair at a time.
# -----------------------------------------------------------------------------


def _build_pair_video(data_path, prev_desc, curr_desc):
    prev_v = _load_segment_video(data_path, prev_desc)
    next_v = _load_segment_video(data_path, curr_desc)

    if tuple(prev_v.shape[:2]) != tuple(next_v.shape[:2]) or tuple(prev_v.shape[3:]) != tuple(next_v.shape[3:]):
        raise ValueError("Disk Final Decode: video latent geometry mismatch.")

    previous_frames = _frames_from_video_t(int(prev_v.shape[2]))
    next_frames = _frames_from_video_t(int(next_v.shape[2]))
    trim = int(curr_desc["trim_frames"])
    video_trim_t = 0 if trim == 0 else _steps_for_frames(trim)
    if trim > 0 and video_trim_t is None:
        raise ValueError(f"Disk Final Decode: trim_frames={trim} is not on H3 grid.")
    if int(video_trim_t) >= int(next_v.shape[2]):
        raise ValueError("Disk Final Decode: trim removes entire continuation.")

    warmup_video_t = 5 if int(video_trim_t) >= 7 else 0
    decode_start_t = int(video_trim_t) - int(warmup_video_t)
    warmup_start_frames = _pixel_frames(decode_start_t) if decode_start_t > 0 else 0
    warmup_frames = trim - warmup_start_frames
    if warmup_frames < 0:
        raise RuntimeError("Disk Final Decode: negative VAE warm-up.")

    if (int(prev_v.shape[2]) - decode_start_t) % 5 != 0:
        raise RuntimeError("Disk Final Decode: H3 temporal phase mismatch.")

    chain = torch.cat((prev_v, next_v[:, :, decode_start_t:, :, :]), dim=2)
    decode_frames = _frames_from_video_t(int(chain.shape[2]))
    expected = previous_frames + next_frames - warmup_start_frames
    if decode_frames != expected:
        raise RuntimeError(
            f"Disk Final Decode: pair decode frames {decode_frames} != {expected}."
        )

    meta = {
        "previous_frames": int(previous_frames),
        "next_frames": int(next_frames),
        "trim_frames": int(trim),
        "warmup_frames": int(warmup_frames),
        "continued_frames": int(next_frames - trim),
        "decode_frames": int(decode_frames),
    }
    return chain, meta


def _decode_pair_video(vae, chain, meta):
    decoded = vae.decode(chain)
    if decoded.ndim == 5:
        decoded = decoded.reshape(
            -1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1]
        )
    if int(decoded.shape[0]) != int(meta["decode_frames"]):
        raise RuntimeError(
            f"Disk Final Decode: VAE returned {decoded.shape[0]}, "
            f"expected {meta['decode_frames']}."
        )

    prev_frames = int(meta["previous_frames"])
    warmup = int(meta["warmup_frames"])
    shift = _auto_early_seam_shift(
        decoded,
        previous_frames=prev_frames,
        warmup_frames=warmup,
        max_early=2,
    )
    start = prev_frames + warmup + int(shift)
    end = start + int(meta["continued_frames"])
    if start < 0 or end > int(decoded.shape[0]):
        raise RuntimeError("Disk Final Decode: seam crop lies outside decoded pair.")

    previous_raw = decoded[:prev_frames]
    current_raw = decoded[start:end]
    return decoded, previous_raw, current_raw, int(shift)


def _correct_current_segment(previous_raw, current_raw):
    tail_n = min(4, int(previous_raw.shape[0]))
    if tail_n < 1:
        return current_raw
    pair = torch.cat((previous_raw[-tail_n:], current_raw), dim=0)
    pair = _photometric_match_segment(
        pair,
        seam_frame=tail_n,
        detect_window=4,
        end_frame=int(pair.shape[0]),
    )
    return pair[tail_n:]


def _find_ffmpeg():
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return str(exe)
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError("MiniMax H3 Disk Final Decode: ffmpeg executable not found.")


def _next_output_path(output_dir, prefix, extension):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(prefix)
    first = output_dir / f"{stem}.{extension}"
    if not first.exists():
        return first
    for i in range(1, 1000000):
        p = output_dir / f"{stem}_{i:05d}.{extension}"
        if not p.exists():
            return p
    raise RuntimeError("Disk Final Decode: could not allocate output filename.")


def _replace_output_from_preview(preview_path, output_dir, filename_prefix):
    """Atomically update the clip-by-clip autosave from the current full preview.

    The progressive preview is already a complete H.264/AAC MP4 containing the
    validated prefix plus the current candidate. Reusing it avoids any second
    VAE decode or sampling pass just to persist the current sequence.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{_safe_name(filename_prefix)}.mp4"
    tmp = output_dir / f".{destination.name}.{uuid.uuid4().hex[:10]}.tmp"
    try:
        shutil.copy2(Path(preview_path), tmp)
        os.replace(tmp, destination)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return destination


def _start_video_encoder(ffmpeg, temp_video, width, height, fps, codec, crf, preset, log_path):
    if str(codec) == "H.265 / HEVC":
        enc = ["-c:v", "libx265", "-preset", str(preset), "-crf", str(int(crf)), "-pix_fmt", "yuv420p"]
    elif str(codec) == "FFV1 lossless":
        enc = ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp"]
    else:
        enc = ["-c:v", "libx264", "-preset", str(preset), "-crf", str(int(crf)), "-pix_fmt", "yuv420p"]

    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s:v", f"{int(width)}x{int(height)}",
        "-r", f"{float(fps):.9f}",
        "-i", "pipe:0",
        "-an",
        *enc,
        str(temp_video),
    ]
    log_f = open(log_path, "wb")
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_f
    )
    return proc, log_f


def _write_image_frames(proc, images, batch_frames=8):
    if proc.stdin is None:
        raise RuntimeError("Disk Final Decode: ffmpeg stdin is closed.")
    n = int(images.shape[0])
    for i in range(0, n, int(batch_frames)):
        part = images[i:i + int(batch_frames), ..., :3]
        part = (
            part.detach().float().clamp(0.0, 1.0)
            .mul(255.0).add_(0.5).to(torch.uint8)
            .cpu().contiguous()
        )
        proc.stdin.write(part.numpy().tobytes(order="C"))
        del part


def _finish_process(proc, log_f, log_path, label):
    if proc.stdin is not None and not proc.stdin.closed:
        proc.stdin.close()
    code = proc.wait()
    log_f.close()
    if code != 0:
        tail = ""
        try:
            tail = Path(log_path).read_bytes()[-12000:].decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"{label} failed with code {code}.\n{tail}")


def _decode_audio_latent(audio_vae, latent, frames, fps):
    waveform = audio_vae.decode(latent).movedim(-1, 1)
    std = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    waveform = waveform / std
    sr = int(
        getattr(
            audio_vae,
            "audio_sample_rate_output",
            getattr(audio_vae, "audio_sample_rate", 32000),
        )
    )
    return _audio_exact_frames(
        {"waveform": waveform, "sample_rate": sr}, int(frames), float(fps)
    )


def _decode_single_audio(data_path, desc, audio_vae, fps):
    latent = _load_segment_audio(data_path, desc)
    return _decode_audio_latent(audio_vae, latent, int(desc["frames"]), fps)


def _decode_pair_audio(data_path, prev_desc, curr_desc, audio_vae, fps, seam_shift):
    prev_a = _load_segment_audio(data_path, prev_desc)
    next_a = _load_segment_audio(data_path, curr_desc)

    previous_frames = int(prev_desc["frames"])
    next_frames = int(curr_desc["frames"])
    trim = int(curr_desc["trim_frames"])
    video_trim_t = 0 if trim == 0 else _steps_for_frames(trim)
    if trim > 0 and video_trim_t is None:
        raise ValueError("Disk Final Decode audio: invalid H3 trim grid.")

    warmup_video_t = 5 if int(video_trim_t) >= 7 else 0
    decode_start_t = int(video_trim_t) - int(warmup_video_t)
    warmup_start_frames = _pixel_frames(decode_start_t) if decode_start_t > 0 else 0
    warmup_frames = trim - warmup_start_frames

    decode_frames = previous_frames + next_frames - warmup_start_frames
    target_audio_t = _audio_t_for_frames(decode_frames)
    audio_start_t = int(prev_a.shape[-1]) + int(next_a.shape[-1]) - int(target_audio_t)
    if audio_start_t < 0 or audio_start_t >= int(next_a.shape[-1]):
        raise RuntimeError("Disk Final Decode audio: invalid warm-up start.")

    chain_audio = torch.cat((prev_a, next_a[..., audio_start_t:]), dim=-1)
    decoded = _decode_audio_latent(audio_vae, chain_audio, decode_frames, fps)
    w = decoded["waveform"]
    sr = int(decoded["sample_rate"])

    prev_n = int(round(float(previous_frames) / float(fps) * sr))
    effective_warm = max(0, int(warmup_frames) + int(seam_shift))
    warm_n = int(round(float(effective_warm) / float(fps) * sr))
    cut_b = prev_n + warm_n
    if cut_b >= int(w.shape[-1]):
        raise RuntimeError("Disk Final Decode audio: warm-up removes continuation.")

    pair = {
        "waveform": torch.cat((w[..., :prev_n], w[..., cut_b:]), dim=-1),
        "sample_rate": sr,
    }
    final_frames = previous_frames + next_frames - trim
    pair = _audio_exact_frames(pair, final_frames, fps)
    return pair, previous_frames, next_frames - trim


def _declick_segment(previous_tail, current, sample_rate, milliseconds=5.0):
    if previous_tail is None or int(previous_tail.shape[-1]) < 2 or int(current.shape[-1]) < 2:
        return current
    n = int(round(float(milliseconds) * 0.001 * int(sample_rate)))
    n = max(2, min(n, int(current.shape[-1])))
    out = current.clone()
    prev2 = previous_tail[..., -2]
    prev1 = previous_tail[..., -1]
    first = out[..., 0]
    target = prev1 + (prev1 - prev2)
    correction = first - target
    k = torch.arange(n, device=out.device, dtype=out.dtype)
    decay = 0.5 * (
        1.0 + torch.cos(
            torch.tensor(math.pi, device=out.device, dtype=out.dtype)
            * k / float(n - 1)
        )
    )
    out[..., :n] = out[..., :n] - correction.unsqueeze(-1) * decay
    return out


def _fit_audio_segment_to_cumulative(wave, target_total, written_total):
    need = max(0, int(target_total) - int(written_total))
    if int(wave.shape[-1]) > need:
        wave = wave[..., :need]
    elif int(wave.shape[-1]) < need:
        wave = torch.nn.functional.pad(wave, (0, need - int(wave.shape[-1])))
    return wave


def _write_audio_raw(file_obj, wave):
    x = wave[0].detach().float().transpose(0, 1).cpu().contiguous()
    file_obj.write(x.numpy().astype("float32", copy=False).tobytes(order="C"))


def _mux_final(ffmpeg, temp_video, raw_audio, output_path, sr, channels, codec, audio_bitrate, log_path):
    audio_args = ["-c:a", "flac"] if str(codec) == "FFV1 lossless" else ["-c:a", "aac", "-b:a", str(audio_bitrate)]
    cmd = [
        ffmpeg, "-y",
        "-i", str(temp_video),
        "-f", "f32le",
        "-ar", str(int(sr)),
        "-ac", str(int(channels)),
        "-i", str(raw_audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        *audio_args,
    ]
    if str(output_path).lower().endswith(".mp4"):
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(output_path))

    with open(log_path, "wb") as log_f:
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=log_f)
    if p.returncode != 0:
        tail = ""
        try:
            tail = Path(log_path).read_bytes()[-12000:].decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"Disk Final Decode mux failed with code {p.returncode}.\n{tail}")


def _comfy_media_item(path, fps, media_type):
    """
    Build a normal ComfyUI /view media descriptor.
    media_type must be 'temp' or 'output'.
    """
    path = Path(path).resolve()

    if folder_paths is None:
        return {
            "filename": path.name,
            "subfolder": "",
            "type": str(media_type),
            "format": "video/mp4",
            "frame_rate": float(fps),
        }

    if str(media_type) == "temp":
        root = Path(folder_paths.get_temp_directory()).resolve()
    else:
        root = Path(folder_paths.get_output_directory()).resolve()

    try:
        rel = path.relative_to(root)
        subfolder = "" if str(rel.parent) == "." else str(rel.parent).replace("\\", "/")
        filename = rel.name
    except Exception:
        # Custom output directories are not served by /view. The caller should
        # publish a temp preview copy first.
        subfolder = ""
        filename = path.name

    return {
        "filename": filename,
        "subfolder": subfolder,
        "type": str(media_type),
        "format": "video/mp4",
        "frame_rate": float(fps),
    }


def _preview_temp_path(unique_id):
    if folder_paths is not None:
        root = Path(folder_paths.get_temp_directory()).resolve()
    else:
        root = _ensure_cache_root() / "_preview"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"h3_motion_preview_{_safe_name(unique_id)}.mp4"


def _publish_full_preview(output_path, unique_id):
    """
    Put the final file under ComfyUI temp so the in-node browser player always
    has a /view-compatible URL, including when output_directory is custom.
    Only one preview file per Final Decode node is kept.
    """
    src = Path(output_path).resolve()
    dst = _preview_temp_path(unique_id)
    if dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass

    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)
    return dst



# -----------------------------------------------------------------------------
# v12.5 - progressive FULL decoded preview cache
# -----------------------------------------------------------------------------

def _decoded_preview_cache_path(data_path):
    return Path(data_path).with_suffix(".preview.mp4")


def _validated_prefix_count(segments):
    n = 0
    for desc in segments:
        if not bool(desc.get("validated", False)):
            break
        n += 1
    return n


def _latent_payload_end(desc):
    a = desc["audio"]
    return int(a["offset"]) + int(a["nbytes"])


def _write_blob_raw(file_obj, source_path):
    source_path = Path(source_path)
    spec = {
        "offset": int(file_obj.tell()),
        "nbytes": int(source_path.stat().st_size),
    }
    with open(source_path, "rb") as src:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            file_obj.write(chunk)
    return spec


def _copy_blob_to_file(data_path, spec, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    remaining = int(spec["nbytes"])
    with open(data_path, "rb") as src, open(destination, "wb") as dst:
        src.seek(int(spec["offset"]))
        while remaining > 0:
            chunk = src.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise IOError("H3 decoded-render blob is truncated.")
            dst.write(chunk)
            remaining -= len(chunk)


def _cache_candidate_render(
    data_path,
    manifest_path,
    manifest,
    clip_index,
    rendered_mp4,
):
    segments = [dict(x) for x in manifest.get("segments", [])]
    idx = int(clip_index)
    if idx != len(segments) - 1:
        raise RuntimeError(
            "H3 progressive preview can cache only the current tail candidate."
        )

    desc = dict(segments[idx])
    latent_end = int(desc.get("latent_end", _latent_payload_end(desc)))

    with open(data_path, "r+b", buffering=0) as f:
        f.truncate(latent_end)
        f.seek(latent_end)
        render_spec = _write_blob_raw(f, rendered_mp4)
        segment_end = int(f.tell())
        f.flush()
        os.fsync(f.fileno())

    desc["latent_end"] = latent_end
    desc["decoded_mp4_blob"] = render_spec
    desc["segment_end"] = segment_end
    segments[idx] = desc

    updated = dict(manifest)
    updated["segments"] = segments
    updated["build"] = BUILD
    updated["updated_at"] = time.time()
    _write_json_atomic(manifest_path, updated)
    return updated, desc


def _encode_corrected_segment_mp4(
    ffmpeg,
    video,
    audio,
    fps,
    target_path,
    token,
):
    root = _ensure_cache_root()
    video_only = root / f"_{token}_video.mp4"
    raw_audio = root / f"_{token}_audio.f32le"
    video_log = root / f"_{token}_video.log"
    mux_log = root / f"_{token}_mux.log"
    proc = None
    log_f = None

    try:
        h, w = int(video.shape[1]), int(video.shape[2])
        proc, log_f = _start_video_encoder(
            ffmpeg,
            video_only,
            w,
            h,
            fps,
            "H.264",
            17,
            "ultrafast",
            video_log,
        )
        _write_image_frames(proc, video)
        _finish_process(proc, log_f, video_log, "H3 progressive cache encoder")
        proc = None
        log_f = None

        wave = audio["waveform"]
        with open(raw_audio, "wb") as af:
            _write_audio_raw(af, wave)

        if Path(target_path).exists():
            Path(target_path).unlink()

        _mux_final(
            ffmpeg,
            video_only,
            raw_audio,
            target_path,
            int(audio["sample_rate"]),
            int(wave.shape[1]),
            "H.264",
            "192k",
            mux_log,
        )
    finally:
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
        if log_f is not None:
            try:
                log_f.close()
            except Exception:
                pass
        for p in (video_only, raw_audio, video_log, mux_log):
            try:
                if Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass


def _render_one_final_segment(
    data_path,
    segments,
    index,
    vae,
    audio_vae,
    fps,
    progress=None,
):
    i = int(index)
    curr = segments[i]

    if i == 0:
        v = _load_segment_video(data_path, curr)
        video = vae.decode(v)
        if progress is not None:
            progress.advance()
        if video.ndim == 5:
            video = video.reshape(
                -1, video.shape[-3], video.shape[-2], video.shape[-1]
            )
        expected = int(curr["frames"])
        if int(video.shape[0]) != expected:
            raise RuntimeError(
                f"H3 progressive preview: clip 1 decoded {video.shape[0]}, "
                f"expected {expected}."
            )
        audio = _decode_single_audio(data_path, curr, audio_vae, fps)
        if progress is not None:
            progress.advance()
        return video, audio, 0

    prev = segments[i - 1]
    chain, meta = _build_pair_video(data_path, prev, curr)
    decoded, previous_raw, current_raw, shift = _decode_pair_video(
        vae, chain, meta
    )
    if progress is not None:
        progress.advance()
    current_video = _correct_current_segment(previous_raw, current_raw)

    pair_audio, prev_frames, curr_frames = _decode_pair_audio(
        data_path,
        prev,
        curr,
        audio_vae,
        fps,
        int(shift),
    )
    if progress is not None:
        progress.advance()
    sr = int(pair_audio["sample_rate"])
    wave = pair_audio["waveform"]

    prev_samples = int(round(float(prev_frames) / float(fps) * sr))
    previous_audio = wave[..., :prev_samples]
    current_audio = wave[..., prev_samples:]

    previous_tail = (
        previous_audio[..., -2:].detach().clone()
        if int(previous_audio.shape[-1]) >= 2
        else None
    )
    current_audio = _declick_segment(
        previous_tail,
        current_audio,
        sr,
        5.0,
    )

    wanted = int(round(float(curr_frames) / float(fps) * sr))
    current_audio = _fit_audio_segment_to_cumulative(
        current_audio, wanted, 0
    )

    audio = {
        "waveform": current_audio,
        "sample_rate": sr,
    }

    del chain, decoded, previous_raw, current_raw, pair_audio, wave, previous_audio
    return current_video, audio, int(shift)


def _concat_mp4_stream_copy(ffmpeg, inputs, output_path, log_path):
    inputs = [Path(p) for p in inputs if p is not None and Path(p).exists()]
    if not inputs:
        raise ValueError("H3 progressive preview concat has no input.")

    if len(inputs) == 1:
        if Path(output_path).resolve() != inputs[0].resolve():
            shutil.copy2(inputs[0], output_path)
        return

    list_path = Path(log_path).with_suffix(".concat.txt")
    try:
        lines = []
        for p in inputs:
            escaped = str(p.resolve()).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cmd = [
            ffmpeg, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-map", "0:v:0",
            "-map", "0:a:0",
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ]
        with open(log_path, "wb") as log_f:
            p = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=log_f,
            )
        if p.returncode != 0:
            tail = ""
            try:
                tail = Path(log_path).read_bytes()[-12000:].decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                pass
            raise RuntimeError(
                f"H3 progressive preview stream-copy concat failed "
                f"with code {p.returncode}.\n{tail}"
            )
    finally:
        try:
            if list_path.exists():
                list_path.unlink()
        except Exception:
            pass


def _append_render_to_committed_preview(
    ffmpeg,
    committed_path,
    segment_path,
    token,
):
    committed_path = Path(committed_path)
    segment_path = Path(segment_path)

    if not committed_path.exists():
        shutil.copy2(segment_path, committed_path)
        return

    temp = committed_path.with_name(
        committed_path.stem + f"_{token}.tmp.mp4"
    )
    log = committed_path.with_name(
        committed_path.stem + f"_{token}.append.log"
    )
    try:
        _concat_mp4_stream_copy(
            ffmpeg,
            [committed_path, segment_path],
            temp,
            log,
        )
        os.replace(temp, committed_path)
    finally:
        for p in (temp, log):
            try:
                if Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass


def _sync_committed_preview(
    data_path,
    manifest_path,
    manifest,
    target_count,
    vae,
    audio_vae,
    fps,
    ffmpeg,
    token,
):
    target = max(0, int(target_count))
    segments = [dict(x) for x in manifest.get("segments", [])]
    committed_path = _decoded_preview_cache_path(data_path)
    current_count = int(manifest.get("preview_committed_count", 0))

    rebuild = (
        current_count > target
        or (current_count > 0 and not committed_path.exists())
    )
    if rebuild:
        try:
            if committed_path.exists():
                committed_path.unlink()
        except OSError:
            pass
        current_count = 0

    root = _ensure_cache_root()

    for i in range(current_count, target):
        desc = segments[i]
        segment_tmp = root / f"_{token}_commit_{i:04d}.mp4"
        try:
            blob = desc.get("decoded_mp4_blob")
            if blob is not None:
                _copy_blob_to_file(data_path, blob, segment_tmp)
            else:
                # Legacy/recovery path only.
                video, audio, _ = _render_one_final_segment(
                    data_path,
                    segments,
                    i,
                    vae,
                    audio_vae,
                    fps,
                )
                _encode_corrected_segment_mp4(
                    ffmpeg,
                    video,
                    audio,
                    fps,
                    segment_tmp,
                    f"{token}_bootstrap_{i}",
                )
                del video, audio

            _append_render_to_committed_preview(
                ffmpeg,
                committed_path,
                segment_tmp,
                f"{token}_{i}",
            )
        finally:
            try:
                if segment_tmp.exists():
                    segment_tmp.unlink()
            except OSError:
                pass

    updated = dict(manifest)
    updated["preview_committed_count"] = target
    updated["build"] = BUILD
    updated["updated_at"] = time.time()
    _write_json_atomic(manifest_path, updated)
    return updated, committed_path



def _export_live_candidate_preview(
    data_path,
    manifest_path,
    manifest,
    segments,
    vae,
    audio_vae,
    fps,
    ffmpeg,
    unique_id,
    progress=None,
):
    segments = [dict(x) for x in segments]
    if not segments:
        raise ValueError("H3 progressive preview: empty chain.")

    validated_count = _validated_prefix_count(segments)
    token = f"v125_{_safe_name(unique_id)}_{uuid.uuid4().hex[:8]}"
    preview_path = _preview_temp_path(unique_id)
    root = _ensure_cache_root()

    # Commit already validated clips into ONE persistent full preview cache.
    manifest, committed_path = _sync_committed_preview(
        data_path,
        manifest_path,
        manifest,
        validated_count,
        vae,
        audio_vae,
        fps,
        ffmpeg,
        token,
    )
    segments = [dict(x) for x in manifest.get("segments", [])]

    # Everything currently present is validated: show cache directly.
    if validated_count >= len(segments):
        if not committed_path.exists():
            raise RuntimeError(
                "H3 progressive preview: validated preview cache is missing."
            )
        if preview_path.exists():
            try:
                preview_path.unlink()
            except OSError:
                pass
        try:
            os.link(committed_path, preview_path)
        except Exception:
            shutil.copy2(committed_path, preview_path)

        # Small compatibility output only: recover last frame from the last
        # segment. This does not affect the full preview cache itself.
        last_i = len(segments) - 1
        video, audio, shift = _render_one_final_segment(
            data_path,
            segments,
            last_i,
            vae,
            audio_vae,
            fps,
            progress=progress,
        )
        last_frame = video[-1:].detach().cpu().clone()
        del video, audio

        return (
            preview_path,
            last_frame,
            int(manifest.get("final_frame_count", 0)),
            int(shift),
            None,
            "committed_full_cache",
        )

    candidate_index = validated_count
    if candidate_index != len(segments) - 1:
        raise RuntimeError(
            "H3 progressive preview expects one unvalidated tail candidate."
        )

    # Normal VAE cost:
    #   clip 1 alone, otherwise ONLY previous + current pair.
    current_video, current_audio, seam_shift = _render_one_final_segment(
        data_path,
        segments,
        candidate_index,
        vae,
        audio_vae,
        fps,
        progress=progress,
    )

    candidate_mp4 = root / f"_{token}_candidate.mp4"
    concat_log = root / f"_{token}_preview.log"

    try:
        _encode_corrected_segment_mp4(
            ffmpeg,
            current_video,
            current_audio,
            fps,
            candidate_mp4,
            token,
        )

        # Save candidate's already decoded/corrected render in .h3cache.
        manifest, _ = _cache_candidate_render(
            data_path,
            manifest_path,
            manifest,
            candidate_index,
            candidate_mp4,
        )

        if preview_path.exists():
            try:
                preview_path.unlink()
            except OSError:
                pass

        # FULL VIDEO preview: cached validated prefix + current candidate.
        if committed_path.exists() and validated_count > 0:
            _concat_mp4_stream_copy(
                ffmpeg,
                [committed_path, candidate_mp4],
                preview_path,
                concat_log,
            )
        else:
            shutil.copy2(candidate_mp4, preview_path)

        last_frame = current_video[-1:].detach().cpu().clone()
        total_frames = int(manifest.get("final_frame_count", 0))

        _LOG.info(
            "H3 v12.5 FULL preview: cached validated clips=%d, candidate=%d, "
            "shift=%d; VAE decoded only %s",
            validated_count,
            candidate_index + 1,
            int(seam_shift),
            (
                "clip 1"
                if candidate_index == 0
                else f"pair {candidate_index}->{candidate_index + 1}"
            ),
        )

        return (
            preview_path,
            last_frame,
            total_frames,
            int(seam_shift),
            None if candidate_index == 0 else int(candidate_index),
            "full_cached_prefix_plus_candidate",
        )

    finally:
        for p in (candidate_mp4, concat_log):
            try:
                if Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass



class MiniMaxH3MotionContextDiskFinalDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache": (CACHE_TYPE,),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "filename_prefix": ("STRING", {"default": "MiniMax_H3_cached"}),
                "output_directory": ("STRING", {"default": ""}),
                "codec": (["H.264", "H.265 / HEVC", "FFV1 lossless"], {"default": "H.264"}),
                "crf": ("INT", {"default": 17, "min": 0, "max": 51, "step": 1}),
                "preset": (["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"], {"default": "fast"}),
                "audio_bitrate": (["128k", "192k", "256k", "320k"], {"default": "192k"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "export"
    CATEGORY = "MiniMax H3"
    OUTPUT_NODE = True

    def export(
        self,
        cache,
        vae,
        audio_vae,
        fps,
        filename_prefix,
        output_directory,
        codec,
        crf,
        preset,
        audio_bitrate,
        unique_id=None,
    ):
        data_path, manifest_path, manifest = _load_manifest(cache)
        if abs(float(manifest["fps"]) - float(fps)) > 1e-6:
            raise ValueError(
                f"Disk Final Decode fps is {manifest['fps']}, export requested {fps}."
            )
        segments = [dict(x) for x in manifest.get("segments", [])]
        if not segments:
            raise ValueError("Disk Final Decode: empty cache.")

        ffmpeg = _find_ffmpeg()

        if str(output_directory).strip():
            out_dir = Path(str(output_directory).strip()).expanduser().resolve()
        elif folder_paths is not None:
            out_dir = Path(folder_paths.get_output_directory()).resolve()
        else:
            out_dir = (Path.cwd() / "output").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # clip_by_clip keeps its fast progressive-preview path, but now also
        # persists that complete current sequence after every rendered clip.
        # The already encoded preview is reused directly: no extra sampling and
        # no second VAE decode are performed for the autosave.
        effective_mode = str(cache.get("run_mode", "full_batch")) if isinstance(cache, dict) else "full_batch"
        if effective_mode == "clip_by_clip":
            progress = _FinalDecodeNativeProgress(unique_id, total=6)
            (
                preview_path,
                last_frame,
                preview_frames,
                seam_shift,
                previous_clip,
                preview_cache_mode,
            ) = _export_live_candidate_preview(
                data_path=data_path,
                manifest_path=manifest_path,
                manifest=manifest,
                segments=segments,
                vae=vae,
                audio_vae=audio_vae,
                fps=float(fps),
                ffmpeg=ffmpeg,
                unique_id=unique_id,
                progress=progress,
            )
            progress.advance()  # preview encode/cache/concat completed

            # Keep one continuously updated real file in the requested output
            # directory.  Re-rendering the same candidate replaces it atomically,
            # so clip-by-clip testing never creates a pile of numbered files.
            autosave_path = _replace_output_from_preview(
                preview_path, out_dir, filename_prefix
            )
            progress.advance()

            total_frames = int(manifest.get("final_frame_count", 0))
            total_duration = float(total_frames / float(fps))
            size = _cache_size_mb(data_path, manifest_path)
            item = _comfy_media_item(preview_path, fps, "temp")
            progress.finish()
            status_shift = (
                f"full_preview_cached_{len(segments)}_clips_shift_{int(seam_shift)}"
            )

            return {
                "ui": {
                    "h3_video": [item],
                    "h3_preview_info": [{
                        "mode": "clip_by_clip",
                        "clip": int(len(segments)),
                        "previous_clip": (
                            None if previous_clip is None else int(previous_clip)
                        ),
                        "seam_shift": int(seam_shift),
                        "cache_mode": str(preview_cache_mode),
                        "preview_frames": int(preview_frames),
                        "total_clips": int(len(segments)),
                        "autosave_path": str(autosave_path),
                    }],
                },
                "result": (),
            }

        extension = "mkv" if str(codec) == "FFV1 lossless" else "mp4"
        output_path = _next_output_path(out_dir, filename_prefix, extension)
        expected_frames = int(manifest["final_frame_count"])
        decode_units = max(1, len(segments) - 1)
        progress = _FinalDecodeNativeProgress(
            unique_id, total=4 + (2 * decode_units)
        )
        seam_shifts = {}
        written_frames = 0
        last_frame = None
        video_proc = None
        video_log_f = None

        # Temp artifacts are one set only and are always removed afterwards.
        temp_root = _ensure_cache_root()
        token = uuid.uuid4().hex[:10]
        temp_video = temp_root / f"_export_{token}_video.{extension}"
        raw_audio = temp_root / f"_export_{token}_audio.f32le"
        video_log = temp_root / f"_export_{token}_video.log"
        mux_log = temp_root / f"_export_{token}_mux.log"

        try:
            # VIDEO - strict constant-memory path: one clip for N=1, otherwise
            # exactly one adjacent pair per seam. Nothing accumulated in IMAGE.
            if len(segments) == 1:
                v = _load_segment_video(data_path, segments[0])
                decoded = vae.decode(v)
                progress.advance()
                if decoded.ndim == 5:
                    decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
                expected0 = int(segments[0]["frames"])
                if int(decoded.shape[0]) != expected0:
                    raise RuntimeError(
                        f"Disk Final Decode: VAE returned {decoded.shape[0]}, expected {expected0}."
                    )
                h, w = int(decoded.shape[1]), int(decoded.shape[2])
                video_proc, video_log_f = _start_video_encoder(
                    ffmpeg, temp_video, w, h, fps, codec, crf, preset, video_log
                )
                _write_image_frames(video_proc, decoded)
                written_frames = int(decoded.shape[0])
                last_frame = decoded[-1:].detach().cpu().clone()
                del decoded, v
            else:
                for i in range(1, len(segments)):
                    chain, meta = _build_pair_video(data_path, segments[i - 1], segments[i])
                    decoded, previous_raw, current_raw, shift = _decode_pair_video(vae, chain, meta)
                    progress.advance()
                    seam_shifts[i] = int(shift)

                    if video_proc is None:
                        h, w = int(previous_raw.shape[1]), int(previous_raw.shape[2])
                        video_proc, video_log_f = _start_video_encoder(
                            ffmpeg, temp_video, w, h, fps, codec, crf, preset, video_log
                        )
                        # First pair supplies clip 1 exactly once.
                        _write_image_frames(video_proc, previous_raw)
                        written_frames += int(previous_raw.shape[0])
                        last_frame = previous_raw[-1:].detach().cpu().clone()

                    current_out = _correct_current_segment(previous_raw, current_raw)
                    _write_image_frames(video_proc, current_out)
                    written_frames += int(current_out.shape[0])
                    last_frame = current_out[-1:].detach().cpu().clone()
                    del current_out, previous_raw, current_raw, decoded, chain

            if video_proc is None or video_log_f is None:
                raise RuntimeError("Disk Final Decode: encoder never started.")
            _finish_process(video_proc, video_log_f, video_log, "Disk Final Decode encoder")
            progress.advance()
            video_proc = None
            video_log_f = None

            if int(written_frames) != int(expected_frames):
                raise RuntimeError(
                    f"Disk Final Decode wrote {written_frames} frames, expected {expected_frames}."
                )

            # AUDIO - same pair-local policy, exact video seam shift, streaming
            # raw f32 to disk. Only the current pair waveform exists in RAM.
            written_samples = 0
            cumulative_frames = 0
            previous_tail = None
            sample_rate = None
            channels = None

            with open(raw_audio, "wb") as af:
                if len(segments) == 1:
                    audio0 = _decode_single_audio(data_path, segments[0], audio_vae, fps)
                    progress.advance()
                    sample_rate = int(audio0["sample_rate"])
                    wave0 = audio0["waveform"]
                    channels = int(wave0.shape[1])
                    cumulative_frames = int(segments[0]["frames"])
                    target = int(round(float(cumulative_frames) / float(fps) * sample_rate))
                    wave0 = _fit_audio_segment_to_cumulative(wave0, target, written_samples)
                    _write_audio_raw(af, wave0)
                    written_samples += int(wave0.shape[-1])
                    previous_tail = wave0[..., -2:].detach().clone()
                    del audio0, wave0
                else:
                    for i in range(1, len(segments)):
                        pair, prev_frames, curr_frames = _decode_pair_audio(
                            data_path,
                            segments[i - 1],
                            segments[i],
                            audio_vae,
                            fps,
                            int(seam_shifts.get(i, 0)),
                        )
                        progress.advance()
                        sr = int(pair["sample_rate"])
                        w = pair["waveform"]
                        if sample_rate is None:
                            sample_rate = sr
                            channels = int(w.shape[1])
                            prev_n = int(round(float(prev_frames) / float(fps) * sr))
                            first = w[..., :prev_n]
                            cumulative_frames = int(prev_frames)
                            target = int(round(float(cumulative_frames) / float(fps) * sr))
                            first = _fit_audio_segment_to_cumulative(first, target, written_samples)
                            _write_audio_raw(af, first)
                            written_samples += int(first.shape[-1])
                            previous_tail = first[..., -2:].detach().clone()
                            current = w[..., prev_n:]
                            del first
                        else:
                            if sr != sample_rate:
                                raise RuntimeError("Disk Final Decode: audio sample rate changed.")
                            prev_n = int(round(float(prev_frames) / float(fps) * sr))
                            current = w[..., prev_n:]

                        current = _declick_segment(previous_tail, current, sr, 5.0)
                        cumulative_frames += int(curr_frames)
                        target = int(round(float(cumulative_frames) / float(fps) * sr))
                        current = _fit_audio_segment_to_cumulative(current, target, written_samples)
                        _write_audio_raw(af, current)
                        written_samples += int(current.shape[-1])
                        previous_tail = current[..., -2:].detach().clone()
                        del pair, w, current

            if int(cumulative_frames) != int(expected_frames):
                raise RuntimeError(
                    f"Disk Final Decode audio represents {cumulative_frames} frames, "
                    f"expected {expected_frames}."
                )

            _mux_final(
                ffmpeg,
                temp_video,
                raw_audio,
                output_path,
                sample_rate,
                channels,
                codec,
                audio_bitrate,
                mux_log,
            )
            progress.advance()

            shifts_text = ",".join(
                f"{i}:{int(seam_shifts.get(i, 0))}" for i in range(1, len(segments))
            )
            size = _cache_size_mb(data_path, manifest_path)
            duration = float(expected_frames / float(fps))
            _LOG.info(
                "H3 Disk Final Decode: clips=%d frames=%d duration=%.3fs output=%s",
                len(segments), expected_frames, duration, output_path,
            )

            preview_path = _publish_full_preview(output_path, unique_id)
            item = _comfy_media_item(preview_path, fps, "temp")
            progress.finish()

            return {
                "ui": {
                    "h3_video": [item],
                    "h3_preview_info": [{
                        "mode": "full_batch",
                        "clip": int(len(segments)),
                        "preview_frames": int(expected_frames),
                        "total_clips": int(len(segments)),
                    }],
                },
                "result": (),
            }

        finally:
            if video_proc is not None:
                try:
                    if video_proc.stdin is not None:
                        video_proc.stdin.close()
                except Exception:
                    pass
                try:
                    video_proc.kill()
                except Exception:
                    pass
            if video_log_f is not None:
                try:
                    video_log_f.close()
                except Exception:
                    pass
            for p in (temp_video, raw_audio, video_log, mux_log):
                try:
                    if Path(p).exists():
                        Path(p).unlink()
                except Exception:
                    pass


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MotionContextDiskJoin": MiniMaxH3MotionContextDiskJoin,
    "MiniMaxH3MotionContextDiskFinalDecode": MiniMaxH3MotionContextDiskFinalDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MotionContextDiskJoin": "MiniMax H3 Motion Context Disk Join",
    "MiniMaxH3MotionContextDiskFinalDecode": "MiniMax H3 Extender Final Decode / Preview",
}
