"""ReCaVSR final-video upscale for the MiniMax H3 Extender.

Refine upscales the H3 latent before sampling, so it changes what the sampler
produces and therefore the clip cache. ReCaVSR is a pixel post-process: it reads
a finished video and writes a larger one. This module therefore runs after the
Final Decode node has written the final container, never touches the cache, and
leaves the original file beside the upscaled one.

The ReCaVSR custom node is optional. Nothing here imports it at module load, so
the Extender keeps working when the node is absent.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

RECAVSR_FOLDER = "recavsr"
RECAVSR_SUFFIX = "_recavsr"
COLOR_FIX_CHOICES = ("none", "adain", "wavelet")
DEFAULT_SCALE = 2.0
DEFAULT_SEED = 42
SCALE_MIN = 1.0
SCALE_MAX = 4.0
SEED_MIN = 0
SEED_MAX = 0xFFFFFFFF

# Loaded transformers stay resident for the session, exactly like a node run.
_MODEL_CACHE = {}


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "off", "no")
    return bool(value)


def normalize_settings(settings):
    """Clamp the stored ReCaVSR settings into one stable payload."""
    data = settings if isinstance(settings, dict) else {}
    try:
        scale = float(data.get("scale", DEFAULT_SCALE))
    except (TypeError, ValueError):
        scale = DEFAULT_SCALE
    if scale != scale:  # NaN
        scale = DEFAULT_SCALE
    scale = max(SCALE_MIN, min(SCALE_MAX, scale))
    color_fix = str(data.get("color_fix", "none") or "none").strip().lower()
    if color_fix not in COLOR_FIX_CHOICES:
        color_fix = "none"
    try:
        seed = int(data.get("seed", DEFAULT_SEED))
    except (TypeError, ValueError):
        seed = DEFAULT_SEED
    seed = max(SEED_MIN, min(SEED_MAX, seed))
    return {
        "enabled": _as_bool(data.get("enabled"), False),
        "scale": round(scale, 6),
        "color_fix": color_fix,
        "seed": seed,
    }


def _node_classes():
    """Return the installed ReCaVSR node classes, or (None, None).

    ComfyUI registers a custom node directory under its full path
    (``nodes.py`` ``load_custom_node``), so the package name is not stable. The
    directory name is.
    """
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        try:
            parts = Path(filename).resolve().parts
        except OSError:
            continue
        if "ComfyUI-ReCaVSR" not in parts:
            continue
        loader = getattr(module, "ReCaVSRLoader", None)
        upscale = getattr(module, "ReCaVSRUpscale", None)
        if loader is not None and upscale is not None:
            return loader, upscale
    return None, None


def transformer_names():
    """Every transformer the ReCaVSR loader can see, through folder_paths."""
    try:
        import folder_paths
    except Exception:
        return []
    try:
        names = folder_paths.get_filename_list(RECAVSR_FOLDER)
    except Exception:
        return []
    return [str(name) for name in names if str(name).endswith(".safetensors")]


def _pick_transformer(names):
    if not names:
        return None
    if "transformer.safetensors" in names:
        return "transformer.safetensors"
    for name in names:
        if "int8" in name.lower():
            return name
    return names[0]


def status():
    """Availability for the UI: node present, transformer found."""
    loader, _upscale = _node_classes()
    names = transformer_names() if loader is not None else []
    return {
        "node": loader is not None,
        "transformer": _pick_transformer(names),
        "count": len(names),
    }


def _first_output(result):
    if result is None:
        return None
    args = getattr(result, "args", None)
    if args is None:
        args = getattr(result, "result", None)
    if args is None:
        args = result
    try:
        return tuple(args)[0]
    except TypeError:
        return args


def _disk():
    """The Final Decode module owns output naming and the VIDEO wrapper."""
    from . import motion_context_disk

    return motion_context_disk


def _video_input(source):
    return _disk()._video_output_from_path(source)


def _model(loader_cls, transformer):
    key = (str(getattr(loader_cls, "__module__", "")), str(transformer))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    model = _first_output(loader_cls.execute(transformer))
    if model is None:
        raise ValueError("ReCaVSR loader returned no model.")
    _MODEL_CACHE[key] = model
    return model


def _stream_source_path(produced):
    getter = getattr(produced, "get_stream_source", None)
    if getter is None:
        raise ValueError("ReCaVSR returned no video file.")
    stream = getter()
    if isinstance(stream, (bytes, bytearray)):
        raise ValueError("ReCaVSR returned an in-memory video; a file-backed result is required.")
    path = Path(str(stream))
    if not path.is_file():
        raise ValueError(f"ReCaVSR reported a missing video file: {path}")
    return path


def _output_path(out_dir, filename_prefix):
    prefix = f"{filename_prefix}{RECAVSR_SUFFIX}"
    return _disk()._next_output_path(out_dir, prefix, "mp4")


def apply_recavsr(video_path, settings, *, out_dir, filename_prefix):
    """Upscale one finished final video with ReCaVSR.

    Returns the upscaled path, or ``video_path`` unchanged when the mode is off.
    The source file stays on disk beside the upscaled one.
    """
    resolved = normalize_settings(settings)
    if not resolved["enabled"]:
        return str(video_path)

    source = Path(str(video_path)).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"ReCaVSR upscale: the final video {source} does not exist.")

    loader_cls, upscale_cls = _node_classes()
    if loader_cls is None or upscale_cls is None:
        raise ValueError("Install ReCaVSR first.")

    transformer = _pick_transformer(transformer_names())
    if transformer is None:
        raise ValueError(
            "ReCaVSR is installed, but no transformer is visible in the recavsr model folder. "
            "Add 'recavsr: comfyui-models\\recavsr\\' to extra_model_paths.yaml and restart ComfyUI, "
            "or copy transformer.safetensors, prompt.safetensors and flashdecoder.safetensors into "
            "ComfyUI/models/recavsr."
        )

    model = _model(loader_cls, transformer)
    video = _video_input(source)
    try:
        produced = upscale_cls.execute(
            model,
            scale=float(resolved["scale"]),
            color_fix=str(resolved["color_fix"]),
            seed=int(resolved["seed"]),
            decoder="flashdecoder",
            kernels="fast",
            kv_offload=False,
            encoder="auto",
            video=video,
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"ReCaVSR upscale failed: {exc}") from exc

    produced_path = _stream_source_path(_first_output(produced))
    target = _output_path(out_dir, filename_prefix)
    shutil.move(str(produced_path), str(target))
    return str(target)
