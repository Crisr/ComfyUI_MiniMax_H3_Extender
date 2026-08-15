"""
MiniMax H3 Extender
===================

One horizontal, JS-driven sequence node that replaces the repeated
Ref2VA -> Motion Context -> Sampler -> Disk Join graph while keeping the
validated disk cache and the separate Final Decode / Preview node.

The node intentionally accepts an already-patched H3 MODEL. Sigma-shift,
LoRA, Spectrum or other model patches therefore remain external and compose
normally before the Extender.
"""

from __future__ import annotations

import json
import math
import secrets
import torch
import torchaudio
import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import node_helpers
from server import PromptServer

from .motion_context_ram import MiniMaxH3MotionContextRAM
from .motion_context_disk import (
    CACHE_TYPE,
    MiniMaxH3MotionContextDiskJoin,
    _cache_size_mb,
    _load_manifest_from_paths,
    _manifest_for_first,
    _truncate_chain,
)

BUILD = "minimax-h3-extender-v14.19-restore-clip-ui-state"
FPS = 24
AUDIO_LATENT_FPS = 40
CANVAS_MULTIPLE = 32
REF_IMAGE_SHORT_EDGE = 2048
MAX_CLIPS = 512
DEFAULT_DURATION = 10.0
DEFAULT_SEED_MAX = (1 << 53) - 1  # exact integer range in browser JS


def _align_frame_count(n: int) -> int:
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def _video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _duration_to_frames(seconds: float) -> int:
    raw = max(5, int(round(float(seconds) * FPS)))
    return _align_frame_count(raw)


def _empty_av_latent(width: int, height: int, frame_count: int):
    frame_count = _align_frame_count(frame_count)
    latent_t = _video_latent_t(frame_count)
    duration = frame_count / float(FPS)
    audio_t = round(duration * AUDIO_LATENT_FPS)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros(
        [1, 24, latent_t, int(height) // 16, int(width) // 16],
        device=device,
    )
    audio = torch.zeros(
        [1, 32, 2, int(audio_t)],
        device=device,
    )
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _resize(image, width: int, height: int):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(
        samples, int(width), int(height), "lanczos", "disabled"
    )
    return samples.movedim(1, -1)


def _encode_ref_audio(audio_vae, audio):
    """Encode one standalone Ref2VA audio reference exactly like ComfyUI native H3."""
    waveform = audio["waveform"]  # [B, C, L]
    sr = int(audio["sample_rate"])
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    latent = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return latent, int(latent.shape[-1])


def _prepare_shared_refs(
    vae,
    audio_vae,
    width: int,
    height: int,
    ref_image_size: str,
    refs,
    ref_audio=None,
):
    """Encode shared Ref2VA image refs plus one optional standalone audio ref."""
    active = []
    seen_gap = False
    for i, image in enumerate(refs, start=1):
        if image is None:
            seen_gap = True
            continue
        if seen_gap:
            raise ValueError(
                f"MiniMax H3 Extender: ref_{i} is connected after an empty "
                "reference slot. Connect shared refs contiguously from ref_1."
            )
        active.append(image)

    ref_items = []
    ref_blocks = []
    for img in active:
        h, w = int(img.shape[1]), int(img.shape[2])
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((int(width) * int(height)) / float(w * h)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / float(min(w, h)))

        tw = max(
            CANVAS_MULTIPLE,
            round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
        )
        th = max(
            CANVAS_MULTIPLE,
            round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
        )

        resized = _resize(img[:1], tw, th)
        z = vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append(
            {
                "kind": "image",
                "latent_h": th // 16,
                "latent_w": tw // 16,
                "latent": z,
            }
        )

    # Official Ref2VA presentation order is images first, then standalone audio.
    # The tokenizer assigns the standalone input the prompt label <Audio 1>.
    if ref_audio is not None:
        if audio_vae is None:
            raise ValueError(
                "MiniMax H3 Extender: ref_audio is connected but audio_vae is not. "
                "Connect the MiniMax H3 Audio VAE to audio_vae."
            )
        if not active:
            raise ValueError(
                "MiniMax H3 Extender: MiniMax H3 Ref2VA requires an audio "
                "reference to be used together with at least one image reference."
            )
        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, ref_audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append(
            {
                "kind": "audio",
                "ref_audio_t": int(ref_audio_t),
                "audio_latent": audio_latent,
            }
        )

    return ref_items, ref_blocks


def _make_ref2va_conditioning(
    clip,
    vae,
    prompt: str,
    width: int,
    height: int,
    frame_count: int,
    ref_items,
    ref_blocks,
):
    latent = _empty_av_latent(width, height, frame_count)
    tokens = clip.tokenize(str(prompt), minimax_ref_items=ref_items)
    cond = clip.encode_from_tokens_scheduled(tokens)
    if ref_blocks:
        cond = node_helpers.conditioning_set_values(
            cond, {"minimax_refs": ref_blocks}
        )
    return cond, latent


class _BasicGuider(comfy.samplers.CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})


def _sigmas(model, scheduler: str, steps: int, denoise: float):
    steps = max(1, int(steps))
    denoise = float(denoise)
    if denoise <= 0.0:
        return torch.FloatTensor([])
    total_steps = steps
    if denoise < 1.0:
        total_steps = max(steps, int(steps / denoise))
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"),
        str(scheduler),
        total_steps,
    ).cpu()
    return sigmas[-(steps + 1):]


def _sample_h3(model, conditioning, latent, seed: int, sampler_name: str, scheduler: str, steps: int, denoise: float):
    if int(steps) < 1:
        raise ValueError("MiniMax H3 Extender: steps must be >= 1.")

    guider = _BasicGuider(model)
    guider.set_conds(conditioning)
    sampler = comfy.samplers.sampler_object(str(sampler_name))
    sigmas = _sigmas(model, scheduler, steps, denoise)

    latent_out = latent.copy()
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    latent_out["samples"] = latent_image

    batch_inds = latent_out.get("batch_index", None)
    noise = comfy.sample.prepare_noise(latent_image, int(seed), batch_inds)
    noise_mask = latent_out.get("noise_mask", None)

    x0_output = {}
    callback = latent_preview.prepare_callback(
        model, sigmas.shape[-1] - 1, x0_output
    )
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = guider.sample(
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=int(seed),
    )
    samples = samples.to(comfy.model_management.intermediate_device())

    out = latent_out.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


def _default_clip(index: int = 0):
    return {
        "id": f"clip_{index + 1}",
        "prompt": "",
        "seed": int(secrets.randbelow(DEFAULT_SEED_MAX)),
        "seed_mode": "randomize",
        "duration": DEFAULT_DURATION,
        "validated": False,
    }


def _parse_clips_json(value: str):
    try:
        payload = json.loads(value or "{}")
    except Exception as exc:
        raise ValueError(f"MiniMax H3 Extender: invalid clips JSON: {exc}") from exc

    if isinstance(payload, list):
        clips = payload
    elif isinstance(payload, dict):
        clips = payload.get("clips", [])
    else:
        clips = []

    if not clips:
        clips = [_default_clip(0)]
    if len(clips) > MAX_CLIPS:
        raise ValueError(
            f"MiniMax H3 Extender: {len(clips)} clips requested; max is {MAX_CLIPS}."
        )

    out = []
    for i, raw in enumerate(clips):
        raw = raw if isinstance(raw, dict) else {}
        try:
            seed = int(raw.get("seed", 0))
        except Exception:
            seed = 0
        seed = max(0, min(DEFAULT_SEED_MAX, seed))

        try:
            duration = float(raw.get("duration", DEFAULT_DURATION))
        except Exception:
            duration = DEFAULT_DURATION
        duration = max(0.25, min(150.0, duration))

        seed_mode = str(raw.get("seed_mode", "randomize"))
        if seed_mode not in {"randomize", "fixed", "increment", "decrement"}:
            seed_mode = "randomize"

        out.append(
            {
                "id": str(raw.get("id") or f"clip_{i + 1}"),
                "prompt": str(raw.get("prompt", "")),
                "seed": seed,
                "seed_mode": seed_mode,
                "duration": duration,
                "validated": bool(raw.get("validated", False)),
            }
        )

    # Validation must always be a continuous prefix. Anything after the first
    # unvalidated clip depends on a clip that is not frozen yet.
    found_open = False
    for clip in out:
        if found_open:
            clip["validated"] = False
        elif not clip["validated"]:
            found_open = True

    return out


def _state_json(clips):
    return json.dumps(
        {"version": 1, "clips": clips},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _manifest_for_extender(owner_id, fps=24.0):
    return _manifest_for_first(f"extender_{owner_id}", fps)


EXTENDER_PROGRESS_EVENT = "h3_extender_progress"


def _send_extender_progress(
    node_id,
    clip_index,
    clip_count,
    phase,
    message="",
):
    """
    Send a tiny live UI event while the single Extender node is still running.

    clip_index is 0-based internally. Use -1 to clear the active card.
    """
    try:
        server = PromptServer.instance
        if server is None:
            return
        payload = {
            "node": str(node_id),
            "clip_index": int(clip_index),
            "clip_count": int(clip_count),
            "phase": str(phase),
            "message": str(message),
        }
        # Target the currently connected execution client when available.
        server.send_sync(
            EXTENDER_PROGRESS_EVENT,
            payload,
            getattr(server, "client_id", None),
        )
    except Exception:
        # Progress UI must never be able to break generation.
        pass



class MiniMaxH3Extender:
    @classmethod
    def INPUT_TYPES(cls):
        sampler_names = list(comfy.samplers.SAMPLER_NAMES)
        scheduler_names = list(comfy.samplers.SCHEDULER_NAMES)
        default_sampler = "euler" if "euler" in sampler_names else sampler_names[0]
        default_scheduler = "simple" if "simple" in scheduler_names else scheduler_names[0]

        required = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "run_mode": (["clip_by_clip", "full_batch"], {"default": "clip_by_clip"}),
            "width": ("INT", {"default": 896, "min": 32, "max": 4096, "step": 32}),
            "height": ("INT", {"default": 576, "min": 32, "max": 4096, "step": 32}),
            "ref_image_size": (["match", "max"], {"default": "match"}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 10000, "step": 1}),
            "sampler_name": (sampler_names, {"default": default_sampler}),
            "scheduler": (scheduler_names, {"default": default_scheduler}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
            "context_length": (["22", "5", "39", "56"], {"default": "22"}),
            "audio_context_length": ("INT", {"default": 0, "min": 0, "max": 240, "step": 1}),
            "clips_json": (
                "STRING",
                {
                    "default": _state_json([_default_clip(0)]),
                    "multiline": True,
                },
            ),
        }

        # audio_vae/ref_audio stay optional so existing image-only Extender
        # workflows continue to load and execute unchanged.  When ref_audio is
        # connected, audio_vae becomes functionally required.
        optional = {
            "audio_vae": ("VAE", {"forceInput": True}),
            "ref_audio": ("AUDIO", {"forceInput": True}),
        }
        optional.update({
            f"ref_{i}": ("IMAGE", {"forceInput": True})
            for i in range(1, 10)
        })

        return {
            "required": required,
            "optional": optional,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (CACHE_TYPE, "INT", "INT", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = (
        "cache",
        "clip_count",
        "validated_count",
        "status",
        "cache_size_mb",
        "build",
    )
    FUNCTION = "extend"
    CATEGORY = "MiniMax H3"
    OUTPUT_NODE = False

    def extend(
        self,
        model,
        clip,
        vae,
        run_mode,
        width,
        height,
        ref_image_size,
        steps,
        sampler_name,
        scheduler,
        denoise,
        context_length,
        audio_context_length,
        clips_json,
        unique_id=None,
        **kwargs,
    ):
        clips = _parse_clips_json(clips_json)
        owner = str(unique_id if unique_id is not None else "h3_extender")
        data_path, manifest_path, manifest = _manifest_for_extender(owner, FPS)

        # If cards were removed, trim the physical cache immediately.
        if len(manifest.get("segments", [])) > len(clips):
            manifest = _truncate_chain(
                data_path, manifest_path, manifest, len(clips)
            )

        segments = manifest.get("segments", [])

        # A TRUE toggle can only validate a clip that actually exists on disk.
        # This also protects old workflow JSON with stale downstream TRUE values.
        if len(segments) < len(clips):
            for i in range(len(segments), len(clips)):
                if clips[i]["validated"]:
                    for j in range(i, len(clips)):
                        clips[j]["validated"] = False
                    break

        refs = [kwargs.get(f"ref_{i}") for i in range(1, 10)]
        audio_vae = kwargs.get("audio_vae")
        ref_audio = kwargs.get("ref_audio")
        ref_items, ref_blocks = _prepare_shared_refs(
            vae,
            audio_vae,
            int(width),
            int(height),
            str(ref_image_size),
            refs,
            ref_audio=ref_audio,
        )

        disk_join = MiniMaxH3MotionContextDiskJoin()
        motion = MiniMaxH3MotionContextRAM()

        previous_handle = None
        previous_proxy = None
        generated = []
        statuses = []

        # Walk the card list in order. Cached TRUE clips are metadata-only;
        # active clips sample and are written immediately to disk.
        for i, cfg in enumerate(clips):
            # Refresh manifest state every iteration because Disk Join can
            # truncate or append the physical chain.
            current_manifest = _load_manifest_from_paths(data_path, manifest_path)
            existing_count = len(current_manifest.get("segments", [])) if current_manifest else 0
            existing = i < existing_count

            if cfg["validated"] and existing:
                result = disk_join.join(
                    samples=None,
                    trim_frames=None,
                    validated=True,
                    run_mode=str(run_mode),
                    fps=float(FPS),
                    previous_cache=previous_handle,
                    unique_id=f"extender_{owner}",
                )
                previous_handle = result[0]
                previous_proxy = result[1]
                statuses.append(result[4])
                continue

            # Any active clip is unvalidated. Make sure everything after it is
            # false in the serialized state as well.
            _send_extender_progress(
                owner,
                i,
                len(clips),
                "preparing",
                f"Preparing clip {i + 1}/{len(clips)}",
            )
            cfg["validated"] = False
            for j in range(i + 1, len(clips)):
                clips[j]["validated"] = False

            frame_count = _duration_to_frames(cfg["duration"])
            positive, latent = _make_ref2va_conditioning(
                clip,
                vae,
                cfg["prompt"],
                int(width),
                int(height),
                frame_count,
                ref_items,
                ref_blocks,
            )

            trim_frames = None
            if i > 0:
                if previous_proxy is None:
                    raise RuntimeError(
                        "MiniMax H3 Extender: previous cached latent is unavailable."
                    )
                positive, trim_frames, _, _, _ = motion.apply(
                    positive,
                    latent,
                    previous_proxy,
                    str(context_length),
                    int(audio_context_length),
                )

            _send_extender_progress(
                owner,
                i,
                len(clips),
                "sampling",
                f"Rendering clip {i + 1}/{len(clips)}",
            )

            sampled = _sample_h3(
                model,
                positive,
                latent,
                cfg["seed"],
                str(sampler_name),
                str(scheduler),
                int(steps),
                float(denoise),
            )

            result = disk_join.join(
                samples=sampled,
                trim_frames=trim_frames,
                validated=False,
                run_mode=str(run_mode),
                fps=float(FPS),
                previous_cache=previous_handle,
                unique_id=f"extender_{owner}",
            )
            previous_handle = result[0]
            previous_proxy = result[1]
            statuses.append(result[4])
            generated.append(i)

            _send_extender_progress(
                owner,
                i,
                len(clips),
                "complete",
                f"Clip {i + 1}/{len(clips)} complete",
            )

            # Drop full sampled/conditioning references before the next clip.
            del sampled, positive, latent

            if str(run_mode) == "clip_by_clip":
                break

        # A clip_by_clip run may stop before walking all cards; the last handle
        # is still exactly the active cached prefix expected by Final Decode.
        if previous_handle is None:
            # This can only happen if the workflow contains no valid cards,
            # which _parse_clips_json prevents. Keep a defensive error anyway.
            raise RuntimeError("MiniMax H3 Extender: sequence produced no cache handle.")

        final_manifest = _load_manifest_from_paths(data_path, manifest_path)
        cached_count = len(final_manifest.get("segments", []))
        validated_count = 0
        for desc in final_manifest.get("segments", []):
            if bool(desc.get("validated", False)):
                validated_count += 1
            else:
                break

        normalized_json = _state_json(clips)
        status = (
            f"{str(run_mode)} | cached {cached_count}/{len(clips)} | "
            f"validated {validated_count} | "
            + (
                "generated " + ",".join(str(i + 1) for i in generated)
                if generated
                else "disk only"
            )
        )
        cache_mb = _cache_size_mb(data_path, manifest_path)

        _send_extender_progress(
            owner,
            -1,
            len(clips),
            "idle",
            status,
        )

        ui_state = {
            "clips_json": normalized_json,
            "clip_count": len(clips),
            "cached_count": cached_count,
            "validated_count": validated_count,
            "generated": [i + 1 for i in generated],
            "status": status,
            "build": BUILD,
        }

        return {
            "ui": {"h3_extender_state": [ui_state]},
            "result": (
                previous_handle,
                int(len(clips)),
                int(validated_count),
                status,
                float(cache_mb),
                BUILD,
            ),
        }


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Extender": MiniMaxH3Extender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Extender": "MiniMax H3 Extender",
}
