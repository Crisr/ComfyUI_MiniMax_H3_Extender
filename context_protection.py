"""Latent-only context protection helpers for the H3 Extender.

This module is intentionally independent of the Geiru Context Suite
implementation. It provides configuration, noise masks, step lookup, and
cache signatures used by the masked-context path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch


@dataclass
class MaskedContext:
    video: torch.Tensor
    audio: torch.Tensor
    video_mask: torch.Tensor
    audio_mask: torch.Tensor
    held_video: torch.Tensor
    held_audio: torch.Tensor
    video_steps: int
    audio_steps: int
    trim_frames: int


@dataclass(frozen=True)
class ProtectionSettings:
    enabled: bool
    keystone: bool
    capstone: bool
    late_step: int
    noise_strength: float
    noise_ease: bool


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        number = int(default)
    return max(low, min(high, number))


def _clamp_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = float(default)
    if not torch.isfinite(torch.tensor(number)):
        number = float(default)
    return max(low, min(high, number))


def normalize_protection_settings(
    enabled=False,
    keystone=True,
    capstone=True,
    late_step=1,
    noise_strength=0.5,
    noise_ease=True,
) -> ProtectionSettings:
    return ProtectionSettings(
        enabled=bool(enabled),
        keystone=bool(keystone),
        capstone=bool(capstone),
        late_step=_clamp_int(late_step, 1, 0, 999),
        noise_strength=_clamp_float(noise_strength, 0.5, 0.0, 1.0),
        noise_ease=bool(noise_ease),
    )


def capstone_index(video_latent_length: int) -> int:
    """Return the start of the end-aligned five-step H3 latent block."""
    length = max(0, int(video_latent_length))
    if length <= 1:
        return 0
    return 5 * ((length - 1) // 5)


def build_context_noise_mask(
    count: int,
    strength: float,
    ease: bool,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Build a held-context ramp that reaches zero at the newest slot."""
    steps = max(0, int(count))
    peak = _clamp_float(strength, 0.5, 0.0, 1.0)
    values = []
    for index in range(steps):
        position = index / max(1, steps - 1)
        remaining = 1.0 - position
        weight = remaining * remaining if ease else remaining
        values.append(peak * weight)
    return torch.tensor(values, device=device, dtype=dtype).reshape(1, 1, steps, 1, 1)


def build_masked_context(
    target_video,
    target_audio,
    held_video,
    held_audio,
    noise_strength=0.5,
    noise_ease=True,
    trim_frames=0,
) -> MaskedContext:
    """Overlay exact held latent values and their sampling masks."""
    if target_video.ndim != 5 or held_video.ndim != 5:
        raise ValueError("Context protection: video latents must be 5D.")
    if target_audio.ndim != 4 or held_audio.ndim != 4:
        raise ValueError("Context protection: audio latents must be 4D.")
    if target_video.shape[:2] != held_video.shape[:2] or target_video.shape[3:] != held_video.shape[3:]:
        raise ValueError("Context protection: held video latent geometry differs from the target.")
    if target_audio.shape[:3] != held_audio.shape[:3]:
        raise ValueError("Context protection: held audio latent geometry differs from the target.")
    held_steps = int(held_video.shape[2])
    held_audio_steps = int(held_audio.shape[-1])
    if held_steps < 1 or held_audio_steps < 1:
        raise ValueError("Context protection: held context is empty.")
    if held_steps > int(target_video.shape[2]) or held_audio_steps > int(target_audio.shape[-1]):
        raise ValueError("Context protection: held context is longer than the target.")

    video = target_video.clone()
    audio = target_audio.clone()
    video[:, :, :held_steps] = held_video
    audio[..., :held_audio_steps] = held_audio

    # H3 mask convention: 1 = generate, 0 = preserve the clean latent. Every
    # frame the shot has to make therefore starts at 1; only the held prefix is
    # lowered, and its ramp reaches 0 at the newest join.
    video_mask = torch.ones(
        (1, 1, int(target_video.shape[2]), int(target_video.shape[3]), int(target_video.shape[4])),
        device=target_video.device,
        dtype=target_video.dtype,
    )
    ramp = build_context_noise_mask(
        held_steps,
        noise_strength,
        noise_ease,
        device=target_video.device,
        dtype=target_video.dtype,
    )
    video_mask[:, :, :held_steps] = ramp

    audio_mask = torch.ones(
        (1, 1, 1, int(target_audio.shape[-1])),
        device=target_audio.device,
        dtype=target_audio.dtype,
    )
    audio_mask[..., :held_audio_steps] = 0.0
    return MaskedContext(
        video=video,
        audio=audio,
        video_mask=video_mask,
        audio_mask=audio_mask,
        held_video=held_video.clone(),
        held_audio=held_audio.clone(),
        video_steps=held_steps,
        audio_steps=held_audio_steps,
        trim_frames=int(trim_frames),
    )


@dataclass
class ProtectionPlan:
    masked: MaskedContext
    keystone: torch.Tensor | None
    capstone: torch.Tensor | None
    late_step: int
    source_clip_id: str
    source_index: int
    signature: str


def validate_late_step(settings: ProtectionSettings, sigmas) -> None:
    count = int(len(sigmas)) if sigmas is not None else 0
    if int(settings.late_step) < 0 or int(settings.late_step) >= count - 1:
        raise ValueError(
            "Context protection: Late Lock step "
            f"{int(settings.late_step)} is outside the active sigma schedule "
            f"({count} values)."
        )


def extract_capstone(source_video, block_steps=5):
    if not torch.is_tensor(source_video) or source_video.ndim != 5:
        raise ValueError("Context protection: capstone source must be a 5D video latent.")
    length = int(source_video.shape[2])
    start = capstone_index(length)
    if start <= 0 or start >= length:
        return None
    capstone = source_video[:, :, start:start + int(block_steps)].clone()
    if int(capstone.shape[2]) == 0:
        return None
    return capstone


def make_protection_plan(
    masked: MaskedContext,
    source_video,
    source_clip_id,
    source_index,
    settings: ProtectionSettings,
    sigmas,
    context_length,
    audio_context_length,
    refine_scale=1.0,
) -> ProtectionPlan:
    validate_late_step(settings, sigmas)
    if not torch.is_tensor(source_video) or source_video.ndim != 5:
        raise ValueError("Context protection: source must be a 5D video latent.")
    if int(source_video.shape[2]) < 1:
        raise ValueError("Context protection: source video latent is empty.")
    if masked.video.shape[:2] != source_video.shape[:2] or masked.video.shape[3:] != source_video.shape[3:]:
        raise ValueError("Context protection: source and target latent geometry differ.")

    # held_video is the restore record: the chain's real previous-tail frames,
    # put back after sampling so the saved clip keeps a clean join.
    held_video = masked.held_video.clone()
    sampling_video = masked.video.clone()
    video_mask = masked.video_mask.clone()
    keystone = None
    if settings.keystone:
        keystone = source_video[:, :, :1].clone()
        sampling_video[:, :, :1] = keystone.to(device=sampling_video.device, dtype=sampling_video.dtype)
    # Slot 0 is owned by Keystone / Late Lock: locked unless Late Lock opens it.
    video_mask[:, :, 0] = 0.0

    capstone = extract_capstone(source_video) if settings.capstone else None
    if capstone is not None:
        capstone = capstone.to(device=sampling_video.device, dtype=sampling_video.dtype)
    updated = MaskedContext(
        video=sampling_video,
        audio=masked.audio,
        video_mask=video_mask,
        audio_mask=masked.audio_mask,
        held_video=held_video,
        held_audio=masked.held_audio,
        video_steps=masked.video_steps,
        audio_steps=masked.audio_steps,
        trim_frames=masked.trim_frames,
    )
    signature = protection_signature(
        source_clip_id, settings, sigmas, context_length,
        audio_context_length, refine_scale,
    )
    return ProtectionPlan(
        masked=updated,
        keystone=keystone,
        capstone=capstone,
        late_step=int(settings.late_step),
        source_clip_id=str(source_clip_id),
        source_index=int(source_index),
        signature=signature,
    )


def _stream_pair(value):
    if isinstance(value, (tuple, list)):
        if len(value) < 2:
            raise ValueError("Context protection: expected video and audio streams.")
        return value[0], value[1]
    if bool(getattr(value, "is_nested", False)):
        tensors = list(getattr(value, "tensors", ()))
        if len(tensors) < 2:
            raise ValueError("Context protection: expected video and audio streams.")
        return tensors[0], tensors[1]
    raise ValueError("Context protection: unsupported sample container.")


def apply_masked_samples(samples, masked):
    """Return samples with the masked video and audio streams installed."""
    video, audio = _stream_pair(samples)
    masked_video = masked.video.to(device=video.device, dtype=video.dtype)
    masked_audio = masked.audio.to(device=audio.device, dtype=audio.dtype)
    if isinstance(samples, tuple):
        return (masked_video, masked_audio, *samples[2:])
    if isinstance(samples, list):
        return [masked_video, masked_audio, *samples[2:]]
    if bool(getattr(samples, "is_nested", False)):
        tensors = list(samples.tensors)
        tensors[0] = masked_video
        tensors[1] = masked_audio
        return type(samples)(tensors)
    raise ValueError("Context protection: unsupported sample container.")


def protected_denoise_mask(samples, plan):
    """Return the sampler denoise mask for a protection plan.

    The mask must reach ``guider.sample``: ``KSamplerX0Inpaint`` uses it to mix
    the clean held latents into x every step, which is what keeps held rows
    clean. Passing it only to the model is not enough.
    """
    if plan is None:
        return None
    video, audio = _stream_pair(samples)
    video_mask = plan.masked.video_mask.to(device=video.device, dtype=torch.float32)
    audio_mask = plan.masked.audio_mask.to(device=audio.device, dtype=torch.float32)
    if bool(getattr(samples, "is_nested", False)):
        return type(samples)((video_mask, audio_mask))
    return video_mask


def make_late_lock_mask_function(plan: ProtectionPlan):
    """Build the per-step mask hook Late Lock needs.

    ``KSamplerX0Inpaint`` is the only consumer of ``denoise_mask_function``
    (comfy/samplers.py), so the keystone slot can only change per step there.
    Values follow the mask convention: 1 before ``late_step`` (free), 0 at and
    after it (locked to the keystone).
    """

    def lock(sigma, mask, extra_options=None):
        if not torch.is_tensor(mask):
            return mask
        sigmas = (extra_options or {}).get("sigmas")
        step = resolve_sampler_step(sigma, sigmas)
        value = 1.0 if step < int(plan.late_step) else 0.0
        video = plan.masked.video
        channels, steps, height, width = (int(v) for v in video.shape[1:])
        n_video = channels * steps * height * width
        result = mask.contiguous().clone()
        if result.ndim == 3 and int(result.shape[-1]) >= n_video:
            # Packed AV mask [B, 1, N].
            vm = result[:, :, :n_video].reshape(result.shape[0], channels, steps, height, width).clone()
            vm[:, :, 0] = value
            result = torch.cat(
                (vm.reshape(result.shape[0], 1, n_video), result[:, :, n_video:]),
                dim=-1,
            )
        elif result.ndim == 5 and int(result.shape[2]) == steps:
            result[:, :, 0] = value
        return result

    return lock


def _replace_video(value, video):
    if isinstance(value, tuple):
        return (video, *value[1:])
    if isinstance(value, list):
        return [video, *value[1:]]
    if bool(getattr(value, "is_nested", False)):
        tensors = list(value.tensors)
        tensors[0] = video
        return type(value)(tensors)
    return video


def _fit_mask_length(mask, length, dim):
    current = int(mask.shape[dim])
    wanted = int(length)
    if current == wanted:
        return mask
    if current > wanted:
        return mask.narrow(dim, 0, wanted)
    shape = list(mask.shape)
    shape[dim] = wanted - current
    tail = torch.zeros(shape, device=mask.device, dtype=mask.dtype)
    return torch.cat((mask, tail), dim=dim)


def make_protection_wrapper(plan: ProtectionPlan):
    def wrapper(
        executor,
        x,
        timestep,
        context,
        transformer_options,
        minimax_payload=None,
        denoise_mask=None,
        audio_denoise_mask=None,
        **kwargs,
    ):
        video, audio = _stream_pair(x)
        sigma = float(torch.as_tensor(timestep).detach().flatten()[0].item()) / 1000.0
        sigmas = transformer_options["sample_sigmas"]
        step = resolve_sampler_step(sigma, sigmas)
        original_video_steps = int(video.shape[2])

        video_dtype = denoise_mask.dtype if torch.is_tensor(denoise_mask) else video.dtype
        plan_video_mask = plan.masked.video_mask.to(device=video.device, dtype=video_dtype)

        if torch.is_tensor(denoise_mask):
            active_video_mask = _fit_mask_length(
                denoise_mask, int(video.shape[2]), 2
            ).clone()
        else:
            active_video_mask = torch.zeros_like(video[:, :1])
        held = min(int(plan.masked.video_steps), int(active_video_mask.shape[2]))
        if held > 0:
            plan_prefix = _fit_mask_length(
                plan_video_mask, int(video.shape[2]), 2
            )[:, :, :held]
            active_video_mask[:, :, :held] = torch.maximum(
                active_video_mask[:, :, :held], plan_prefix
            )
            # Late Lock owns the keystone slot: locked before the step, free at
            # and after it. Existing caller masks remain chained elsewhere.
            active_video_mask[:, :, 0] = 1.0 if step < int(plan.late_step) else 0.0
        active_audio_mask = audio_denoise_mask

        if step < int(plan.late_step):
            return executor(
                x, timestep, context, transformer_options,
                minimax_payload=minimax_payload,
                denoise_mask=active_video_mask,
                audio_denoise_mask=active_audio_mask,
                **kwargs,
            )

        model_x = x
        if plan.capstone is not None:
            capstone = plan.capstone.to(device=video.device, dtype=video.dtype)
            model_video = torch.cat((video, capstone), dim=2)
            mask_tail = torch.zeros(
                (1, 1, int(capstone.shape[2]), int(video.shape[3]), int(video.shape[4])),
                device=active_video_mask.device,
                dtype=active_video_mask.dtype,
            )
            active_video_mask = torch.cat((active_video_mask, mask_tail), dim=2)
            model_x = _replace_video(x, model_video)

        out = executor(
            model_x, timestep, context, transformer_options,
            minimax_payload=minimax_payload,
            denoise_mask=active_video_mask,
            audio_denoise_mask=active_audio_mask,
            **kwargs,
        )
        if plan.capstone is not None:
            out_video, _ = _stream_pair(out)
            out = _replace_video(
                out, out_video[:, :, :original_video_steps].contiguous()
            )
        return out

    return wrapper


def install_protection_wrapper(model, plan):
    if plan is None or plan.signature == "off":
        return model
    from comfy.patcher_extension import WrappersMP

    patched = model.clone()
    patched.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL,
        "h3_context_protection",
        make_protection_wrapper(plan),
    )
    previous = patched.model_options.get("denoise_mask_function")
    late_lock = make_late_lock_mask_function(plan)

    def denoise_mask_function(sigma, mask, extra_options=None):
        if previous is not None:
            mask = previous(sigma, mask, extra_options)
        return late_lock(sigma, mask, extra_options)

    patched.set_model_denoise_mask_function(denoise_mask_function)
    return patched


def restore_samples(samples, plan):
    if plan is None:
        return samples
    video, audio = _stream_pair(samples)
    restored_video = video.clone()
    restored_audio = audio.clone()
    restored_video[:, :, :plan.masked.video_steps] = plan.masked.held_video
    restored_audio[..., :plan.masked.audio_steps] = plan.masked.held_audio
    if isinstance(samples, tuple):
        return (restored_video, *samples[1:])
    if isinstance(samples, list):
        return [restored_video, *samples[1:]]
    if bool(getattr(samples, "is_nested", False)):
        tensors = list(samples.tensors)
        tensors[0] = restored_video
        tensors[1] = restored_audio
        return type(samples)(tensors)
    raise ValueError("Context protection: unsupported sample container.")


def normalize_protection_source(raw_clip, clip_index, first_clip_id):
    if int(clip_index) <= 0:
        return None
    raw = raw_clip if isinstance(raw_clip, dict) else {}
    if "protection_source" in raw and raw.get("protection_source") is None:
        return None
    source = str(raw.get("protection_source") or "").strip()
    if source:
        return source
    first = str(first_clip_id or "").strip()
    return first or None


def resolve_source_index(clips, clip_index, source_clip_id):
    wanted = str(source_clip_id or "")
    for index, clip in enumerate(clips):
        if str((clip or {}).get("id") or "") != wanted:
            continue
        if index >= int(clip_index):
            raise ValueError(
                f"Context protection: clip {int(clip_index) + 1} source "
                f"{wanted!r} must be an earlier clip."
            )
        return index
    raise ValueError(
        f"Context protection: clip {int(clip_index) + 1} source "
        f"{wanted!r} was not found in the clip list."
    )


def protection_cache_mismatch(segment_meta, expected_signature) -> bool:
    meta = segment_meta if isinstance(segment_meta, dict) else {}
    previous = str(meta.get("protection_signature", "off") or "off")
    expected = str(expected_signature or "off")
    return previous != expected


def resolve_sampler_step(sigma, sigmas) -> int:
    """Return the sigma index nearest to the current sampler sigma."""
    if not torch.is_tensor(sigmas) or int(sigmas.numel()) == 0:
        return 0
    flat = sigmas.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    value = float(torch.as_tensor(sigma).detach().to(dtype=torch.float32, device="cpu").reshape(-1)[0])
    distances = (flat - value).abs()
    return int(torch.argmin(distances).item())


def protection_signature(
    source_clip_id,
    settings: ProtectionSettings,
    sigmas,
    context_length,
    audio_context_length,
    refine_scale=1.0,
) -> str:
    """Return a stable cache key for every effective protection input."""
    source = "" if source_clip_id is None else str(source_clip_id).strip()
    if not source or not settings.enabled:
        return "off"
    sigma_values = []
    if torch.is_tensor(sigmas):
        sigma_values = [
            float(value)
            for value in sigmas.detach().to(dtype=torch.float32, device="cpu").reshape(-1).tolist()
        ]
    payload = {
        "source_clip_id": source,
        "enabled": bool(settings.enabled),
        "keystone": bool(settings.keystone),
        "capstone": bool(settings.capstone),
        "late_step": int(settings.late_step),
        "noise_strength": float(settings.noise_strength),
        "noise_ease": bool(settings.noise_ease),
        "sigmas": sigma_values,
        "context_length": str(context_length),
        "audio_context_length": int(audio_context_length),
        "refine_scale": float(refine_scale),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
