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

    video_mask = torch.zeros(
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

    audio_mask = torch.zeros(
        (1, 1, 1, int(target_audio.shape[-1])),
        device=target_audio.device,
        dtype=target_audio.dtype,
    )
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
