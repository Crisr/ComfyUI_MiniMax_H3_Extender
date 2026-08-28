
"""
MiniMax H3 payload patch allowing motion-context keyframes and Ref2VA refs
to coexist.

Derived from:
https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context
(GPL-3.0)
"""
import logging

import comfy.model_base as model_base

MC_KEY = "motion_context_index"
MC_AUDIO_KEY = "motion_context_audio_end_frame"
PATCH_MARKER = "_h3_motion_context_payload_patch"

_LOG = logging.getLogger("minimax_h3_tail_from_latent.motion_context")
_orig_extra_conds = None
_applied = False


def _patched_extra_conds(self, **kwargs):
    out = _orig_extra_conds(self, **kwargs)

    keyframes = kwargs.get("minimax_keyframes", None)
    refs = kwargs.get("minimax_refs", None)

    if not keyframes or not refs:
        return out

    cond = out.get("minimax_payload", None)
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        _LOG.warning(
            "MiniMax H3 Motion Context RAM: could not access minimax_payload; "
            "Ref2VA refs may overwrite carried conditioning."
        )
        return out

    # Layout order is: keyframe cond rows first, then Ref2VA reference rows.
    # New ComfyUI carries the motion-context audio tail as a keyframe
    # audio_latent, so both lists must keep that keyframes-first order or the
    # packed row count desyncs from PackedLayout. Rebuilding the audio list
    # from refs alone (as older compat wrappers do) drops the carried audio.
    payload["cond_video_latents"] = (
        [kf["latent"] for kf in keyframes if kf.get("latent") is not None]
        + [ref["latent"] for ref in refs if "latent" in ref]
    )
    payload["cond_audio_latents"] = (
        [
            kf["audio_latent"]
            for kf in keyframes
            if kf.get("audio_latent") is not None
        ]
        + [
            ref["audio_latent"]
            for ref in refs
            if ref.get("audio_latent") is not None
        ]
    )

    frame_count = kwargs.get("minimax_frame_count", None)
    if frame_count is not None:
        payload["frame_count"] = frame_count

    return out


setattr(_patched_extra_conds, PATCH_MARKER, True)


def _already_patched(cls):
    fn = getattr(cls, "extra_conds", None)
    if fn is None:
        return None
    if getattr(fn, PATCH_MARKER, False):
        if getattr(fn, "__module__", None) == __name__:
            return "same"
        # Another pack installed a marker-compatible wrapper (e.g.
        # H3-Multishot's AVBank probe). Those rebuild the audio list from
        # refs alone and drop keyframe audio latents; wrap on top and repair.
        return "compatible"
    if getattr(fn, "__name__", "") == "_patched_extra_conds":
        # Vendored copy of this patch from another pack; same refs-only
        # audio gap, so wrap on top.
        return "compatible"
    if hasattr(fn, "__wrapped__"):
        return "foreign"
    home = getattr(cls, "__module__", None)
    where = getattr(fn, "__module__", None)
    if home and where and where != home:
        return "foreign"
    return None


def apply_patch():
    global _orig_extra_conds, _applied

    if _applied:
        return True

    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        _LOG.warning(
            "MiniMax H3 Motion Context RAM: MiniMaxH3.extra_conds not found."
        )
        return False

    who = _already_patched(cls)
    if who == "foreign":
        _LOG.warning(
            "MiniMax H3 Motion Context RAM: another custom node already owns "
            "MiniMaxH3.extra_conds; refusing to stack incompatible patches."
        )
        return False

    if who == "same":
        _applied = True
        return True

    _orig_extra_conds = cls.extra_conds
    cls.extra_conds = _patched_extra_conds
    _applied = True
    if who == "compatible":
        _LOG.info(
            "MiniMax H3 Motion Context RAM: wrapped a compatible "
            "extra_conds patch to keep keyframe audio latents"
        )
    else:
        _LOG.info(
            "MiniMax H3 Motion Context RAM: keyframe/ref coexistence enabled"
        )
    return True


def is_applied():
    return _applied
