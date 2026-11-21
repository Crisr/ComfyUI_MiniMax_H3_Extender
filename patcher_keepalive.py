"""Keep the patcher behind a loaded H3 model entry alive.

ComfyUI records a ``LoadedModel`` entry for the patcher that
``load_models_gpu()`` receives. H3 sampling runs on a temporary patcher: the
context-protection clone from ``install_protection_wrapper()``, or the per-clip
LoRA clone. When the sampling call drops that patcher, the entry becomes dead:
the ``nn.Module`` is still alive in the executor cache, so
``cleanup_models()`` cannot remove the entry, and ``can_unload()`` skips it.
The entry then stays forever, keeps stale VRAM accounting, and makes
``cleanup_models_gc()`` log "Potential memory leak" on every later load.

Holding the loaded patcher keeps its entry valid. The next load replaces the
entry while the patcher is still alive, which is the removal path ComfyUI
expects.
"""

# One current patcher plus the previous one: a new load pops the previous
# entry, and that pop must find the previous patcher alive.
KEEP_LIMIT = 2

_kept = []


def keep_loaded_patcher(model, keep=KEEP_LIMIT, manager=None):
    """Keep the patcher that owns ``model``'s LoadedModel entry alive.

    ``model`` is the patcher handed to sampling. ``manager`` defaults to
    ``comfy.model_management`` and exists so tests can pass a stub.
    Returns the list of kept patchers.
    """
    if manager is None:
        import comfy.model_management as manager

    if model is None:
        return list(_kept)
    real_model = getattr(model, "model", None)
    if real_model is None:
        return list(_kept)

    loaded = [model]
    for entry in list(getattr(manager, "current_loaded_models", ()) or ()):
        patcher = getattr(entry, "model", None)
        if patcher is None or getattr(patcher, "model", None) is not real_model:
            continue
        if patcher is not model and all(patcher is not kept for kept in loaded):
            loaded.append(patcher)

    _kept.extend(loaded)
    del _kept[:-max(1, int(keep))]
    return list(_kept)


def kept_patchers():
    """Return the patchers that are currently held."""
    return list(_kept)


def reset():
    """Drop every held patcher. Tests use this to start from a known state."""
    _kept.clear()
