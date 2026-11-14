# Refine Upscaler Selector — 2026-11-14

## Summary

Refine now has a **Upscaler** selector. It sits in the **REFINE SETTINGS** panel
next to Scale, Steps and Denoise, and the panel opens when REFINE is ON.

| Value | Behaviour |
|---|---|
| `learned` (default) | Runs the H3 3D conv latent upscaler. Same as every earlier version. |
| `trilinear` | Skips the conv model and interpolates the video latent directly. |

`trilinear` removes three fixed costs:

1. No conv pass over the clip.
2. No H3 unload and reload around the upscaler, because the 690 MB model never loads.
3. No latent upscaler download on first use.

The result is softer, so raise **Refine Denoise** to about `0.35`–`0.40` when you
use it.

## Cache behaviour

Changing the upscaler invalidates generated clips, in both directions. The clip
cache stores the choice in the refine settings payload.

`learned` writes no extra field. Caches written before this change therefore keep
the signature they were generated with, and stay valid.

## Changes by file

### `latent_upscaler.py`

- Add `UPSCALER_METHODS = ("learned", "trilinear")`.
- Add `normalize_upscaler_method(method)`. Unknown values fall back to `learned`.
- `upscale_video_latent(video, scale, progress=None, method="learned")` gains a
  `method` argument and a trilinear branch. The branch runs after the
  "no size change" early return. It moves the latent to the compute device,
  interpolates with `F.interpolate(mode="trilinear", align_corners=False)`, and
  returns the original device and dtype.

### `latent_refine_engine.py`

- `upscale_for_refine(sampled, scale, progress=None, upscaler="learned")` passes
  the method to `upscale_video_latent`.
- The first-pass audio is still preserved unchanged.

### `extender.py`

- Import `normalize_upscaler_method`.
- New widget `refine_upscaler` with values `["learned", "trilinear"]`, default
  `learned`. It is declared last in `INPUT_TYPES`, so workflows saved earlier keep
  their positional widget values.
- Add `"refine_upscaler"` to `PROJECT_SETTING_NAMES`.
- Clamp the value with `normalize_upscaler_method` next to the Denoise clamp.
- Pass the value through `_refine_ref2va_sample`, `_refine_fl2va_sample`,
  `_extend_fl2va`, `_run_ref2va_independent`, and both refine call sites of the
  main `extend` path.
- `_refine_settings_payload` gains an `upscaler` argument. It records the method
  only when the method is not `learned`.
- Release the protection source mmap before the join. See the next section.

### `ref2va_independent.py`

- Add the `refine_upscaler` parameter, and pass it to the shared refine payload and
  sample helpers.

### `web/extender.js`

- Add `refine_upscaler` to `PROJECT_WIDGETS`.
- Hide the native widget row with two CSS rules.
- Add `makeRefineSelect(labelText, widget, options)`. It builds a `<select>` that
  matches the style of the number inputs.
- Seed the widget value when a saved workflow has no value for it.
- Invalidate the generated clips when the value changes.
- Sync the control in `render()` and disable it while the project is busy.
- Update the panel tooltip.

### `README.md`

- Document the new option under Refine settings.

## Related fix in the same working tree

### `extender.py` — release the protection source mmap

`source_video` comes from `_streams_from_latent()`, which returns a tensor backed by
an open `np.memmap` of the cache data file. The variable stayed alive for the rest
of `extend()`, so `f.truncate()` in `motion_context_disk.join()` failed with
`PermissionError: [Errno 13]` on Windows.

`make_protection_plan()` keeps only clones. The code now deletes `source_video`
right after the plan is built.

**Status:** compiled, but not yet confirmed by a Windows run.

### `motion_context_disk.py` — survive a locked output file

`_embed_final_metadata_in_place()` now returns `_replace_locked_source(temp_path,
source)` instead of calling `os.replace`. A player or a scanner can hold the
finished MP4 open on Windows and make `os.replace` fail with `WinError 5`. The
helper retries, then falls back to a new file name. The caller stores the returned
path.

## Verification

| Check | Result |
|---|---|
| `python3 -m py_compile extender.py ref2va_independent.py latent_upscaler.py latent_refine_engine.py` | OK |
| `node --test tests/` | 35 pass, 0 fail |

Python unit tests did not run locally, because this machine has no `torch`. Run
them on the Windows machine:

```
python -m unittest discover -s tests -p "test_*.py"
node --test tests/
```

## How to use

1. Press **REFINE: ON**.
2. Open **⚙ REFINE SETTINGS**.
3. Set **Scale** and **Steps** as usual.
4. Select **Upscaler**:
   - `learned` for the best quality.
   - `trilinear` for speed and low VRAM.
5. With `trilinear`, set **Denoise** to `0.35`–`0.40`.
6. Uncheck **Validated** on the clip cards before you change an existing project.
