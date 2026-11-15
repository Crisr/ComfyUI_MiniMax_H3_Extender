# Context protection

Context protection lowers colour drift and over-sharpening in a chained clip.
Each clip starts from the previous clip's held tail. The sampler would reuse
that tail as if it were its own early output, so colour and detail drift down
the chain.

Protection changes which latent steps the sampler may touch. It does not change
the model, the prompts or the references.

## Where it runs

- Ref2VA with Motion Context ON only. Other modes ignore the switch.
- Twice per clip: once in the base pass, once in the refine pass when Refine is
  on. The refine pass clamps a late step beyond its schedule instead of failing.
- In both motion context modules: `motion_context_ram.py` and
  `motion_context_disk.py`.

## Settings

| Widget | Default | Meaning |
|---|---|---|
| `context_protection` | `false` | Master switch |
| `protection_keystone` | `true` | Put the source clip's first video latent into held slot 0 |
| `protection_capstone` | `true` | Append the source clip's end-aligned 17-frame block during late steps |
| `protection_late_step` | `1` | Step where the Capstone engages and Late Lock closes |
| `protection_noise_strength` | `0.5` | Peak mask weight on the oldest held slot |
| `protection_noise_ease` | `true` | Legacy curve switch, read only when `protection_noise_fade` is `auto` |
| `protection_noise_fade` | `auto` | Curve for the held-context ramp: `auto`, `linear`, `ease`, `ethanfel` |
| `protection_audio_feather` | `0` | Half-cosine release over N held audio steps at the join |
| `protection_late_lock_slots` | `both` | Slots Late Lock owns: `both`, `capstone`, `keystone` |

Mask convention: `1` means the sampler generates the step, `0` means it keeps
the clean latent. Only the held prefix is lowered.

## Held-context ramp

The ramp is maximum on the oldest held step and reaches `0.0` at the newest held
step, which is the join. The sampler therefore keeps the join exact and gets
freer as it moves toward the older context.

| Mode | Weight at held slot `i` of `n` |
|---|---|
| `linear` | `1 - i / (n - 1)` |
| `ease` | `(1 - i / (n - 1)) ** 2` |
| `ethanfel` | `1.0` until the last 4 steps, then `remaining / 4` |

`auto` follows the legacy `protection_noise_ease` switch: `true` gives `ease`,
`false` gives `linear`. Both curves end at `0.0`.

### Ethanfel

Ethanfel rebuilds the ramp on every sampler step instead of once per pass:

```
weight  = ethanfel_weights(held)                 # 1.0, then a 4-step taper to 0.0
scale   = sigmas[step + 1] / sigmas[step]        # 0.0 when the denominator is 0
mask    = noise_strength * weight * scale
```

Early steps carry more noise, so they get the full taper. Late steps carry
little noise, so the same held steps receive almost none. The per-step ramp owns
the held prefix: it replaces the fixed ramp from `apply_masked_samples` instead
of merging with it.

The recipe comes from ethanfel's `drift_control` work, as documented in the
Geiru suite.

## Audio feather

The held audio is preserved byte for byte. Without a feather the sound stops
dead at the join, which sounds like a click and can build up down a chain.

`protection_audio_feather = N` writes a half-cosine release over the last `N`
held audio steps:

```
for k in range(N):
    phase  = (N - k) / (N + 1)
    weight = 0.5 - 0.5 * cos(pi * phase)
    audio_mask[held - 1 - k] = weight
```

`k = 0` is the step next to the free audio and gets the largest weight. Deeper
steps keep the held value.

The sampler now owns part of those steps, so `restore_samples` blends rather
than overwrites:

```
restored[column] = held_audio[column] * (1 - weight) + sampled[column] * weight
```

8 steps is 0.2 s at H3's 40 audio steps per second. `0` keeps the hard edge and
the previous behaviour.

## Late Lock slots

Before `protection_late_step`, the keystone slot is open so the shot can find
its own pose. At and after that step it is locked to the keystone latent.

`protection_late_lock_slots` decides which slots Late Lock owns:

| Value | Keystone slot (front) | Capstone (back) |
|---|---|---|
| `both` | open before the step, locked after | from the step |
| `capstone` | locked for the whole pass | from the step |
| `keystone` | open before the step, locked after | not appended |

The Geiru suite defaults to the capstone-only shape, because a locked front slot
costs pose freedom. `both` is the Extender's previous behaviour.

## Comparison with Geiru Context Suite

Reference: `github.com/roycho87/geiru_context_suite`, commit `e03cda9`.

| Idea | Geiru | This node |
|---|---|---|
| Keystone (source clip's first latent) | yes, several placements | yes, held slot 0 |
| Capstone (source clip's end block) | yes | yes, 17-frame block |
| Late Lock | `swap_step`, slot choice | `protection_late_step`, slot choice |
| Context noise curve | `linear` / `ease` / `ethanfel` | `auto` / `linear` / `ease` / `ethanfel` |
| Held audio feather | `audio_feather_ticks` | `protection_audio_feather` |
| Per-clip source | `source_clip` | `protection_source` |
| Restore the real held latents | yes | yes |
| Master switch | `degradation_protection` | `context_protection` |

Geiru reports colour drift from clip 1 falling from `6.4` to `3.9` with the
keystone, and to `2.9` with the keystone plus a late-lock capstone. It does not
report the measurement method, so treat the numbers as relative.

Geiru Hybrid Windows features (`finished_handoff`, `sound_first`, window fusion)
need one long clip. This node chains clips, so those do not apply.

## Compatibility

The three new widgets are appended after every earlier widget, so saved
positional arrays stay valid. Their defaults (`auto`, `0`, `both`) reproduce the
previous output exactly:

- `auto` reads the old `protection_noise_ease` switch.
- `0` keeps the hard audio edge.
- `both` keeps the previous Late Lock behaviour.

`protection_signature` includes the three values, so changing one drops stale
protected clips from the cache.

ComfyUI restores a saved workflow's widget values by position, not by name. A
widget inserted anywhere before the end of `INPUT_TYPES["required"]` therefore
shifts every older value onto the wrong widget. New widgets belong at the very
end of that dict; `tests/test_widget_order.py` enforces this.

## Clip length advice

Geiru notes that H3 audio runs 40 steps a second against 24 video frames. A
clip length that is a multiple of 3 frames lands exactly on the audio grid. The
Extender does not enforce this. Prefer 141, 192, 243, 294 or 345 frames when
protection is on, and when the audio context is held.

## Tests

```
$env:PYTHONPATH="."; G:/comfy-cu133/env/python.exe -m unittest discover -s tests -p "test_context_protection.py"
node tests/context-protection.test.mjs
```

The Python suite covers the ramp curves, the ethanfel step scaling, the audio
feather mask and its blend on restore, the Late Lock slot choice, the signature,
and the default behaviour.

No GPU is needed. The tests build small tensors directly.

## Files

| File | Change |
|---|---|
| `context_protection.py` | New settings, `resolve_noise_fade`, `ethanfel_weights`, `audio_feather_weights`, `_ethanfel_step_mask`, feather in `build_masked_context`, blend in `restore_samples`, slots in `make_late_lock_mask_function` |
| `motion_context_ram.py` | `apply_masked` forwards `noise_fade` and `audio_feather` |
| `extender.py` | Three widgets, `extend()` arguments, both `apply_masked` call sites, `PROJECT_SETTING_NAMES` |
| `web/extender.js` | Widget defaults for the project and the seed helper |
| `tests/test_context_protection.py` | New `ProtectionFadeTests` class |
| `tests/context-protection.test.mjs` | Defaults and project-widget assertions |

Two defects were fixed on the way:

- `restore_samples` returned the restored audio only for a `NestedTensor`. Tuple
  and list callers got the video alone.
- The held-context mask test expected `1.0` at the join, but the curve and the
  docstring both reach `0.0` there.
