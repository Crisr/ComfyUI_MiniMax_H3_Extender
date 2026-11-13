# H3 Context Protection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add latent-only color-drift protection to the Ref2VA Motion Context path of MiniMax H3 Extender.

**Architecture:** A new `context_protection.py` owns settings, masks, source selection, keystone/capstone, sampler wrappers, and restoration. `motion_context_ram.py` gains a masked-context path that copies previous video and audio tails into the target latent instead of attaching keyframes. `extender.py` wires the feature into the Ref2VA Motion Context ON path only, for base and latent-refine passes.

**Tech Stack:** Python 3, PyTorch, ComfyUI patcher wrappers, vanilla JavaScript, `unittest`, `node:test`.

**Spec:** `docs/superpowers/specs/2026-11-03-h3-context-protection-design.md`

## Global Constraints

- Version 1 supports Ref2VA with Motion Context ON only.
- Protection is off by default and does not change Motion OFF, FL2VA, independent Ref2VA, or Clip 0 behavior.
- Do not add LAB or RGB color correction. This feature is latent-only.
- Do not copy GPL code from `geiru_context_suite`. Write an independent Apache-2.0-compatible implementation and attribute the mechanism.
- Keep existing widget order. Append the six new widgets after `color_match_process_clip0`.
- Apply identical protection to the base pass and the latent-refine pass.
- Store `protection_signature` on every Ref2VA segment. Missing signature means protection off.
- Preserve validated clips when only the protection signature changed. Warn and stop reuse from the first mismatch.
- Use stable clip IDs for `protection_source`. Never silently remap a source after reorder.
- Restore held latents whenever sampling returns output. Remove every temporary capstone before the model call returns.
- Keep CPU-only unit tests. Do not require a GPU or a running ComfyUI.

## Review Focus

1. A source clip is missing, current, or later in the list. Expect a clear pre-sample error, not a silent fallback.
2. `protection_late_step` is outside the effective sigma schedule. Expect a pre-sample error.
3. The source clip is too short for a 17-frame capstone. Expect capstone to be skipped, not a crash.
4. An exception occurs during sampling. Expect no temporary capstone to survive in model state.
5. Refine geometry differs from base geometry. Expect source and held context to resize to the refine latent.

---

### Task 1: Protection Settings, Masks, and Signature

**Files:**
- Create: `context_protection.py`
- Test: `tests/test_context_protection.py`

**Interfaces:**
- Consumes: none.
- Produces:
  - `ProtectionSettings(enabled, keystone, capstone, late_step, noise_strength, noise_ease)`
  - `normalize_protection_settings(enabled=False, keystone=True, capstone=True, late_step=1, noise_strength=0.5, noise_ease=True) -> ProtectionSettings`
  - `capstone_index(video_latent_length: int) -> int`
  - `build_context_noise_mask(count: int, strength: float, ease: bool, *, device=None, dtype=torch.float32) -> torch.Tensor`
  - `protection_signature(source_clip_id, settings, sigmas, context_length, audio_context_length, refine_scale=1.0) -> str`
  - `resolve_sampler_step(sigma, sigmas) -> int`

- [ ] **Step 1: Write the failing tests**

```python
import math
import unittest

import torch

from context_protection import (
    ProtectionSettings,
    build_context_noise_mask,
    capstone_index,
    normalize_protection_settings,
    protection_signature,
    resolve_sampler_step,
)


class ContextProtectionCoreTests(unittest.TestCase):
    def test_settings_are_normalized(self):
        settings = normalize_protection_settings(
            enabled=1, keystone=0, capstone=True,
            late_step=-4, noise_strength=1.7, noise_ease=0,
        )
        self.assertTrue(settings.enabled)
        self.assertFalse(settings.keystone)
        self.assertTrue(settings.capstone)
        self.assertEqual(settings.late_step, 0)
        self.assertEqual(settings.noise_strength, 1.0)
        self.assertFalse(settings.noise_ease)

    def test_capstone_index_uses_end_aligned_h3_block(self):
        self.assertEqual(capstone_index(22), 20)
        self.assertEqual(capstone_index(25), 20)
        self.assertEqual(capstone_index(17), 15)
        self.assertEqual(capstone_index(1), 0)

    def test_noise_ramp_linear_and_eased(self):
        linear = build_context_noise_mask(5, 0.5, False)[0, 0, :, 0, 0].tolist()
        eased = build_context_noise_mask(5, 0.5, True)[0, 0, :, 0, 0].tolist()
        self.assertEqual(linear, [0.5, 0.375, 0.25, 0.125, 0.0])
        self.assertEqual(eased, [0.5, 0.28125, 0.125, 0.03125, 0.0])

    def test_single_context_step_uses_full_strength(self):
        values = build_context_noise_mask(1, 0.5, True)[0, 0, :, 0, 0].tolist()
        self.assertEqual(values, [0.5])

    def test_signature_is_off_without_an_active_source(self):
        settings = ProtectionSettings(True, True, True, 1, 0.5, True)
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        self.assertEqual(
            protection_signature(None, settings, sigmas, "22", 0),
            "off",
        )
        self.assertEqual(
            protection_signature("clip_1", ProtectionSettings(False, True, True, 1, 0.5, True), sigmas, "22", 0),
            "off",
        )

    def test_signature_changes_with_each_effective_input(self):
        settings = ProtectionSettings(True, True, True, 1, 0.5, True)
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        base = protection_signature("clip_1", settings, sigmas, "22", 0)
        self.assertNotEqual(base, protection_signature("clip_2", settings, sigmas, "22", 0))
        self.assertNotEqual(base, protection_signature("clip_1", settings, sigmas, "17", 0))
        self.assertNotEqual(base, protection_signature("clip_1", settings, sigmas, "22", 1))
        self.assertNotEqual(base, protection_signature("clip_1", settings, sigmas, "22", 0, 1.5))

    def test_resolve_sampler_step_uses_nearest_sigma(self):
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        self.assertEqual(resolve_sampler_step(1.0, sigmas), 0)
        self.assertEqual(resolve_sampler_step(0.49, sigmas), 1)
        self.assertEqual(resolve_sampler_step(0.0, sigmas), 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_context_protection.ContextProtectionCoreTests -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'context_protection'`.

- [ ] **Step 3: Implement `context_protection.py`**

Use a frozen dataclass. Clamp `late_step` to `0..999`, and clamp `noise_strength` to `0.0..1.0`. Ease the ramp with `(1 - t) ** 2`. Return a mask shaped `[1, 1, count, 1, 1]`. Hash a canonical JSON payload with `hashlib.sha256(...).hexdigest()[:16]`, and return `"off"` when `source_clip_id` is empty or `settings.enabled` is false.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_context_protection.ContextProtectionCoreTests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add context_protection.py tests/test_context_protection.py
git commit -m "feat: add context protection settings and masks"
```

---

### Task 2: Masked Context Builder

**Files:**
- Modify: `context_protection.py`
- Modify: `motion_context_ram.py:423`
- Test: `tests/test_context_protection.py`

**Interfaces:**
- Consumes: Task 1 settings and masks.
- Produces:
  - `MaskedContext(video, audio, video_mask, audio_mask, held_video, held_audio, video_steps, audio_steps, trim_frames)`
  - `build_masked_context(target_video, target_audio, held_video, held_audio, noise_strength=0.5, noise_ease=True, trim_frames=0) -> MaskedContext`
  - `MiniMaxH3MotionContextRAM.apply_masked(latent, context_latent, context_length, audio_context_length, noise_strength=0.5, noise_ease=True) -> tuple[MaskedContext, int]`

- [ ] **Step 1: Write the failing test**

Add a new `MaskedContextTests(unittest.TestCase)` class with these methods:

```python
from context_protection import build_masked_context


def test_masked_context_replaces_only_the_held_prefix(self):
    target_video = torch.arange(1 * 2 * 6 * 4 * 4).reshape(1, 2, 6, 4, 4).float()
    target_audio = torch.zeros(1, 2, 2, 30)
    held_video = target_video[:, :, :3].clone() + 100
    held_audio = torch.ones(1, 2, 2, 15)

    ctx = build_masked_context(
        target_video, target_audio, held_video, held_audio,
        noise_strength=0.5, noise_ease=False, trim_frames=9,
    )

    self.assertTrue(torch.equal(ctx.video[:, :, :3], held_video))
    self.assertTrue(torch.equal(ctx.video[:, :, 3:], target_video[:, :, 3:]))
    self.assertTrue(torch.equal(ctx.audio[..., :15], held_audio))
    self.assertTrue(torch.equal(ctx.audio[..., 15:], target_audio[..., 15:]))
    self.assertEqual(ctx.video_mask.shape, (1, 1, 6, 4, 4))
    self.assertEqual(ctx.video_mask[0, 0, :, 0, 0].tolist(), [0.5, 0.25, 1.0, 1.0, 1.0, 1.0])
    self.assertEqual(ctx.audio_mask.shape, (1, 1, 1, 30))
    self.assertEqual(ctx.audio_mask[0, 0, 0, :16].tolist(), [0.0] * 15 + [1.0])
    self.assertEqual(ctx.trim_frames, 9)

def test_masked_context_rejects_geometry_mismatch(self):
    with self.assertRaises(ValueError):
        build_masked_context(
            torch.zeros(1, 2, 6, 4, 4), torch.zeros(1, 2, 2, 30),
            torch.zeros(1, 2, 3, 8, 8), torch.zeros(1, 2, 2, 15),
        )

def test_masked_context_rejects_an_empty_held_context(self):
    with self.assertRaises(ValueError):
        build_masked_context(
            torch.zeros(1, 2, 6, 4, 4), torch.zeros(1, 2, 2, 30),
            torch.zeros(1, 2, 0, 4, 4), torch.zeros(1, 2, 2, 0),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_context_protection.MaskedContextTests -v`

Expected: FAIL with `ImportError: cannot import name 'build_masked_context'`.

- [ ] **Step 3: Implement the builder and RAM method**

`build_masked_context` clones the target streams, validates channel/spatial equality and prefix length, writes held values into the prefix, builds the video ramp over an all-`1` mask, and returns an all-`1` audio mask with a locked held prefix.

`apply_masked` must:

1. Split `latent` and `context_latent` with `_streams_from_latent`.
2. Reject different batch, channel, or spatial geometry.
3. Get blocks with `_video_tail_from_latent(context_latent, frame_count)`.
4. Concatenate blocks on dim 2 into `held_video`.
5. Get `held_audio`, `audio_steps`, `_` with `_audio_tail_from_latent`.
6. Call `build_masked_context` with `trim_frames` from the video tail span.
7. Return `(masked, trim_frames)`.

Do not modify conditioning. Do not attach `minimax_keyframes`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_context_protection.MaskedContextTests -v`

Expected: PASS.

- [ ] **Step 5: Run the existing Motion Context tests**

Run: `python -m unittest tests.test_clip_motion_context -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add context_protection.py motion_context_ram.py tests/test_context_protection.py
git commit -m "feat: build masked motion context latents"
```

---

### Task 3: Keystone, Capstone, Late Lock, and Restoration

**Files:**
- Modify: `context_protection.py`
- Test: `tests/test_context_protection.py`

**Interfaces:**
- Consumes: Task 2 `MaskedContext`.
- Produces:
  - `ProtectionPlan(masked, keystone, capstone, late_step, source_clip_id, source_index, signature)`
  - `validate_late_step(settings, sigmas) -> None`
  - `make_protection_plan(masked, source_video, source_clip_id, source_index, settings, sigmas, context_length, audio_context_length, refine_scale=1.0) -> ProtectionPlan`
  - `extract_capstone(source_video, block_steps=5) -> torch.Tensor | None`
  - `make_protection_wrapper(plan) -> callable`
  - `install_protection_wrapper(model, plan) -> model`
  - `restore_samples(samples, plan) -> samples`

- [ ] **Step 1: Write the failing tests**

Add a new `ProtectionPlanTests(unittest.TestCase)` class with these helpers and methods:

```python
from context_protection import (
    ProtectionSettings,
    build_masked_context,
    make_protection_wrapper,
    make_protection_plan,
    restore_samples,
)

def _fake_masked():
    target_video = torch.zeros(1, 2, 6, 4, 4)
    held_video = torch.ones(1, 2, 3, 4, 4)
    return build_masked_context(
        target_video, torch.zeros(1, 2, 2, 30),
        held_video, torch.ones(1, 2, 2, 15),
        noise_strength=0.5, noise_ease=False, trim_frames=9,
    )

def _plan_with_held_values():
    settings = ProtectionSettings(True, False, True, 1, 0.5, True)
    return make_protection_plan(
        _fake_masked(), torch.zeros(1, 2, 22, 4, 4), "clip_1", 0, settings,
        torch.tensor([1.0, 0.5, 0.0]), "22", 0,
    )

def test_plan_rejects_late_step_outside_schedule(self):
    settings = ProtectionSettings(True, True, True, 3, 0.5, True)
    masked = _fake_masked()
    with self.assertRaises(ValueError):
        make_protection_plan(
            masked, torch.zeros(1, 2, 22, 4, 4), "clip_1", 0, settings,
            torch.tensor([1.0, 0.5, 0.0]), "22", 0,
        )

def test_plan_skips_capstone_for_short_source(self):
    settings = ProtectionSettings(True, True, True, 1, 0.5, True)
    plan = make_protection_plan(
        _fake_masked(), torch.zeros(1, 2, 5, 4, 4), "clip_1", 0, settings,
        torch.tensor([1.0, 0.5, 0.0]), "22", 0,
    )
    self.assertIsNone(plan.capstone)

def test_protection_wrapper_delays_front_mask_before_late_step(self):
    settings = ProtectionSettings(True, True, True, 1, 0.5, True)
    plan = make_protection_plan(
        _fake_masked(), torch.zeros(1, 2, 25, 4, 4), "clip_1", 0, settings,
        torch.tensor([1.0, 0.5, 0.0]), "22", 0,
    )
    wrapper = make_protection_wrapper(plan)
    seen = {}

    def executor(x, timestep, context, transformer_options, **kwargs):
        seen["t"] = x[0].shape[2]
        seen["front"] = kwargs["denoise_mask"][0, 0, :3, 0, 0].tolist()
        seen["audio_front"] = kwargs["audio_denoise_mask"][0, 0, 0, :3].tolist()
        return [torch.zeros_like(x[0]), torch.zeros_like(x[1])]

    x = [torch.zeros(1, 2, 22, 4, 4), torch.zeros(1, 2, 2, 100)]
    mask = torch.empty(1, 1, 22, 4, 4).fill_(0.5)
    audio_mask = torch.zeros(1, 1, 1, 100)
    wrapper(executor, x, torch.tensor(1000.0), None,
            {"sample_sigmas": torch.tensor([1.0, 0.5, 0.0])},
            denoise_mask=mask, audio_denoise_mask=audio_mask)
    self.assertEqual(seen["t"], 22)
    self.assertEqual(seen["front"], [1.0, 1.0, 1.0])
    self.assertEqual(seen["audio_front"], [1.0, 1.0, 1.0])

def test_protection_wrapper_appends_then_trims_after_late_step(self):
    settings = ProtectionSettings(True, True, True, 1, 0.5, True)
    plan = make_protection_plan(
        _fake_masked(), torch.zeros(1, 2, 25, 4, 4), "clip_1", 0, settings,
        torch.tensor([1.0, 0.5, 0.0]), "22", 0,
    )
    wrapper = make_protection_wrapper(plan)
    seen = {}

    def executor(x, timestep, context, transformer_options, **kwargs):
        seen["t"] = x[0].shape[2]
        seen["mask_t"] = kwargs["denoise_mask"].shape[2]
        seen["front"] = kwargs["denoise_mask"][0, 0, :3, 0, 0].tolist()
        return [torch.zeros_like(x[0]), torch.zeros_like(x[1])]

    x = [torch.zeros(1, 2, 22, 4, 4), torch.zeros(1, 2, 2, 100)]
    mask = torch.ones(1, 1, 22, 4, 4)
    out = wrapper(executor, x, torch.tensor(500.0), None, {"sample_sigmas": torch.tensor([1.0, 0.5, 0.0])},
                  denoise_mask=mask, audio_denoise_mask=torch.ones(1, 1, 1, 100))
    self.assertEqual(seen["t"], 27)
    self.assertEqual(seen["mask_t"], 27)
    self.assertEqual(seen["front"], [0.5, 0.25, 0.0])
    self.assertEqual(out[0].shape[2], 22)

def test_capstone_wrapper_cleans_up_when_executor_raises(self):
    settings = ProtectionSettings(True, True, True, 1, 0.5, True)
    plan = make_protection_plan(
        _fake_masked(), torch.zeros(1, 2, 25, 4, 4), "clip_1", 0, settings,
        torch.tensor([1.0, 0.5, 0.0]), "22", 0,
    )
    wrapper = make_protection_wrapper(plan)
    x = [torch.zeros(1, 2, 22, 4, 4), torch.zeros(1, 2, 2, 100)]
    mask = torch.ones(1, 1, 22, 4, 4)

    def executor(*args, **kwargs):
        raise RuntimeError("sampler failed")

    with self.assertRaises(RuntimeError):
        wrapper(executor, x, torch.tensor(500.0), None,
                {"sample_sigmas": torch.tensor([1.0, 0.5, 0.0])},
                denoise_mask=mask,
                audio_denoise_mask=torch.ones(1, 1, 1, 100))

    self.assertEqual(x[0].shape[2], 22)
    self.assertEqual(mask.shape[2], 22)

def test_restore_samples_replaces_the_held_prefix(self):
    plan = _plan_with_held_values()
    video = torch.full((1, 2, 6, 4, 4), 9.0)
    audio = torch.full((1, 2, 2, 30), 9.0)
    restored = restore_samples((video, audio), plan)
    self.assertTrue(torch.equal(restored[0][:, :, :3], plan.masked.held_video))
    self.assertTrue(torch.equal(restored[0][:, :, 3:], video[:, :, 3:]))
    self.assertEqual(float(restored[0][:, :, :3].min()), 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_context_protection.ProtectionPlanTests -v`

Expected: FAIL with `NameError: make_protection_plan is not defined`.

- [ ] **Step 3: Implement the plan and wrapper**

`make_protection_plan` must replace held video slot 0 with source video latent 0 when keystone is on. Store the updated `MaskedContext`. A capstone is a slice of up to 5 source video latent steps starting at `capstone_index(source_video.shape[2])`. A zero-length or index-zero slice returns `None`.

`validate_late_step` raises `ValueError` when `settings.late_step >= len(sigmas) - 1` or is negative. `make_protection_plan` calls it. It also rejects a capstone index outside source length and a source geometry mismatch.

`make_protection_wrapper` has this exact signature:

```python
def wrapper(executor, x, timestep, context, transformer_options,
            minimax_payload=None, denoise_mask=None,
            audio_denoise_mask=None, **kwargs):
```

Convert `timestep` to sigma with `float(timestep.flatten()[0]) / 1000.0`, then call `resolve_sampler_step(sigma, transformer_options["sample_sigmas"])`. Before `late_step`, clone both masks and set the held video and audio prefixes to `1.0`, then call `executor` without a capstone. At or after `late_step`, use the plan masks, concatenate the capstone on dim 2 of `x[0]`, concatenate an equal zero region on dim 2 of `denoise_mask`, call `executor`, and trim the returned video back to the original length. Do not mutate the caller's tensors.

`install_protection_wrapper` always adds the wrapper under key `h3_context_protection` to `WrappersMP.DIFFUSION_MODEL` for an active plan. The wrapper handles the `capstone is None` case.

`restore_samples` accepts nested or tuple/list `[video, audio]`. It clones the streams, overwrites the first `masked.video_steps` video slots and first `masked.audio_steps` audio slots with held originals, and returns the same container type.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_context_protection.ProtectionPlanTests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add context_protection.py tests/test_context_protection.py
git commit -m "feat: add keystone capstone and latent restoration"
```

---

### Task 4: Source Resolution and Cache Contract

**Files:**
- Modify: `context_protection.py`
- Test: `tests/test_context_protection.py`

**Interfaces:**
- Consumes: Task 1 signature, Task 3 plan.
- Produces:
  - `normalize_protection_source(raw_clip, clip_index, first_clip_id) -> str | None`
  - `resolve_source_index(clips, clip_index, source_clip_id) -> int`
  - `protection_cache_mismatch(segment_meta, expected_signature) -> bool`

- [ ] **Step 1: Write the failing tests**

Add a new `SourceResolutionTests(unittest.TestCase)` class with these methods:

```python
from context_protection import (
    normalize_protection_source,
    protection_cache_mismatch,
    resolve_source_index,
)

def test_source_resolution_rejects_missing_current_and_future(self):
    clips = [{"id": "clip_a"}, {"id": "clip_b"}, {"id": "clip_c"}]
    self.assertEqual(resolve_source_index(clips, 2, "clip_a"), 0)
    for bad in ("missing", "clip_b", "clip_c"):
        with self.assertRaises(ValueError):
            resolve_source_index(clips, 2, bad)

def test_missing_source_defaults_to_first_clip_but_explicit_null_stays_null(self):
    self.assertEqual(normalize_protection_source({}, 2, "clip_a"), "clip_a")
    self.assertEqual(normalize_protection_source({"protection_source": ""}, 2, "clip_a"), "clip_a")
    self.assertEqual(normalize_protection_source({"protection_source": "clip_b"}, 2, "clip_a"), "clip_b")
    self.assertIsNone(normalize_protection_source({"protection_source": None}, 2, "clip_a"))
    self.assertIsNone(normalize_protection_source({}, 0, "clip_a"))

def test_cache_mismatch_treats_missing_signature_as_off(self):
    self.assertFalse(protection_cache_mismatch({}, "off"))
    self.assertTrue(protection_cache_mismatch({}, "abc"))
    self.assertFalse(protection_cache_mismatch({"protection_signature": "abc"}, "abc"))
    self.assertTrue(protection_cache_mismatch({"protection_signature": "abc"}, "off"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_context_protection.SourceResolutionTests -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the helpers**

`resolve_source_index` uses exact stable ID matching and requires `source_index < clip_index`. Raise messages that name the clip number and source ID. `normalize_protection_source` returns `None` for clip 0. It returns `None` when the key is present and null. It keeps a nonempty string. It uses `first_clip_id` when the key is absent or blank. `protection_cache_mismatch` compares `segment_meta.get("protection_signature", "off")` with the expected value.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_context_protection.SourceResolutionTests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add context_protection.py tests/test_context_protection.py
git commit -m "feat: validate protection source and cache signature"
```

---

### Task 5: Base-Pass Integration

**Files:**
- Modify: `extender.py:2554` (`_sample_h3`)
- Modify: `extender.py:3063` (`_parse_clips_json`)
- Modify: `extender.py:4935` (`PROJECT_SETTING_NAMES`)
- Modify: `extender.py:5040` (node widgets)
- Modify: `extender.py:6008` (`extend`)
- Modify: `extender.py:6415-6836` (Ref2VA loop)
- Modify: `motion_context_disk.py:953`
- Test: `tests/test_context_protection.py`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces:
  - `motion_context_disk.latent_at(data_path, manifest, index)`
  - `_sample_h3(..., protection=None)`
  - per-clip fields `protection_source`, `protection_signature`
  - `PROJECT_SETTING_NAMES` includes the six settings

- [ ] **Step 1: Write failing static contract tests**

Add a new `IntegrationContractTests(unittest.TestCase)` class with:

```python
from pathlib import Path


def test_extender_widget_contract(self):
    text = Path("extender.py").read_text(encoding="utf-8")
    for name in (
        "context_protection", "protection_keystone", "protection_capstone",
        "protection_late_step", "protection_noise_strength", "protection_noise_ease",
    ):
        self.assertIn(f'"{name}"', text)

def test_parser_exposes_protection_source(self):
    text = Path("extender.py").read_text(encoding="utf-8")
    self.assertIn('"protection_source"', text)
    self.assertIn('"protection_signature"', text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_context_protection.IntegrationContractTests -v`

Expected: FAIL because the names are absent.

- [ ] **Step 3: Add public disk access and sampling restoration**

Add to `motion_context_disk.py`:

```python
def latent_at(data_path, manifest, index):
    """Return the lazy cached latent for one segment, or raise KeyError."""
    return _proxy_at(data_path, manifest, index)
```

Add `protection=None` to `_sample_h3`. After `guider.sample(...)` returns and before the device move, call `restore_samples(samples, protection)` when protection is not `None`.

- [ ] **Step 4: Add widgets, parser fields, and project names**

Append the six widgets after `color_match_process_clip0` with the exact defaults and limits from the spec. Add the six names to `PROJECT_SETTING_NAMES`. In `_parse_clips_json`, add:

```python
"protection_source": normalize_protection_source(
    raw, i, out[0]["id"] if out else None
)
```

Keep `protection_source` null for clip 0.

- [ ] **Step 5: Wire the base Ref2VA path**

At the start of each Ref2VA clip iteration:

1. Read `protection_settings = normalize_protection_settings(...)`.
2. Set `protection_active = bool(protection_settings.enabled and clip_motion and i > 0 and protection_source is not None)`.
3. When active, resolve `source_index` from `protection_source`.
4. Compute `expected_signature` with `protection_signature(...)`.
5. Call `validate_late_step(protection_settings, sample_sigmas)` and, when refine is enabled, `validate_late_step(protection_settings, _resolve_sample_sigmas(external_sigmas, float(refine_denoise)))` before sampling.
6. Apply cache checks with `protection_cache_mismatch`.

For a validated segment with a protection mismatch, log a warning that names the clip and keep the validated result immutable. For a non-validated segment, set the reuse cutoff to the first mismatch and truncate the causal suffix from that clip. A COMPUTED segment is reusable only when both Motion and protection signatures match.

In the base sampling block, when active:

```python
source_latent = motion_context_disk.latent_at(data_path, current_manifest, source_index)
base_source = _resize_context_latent_spatial(source_latent, resolved_width, resolved_height) if refine_enabled else source_latent
source_video = _streams_from_latent(base_source, "protection source")[0]
masked, trim_frames = motion.apply_masked(
    latent, base_context_proxy, str(context_length), int(audio_context_length),
    protection_settings.noise_strength, protection_settings.noise_ease,
)
base_plan = make_protection_plan(
    masked, source_video, protection_source, source_index,
    protection_settings, sample_sigmas, str(context_length),
    int(audio_context_length), refine_scale,
)
clip_model = install_protection_wrapper(clip_model, base_plan)
```

Then call `_sample_h3(..., protection=base_plan)`. Use `before_sampling` validation so a bad source, geometry, capstone, or late step raises before the sampler starts.

When not active, leave the current `motion.apply` path unchanged.

- [ ] **Step 6: Write segment metadata**

Set:

```python
segment_meta={
    "motion_context": bool(clip_motion),
    "protection_source": protection_source if protection_active else None,
    "protection_signature": expected_signature,
}
```

For protected segments, the signature must match the cache check. For inactive segments, write `"off"`.

- [ ] **Step 7: Run tests**

Run: `python -m unittest tests.test_context_protection tests.test_clip_motion_context -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add extender.py motion_context_disk.py tests/test_context_protection.py
git commit -m "feat: protect ref2va base sampling from context drift"
```

---

### Task 6: Refine-Pass Integration

**Files:**
- Modify: `extender.py:2755` (`_refine_ref2va_sample`)
- Modify: `extender.py:6807` (refine call)
- Test: `tests/test_context_protection.py`

**Interfaces:**
- Consumes: Task 5 base plan and source latent.
- Produces:
  - `_refine_ref2va_sample(..., protection_source_latent=None, protection_settings=None, protection_source_id=None, protection_source_index=None)`
  - refined plan with resized source and held context

- [ ] **Step 1: Write a failing contract test**

```python
def test_refine_accepts_protection_context(self):
    text = Path("extender.py").read_text(encoding="utf-8")
    self.assertIn("protection_source_latent", text)
    self.assertIn("make_protection_plan", text)
    self.assertIn("_resize_context_latent_spatial(protection_source_latent", text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_context_protection.IntegrationContractTests.test_refine_accepts_protection_context -v`

Expected: FAIL.

- [ ] **Step 3: Implement the refine path**

After `upscale_for_refine` and sigma resolution, when protection is active:

1. Resize `protection_source_latent` to the refined width and height with `_resize_context_latent_spatial`.
2. Require `previous_refined_proxy`; raise `RuntimeError` when it is missing.
3. Call `motion.apply_masked` on `up_latent` and the refined proxy.
4. Build the plan with `refine_scale` and the refine sigmas.
5. Call `install_protection_wrapper(clip_model, refine_plan)`.
6. Call `_sample_h3(..., protection=refine_plan)`.

When protection is inactive, keep the current `motion.apply` and keyframe path. Do not apply both paths at once.

- [ ] **Step 4: Pass source state into the refine call**

Pass `protection_source_latent`, `protection_settings`, `protection_source_id`, and `protection_source_index` from the base loop. The source latent must be the original cached source, not the base-resized copy.

- [ ] **Step 5: Run tests**

Run: `python -m unittest tests.test_context_protection tests.test_clip_motion_context -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add extender.py tests/test_context_protection.py
git commit -m "feat: protect ref2va refine sampling from context drift"
```

---

### Task 7: Frontend Source Selector and Project Settings

**Files:**
- Modify: `web/extender.js:81` (`PROJECT_WIDGETS`)
- Modify: `web/extender.js:1085` (`newClip`)
- Modify: `web/extender.js:1105` (`normalizeClipList`)
- Modify: `web/extender.js:6872` (card controls)
- Test: `tests/context-protection.test.mjs`

**Interfaces:**
- Consumes: backend field `protection_source`.
- Produces:
  - six entries in `PROJECT_WIDGETS`
  - `protectionSourceOptions(clips, index)`
  - per-clip selector serialized through `serializeState`

- [ ] **Step 1: Write the failing Node tests**

Copy the VM bootstrap from `tests/clip-motion-context.test.mjs`, then add:

```javascript
test("new clips start with no explicit protection source", () => {
  assert.equal(newClip(0).protection_source, null);
  assert.equal(newClip(1).protection_source, null);
});

test("normalization defaults a missing source to the first clip", () => {
  const clips = normalizeClipList([{ id: "a" }, { id: "b" }, { id: "c", protection_source: null }]);
  assert.equal(clips[0].protection_source, null);
  assert.equal(clips[1].protection_source, "a");
  assert.equal(clips[2].protection_source, null);
});

test("options expose only earlier clips plus None", () => {
  const options = protectionSourceOptions([{ id: "a" }, { id: "b" }, { id: "c" }], 2);
  assert.deepEqual(options.map((item) => item.value), ["", "a", "b"]);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test tests/context-protection.test.mjs`

Expected: FAIL because `protection_source` and `protectionSourceOptions` are absent.

- [ ] **Step 3: Implement serialization and defaults**

Add `protection_source: null` to `newClip`. In `normalizeClipList`, force clip 0 to `null`. For later clips, preserve an explicit `null`, preserve any nonempty string, and use `clips[0].id` when the field is missing or blank.

Add the six protection widget names to `PROJECT_WIDGETS`.

- [ ] **Step 4: Add the card selector**

Add a compact `select` after the Motion button. `protectionSourceOptions(clips, index)` returns `[{value: "", label: "Previous tail"}, ...earlier clips]`. Disable the selector for clip 0, Motion OFF, global protection OFF, and busy states. On change, set `clip.protection_source` to the selected ID or `null`, clear this clip's COMPUTED marker, call `captureNativeWorkflowState`, and call `render`.

- [ ] **Step 5: Run tests**

Run: `node --test tests/context-protection.test.mjs`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add web/extender.js tests/context-protection.test.mjs
git commit -m "feat: add per-clip context protection source selector"
```

---

### Task 8: Documentation and Manual Acceptance

**Files:**
- Modify: `README.md`
- Modify: `THIRD_PARTY_NOTICES.md` (create if absent)

**Interfaces:**
- Consumes: completed implementation.
- Produces: user documentation and attribution.

- [ ] **Step 1: Document behavior**

Add a "Context Protection" section that states:

1. It works only in Ref2VA with Motion Context ON.
2. It is latent-only and does not correct LAB or RGB color.
3. It is off by default.
4. `protection_source` selects an earlier clip. Reordering never remaps it.
5. Keystone, Capstone, Late Lock, and Context Noise map to the six widgets.
6. Changing protection settings invalidates cache from the first affected clip.
7. Validated clips remain valid, but reuse stops at the first mismatch.

- [ ] **Step 2: Add attribution**

Record the Geiru Context Suite URL, GPL-3.0 license, and the fact that this implementation is independent and does not copy its code.

- [ ] **Step 3: Run the full automated suite**

Run:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
node --test tests/context-protection.test.mjs
node --test tests/clip-motion-context.test.mjs
```

Expected: all tests pass.

- [ ] **Step 4: Perform manual acceptance**

Run a three-clip Ref2VA project with Motion Context ON and protection ON:

1. Clip 2 uses Clip 1 as source.
2. Clip 3 uses Clip 1 as source.
3. Compare the clip boundary before and after enabling protection.
4. Verify Clip 1 and Motion OFF runs are unchanged.
5. Verify a source set to the current clip fails before sampling.
6. Verify a reordered source fails instead of silently changing.
7. Verify a cached clip with a changed signature rerenders from that clip.
8. Verify refine-enabled output uses the same protection and restores the held prefix.

- [ ] **Step 5: Commit**

```bash
git add README.md THIRD_PARTY_NOTICES.md
git commit -m "docs: document context protection and attribution"
```

---

## Implementation Record

**Status:** Implemented, merged, and pushed. Manual video acceptance is pending.

**Merge commit:** `440cd61` (`main`).

**Feature commits:**

- `e09c97f` — settings, masks, step lookup, and protection signature
- `366676b` — masked context builder and `apply_masked`
- `24cb648` — Keystone, Capstone, Late Lock, wrapper, and restoration
- `a5910f2` — source resolution and cache signature checks
- `f04456e` — Ref2VA base-pass integration
- `3b14d78` — Ref2VA Refine-pass integration
- `7451836` — per-clip source selector and project settings
- `542ef6d` — README and third-party notices
- `31080cf` — final review fixes for masked samples, nested audio, and mask chaining

### Files added

- `context_protection.py` — settings, masks, plans, wrapper, source resolution, signature, masked-sample installation, and restoration.
- `tests/test_context_protection.py` — CPU-only unit and contract tests.
- `tests/context-protection.test.mjs` — frontend source and serialization tests.
- `THIRD_PARTY_NOTICES.md` — independent-implementation attribution.

### Files changed

- `extender.py` — widgets, parser fields, project settings, base and Refine integration, cache invalidation, segment metadata, and masked-sample sampling.
- `motion_context_ram.py` — masked target-latent context path.
- `motion_context_disk.py` — public `latent_at()` source accessor.
- `web/extender.js` — clip source field, source options, card selector, widget defaults, and master-switch refresh.
- `tests/test_clip_motion_context.py` — parser-test namespace update.
- `README.md` — Context Protection usage, behavior, and manual acceptance steps.

### Implemented behavior

- Protection is active only for Ref2VA with Motion Context ON.
- Protection is off by default.
- The source is a stable earlier clip ID. Missing, current, and future sources fail before sampling.
- Clip 1, Motion Context OFF clips, and explicit `null` sources do not use protection.
- Keystone replaces the oldest held video slot with the source's first video latent.
- Capstone appends the end-aligned source latent block only at and after Late Lock.
- Late Lock locks the Keystone slot before the configured step and frees it at and after that step.
- Context Noise uses a linear or eased ramp and reaches zero at the newest held slot.
- Held audio masks stay unchanged. Held audio latents are restored after sampling.
- Masked video and audio streams are installed into the sampler input.
- Capstone handling preserves tuple, list, and nested audio containers.
- Base and Refine passes use separate protection plans and separate resized source latents.
- Held latents are restored whenever an output exists. Temporary capstones are removed on success and failure.
- Segment metadata stores `motion_context`, `protection_source`, and `protection_signature`.
- A missing signature reads as `off`.
- A non-validated signature mismatch truncates the causal chain from that clip.
- A validated mismatch logs a warning and keeps the validated result immutable.
- COMPUTED reuse requires both Motion Context and protection signatures to match.
- Changing the master switch refreshes the per-clip selector state.

### Implementation rulings

- The effective sigma schedule is derived when the optional `SIGMAS` input is disconnected. This lets signature checks and Late Lock validation use the schedule the sampler will actually run.
- Base sampling installs its protection wrapper on a temporary model clone. The Refine pass therefore cannot inherit the base-resolution plan or stack a second wrapper.
- The empty source option is labelled `None`, because `null` disables protection.
- The six protection widgets remain visible as native rows. No custom settings panel exists.
- Clips added through either Add button default to the first clip ID.
- Manual video acceptance is deferred to the human partner.
- The source selector does not visibly mark an invalid stored source. The backend still fails before sampling.

### Verification

Run on merged `main`:

```text
python -m unittest discover -s tests -p "test_*.py"
→ 102 tests, OK

node --test tests/context-protection.test.mjs
→ 4 pass, 0 fail

node --test tests/clip-motion-context.test.mjs
→ 3 pass, 0 fail
```

Manual acceptance remains open:

1. Generate three Ref2VA Motion Context ON clips with protection ON.
2. Set Clip 2 and Clip 3 to use Clip 1.
3. Compare the boundaries with protection off and on.
4. Confirm Clip 1 and Motion OFF runs are unchanged.
5. Confirm a current-clip or reordered invalid source fails before sampling.
6. Confirm a changed signature rerenders from the first affected clip.
7. Confirm Refine output uses the same protection and restores the held prefix.
