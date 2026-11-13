# H3 Extender Latent Context Protection Design

- Date: 2026-11-03
- Status: Awaiting user review
- Scope: Ref2VA with Motion Context ON
- Source reference: [Geiru Context Suite](https://github.com/roycho87/geiru_context_suite)

## 1. Purpose

Long H3 chains can lose detail, texture, and color consistency. Each clip starts from the output of the previous clip. The loss therefore accumulates.

This design adds latent-based protection to the MiniMax H3 Extender. The feature uses fixed source latents, delayed guidance, and controlled context noise to reduce cumulative drift.

The feature does not perform color grading. It does not use LAB conversion, histogram matching, or reverse-engineered model behavior.

## 2. Goals

- Protect long chains of Ref2VA clips with Motion Context ON.
- Preserve motion continuity across clip joins.
- Pull later clips toward the appearance of a selected earlier source clip.
- Reduce exact copying of held context frames.
- Allow a late sampler-step lock to preserve new poses.
- Keep protection off by default.
- Keep existing workflows valid.
- Keep the current keyframe path unchanged when protection is inactive.

## 3. Non-goals

- No LAB color correction.
- No direct RGB grading.
- No reverse engineering of undocumented model behavior.
- No protection for FL2VA.
- No protection for independent Ref2VA with Motion Context OFF.
- No protection for Clip 0.
- No automatic hard-cut detection.
- No copying of GPL-3.0 source code.

## 4. Success criteria

- A 20-clip continuous scene keeps appearance near clip 1.
- The protected chain has less drift than the same chain without protection.
- Motion crosses each clip join without a visible freeze.
- A hard cut can use a new source clip without pulling the old scene into the new one.
- Protection OFF produces the existing sampling path.
- A changed protection setting invalidates affected cached segments.
- A failed sample does not leave a modified latent or model behind.

## 5. How the protection works

The design adapts three mechanisms from Geiru Context Suite.

### 5.1 Context protection

The previous shot’s tail enters the next shot’s target latent as held context. A denoising mask locks the held latents.

This differs from the current H3 Extender Motion Context path. The current path attaches the previous tail as `minimax_keyframes` conditioning. The protected path uses real target-latent context. This change is necessary for mask ramps and exact post-sampling restoration.

### 5.2 Keystone and capstone

The keystone is the first video latent of a selected source clip. It replaces the earliest held latent in each protected shot.

The capstone is a block from the source clip’s final latent region. It temporarily joins the model input near the end. It guides attention but never becomes an output frame.

The keystone and capstone pull every shot toward the same source appearance. Late Lock delays their full effect.

### 5.3 Late Lock and Context Noise

Late Lock changes the front mask and capstone engagement by sampler step. Early steps can establish the new pose. Later steps receive stronger source guidance.

Context Noise gives held context latents a fractional mask. The mask decreases toward the context join. H3 applies sampler noise through the partial masks. This reduces exact copying.

The original held latents are restored after sampling. Protected output therefore keeps a clean context join.

## 6. Architecture

### 6.1 New module

Add `context_protection.py`.

This module owns:

- Protection configuration normalization.
- Source clip resolution.
- Keystone and capstone extraction.
- Context Noise ramp calculation.
- Mask construction.
- Model wrapper installation.
- Late Lock step detection.
- Held-latent restoration.
- Protection signature generation.

Keep the module independent from UI code and node registration.

### 6.2 Existing modules

Modify these modules:

- `extender.py`: add node controls, orchestrate the protected path, pass the plan to sampling.
- `motion_context_ram.py`: add a masked-context method that reuses tail extraction and block alignment.
- `motion_context_disk.py`: store protection metadata in segment descriptors when required.
- `web/extender.js`: add global controls, per-clip source selection, normalization, and serialization.
- `README.md`: document the feature and its limits.
- `THIRD_PARTY_NOTICE.md`: add Geiru attribution and explain that the implementation is independent.

Do not add an external runtime dependency on Geiru Context Suite.

### 6.3 Protection plan

Use a dataclass named `ProtectionPlan`.

It contains:

- Source clip ID.
- Source keystone tensor.
- Source capstone tensor, or `None`.
- Held video originals.
- Held audio originals.
- Video mask ramp.
- Effective Late Lock step.
- Protection signature.
- Flags for Keystone, Capstone, and Context Noise.

The plan is immutable after construction. Sampling uses a separate mutable restoration record.

### 6.4 Data flow

For each Ref2VA clip with Motion Context ON and an active source:

1. Resolve the source clip by stable ID.
2. Verify that the source precedes the current clip.
3. Verify source geometry and channels.
4. Extract previous-shot video and audio tails.
5. Copy those tails into the target AV latent.
6. Store exact copies of all held original latent positions.
7. Build nested video and audio noise masks.
8. Apply the Context Noise ramp to video masks.
9. Keep audio masks fully locked.
10. Replace the earliest held video latent with the source keystone.
11. Extract the source capstone for model-input guidance.
12. Clone the per-clip model.
13. Install capstone, Late Lock, and restoration wrappers.
14. Sample with the resolved sigma schedule.
15. Restore all held latent positions to their original values.
16. Remove temporary capstone data.
17. Return the normal H3 Extender latent shape.
18. Store the protection signature in the segment.

When protection is inactive, use the current `MiniMaxH3MotionContextRAM.apply()` keyframe path.

## 7. Configuration and UI

### 7.1 Node controls

Append these widgets after existing widgets:

| Widget | Type | Default | Meaning |
| --- | --- | --- | --- |
| `context_protection` | Boolean | Off | Master switch |
| `protection_keystone` | Boolean | On | Enable front source latent |
| `protection_capstone` | Boolean | On | Enable back source guidance |
| `protection_late_step` | Integer | 1 | Zero-based sampler step for Late Lock |
| `protection_noise_strength` | Float | 0.5 | Maximum Context Noise mask value |
| `protection_noise_ease` | Boolean | On | Use quadratic mask decay |

Use these limits:

- `protection_late_step`: minimum `0`.
- `protection_noise_strength`: range `0.0` to `1.0`.

Append the widgets to preserve positional `widgets_values` compatibility.

### 7.2 Per-clip source

Add `protection_source` to each clip record.

- Store a source clip’s stable ID.
- Store `null` for no protection on that clip.
- Clip 1 always uses `null`.
- New clip 2 and later default to the first clip ID.
- A dropdown lists earlier clips and `None`.
- Do not list the current clip or future clips.
- Do not silently rewrite a source after reorder.
- Show invalid or missing sources as an error.

The transition clip at a hard cut should use `None`. The next clip should use the first clip of the new scene.

### 7.3 Serialization

- Include `protection_source` in clip normalization.
- Include it in project save and load.
- Include it in `clips_json`.
- Preserve it across mode switches.
- Remove it only if the project is reset.

## 8. Temporal and mask rules

### 8.1 Held context

- Reuse current H3 block alignment.
- Align context to complete latent blocks.
- Keep the existing `context_length` and `audio_context_length` semantics.
- Reject a protected clip when no real held context exists.
- Reject a protected clip when `context_length` produces zero held blocks.

### 8.2 Keystone index

Use source video latent index `0`.

### 8.3 Capstone index

Use:

```text
capstone_index = 5 * ((source_video_latent_length - 1) // 5)
```

Use a 17-frame source block.

If the capstone index equals the keystone index, skip Capstone and log a warning.

### 8.4 Context Noise ramp

Order held video positions from oldest to newest.

For `n` held positions and index `i`:

```text
position = i / max(1, n - 1)
linear_weight = 1 - position
ease_weight = (1 - position) ** 2
weight = ease_weight if protection_noise_ease else linear_weight
mask_value = protection_noise_strength * weight
```

The newest held position receives mask `0`.

When only one held position exists, use the full configured strength.

Keep held audio masks at `0`.

H3 mask convention: `1` means generate and `0` means preserve the clean latent. The generated region therefore keeps mask `1`. Only the held prefix is lowered by the ramp.

The sampler must receive this mask through `guider.sample(denoise_mask=...)`. `KSamplerX0Inpaint` uses it to mix the clean held latents into `x` at every step. A model-only mask cannot replace that mix.

### 8.5 Front mask during Late Lock

- Before the Late Lock step, set the keystone slot mask to `1`.
- At and after the Late Lock step, set the keystone slot mask to `0`.
- Chain any existing denoising mask function.
- Leave audio mask values unchanged.

## 9. Capstone model input

### 9.1 Append

- Append the capstone to the video latent list during a model call.
- Add a matching zero-valued denoising mask.
- Update multimodal latent shapes.
- Keep audio shapes unchanged.
- Do not modify text conditioning.

### 9.2 Late engagement

- Before the Late Lock step, call the model without the capstone.
- At and after the Late Lock step, call the model with the capstone.
- Determine the step from the active sigma schedule.
- Do not count model calls.
- Support normal and external sigma schedules.

### 9.3 Remove

- Remove the temporary capstone from model output.
- Restore the original output shapes.
- Remove the capstone before returning to the sampler.
- Reuse this behavior when another Extender wrapper is present.
- Avoid duplicate capstone wrappers.

## 10. Restoration

Create a restoration record before any latent modification.

The record maps each held temporal index to its original video or audio latent.

After sampling:

- Replace every held index with its original latent.
- Keep generated frames unchanged.
- Remove temporary capstone slots.
- Validate output shape.
- Log restore failures without discarding a valid sampled output.
- Restore held latents whenever a sampled output exists.
- Remove the temporary capstone on both success and failure paths.
- Never return a latent that still contains the temporary capstone.

Restoration is required because the keystone and Context Noise replace real context positions.

## 11. Sampling and refine passes

Apply protection to both passes:

- Base Ref2VA pass.
- Latent refine pass.

For the refine pass:

- Rebuild the plan for the refined latent geometry.
- Resize the source keystone and capstone with the existing spatial resize helper.
- Rebuild masks for the refined context window.
- Resolve Late Lock from the refine sigma schedule.
- Restore held refined positions after sampling.
- Preserve first-pass audio through the current helper.

Compute the effective step separately for each pass.

## 12. Cache and invalidation

### 12.1 Protection signature

Build a stable signature from:

- Canonical source clip ID.
- Keystone enabled state.
- Capstone enabled state.
- Late Lock step.
- Context Noise strength.
- Context Noise ease setting.
- Effective sigma schedule.
- Protected context length.
- Audio context length.
- Refine scale when protection is active in refine.

Use a short SHA-256 digest.

### 12.2 Segment metadata

- Add `protection_signature` to protected segment descriptors.
- Add `protection_source` to the descriptor.
- Keep these fields out of exported video metadata.
- Include them in project diagnostics.

### 12.3 Reuse

- Reuse a segment only when its signature matches.
- Treat a missing signature as protection off.
- An inactive source produces the canonical `off` signature.
- A signature mismatch invalidates that segment.
- Invalidate later causal Motion ON segments through the existing chain truncation.
- Keep Motion OFF segments unchanged.
- Keep validated clips immutable.
- Warn when a protection change cannot apply because a clip is validated.
- Require explicit de-validation.

## 13. Errors

Fail with a clear message before sampling when:

- The source clip is missing.
- The source is the current clip or a future clip.
- The source has no cached latent.
- Source channels differ.
- Source spatial geometry differs.
- No held context exists.
- The capstone has no valid block.
- The configured Late Lock step is outside the effective schedule.
- A required mask or latent shape is unsupported.

Do not silently fall back to unprotected sampling when protection is requested.

A `None` source is an explicit no-protection choice.

## 14. Compatibility

### 14.1 Preserved behavior

- Protection off keeps the current keyframe Motion Context path.
- Motion Context OFF stays independent.
- FL2VA keeps its current behavior.
- Independent Ref2VA keeps its current behavior.
- Clip 0 keeps its current guide behavior.
- Color Match still runs after decoding.
- Per-clip color controls still run after decoding.
- Existing nodes and outputs keep their contracts.

### 14.2 Model wrappers

- Clone the per-clip model before installing wrappers.
- Apply LoRAs before protection wrappers.
- Chain upstream model wrappers.
- Avoid mutating a shared model.
- Avoid duplicate wrappers after repeated execution.

### 14.3 Widget compatibility

- Append new widgets.
- Normalize missing per-clip fields.
- Keep old workflows loadable.
- Keep old project archives loadable.

## 15. Testing

Add Python tests in `tests/test_context_protection.py`.

Cover:

- Context block alignment.
- Keystone selection.
- Capstone selection.
- Short-source capstone fallback.
- Single held position.
- Linear and eased Context Noise.
- Nested video and audio mask shapes.
- Audio lock values.
- Late Lock before and after the step.
- Nearest-sigma step resolution.
- External sigma schedules.
- Capstone append and removal.
- Output shape preservation.
- Exact restoration of all held positions.
- Restoration after a sampling exception.
- Wrapper composition.
- Refined geometry source resizing.
- Protection signature stability.
- Signature changes after each setting change.

Add Node tests in `tests/context-protection.test.mjs`.

Cover:

- Default clip fields.
- Missing-field normalization.
- Source serialization.
- Project save and load.
- Mode switching.
- Widget append compatibility.

Run the existing test suite after implementation.

Perform a manual acceptance test:

1. Render a 20-clip continuous scene without protection.
2. Render the same scene with protection.
3. Compare clip 1 with clip 20.
4. Confirm reduced drift.
5. Confirm motion continuity.
6. Insert a hard cut.
7. Set the transition clip source to `None`.
8. Set the new scene’s source to its first clip.
9. Confirm the old scene does not pull the new scene.
10. Disable protection and confirm the old path still runs.

## 16. Documentation and licensing

Update `README.md` with:

- The difference between latent protection and color correction.
- Node settings.
- Per-clip source selection.
- Hard-cut guidance.
- Cache invalidation behavior.
- Supported modes.
- Manual acceptance steps.

Update `THIRD_PARTY_NOTICE.md` with:

- Geiru Context Suite attribution.
- The GPL-3.0 source license.
- A statement that this implementation is independently written.
- The specific mechanisms adapted.

Do not copy Geiru source code, comments, or assets into this Apache-2.0 project.

## 17. Risks

- Masked target context can produce different motion than the current keyframe path.
- Late Lock can cause pose attraction when the step is too early.
- Context Noise can reduce seam stability when strength is too high.
- Capstone guidance adds one video latent to model calls.
- Additional saved originals increase peak latent memory.
- Refine support doubles protection paths and test coverage.
- Hard cuts need explicit user source changes.
- Static shots can still darken slowly.
