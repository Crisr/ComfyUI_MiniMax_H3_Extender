# ComfyUI MiniMax H3 Extender

A ComfyUI custom node for **MiniMax H3** designed to generate long, continuous video sequences from multiple clips while preserving motion, visual continuity, and audio continuity between generations.

The node combines **Ref2VA conditioning, Motion Context, disk caching, multi-clip generation, image references, audio references, and final video/audio decoding** into a much simpler workflow.

---

## Features

- Multi-clip MiniMax H3 generation
- Continuous Motion Context between clips
- Video and audio latent continuity
- Disk-based latent cache
- Clip-by-clip generation workflow
- Full batch generation mode
- Per-clip prompt
- Per-clip seed
- Seed modes:
  - Randomize
  - Fixed
  - Increment
  - Decrement
- Per-clip duration
- Clip validation system
- Dynamic image reference inputs
- Up to **9 image references**
- Optional **audio reference**
- Shared references automatically applied to all clips
- Native MiniMax H3 Ref2VA conditioning
- Native ComfyUI sampling progress
- Final video preview
- Final video + audio export
- Seam correction between generated clips
- Audio seam correction / declick
- H.264, H.265 / HEVC and FFV1 export
- Persistent disk cache allowing generation to be resumed

---
# Clip Validation

Each clip has a **Validated** checkbox.

The validation system allows you to progressively build a long sequence without regenerating clips that have already been accepted.

When a clip is validated:

- its generated latent remains stored in the disk cache;
- the clip is locked and reused directly;
- it becomes the Motion Context source for the next clip;
- it will not be sampled again while it remains valid.

A typical workflow is:

    Clip 1 → Generate
    Clip 1 → Validate

    Clip 2 → Generate
    Clip 2 → Retry if needed
    Clip 2 → Validate

    Clip 3 → Generate
    Clip 3 → Retry if needed
    Clip 3 → Validate

This makes it possible to build a sequence one clip at a time while preserving all previously accepted generations.

Validation always forms a continuous chain from the beginning of the sequence.

For example:

    Clip 1 ✅
    Clip 2 ✅
    Clip 3 ⬜
    Clip 4 ⬜

If an earlier clip is unvalidated, every clip after it is automatically unvalidated as well, because each continuation depends on the previous generated clip.

Changing a generation parameter also invalidates the sequence when necessary. This includes changes to:

- prompt
- seed
- duration
- model
- sampling settings
- image references
- audio reference
- Motion Context settings

The affected clip and all dependent clips after it must then be regenerated.

In **clip_by_clip** mode, the Extender works on the first unvalidated clip in the sequence.

The intended workflow is therefore:

    Generate → Preview → Retry if needed → Validate → Continue

This allows long MiniMax H3 sequences to be created progressively without repeatedly regenerating clips that are already approved.
