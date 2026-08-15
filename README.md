# ComfyUI MiniMax H3 Extender

A ComfyUI custom node for **MiniMax H3** designed to generate long, continuous video sequences from multiple clips while preserving motion, visual continuity, and audio continuity between generations.

The node combines **Ref2VA conditioning, Motion Context, disk caching, multi-clip generation, image references, audio references, and final video/audio decoding** into a much simpler workflow.

---
<img width="2557" height="1212" alt="Capture d&#39;écran 2026-08-15 083401" src="https://github.com/user-attachments/assets/99ca1fc4-d8b9-4662-a869-1fa06e5e58e1" />

## Keeping Image References Consistent Across Clips

When using image references, it is strongly recommended to define the subjects explicitly at the beginning of **every clip prompt**.

MiniMax H3 does not automatically remember the semantic role of each reference from one prompt to the next.

To keep the same characters, identities, clothing and environment consistent across multiple generated clips, repeat the same `subject_definitions` block at the beginning of each prompt.

Example:

    subject_definitions:
    <Picture 1> is the reference image defining the exact visual appearance, identity, face, hairstyle, body proportions, clothing, accessories, and overall look of <Subject 1>, as well as the established environment and visual context of the scene.
    <Picture 2> is the reference image defining the exact visual appearance, identity, face, hairstyle, body proportions, clothing, accessories, and overall look of <Subject 2>.
    <Subject 1> is the exact same woman shown in <Picture 1>.
    <Subject 2> is her friend, the exact same woman shown in <Picture 2>.
    <Subject 3> is the same street environment and scene context established in <Picture 1>.

This block should be placed at the **very beginning of every clip prompt**.


The definitions should stay identical from clip to clip unless you intentionally want to change the role of a reference.

This is especially important when several image references are connected.

For example:

    <Picture 1> → <Subject 1>
    <Picture 2> → <Subject 2>
    <Picture 3> → <Subject 3>

Keeping these mappings explicit in every prompt helps MiniMax H3 preserve the intended identity and role of each reference throughout the whole sequence.

The image references themselves are shared by the Extender across all clips. The `subject_definitions` block tells MiniMax H3 how those references should be interpreted in each individual clip.

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

---

# Installation

## ComfyUI Manager

Search for:

    MiniMax H3 Extender

and install it directly from ComfyUI Manager.

## Manual Installation

Open a terminal in:

    ComfyUI/custom_nodes/

Then run:

    git clone https://github.com/tritant/ComfyUI_MiniMax_H3_Extender.git

Restart ComfyUI after installation.
