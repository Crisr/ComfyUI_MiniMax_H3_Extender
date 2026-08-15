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
