# ReCaVSR final-video upscale

This document describes the **RECAVSR** mode of the MiniMax H3 Extender: what
it does, how it differs from **REFINE**, which files implement it, and what it
does not cover.

## 1. Two upscale paths, and why they differ

| | REFINE | RECAVSR |
|---|---|---|
| Works on | the H3 video **latent**, before sampling | the **finished video file**, after Final Decode |
| Second sampler pass | yes (refine steps + denoise) | no |
| Changes cache geometry | yes | no |
| Invalidates generated clips | yes | **no** |
| Needs a second model | the H3 latent upscaler | the ComfyUI-ReCaVSR node and its weights |
| Output | one file | two files: the decoded source and the upscaled copy |

REFINE is part of generation: the clip is sampled again at the larger latent
size, so its cache entry and the whole chain geometry depend on it. RECAVSR is a
post-process on pixels. It cannot run on a latent, and it cannot change the clip
cache. Therefore the RECAVSR button only acts on the final export.

The two modes are exclusive. Turning RECAVSR on turns REFINE off (which
invalidates clips, because Refine owns the latent geometry). Turning REFINE on
turns RECAVSR off, and its setting values are kept.

## 2. Requirements

1. The **ComfyUI-ReCaVSR** custom node, installed in `custom_nodes` and visible
   in ComfyUI.
2. The ReCaVSR weights: a transformer file, plus `prompt.safetensors` and
   `flashdecoder.safetensors` **in the same folder**.
3. A ComfyUI model folder named `recavsr` that contains those files.

The ReCaVSR node registers `ComfyUI/models/recavsr` by itself. A second drive
needs a line in `extra_model_paths.yaml`:

```yaml
comfyui:
    base_path: H:\
    # ... existing keys ...
    recavsr: comfyui-models\recavsr\
```

Restart ComfyUI after that change. The Extender reads the same folder through
`folder_paths`, so any folder the ReCaVSR loader can see also works for the
Extender.

## 3. Using it

1. Build the clips as usual.
2. Press **RECAVSR** in the toolbar, right after **REFINE**.
3. Open **⚙ RECAVSR SETTINGS** and set:

   | Setting | Range | Default | Meaning |
   |---|---|---|---|
   | Scale | 1.0 to 4.0, step 0.05 | 2.0 | Output size = input size x scale, each dimension rounded half up |
   | Colour fix | `none`, `adain`, `wavelet` | `none` | Per-frame colour match to the source |
   | Seed | 0 to 4294967295 | 42 | Seed of the one-shot ReCaVSR noise |

   The panel also shows **Final video: W × H**, the expected upscaled size.
4. Queue with **run_mode = full_batch**. The ReCaVSR pass runs once, after the
   final container exists.

The button is disabled with the tooltip **Install ReCaVSR first.** when the
ReCaVSR node is not loaded. The tooltip also explains a missing transformer file
after the first run reports the model folder as empty.

## 4. Output files

Two files stay on disk:

- `prefix.mp4` (or `prefix_00001.mp4`, following the existing numbering): the
  decoded source.
- `prefix_recavsr.mp4` (or `prefix_recavsr_00001.mp4`): the upscaled video.

The Final Decode node returns the upscaled file, and the workflow metadata is
embedded in it.

The `filename_prefix` token `%date:yyyy-MM-dd%` and `%time:...%` are expanded by
the Extender itself, and a slash in the prefix creates that subfolder. Without a
prefix, files go straight into the output directory.

ReCaVSR encodes the upscaled video with its own encoder. The Final Decode
`codec`, `crf` and `preset` therefore describe the **source** file only.

## 5. Covered paths

| Path | Covered |
|---|---|
| Full Batch, Ref2VA + Motion Context | yes |
| Full Batch, Ref2VA independent | yes |
| Full Batch, FL2VA | yes |
| `clip_by_clip` progressive autosave | no |
| `save_individual_clips` files | no |

`clip_by_clip` keeps its fast preview and autosave un-upscaled, because the pass
belongs to the finished timeline. Run the same cache with `full_batch` to
upscale it.

## 6. Files touched

- `recavsr_pass.py` — new. `apply_recavsr()` normalises the settings, finds the
  ReCaVSR classes, loads and caches the transformer, calls
  `ReCaVSRUpscale.execute()`, and moves the produced temp file to
  `<prefix>_recavsr.mp4`. It never imports ReCaVSR at module load, so the
  Extender works without the node.
- `extender.py` — four widgets appended after every legacy widget
  (`recavsr_enabled`, `recavsr_scale`, `recavsr_color_fix`, `recavsr_seed`);
  `extend()`, `_extend_fl2va()` and `_run_ref2va_independent()` write
  `manifest["recavsr_mode"]`; `PROJECT_SETTING_NAMES` carries the four names so
  `.ext` projects round-trip; the returned UI state gains
  `recavsr_node_available` and `recavsr_transformer`.
- `motion_context_disk.py` — the Full Batch export calls `apply_recavsr()` on
  `output_path` before it embeds metadata and returns the video.
- `fl2va_engine.py` — the FL2VA Full Batch export does the same. Ref2VA
  independent reuses that exporter, so it is covered by the same call.
- `ref2va_independent.py` — receives and stores the `recavsr` settings.
- `web/extender.js` — the **RECAVSR** button inside a tooltip wrapper, the
  `RECAVSR SETTINGS` panel, `recavsrNodeAvailable()` against
  `LiteGraph.registered_node_types`, the exclusivity helper
  `toggleUpscaleMode()`, hidden native widgets, and the extra panel height.

Line numbers move; search for the function names above instead.

## 7. Tests

No GPU, no weights, and no ComfyUI process are needed.

```powershell
# JavaScript: exclusivity, panel height, node probe
node tests/recavsr-mode.test.mjs

# Python: settings clamp, error messages, stub-node copy, wiring
python tests/test_recavsr_pass.py
```

## 8. Memory

ReCaVSR keeps the transformer resident, like any ComfyUI node. The node streams
weights that do not fit. Activations cannot stream, so the output size sets the
floor.

The node's own constant is 2.6 MiB of activation per 32×32 output token. These
values are **derived** from that constant, not measured on this machine:

| Source | Scale | Output | Tokens | Activations |
|---|---|---|---|---|
| 640×480 | 2.0 | 1280×960 | 1200 | about 3.0 GiB |
| 896×576 | 2.0 | 1792×1152 | 2016 | about 5.1 GiB |
| 640×480 | 4.0 | 2560×1920 | 4800 | about 12.2 GiB |
| 896×576 | 4.0 | 3584×2304 | 8064 | about 20.5 GiB |

Weights are 10.6 GB in bf16, or 6.27 GB for the int8 ConvRot file. On a 16 GB
card, scale 2.0 at 896×576 is the practical default; 4.0 at that size needs
about 24 GB total. When the activations do not fit, the ReCaVSR node raises a
`ValueError` that names a workable scale.

## 9. Error messages

| Message | Cause | Fix |
|---|---|---|
| `Install ReCaVSR first.` | the ReCaVSR node is not loaded | install it, restart ComfyUI |
| `ReCaVSR is installed, but no transformer is visible in the recavsr model folder.` | the `recavsr` folder is empty or unregistered | add the `extra_model_paths.yaml` line, or move the weights into `ComfyUI/models/recavsr` |
| `ReCaVSR at WxH needs about N GiB of VRAM ...` | the output is too large for the card | lower Scale |
| `ReCaVSR returned no video file.` | the node changed its return shape | update ComfyUI-ReCaVSR |
