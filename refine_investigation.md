# H3 Extender refine pass — investigation

Status: **root cause found: a VRAM cliff at about 0.7 MP. Not refine-specific. Decision: cap refine at 1056x608 and keep Veda; see section 13**.
Node: `ComfyUI_MiniMax_H3_Extender`
Repo: `G:\comfy-cu133\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Extender`
Log: `G:\comfy-cu133\ComfyUI\user\comfyui.log`
Last update: 2026-11-26 14:19 local. Refine completed at 41:28 (1244 s/it). First pass alone at 1216x672 is also slow (~700 s/step).

---

## 1. Question

1. Does the refine step (not ReCaVSR) use the same model, CLIP, LoRAs and attention method as the first pass?
2. Why did refine become very slow after enabling Veda sparse attention in step 1?

User constraints: keep Veda in step 1, because step 1 is fast.

---

## 2. Answer to question 1: yes, same objects

| Item | First pass | Refine | Evidence |
|---|---|---|---|
| MODEL patcher | `base_sample_model` | `clip_model` | `extender.py:7184`, `extender.py:7202` |
| Per-clip LoRAs | applied once at `extender.py:7046` | same patcher, not re-applied | `_apply_per_clip_loras`, `extender.py:3069` |
| CLIP / text encoder | untouched by LoRAs | same encoder, conditioning rebuilt at the new size | `extender.py:3069-3079` |
| Sampler / scheduler | `sampler_name`, `scheduler` | same strings | `_sample_h3` call, `extender.py:2908` |
| Attention | Veda override in `model_options` | same override, re-installed every step | `veda_comfy/comfy_patch.py:629-644`, `comfy/model_patcher.py:1485` |
| Guider | `_BasicGuider` (subclass of `CFGGuider`) | same | `extender.py:2515`, `extender.py:2590` |

`ModelPatcher.clone()` deep-copies `model_options` (`comfy/model_patcher.py:458`), so the Veda override survives the LoRA clone and the protection clone.
The H3 layout is rebuilt per forward from the live latent shape (`comfy/ldm/minimax/model.py:865-868`).

### Real differences

1. Latent geometry: `refine_scale` 1.25 -> 1216x672 from a 960x544 base.
2. `refine_steps`, `refine_denoise`, trimmed sigma schedule (`_resolve_sample_sigmas`).
3. Upscaler `learned` loads `LatentResizer3D` (658.57 MB). `trilinear` skips it.
4. `_release_models_before_refine` (`extender.py:2724-2770`) offloads VAE + text encoder only. Never the DiT. No `torch.cuda.empty_cache()` anywhere in the pack.
5. Protection wrapper clone, only when Context Protection is active.

---

## 3. Measured data

### 3.1 First pass with full Veda diagnostics (08:05:40 -> 08:09:02, 6 steps, 200 s, 33.7 s/step)

```
Geometry: 960x544, 447 frames (18.6 s) | latent 34x60 x 132, tokens 17x30 x 132
Plan: 16x9_t102 searched on latent 48x84 x 102, tokens 24x42 x 102 | padding 30.5%
Sequence: 73259 tokens | target 132x17x30, ref_img 1x17x30, ref_img 1x28x18, ref_img 1x28x18
Selection: generated 32 tiles | reference 32 tiles
Veda attention time: 116.05 s (19.34 s per model call x 6)
  gather 7.66 s | score 7.77 s | select 2.69 s | kernel 95.44 s | scatter 2.49 s
Backend: Triton INT8 (SM120) on cuda:0
Video tiles kept: 6.5% of video x video tile pairs
Key tiles per query tile: 50 min | 50.0 mean | 50 max
Chunks: 9 per layer, 7 heads each (1.7 GB free)
Attention workspace: 1662 MB at once
Attention calls: 300 sparse
Attention computed: 28.5% of full attention (71.5% skipped)
```

Attention = **116 s of 200 s** (58 %). Non-attention part = **14.4 s/step**.

### 3.2 Earlier passes (verbose off)

| Time | Pass | Geometry | Tokens | Steps | Wall | s/it |
|---|---|---|---|---|---|---|
| 06:53:23 -> 06:55:46 | first | 864x480, 16.5 s | 15x27x117 | 6 | 143 s | 23.8 |
| 07:08:06 -> 07:11:29 | first | 960x544, 18.6 s | 17x30x132 | 6 | 203 s | 33.7 |
| 07:11:48 -> 07:13:23 | refine | 1216x672, 18.6 s | 21x38x132 | 2 | 94.3 s, aborted, no bar | >= 47.2 |
| 07:14:39 -> 07:18:48 | first | 1056x608, 18.6 s | 19x33x132 | 6 | 249 s | 41.6 |
| 07:19:42 -> 07:26:58 | first, clip 2 | 1056x608, 20.0 s | 19x33x142 | 6 | 436.9 s aborted | ~73 (outlier) |
| 07:29:45 -> 07:33:11 | first | 960x544, 18.6 s | 17x30x132 | 6 | 205 s | 33.9 |
| 07:38:12 -> 07:41:58 | first, clip 2 | 960x544, 20.0 s | 17x30x142 | 6 | 226 s | 38.3 |
| 07:44:14 -> 07:47:24 | first, clip 3 | 960x544, 20.0 s | 17x30x142 | 6 | 230 s | 37.5 |

The 07:19 outlier points to VRAM pressure: the same clip and size ran 5 minutes later at 37.7 s/it.

### 3.3 Refine of the current run (completed, 08:09:16 -> 08:50:08)

```
08:09:02.359 Requested to load LatentResizer3D
08:09:02.911 loaded completely; 658.57 MB loaded, full load: True
08:09:08.635 Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14955MB Staged.
08:09:16.332 H3 Refine preflight: offloaded via ComfyUI: MiniMaxH3TEModel_, MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
08:09:16.389 0 models unloaded.
08:09:16.502 Model MiniMaxH3 prepared for dynamic VRAM loading. 15080MB Staged. 208 patches attached.
08:09:16.876 Veda: Veda running | Triton INT8 (SM120)
Video: 1216x672 | 18.6 s
Tile plan: nearest trained size: 1344x768 | 14.4 s (this video: 1216x672 | 18.6 s)
Sparsity: generated 32 tiles | reference 32 tiles
References: 3 spans (tiled, 32 tiles sparse)
08:09:16.877 Veda: 17% of this video's tiles are padding, computed but empty, against 7% at the size this plan was built for.
08:50:08.041 2/2 [41:28<00:00, 1244.39s/it]
08:50:08.133 Veda: Veda done | Triton INT8 (SM120)
```

Diagnostics block at 08:50:08:

```
Attention computed: 16.2% of full attention (83.8% skipped)
Geometry: 1216x672, 447 frames (18.6 s) | latent 42x76 x 132, tokens 21x38 x 132
Plan: 16x9_t102 searched on latent 48x84 x 102, tokens 24x42 x 102 | padding 17.1% (that plan's own grid 7.0%)
Sequence: 111275 tokens | target 132x21x38, ref_img 1x17x30, ref_img 1x28x18, ref_img 1x28x18
Selection: generated 32 tiles | reference 32 tiles
Veda attention time: 1759.21 s (879.60 s per model call x 2)
  gather 10.26 s | score 31.24 s | select 2.08 s | kernel 1711.12 s | scatter 4.51 s
Backend: Triton INT8 (SM120) on cuda:0
Video tiles kept: 5.0% of video x video tile pairs
Key tiles per query tile: 50 min | 50.0 mean | 50 max
Chunks: 19 per layer, 3 heads each
Attention workspace: 2485 MB at once
Attention calls: 100 sparse
Predictor: minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors | 50 layers x 56 heads | trained 90% sparse | step 100
```

Settings: `refine_scale = 1.25`, `refine_upscaler = learned`, `refine_denoise = 0.4`, `refine_steps = 2`.
`ComfyUI-Ref2VA-VSA` and `H3-Optimizations` are disabled in this workflow.
`refine_steps = 2` produced exactly **2 model calls** (2/2 bar), so `refine_denoise = 0.4` adds no call.
Fixed pre-refine overhead = 08:09:02.3 -> 08:09:16.876 = **14.5 s** (upscaler load + TE restage + DiT reload).

### 3.4 Refine vs first pass, per model call

| Item | First pass (3.1) | Refine (3.3) | Ratio |
|---|---|---|---|
| Sequence tokens | 73,259 | 111,275 | 1.52x |
| Computed fraction | 28.5 % | 16.2 % | 0.57x |
| Attention total | 19.34 s | 879.60 s | 45.5x |
| `kernel` | 15.91 s | 855.56 s | **53.8x** |
| `gather` | 1.28 s | 5.13 s | 4.0x |
| `score` | 1.30 s | 15.62 s | 12.0x |
| `scatter` | 0.42 s | 2.26 s | 5.4x |
| Chunks per layer | 9 (7 heads) | 19 (3 heads) | 2.1x |
| Workspace | 1662 MB | 2485 MB | 1.5x |
| Non-attention per step | 14.3 s | 346 s | **24x** |
| Step time | 33.7 s | 1244.39 s | **36.9x** |

### 3.5 First pass alone at 1216x672 (14:12, aborted at 14:17)

Refine was **off**. Same geometry as the refine. Interrupted after 282.7 s of sampling.
Diagnostics printed on abort:

```
Attention computed: 18.8% of full attention (81.2% skipped)
Sequence: 113055 tokens | target 132x21x38, ref_img 1x21x38, ref_img 1x35x23, ref_img 1x35x23
Veda attention time: 152.25 s (152.25 s per model call x 1)
  gather 2.02 s | score 5.45 s | select 0.38 s | kernel 143.82 s | scatter 0.58 s
Chunks: 19 per layer, 3 heads each
Attention workspace: 2525 MB at once
Attention calls: 11 sparse
```

Only **11 of 50 layers** ran in 282.7 s. That is **25.7 s/layer** of wall time and **13.84 s/layer** of attention.
A full step is therefore about **690 s of attention plus non-attention: ~700-800 s/step.**

### 3.6 Kernel cost per layer across sizes

| Run | Geometry | Tokens | chunks / heads | kernel per layer | s/step |
|---|---|---|---|---|---|
| 06:53 first | 864x480 | 15x27x117 | ? | ? | 23.8 |
| 07:29 first | 960x544 | 17x30x132 | 9 / 7 | 0.32 s | 33.7 |
| 07:14 first | 1056x608 | 19x33x132 | ? | ? | 41.6 |
| **14:13 first** | **1216x672** | 21x38x132 | **19 / 3** | **13.07 s** | **~1285 (est)** |
| 08:09 refine | 1216x672 | 21x38x132 | 19 / 3 | 17.11 s | 1244.4 |

Per-layer kernel cost jumps **41x** between 960x544 and 1216x672, and the jump appears in the **first pass too**.
Between 1056x608 and 1216x672 the step time rises 16x while the token count rises only 1.21x.
This is a **cliff**, not a slope. It is where free memory reaches 0.

### 3.7 After the P2 fix (14:27, first pass, aborted at 14:29)

The chunking fix works:

```
H3 VRAM [first pass]: mem_get_info free=1507 / 16275 MB | torch reserved=2880 allocated=35 MB
Chunks: 8 per layer, 8 heads each (0.00 GB free) | budget 256 MB
Attention workspace: 2662 MB at once
Veda attention time: 22.20 s (22.20 s per model call x 1)
  gather 0.43 s | score 2.17 s | select 0.40 s | kernel 19.12 s | scatter 0.09 s
Attention calls: 5 sparse
```

5 layers in 78 s of sampling.

| Measure | 14:13 (before) | 14:27 (after) | Change |
|---|---|---|---|
| Chunks / heads | 19 / 3 | **8 / 8** | fixed |
| Attention per layer | 13.84 s | **4.44 s** | 3.1x faster |
| Workspace | 2525 MB | 2662 MB | +5 % |
| Wall per layer | 25.7 s | **15.6 s** | 1.65x faster |
| Step estimate | 1285 s | **780 s** | 1.65x faster |
| **Non-attention per layer** | 11.9 s | **11.2 s** | **unchanged** |

**The bottleneck moved.** Veda attention is now 4.44 s/layer. The other **11.2 s/layer** is weight streaming: about **560 s per step**.
At 960x544 that same cost is 0.29 s/layer. It is 39x higher here because the 15 GB DiT does not fit next to the activations.

### 3.8 Full attention instead of Veda at this size

Dense attention at 113055 tokens: `4 x S^2 x 56 x 128 x 50 layers` = 1.83e16 FLOPs per step.
At a realistic 100 TFLOPS that is about **183 s/step, or 3.7 s/layer**.

Veda now costs **4.44 s/layer** and holds **2662 MB** of workspace.

So at 1216x672 Veda is **no longer faster than full attention**, and it costs 2.6 GB that the weight streamer cannot use.
The user's proposal (use full attention for refine) is therefore correct on both counts: it is not slower, and it frees memory.

---

## 4. Arithmetic: the observed cost is impossible by geometry

- Token ratio refine / base = (21 * 38) / (17 * 30) = **1.565**.
- Computed-pair ratio = `0.162 / 0.285 * 1.565²` = **1.40x** the base attention work.
- Measured attention cost = **45.5x**, kernel = **53.8x**. Kernel throughput fell **~38x**.
- Full dense fallback, worst case: 19.34 / 0.285 = 67.9 s dense base -> 166 s, plus 22 s = **~188 s/step**. Measured is 1244 s/step = **6.6x over dense**.
- Non-attention work contains no attention at all, yet rose from 14.3 to **346 s/step** (**24x**).

Conclusion: the kernel does 1.4x the work in 54x the time, and the dense part of the same step degrades just as hard.
Tile-plan and sparsity effects cannot do that. Only the memory path is shared by both.

Section 3.5 closes the loop: the **first pass alone** at 1216x672 shows the same 19 chunks, 3 heads and 13 s per layer.
So the cost is **not refine-specific**. It is a VRAM cliff between 1056x608 and 1216x672.

---

## 5. VRAM evidence (leading hypothesis)

### 5.1 Allocator reports disagree during refine

| Source | Value (08:30) |
|---|---|
| `nvidia-smi memory.used` | 15674 / 16303 MiB (**~0.6 GB free**) |
| `torch.cuda.mem_get_info` via `/system_stats` | **7163-7279 MB free** |
| ComfyUI `torch_used` (total - torch free) | 8996 MB of a 15080 MB DiT |

Only ~60 % of the DiT is resident during refine. ComfyUI dynamic VRAM is streaming weights.
In the first pass the DiT was fully resident and Veda saw 1.7 GB free.
ComfyUI 0.46 reports the allocator as `cudaMallocAsync`, so `mem_get_info` describes the pool, not the card.

**Decisive detail (08:50):** `veda_comfy/comfy_patch.py:549-554` prints the free figure only `if free`:

```
free = chunking.get('free_bytes')
... + (f' ({free / 2**30:.1f} GB free)' if free else '')
```

The refine line printed **no free figure**. So `torch.cuda.mem_get_info()` returned **0** at the last chunk call.
The first-pass line printed `(1.7 GB free)`. Same code, same run, different reading.

With `free = 0`, `_chunk_bytes()` (`engine.py:229-235`) returns its **128 MB floor**.
That is the whole chain: `free = 0` -> 128 MB budget -> 3 heads per chunk -> 19 chunks per layer.

The measured peak was **2485 MB**, 19x the 128 MB budget it sized itself from.
`per_head = layout.num_slots * dim * elem` (`engine.py:288`) counts only the gathered q/k/v tiles.
It ignores `block_mask`, `o_t`, and the tile-index temporaries. The real peak is far above the budget.

### 5.2 Veda sizes its work from that number

`veda_comfy/core/engine.py:229-235`:

```
_CHUNK_SHARE = 8 ; _MIN_CHUNK_BYTES = 128 MB ; _MAX_CHUNK_BYTES = 1024 MB
chunk_bytes = clamp(free // 8, 128 MB, 1024 MB)
step = max(1, chunk_bytes // per_head)
step = min(step, max(1, out.numel()*elem // (per_head * 6)))   # _WORKING_SET_SHARE
if head_chunks > 1: step = min(step, ceil(heads / head_chunks))
```

Measured, not inferred:

| Pass | `free_bytes` | `chunk_bytes` | heads/chunk | chunks/layer | real peak |
|---|---|---|---|---|---|
| First pass | 1.7 GB | 217 MB | 7 | 9 | 1662 MB |
| Refine | **0** (floor) | 128 MB | 3 | 19 | 2485 MB |

`head_chunks` exists (KJNodes Low VRAM attention) but this workflow does not wire it.
`comfy_patch.py:396` reads it from `options.get(_HELD_HEAD_CHUNKS, 1)`.

### 5.3 Nothing frees memory before refine

- `_release_models_before_refine` (`extender.py:2724-2770`): offloads `MiniMaxH3VideoVAE`, `MiniMaxH3AudioVAE`, `MiniMaxH3TEModel_` only.
- No `torch.cuda.empty_cache()` or `soft_empty_cache()` call exists anywhere in the pack (grep of `extender.py`, `latent_refine_engine.py`).
- `LatentResizer3D` is cached in a module-level dict and never unloaded (`latent_upscaler.py:48`, `248-265`).
- The first-pass sampled latent, `previous_refined_proxy` and the upscaled latent are all alive during refine.

---

## 6. Hypotheses

| # | Hypothesis | Status | Basis |
|---|---|---|---|
| H1 | Refine silently runs dense attention | **Refuted** | 16.2 % computed, no `ran at full cost` line; dense would be ~188 s/step, still 6x faster than measured |
| H2 | Veda chunk count explodes at the refine geometry | **Confirmed as the trigger** | 19 chunks x 3 heads vs 9 x 7; not the whole 38x on its own |
| H3 | No VRAM headroom + dynamic VRAM streaming per chunk | **Confirmed** | non-attention work rose 24x with no attention in it; only ~60 % of the DiT resident |
| H4 | Perceived-blame: step 1 got fast, refine did not change | **Refuted as the cause** | absolute cost is 6.6x above the dense estimate |
| H5 | Fixed setup cost (14.5-19.5 s) | **Real but small** | 14.5 s of 2451 s |
| H6 | Tile-plan padding at 1216x672 | **Minor** | 17.1 % vs 7.0 % |
| H7 | Allocator mismatch: `cudaMallocAsync` pool vs driver free | **Confirmed, decisive** | refine printed no free figure, so `mem_get_info` returned 0 -> 128 MB floor |
| H8 | Triton INT8 kernel hit a degenerate configuration at this layout | **Contributing, not root** | 3 heads per chunk is a bad shape; but the dense path also fell 24x, so the kernel shape is not the only cause |
| H9 | Veda's budget ignores mask, output and tile temporaries | **Confirmed** | budget 128 MB, measured peak 2485-2525 MB; `out` alone is ~1.6 GB |
| H10 | Refine-only state (upscaler, held latents) is the cause | **Refuted** | the first pass alone at 1216x672 is equally slow: 19 chunks, 3 heads, 13 s/layer (section 3.5) |

---

## 7. Facts that narrow it

1. The 07:11 refine ran the **same geometry**, scale and step count and was aborted after **94.3 s**.
2. The 08:09 refine is the first observed completion: **41:28** for 2 steps.
3. No ComfyUI restart between 07:47 and 08:05 (same process, same session).
4. First appearance of `Veda: backends on RTX 5070 Ti (SM120): triton-int8: ok` and of the diagnostics block at 08:05:40 = verbose turned on for this run.
5. User reports the slow refine is **weeks old**, so a recent update is unlikely to be the cause.
6. User did a ComfyUI update (v0.46.0-54-gef507153a, released 2026-11-25) before this log session started.
7. GPU sat at **100 % SM** with 1-25 % memory-controller use during the slow refine.
8. Megapixels: 960x544 = 0.52 MP, 1216x672 = 0.82 MP, 1440x816 = 1.18 MP. The 1 mpx H3 cap is **not** involved at scale 1.25.
9. Refine completed: **41:28**, `1244.39 s/it`, 2 model calls.
10. Refine `Chunks:` line had **no free figure**, so `mem_get_info` returned **0** (section 5.1).
11. Refine computed **16.2 %**, and `kernel` was **97.3 %** of attention time (1711 of 1759 s).
12. Refine non-attention time was **346 s/step**, against 14.3 s/step in the first pass.

---

## 8. Decisive test: the refine diagnostics block

The block prints at the end **and** on abort. It printed at 08:50 (section 3.3). Read these lines:

1. `Chunks: N per layer, M heads each (X GB free)`
2. `Attention workspace: N MB at once`
3. Phase split `gather / score / select / kernel / scatter`
4. `Veda attention time: X s (Y s per model call x N)`
5. Any `attention calls ran at full cost - <reason>`

Interpretation:

| Observed | Reading | Action |
|---|---|---|
| Large free GB (several) + few chunks + huge `kernel` | H7/H8: Veda trusts an impossible budget; kernel degenerate or paging per chunk | Cap the chunk size; do not trust `mem_get_info` |
| Small free GB + many chunks + huge `kernel` | H3: plain starvation | Free VRAM before refine |
| Small free GB + many chunks + huge `gather`/`scatter` | Re-tiling overhead | Reduce resolution or free VRAM |
| Fallback line present | Veda declined the call | Read the reason |

**Result (08:50):** row 1 and row 2 both. No free figure (so free = 0, a large "impossible" reading inverted into the floor), 19 chunks, and a huge `kernel`.
That is H7 plus H9: Veda trusts a zero reading, falls to the 128 MB floor, and its real peak is 2485 MB.

Second, independent test: the **first pass alone at 1216x672**, refine disabled.

- ~64 s/step -> geometry is fine, the problem is refine-only state (memory held by the first pass).
- ~1250 s/step -> the pack cannot handle this layout at all.

**Done (14:13):** about **1285 s/step** (25.7 s/layer x 50). 11 layers in 282.7 s, 19 chunks, 3 heads. The pack cannot handle this layout.
The refine pass is not special; the layout is the problem (section 3.5).

---

## 9. Candidate fixes

See section 13 for the ordered fix plan. Summary:

1. `refine_upscaler = trilinear` — removes the 658 MB `LatentResizer3D` and its 3D conv entirely. Test first; zero code.
2. Wire KJNodes Low VRAM attention (`head_chunks`) so Veda caps `step` by head count, not by the unreliable `mem_get_info`.
3. Code: drop the upscaler reference and call `comfy.model_management.soft_empty_cache(force=True)` + `torch.cuda.empty_cache()` immediately before `_sample_h3` in `_refine_ref2va_sample` (`extender.py:2900-2910`).
4. Code: clamp `chunk_bytes` inside Veda when the device is `cuda` and the ComfyUI allocator is `cudaMallocAsync`, e.g. use `min(mem_get_info_free, torch.cuda.memory_reserved())`-style budgeting or a user-set cap.
5. Reduce `refine_scale` to 1.15 (0.69 MP) as a stopgap.

---

## 10. Open questions

1. ~~What is the `2/2` s/it for the two-step refine?~~ **Answered: 1244.39 s/it.**
2. ~~Does the refine `Chunks:` line report a large "free" value?~~ **Answered: it reports none, so free = 0.**
3. ~~Is the `kernel` phase dominant?~~ **Answered: yes, 97.3 %.**
4. Did refine ever complete before Veda was introduced? No data: log rotation dropped older sessions.
5. ~~Does `refine_denoise = 0.4` with `refine_steps = 2` produce 2 or 3 model calls?~~ **Answered: 2.**
6. Why does the dense (non-attention) path also lose 24x? Weight streaming is the only shared cause, but it is not yet measured directly.
7. Is 3 heads per chunk enough on its own to cost 38x, or does paging add the rest? Section 13 T3 separates them.

---

## 11. How to inspect (exact commands)

```powershell
# live allocator view vs driver view
Invoke-RestMethod http://127.0.0.1:8188/system_stats | Select-Object -Expand devices |
  Select-Object name, vram_total, vram_free
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
nvidia-smi dmon -s um -c 6

# refine lines
Select-String -Path 'G:\comfy-cu133\ComfyUI\user\comfyui.log' -Pattern 'H3 Refine preflight|Veda running|Veda done|Chunks:|Attention workspace|Veda attention time|ran at full cost|Requested to load'

# last lines
Get-Content 'G:\comfy-cu133\ComfyUI\user\comfyui.log' -Tail 40
```

---

## 12. Code references

- `extender.py:7046` per-clip LoRA application.
- `extender.py:7184` first-pass sampling.
- `extender.py:7198-7225` refine dispatch.
- `extender.py:2818` `_refine_ref2va_sample`.
- `extender.py:2917` `_refine_fl2va_sample`.
- `extender.py:2590` `_sample_h3`.
- `extender.py:2515` `_BasicGuider`.
- `extender.py:2724-2770` `_release_models_before_refine` (VAE + TE only).
- `extender.py:3069` `_apply_per_clip_loras`.
- `extender.py:5583-5587` refine widget defaults (enabled False, scale 1.5, steps 4, denoise 0.30, upscaler learned).
- `latent_refine_engine.py` `upscale_for_refine`, `preserve_first_pass_audio`.
- `latent_upscaler.py:48, 248-265` `_CACHE`, `_load_model` (never unloaded).
- `veda_comfy/comfy_patch.py:629-702` override install and per-step reinstall.
- `veda_comfy/comfy_patch.py:551-556` `Chunks:` / `Attention workspace:` summary lines.
- `veda_comfy/comfy_patch.py:549-554` free figure printed only `if free` (0 hides it).
- `veda_comfy/core/engine.py:229-235` chunk sizing from `mem_get_info`.
- `veda_comfy/core/engine.py:288` `per_head` ignores mask, `o_t` and index temporaries.
- `veda_comfy/core/engine.py:245-326` `attention()` chunk loop and phase timers.
- `comfy/ldm/minimax/model.py:207, 736, 865-868` attention call and per-forward layout.

---

## 13. Fix plan

**Decision (14:35): cap the refine size and keep Veda in every pass.**
`refine_scale = 1.1` gives 1056x608, which was measured at 41.6 s/step. No code change is needed for this path.
P1a (dense attention for refine) is deferred. P1 and P2.9 stay open for later.

**Root cause.** A VRAM cliff at about 0.7 MP. At 1216x672 free memory reads 0. Veda treats 0 as "no budget" and falls to its 128 MB floor. That gives 3 heads per chunk and 19 chunks per layer. Each layer then costs 13-17 s instead of 0.32 s. The DiT also streams weights.
The cliff is in **both** passes (section 3.5), so the refine pass is not the cause.

**Answer to "is it memory related?": yes.**
- The dense part of the step (no attention in it) fell 24x. Only weight streaming can do that.
- Veda's budget fell from 217 MB to 128 MB, because it read 0 free.
- The same 19 x 3 chunking appears in the first pass at 1216x672.

**Why chunking cannot fix it alone.** Per attention call, `out` alone is `113055 x 56 x 128 x 2` = **1.55 GB**, and q/k/v are about **4.9 GB**. Veda allocates those regardless of its chunk budget. On a 16 GB card with the DiT staged, nothing fits. Free VRAM and a smaller sequence are the only real levers.

### P0 — Confirm (one test still open)

1. ~~First pass alone at 1216x672.~~ **Done: ~700 s/step. Not refine-specific.**
2. Optional: refine with `refine_upscaler = trilinear`. It removes only 658 MB, so expect a small change.
3. ~~Add the VRAM log line.~~ **Added at 14:06.** It did not print in the 14:13 run, so that run used the old code. **Restart ComfyUI before the next run.**

### P1a — Refine: full attention instead of Veda (leading candidate)

The user proposed this. Section 3.7/3.8 now support it: after the chunk fix, Veda costs **4.44 s/layer** at 1216x672 and holds **2662 MB**. Dense attention is about **3.7 s/layer** (estimate) and holds no workspace. So Veda is no longer a speed win at this size, and its workspace feeds the paging that costs 11.2 s/layer.

a1. Zero-code test: set the Veda node input `full_attention_steps` to `0`. Run 1216x672, 447 frames, **steps 1**, refine off.
    Expect `Attention calls: 0 sparse; full attention: full-attention step 50`. Compare wall per layer against 15.6 s.
    Risk: if the dense kernel falls back to the math backend, it needs 113055^2 x 2 bytes = 25 GB and fails fast with OOM.
a2. If dense wins, wire it for refine only. `full_attention_steps` is global per sampling run, so `0,1` also makes first-pass steps 0 and 1 dense (about +116 s per clip at 960x544).
    A refine-only switch needs a small path: the extender sets a `transformer_options` flag, and Veda reads it in `_block_reason` (`comfy_patch.py:324`).
a3. Keep Veda in step 1. The user requires it, and at 960x544 Veda gives 33.7 s/step against about 77 s dense.

### P1 — Extender: free VRAM before every sampling call

4. Extend `_release_models_before_refine` (`extender.py:2724-2770`): call `comfy.model_management.soft_empty_cache(force=True)`, `torch.cuda.empty_cache()` and `gc.collect()` after the VAE/TE unload.
5. Call the same release **before the first pass** too. Conditioning is already built at that point, so the TE and VAE can go.
6. Drop the first-pass `sampled` tensor after `upscale_for_refine` returns.
7. Evict `LatentResizer3D` from `latent_upscaler._CACHE` when `refine_upscaler != learned`, and after the upscale.

### P2 — Veda: fix the budget model (the real fix)

8. **Applied.** `_chunk_bytes()` (`engine.py:233-248`): `free <= 0` now uses `_DEFAULT_CHUNK_BYTES` (256 MB), not the 128 MB floor. I used 256 MB and not 1 GB, because a large blind allocation on a full card risks paging.
9. `attention()` (`engine.py:296-299`): include `out` (1.55 GB at this size) and the mask in the budget. The measured peak was 19x the budget. **Not applied yet.**
10. **Applied and measured.** `_MIN_HEADS_PER_CHUNK = 8` (`engine.py:40`), clamped up in `attention()` (`engine.py:301-306`). A node that asks for low VRAM (`head_chunks > 1`) still wins.
    Result at 1216x672: `Chunks: 8 per layer, 8 heads each`, workspace 2662 MB, attention 13.84 -> 4.44 s/layer (3.1x), wall 25.7 -> 15.6 s/layer (1.65x). See section 3.7.

### P3 — Deterministic wiring

11. Set `transformer_options['veda_held_head_chunks'] = 8` for sampling. Then the chunk shape does not depend on the allocator. Read path: `comfy_patch.py:396`.
12. Keep Veda in step 1. It is fast at 960x544, and the user requires it.

### P4 — Verify

13. Re-run at 1216x672 with the new log line. Record `free`, `budget`, chunks, workspace, kernel.
14. Target: 7-8 heads per chunk, kernel under 200 s per call, step under 200 s.
15. Confirm no fallback line and computed fraction near 16-19 %.

### Stopgap (no code, use now)

16. **`refine_scale = 1.1`** gives 1056x608 (0.64 MP), the fastest measured size at 41.6 s/step. Keep `refine_upscaler = trilinear`.
17. Do not use 1.15. It gives about 1104x626, which is still above the cliff.

### Chosen path (14:35)

The user selected the size cap. Run settings:

- Base: 960x544, 447 frames, 6 steps. Veda on, unchanged.
- Refine: on, **scale 1.1** -> 1056x608, 2 steps, denoise 0.4, upscaler `learned` (658 MB fits at this size; use `trilinear` only if free reads 0).
- Expected: about 41.6 s/step in the first pass and about 2x that in refine. Total near 8-10 minutes instead of 41.

Honest limit: 1.1 upscales only 10 %, so the refine gains detail but almost no resolution. More resolution needs more VRAM, or the dense path in P1a.

### Test order note

One change at a time. P1 alone may move the cliff. P2.8 alone should move the chunk count from 19x3 back to about 9x7.

### Decision (implemented 23:40): DenserRefine replaces RECAVSR

The user asked to replace the RECAVSR export path with a **DenserRefine** path: the same
Refine options, without the sparse-attention optimisation. Reason: at 1216x672 the
sparse path is slower than dense, so the only way to refine at that size is to skip Veda.

What the code does now:

- `dense_refine_enabled` (BOOLEAN) replaces the `recavsr_enabled` widget slot, so saved
  workflows keep every later widget in place. The three other RECAVSR slots became inert
  `reserved_float`, `reserved_choice` and `reserved_int` widgets, same position and same
  types, and the frontend hides all four.
- `_sample_h3(..., dense_attention=True)` sets
  `model.model_options['transformer_options']['veda_force_dense'] = True` for the duration
  of one `guider.sample` call and removes it in `finally`.
- Veda reads that key first in `_block_reason()` and returns `'dense requested'`, so every
  attention call in the pass takes the model's own full attention. Nothing else changes:
  the patch stays installed, so step 1 keeps Veda sparse attention.
- `_refine_settings_payload(..., dense=True)` adds `"attention": "dense"`, so a dense pass
  and a sparse pass never share a cache entry.

The frontend shows two exclusive buttons, **REFINE** and **DENSEREFINE**. They share the
REFINE SETTINGS panel; DENSEREFINE has no panel of its own. Pressing DENSEREFINE turns the
refine pass on, so it invalidates clips exactly like REFINE.

Verification: `node --check web/extender.js` OK, `py_compile` OK on `extender.py`,
`ref2va_independent.py`, `fl2va_engine.py`, `motion_context_disk.py`,
`veda_comfy/comfy_patch.py`, `veda_comfy/core/engine.py`; 43 Python tests run with no
failure (7 modules skip on this Mac because `torch` is absent); all 8 node tests pass.

Expected outcome at 1216x672: no Veda workspace at all (saves about 2.6 GB), kernel time
replaced by about 3.7 s/layer of dense attention. That is slower than 1056x608 with Veda,
but it is the only route to 1216x672 on 16 GB.

### 3.9 DenserRefine measured at 1216x672 (2026-11-26 15:21-15:27)

Run: 960x544, 447 frames, 6 steps, Veda sparse; refine on with **DENSEREFINE**,
`refine_scale 1.25` -> 1216x672, 1 step, denoise 0.4. Manifest confirms
`{"scale": 1.25, "steps": 1, "denoise": 0.4, "attention": "dense"}`.

```
first pass  6/6 [03:12, 32.06 s/it]
Veda attention time: 109.44 s (18.24 s per model call x 6)
  gather 7.00 | score 7.70 | select 2.14 | kernel 90.33 | scatter 2.27
Chunks: 8 per layer, 8 heads each (1.99 GB free) | budget 255 MB
Attention workspace: 1869 MB at once | 300 sparse calls

refine      1/1 [01:29, 89.01 s/it]
Veda done, fell back to full attention
Attention calls: 0 sparse; full attention: dense requested 50
Prompt executed in 395.70 seconds
```

| 1216x672 refine | Time per step |
|---|---|
| Veda sparse (08:09 run) | 1244.4 s |
| **Dense (DENSEREFINE)** | **89.0 s**, includes the 15 GB weight reload |

**14x faster.** No Veda workspace is allocated in the dense pass, and the wall
time no longer depends on how much VRAM is free. This settles P1a: the dense
route is the answer above 1056x608, and Veda stays for the first pass and for
refine sizes at or below 1056x608, where it is still the cheaper option
(about 20 s of attention instead of about 70 s).

The P2 budget fix also held at 960x544: 8 chunks of 8 heads with a 255 MB
budget, against 9 chunks of 7 heads before. Attention fell 6 % (116.05 -> 109.44 s),
the step fell 5 % (33.7 -> 32.06 s), and the workspace rose 12 % (1662 -> 1869 MB),
as predicted.

Two reporting defects found in this run, both cosmetic:

1. `H3 VRAM [refine ...]` reads **before** the DiT loads (185 ms later), so it
   showed 13639 MB free while the sampler then loaded 15080 MB. It records the
   pre-load state, not the sampling state.
2. The `Chunks:` line in the dense block is stale from the first-pass engine:
   `_dense_reason()` returns before `_sparse()` ever runs, so the recorded
   chunking is never refreshed. It must not be read as a dense-pass measurement.
