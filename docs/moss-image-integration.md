# MOSS-Image-SFT integration — 2026-10-16

Status: verified on the installed RTX 4090 through the named API.

## What it adds

`moss-image-sft-t2i` is the first route with layout control. A caller can pass a
layout JSON object, serialised into `prompt`, that places text and objects by
pixel box. It also handles long prompts (up to 4096 tokens) and renders English
and Chinese text. It accepts no image inputs, and SFW is the only content mode it
advertises. It is exclusive: it is selected only by `domain`/`style` `moss` or
`layout`, or by its workflow ID, so generic anime and Qwen selection is unchanged.

## Pinned deployment

- Model: `OpenMOSS-Team/MOSS-Image-SFT`, revision
  `daf706d2e1c7a6369c17b6d1658671aee6db1ebe` (Apache-2.0). 5B DiT (BF16, 10.2 GB),
  Qwen3.5 9B encoder (BF16, 16.8 GB), FLUX.2 small-decoder VAE (0.25 GB).
- Files stay in the Hugging Face layout under
  `ComfyUI/models/moss-image/MOSS-Image-SFT/`. Profile `moss-image-sft` downloads
  16 files and checks each one's size and SHA-256.
- Node pack: `OpenMOSS/MOSS-Image` `comfyui/moss-image` at
  `599a039ec6f8108eecdd0d8c107ceade6863145e` (v0.1.1). `setup_comfyui.ps1` makes a
  sparse checkout under `ComfyUI/.vendor/` and copies the pack into `custom_nodes/`.
  The pack needs no extra Python packages. It reuses ComfyUI 0.37.0's native
  `qwen35` encoder, `EmptyFlux2LatentImage`, `ModelSamplingSD3` and
  `SkipLayerGuidanceDiTSimple`.
- Graph: the publisher's API example, with `SaveImageAdvanced` in place of
  `SaveImage`. `MOSSImageHFLoader` reads `models/moss-image/MOSS-Image-SFT`
  relative to the ComfyUI working directory. Sampling is 50 Euler/sgm_uniform
  steps, CFG 3, shift 5, with negative-branch skip at single layer 22.

## Runtime evidence

`scripts/verify_moss_image.py` checks route selection, runs both MOSS jobs and
then switches to Anima Turbo. All checks passed:

| Job | Output | Elapsed | Min sampled free VRAM |
| --- | --- | --- | --- |
| Plain prompt, bilingual chalkboard text (cold load) | 1024×1024 | 59.95 s | 1.6 GiB |
| Layout JSON, 4 items | 1024×1024 | 53.75 s | 2.0 GiB |
| Anima Turbo after the switch | 512×512 | 5.77 s | 15.8 GiB |

Sampling ran at about 1.15 it/s. ComfyUI staged the encoder in FP16 with dynamic
VRAM loading, and the output showed no overflow artefacts. Every layout box landed
within about 10 px of its requested coordinates. "MOSS IMAGE", "开放模型" and
"布局控制测试" all rendered correctly.

## Head-to-head with Qwen-Image-2.1 and the routing decision

Twelve shared prompts were run through both named routes at seed 4242, plus one
layout-JSON prompt for MOSS only. All 25 jobs succeeded. Images and scoring notes
are in the published capability report.

| Test | Winner | Deciding detail |
| --- | --- | --- |
| Photo portrait | Tie | All elements in both; MOSS lighting closer to "golden hour" |
| Product label | Tie | MOSS exact incl. "·"; Qwen hyphen but a more believable label |
| Chinese poster | Qwen | MOSS invented a garbled extra subtitle and seal |
| Chalkboard menu (11 strings) | Tie | Both exact |
| Counting and spatial relations | Tie | Both 6/6 |
| 12-detail scene | MOSS | MOSS 12/12; Qwen's robot holds no bowl |
| Anime girl (tags) | Tie | Both clean; Qwen blazer halves mismatch |
| NZ landscape 1216×832 | Qwen | Qwen looks like Central Otago; MOSS generic alpine |
| Weather-app UI mock-up | MOSS | Qwen sparse, garbled glyph, duplicated value |
| Macro at 1456² | Qwen | More authentic photo, 87 s vs 110 s |
| "a cat" | Qwen | MOSS deformed the face |
| Poster as prose | Tie | Same composition; sizes uncontrolled in both |
| Poster as layout JSON | MOSS only | Every box within ~10 px, two seeds |

Mean 1 MP time: MOSS 50.0 s, Qwen 47.5 s.

Decision: both stay in the heavy general-purpose SFW bucket. Qwen is the default
for real-world requests (`photo`, `product`, `landscape`, `poster`, `realistic`,
style `photorealistic`) because it tolerates short prompts, knows real places,
does not invent text, and is also NSFW-capable and editable. MOSS owns `layout`
and `ui mockup` and accepts only structured, detailed or expert prompts.
`route_table[].selectors` and `best_for` now publish each route's match words, so
agents choose from a menu instead of guessing. Unmatched requests still fall back
to the anime routes.

## Not covered

- 1456² (2.1 MP) ran cleanly; nothing above that was tried. The 2.36 MP budget is advisory.
- One seed per prompt (two for layout JSON). Ties may break either way on other seeds.
- LoRAs and the Base checkpoint are not wired in. NSFW behaviour is untested.
- Native sgm_uniform sampling does not match the publisher's diffusers grid, so
  seeds do not reproduce diffusers images.
