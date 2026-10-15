# Qwen-Image-2.1 Uncensored GGUF migration — 2026-09-26

## Existing deployment

- Windows project-owned ComfyUI checkout at `ComfyUI/`, pinned to commit
  `b0f4b7b294ce482a2e071d9d762c133d38c7aa07` (0.37.0), Python 3.12.10, and
  PyTorch 2.11.0+cu128.
- `scripts/manage_comfyui.ps1` starts the service on port 8191 through
  `scripts/launch_comfyui.py`. The launcher restricts CUDA visibility to the
  configured RTX 4090 UUID; the manager enables CPU VAE, one GiB VRAM headroom,
  and two async offload streams.
- The project API reads `workflows/catalog.json` and `workflows/models.json`.
  Named jobs pass their graph through ComfyUI's serial worker and model-release
  boundary in `project_nodes/runtime_guard.py`.
- Both Qwen routes shared one official INT8 ConvRot diffusion model. Text
  conditioning uses `TextEncodeQwenImage21` and the Qwen3VL INT8 encoder; image
  decoding uses the Qwen-Image-2.1 BF16 VAE. The edit route also feeds its
  uploaded reference through the same Qwen encoder and a QwenImage21 cache node.
  Other model families use their own encoders and VAEs.
- Before migration, the official diffusion file was present at
  `ComfyUI/models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors`
  (7,256,783,064 bytes, SHA-256
  `cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d`).

## Replacement

- Qwen T2I and image-edit graph IDs and API contracts are retained. Their
  diffusion loader is changed from `UNETLoader` to `UnetLoaderGGUF`, using the
  same Qwen-Image-2.1 architecture.
- The active diffusion asset is `qwen-image-2.1-UC-Q4_K_M.gguf`, pinned to
  Hugging Face revision `40319fb15542f0ad22921e0124a191a8a935a60a` (4,604,558,112
  bytes; SHA-256
  `e79c8a009f2ecbdb6c70fd663d9aea9ee304a0d91f347e4169a756b8ad141b41`).
- The accompanying INT8 Qwen3VL encoder and BF16 VAE retain their current local
  filenames and files; their hashes match the versions published in the target
  repository. The original official diffusion file remains installed as a
  rollback copy but is no longer referenced by active workflows.
- `scripts/setup_comfyui.ps1` installs the `leejet/ComfyUI-GGUF` custom node at
  commit `373048b8403a7820620065210a691263d4da0a61` and its declared Python
  requirements, with Python package `gguf==0.19.0`. The existing
  `qwen-image-2.1` model profile downloads and verifies the uncensored GGUF and
  matching companion assets.
- Qwen routes now advertise SFW, NSFW, and general content modes. The linked
  model card describes this release as having no built-in safety checker or
  content filter; output still depends on the submitted prompt.
- Explicit domain matches now outrank generic fallback routes. This keeps an
  NSFW request for `qwen` on Qwen-Image-2.1 instead of a generic anime route.

## Runtime validation

On 2026-09-26, the local RTX 4090 completed the named Qwen T2I route and then
edited its generated reference through the image-edit route. Both outputs were
1024x1024. T2I took 51,907 ms and edit took 63,922 ms. The same probe confirmed
that an NSFW `qwen` domain request resolves to the Qwen route. Detailed job IDs
and results are recorded in `workflows/catalog.json` and `workflows/models.json`.
