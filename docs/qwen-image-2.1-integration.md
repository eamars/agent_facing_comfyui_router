# Qwen-Image-2.1 integration — 2026-09-22

Status: verified on the installed RTX 4090, with CPU VAE and RAM offload.

## Project contract

`workflows/catalog.json` describes named API graphs, parameter bindings,
input contracts, model dependencies, route selectors and resource budgets.
`workflows/models.json` describes installed components and exact file sizes.
`project_nodes/workflow_api.py` reads these catalogs on requests, scores route
metadata, checks readiness, binds parameters and validates the resulting graph
through ComfyUI before queueing. Model discovery does not load weights.

Named jobs pass through `ProjectWorkflowCoordinator` into ComfyUI's native
serial prompt worker. `project_nodes/runtime_guard.py` wraps that worker:
different workflow IDs reset executor caches and unload models before execution;
same-workflow jobs reuse caches. Native jobs participate in the same boundary,
and failed jobs release models. `resident_group` is descriptive metadata, not
the worker's switch key. This integration leaves the coordinator and guard intact.

## Pinned deployment

- ComfyUI: `7972b5ba` (0.30.0) →
  `b0f4b7b294ce482a2e071d9d762c133d38c7aa07` (0.37.0).
- Frontend 1.53.6; workflow templates 0.11.66; comfy-kitchen 0.2.35;
  comfy-aimdo 0.5.5. Existing PyTorch 2.11.0+cu128 is retained.
- Model package: `Comfy-Org/Qwen-Image-2.1`, revision
  `ace0edeb3791a594ddfa36ed5f41a178a394e921`.
- Official template repository observed at
  `371a7b7171bbd11e9cc92ef615ba5ad223d7e5b4`.
- GPU visibility: `GPU-eed52936-813f-8d68-1654-bfb56cb42bc3` only.
  The launcher validates that this UUID is an RTX 4090 and rejects device
  override arguments. During inference, PID 20984 appeared only against this
  UUID in `nvidia-smi` compute-process reporting.
- Existing trusted-LAN address `0.0.0.0:8191` is retained.

All three model files were downloaded and SHA-256 checked by the project
downloader. The official template-selected components are the INT8 ConvRot
diffusion model and Qwen3VL 8B encoder, plus the BF16 Qwen-Image-2.1 VAE.
The original Qwen VAE and Qwen3 encoders used by existing routes remain intact.

The official UI templates are flattened into API graphs. Both named routes
use 25 steps, CFG 1, Euler/simple, denoise 1, and 8-bit sRGB PNG output.
Text-to-image defaults to 1024×1024 and batch 1. Image edit exposes one uploaded
reference, uses official resolution 0 and the reference-sized latent, and keeps
KV cache placement at auto/default. UI comparison controls and the optional
custom-size branch are not exposed. Existing multi-image native-node support
does not imply a multi-image contract in this named API.

The new routes match explicit Qwen domain/style requests or exact workflow IDs.
Their exclusive metadata avoids changing existing generic anime selection.
NSFW behavior is unverified and is not advertised as a route capability.

## Runtime evidence

Before upgrade: 43 public-HTTP checks passed for the 13 existing routes.
After upgrade: the same 43 checks passed. All 13 existing routes then rendered
successfully in a 23-job, 91-check regression run. This also covered six queued
submissions, native/project interleaving, inpaint preservation and recovery
after a queued reference file disappeared.

After integration: all 15 routes passed the 47-check public-HTTP audit.
The existing 13 catalog entries and all existing model profiles were compared
against Git HEAD and are unchanged. All 20 contract tests passed; `pip check`
reported no broken requirements.

The same server process completed this sequence without a restart:

| Workflow | Output | Elapsed, including switch and CPU VAE |
| --- | --- | --- |
| Anima Turbo | 512×512 | 5.015 s |
| Qwen-Image-2.1 T2I | 1024×1024 | 52.813 s |
| Qwen-Image-2.1 Edit | 1024×1024 | 63.125 s |
| JANKU v5 | 512×512 | 8.329 s |
| Qwen-Image-2.1 T2I | 1024×1024 | 51.579 s |
| Anima Turbo | 512×512 | 4.125 s |

Worker logs recorded each model boundary. Ten-second resource samples reported
a minimum 3,099,590,656 bytes free VRAM and 30,662,287,360 bytes free system RAM.
These samples are not continuous peak measurements or a guarantee for larger
images. Logs confirmed dynamic VRAM loading, two async offload streams and
CPU VAE. Visual inspection confirmed a blue teapot became red while the scene
and card text were largely preserved; this is not a general fidelity benchmark.

Reports are under `ComfyUI/.codex-runtime/`:
`qwen21-baseline.json`, `qwen21-upgrade.json`,
`qwen21-existing-generation.json`, `qwen21-integrated.json`,
and `qwen21-hot-switch.json`. Reproduce with the project Python and
`scripts/verify_agent_api.py --generate` / `scripts/verify_qwen21.py`.

## Local state and limits

No local ComfyUI source modifications needed removal. The pre-existing deleted
zero-byte output placeholder and `.codex-runtime` directory were left intact.
No existing models or images were deleted. Probe-created uploaded references
were deleted after their jobs completed; generated probe images were retained.

Upstream migration 0007 rebuilt the asset database and automatically saved the
old database as `ComfyUI/user/comfyui.db.bkp`. Read-only inspection of the backup
found zero assets, asset references, reference metadata and reference-tag rows
(24 tag definitions). The backup remains available.

Verified scope is 1024-square Qwen rendering and single-reference editing,
with the original 13 routes exercised at regression-probe sizes. Larger Qwen
resolutions, multiple references, transparency fidelity and NSFW content were
not tested. The named edit route intentionally rejects width/height/batch
overrides; its budget calculation uses the same 32-pixel rounding as the
official node.
