# Generic ComfyUI text/image generation deployment

This repository wraps an isolated ComfyUI checkout with a small named-workflow
catalog. A remote agent can discover the available workflows, inspect their
model requirements, runtime probes and parameter bindings, then queue the
workflow it chose. Each generation route can use text-to-image or latent
image-to-image mode when its `input_contract` declares it; the catalog also
includes masked inpaint, face-centric IPAdapter and OpenPose ControlNet routes.
The original Moody Desire Mix workflow remains available alongside the
official Anima Turbo v1.1 routes, the retained Moody Anima Mix comparison
route, Animagine XL 3.1 and Blue Archive routes.

The agent API is version **1.5.0**. See the
[interface audit and verification notes](docs/interface-audit-2026-09-17.md)
for the current contract, routing and memory fixes.

## Runtime policy

- ComfyUI chooses the default CUDA device. The launcher does not pass a GPU
  name or `--cuda-device`, so the deployment is not tied to an RTX 4090.
- The managed launcher passes `--vram-headroom 1` for dynamic weight loading,
  `--reserve-vram 1` for the standard memory manager, `--cpu-vae` and two
  async offload streams. These safeguards do not guarantee an OOM-free run.
- `GET /project/resources` reports the selected CUDA device and memory estimates.
  NVIDIA physical memory is matched by UUID because CUDA and `nvidia-smi` GPU
  indices can differ. Each route publishes its configurable pixel/RAM/VRAM
  admission budgets in `resource_limits`; total pixels include batch size.
- The default bind is `0.0.0.0:8191` and there is no authentication layer.
  Treat this as a trusted-LAN service; do not expose it directly to the
  internet. Use `-BindHost 127.0.0.1` for local-only access.

## Setup and launch

```powershell
.\scripts\setup_comfyui.ps1
.\scripts\download_models.ps1 -Profile all
.\scripts\manage_comfyui.ps1 start
```

The model downloader is resumable and verifies file size and SHA-256. Profiles
can be downloaded independently:

```powershell
.\scripts\download_models.ps1 -Profile moody-desire-v30
.\scripts\download_models.ps1 -Profile moody-anima-mix
.\scripts\download_models.ps1 -Profile anima-turbo
.\scripts\download_models.ps1 -Profile animagine-xl-3.1-nsfw
.\scripts\download_models.ps1 -Profile blue-archive-style-sdxl
.\scripts\download_models.ps1 -Profile blue-archive-character-illustrious-xl
.\scripts\download_models.ps1 -Profile ipadapter-face-sdxl
.\scripts\download_models.ps1 -Profile controlnet-openpose-sdxl
```

Service management:

```powershell
.\scripts\manage_comfyui.ps1 status
.\scripts\manage_comfyui.ps1 stop
```

The server is intended to be reachable by another trusted machine at
`http://<host-LAN-IP>:8191`. If Windows Firewall blocks it, allow inbound TCP
8191 on the appropriate Private network profile.

## Agent-facing workflow API

The project custom-node extension adds these unauthenticated routes on the
same ComfyUI port:

```text
GET  /.well-known/agent-manifest.json
GET  /project/agent-manifest
GET  /project/openapi.json
GET  /project/resources
GET  /project/resources
GET  /project/routes
GET  /project/inputs
DELETE /project/inputs/{name}
POST /project/resolve
POST /upload/image
GET  /project/models
GET  /project/models/{model_id}
GET  /project/workflows
GET  /project/workflows/{workflow_id}
POST /project/generate
POST /project/prompt
GET  /project/jobs/{prompt_id}
```

An independent agent should first call `/.well-known/agent-manifest.json`.
`/project/agent-manifest` is a compatibility alias and `/project/openapi.json`
is the importable contract. The root path remains the native ComfyUI UI, so an
agent should not infer the project API from the root page.

The manifest and workflow catalog expose two complementary machine-readable
hint layers:

- `routing` describes when a workflow/model combination is a match: domain,
  content mode, style, input type/mode, aliases, priority and accepted prompt
  granularity.
  Routing is intentionally at the workflow/model level, not per character or
  outfit.
- `anima-turbo-v1-1` is the preferred official generic Anima/Turbo route.
  `moody-anima-mix` remains available as a separate comparison/fallback route.
  Explicit Blue Archive or
  NIKKE requests are routed to their specialized routes first; the agent keeps
  the character, outfit and pose in the prompt after route selection.
- `capabilities`, `prompting`, `agent_hints` and the referenced `models`
  describe what the route can control, which prompt dialect it expects, and
  where its limits are.
- `/project/resources` is an advisory selected-device memory snapshot plus
  configured queue/input limits; it is diagnostic information, not a resource
  reservation.
- `capabilities.input_types`, `capabilities.input_modes` and
  `input_contract` describe executable input behavior. Latent img2img routes
  use `LoadImage -> VAEEncode`; the inpaint route uses
  `LoadImageMask -> VAEEncodeForInpaint`, the face route uses `face_image` with
  IPAdapter, and the OpenPose route preprocesses `control_image` before
  applying ControlNet.

The model registry is stored in `workflows\models.json`. It includes the
checkpoint, text encoder, VAE and adapter profiles used by each installed
route. `ready` checks file presence and recorded sizes, registry eligibility and
registered graph node classes. The downloader performs SHA-256 verification;
discovery does not rehash weights or promise runtime fit or visual quality.
The full `qwen_3_8b.safetensors` text encoder is also installed as a
non-routeable high-precision asset; the generic Moody route intentionally keeps
the verified FP8-mixed encoder because the full encoder plus checkpoint may not
fit a 24 GB device with the one-GB headroom target.

The full `qwen_3_8b.safetensors` encoder was also tested. It succeeded when the
text encoder was explicitly placed on CPU, but the default dynamic-GPU
placement OOMed while transferring weights alongside Moody on the selected
24-GB device. It therefore remains installed but `routeable: false`; the
generic route uses the FP8-mixed encoder to preserve the one-GB VRAM headroom
target.

The preferred route-selection call is:

```json
{
  "domain": "blue archive",
  "content_mode": "sfw",
  "input_type": "text",
  "input_mode": "text-to-image",
  "prompt_granularity": "detailed",
  "style": "game cg"
}
```

`POST /project/resolve` returns ranked ready workflows, reasons for each match,
the complete route metadata and the prompt/model hints needed to construct the
final request. It does not load a model or queue a generation. The resolver
selects the workflow/model combination; character, outfit, pose and appearance
remain inputs to the agent's final prompt. Generic Blue Archive requests select
the style route. The Asuna identity route requires an explicit `workflow`
choice so it cannot impose Asuna on other characters. Optional `family`
selects an exact catalog family, such as `sdxl-ipadapter` for SDXL face reference.

Capability-specific routes declare `routing.required_selectors`. In particular,
inpaint, face-reference and OpenPose routes require the resolver request to
include the matching `input_mode`; this prevents a normal text-only portrait
request from accidentally selecting a reference/control workflow.

For an image-to-image request, use the same selected workflow with
`input_type: "image"` and `input_mode: "image-to-image"`, then pass a relative
filename from `GET /project/inputs`. If the reference is not present, upload it
to the native ComfyUI endpoint on the same port first:

```text
POST /upload/image
Content-Type: multipart/form-data
Fields: image=<file>, type=input, overwrite=false
```

The response contains `name`, `subfolder` and `type`. Join `subfolder` and
`name` with `/` and pass the result as `parameters.input_image`:

```json
{
  "workflow": "blue-archive-character-illustrious-xl",
  "parameters": {
    "input_image": "example.png",
    "prompt": "preserve the subject and pose, change the lighting to soft evening light",
    "denoise": 0.65
  }
}
```

For example, a command-line agent can upload a local file with:

```powershell
curl.exe -X POST http://<host>:8191/upload/image `
  -F "image=@.\xiaoman_ref.png" -F "type=input" -F "overwrite=false"
```

`input_image` is intentionally not an arbitrary host path, URL or output job
reference. The named API does not yet implement automatic output-to-input
copying between jobs; use the upload response or the `/project/inputs` listing
to obtain the relative filename before submitting the workflow.

Use a unique basename for each new upload. With `overwrite=false`, native
ComfyUI may return an already-existing same-name file instead of creating a
new one; only delete input filenames owned by the calling agent.

To remove a reference or a temporary probe, delete the exact relative filename
returned by `/project/inputs`:

```text
DELETE /project/inputs/xiaoman_ref.png
```

The delete operation removes one regular file below `ComfyUI/input`; it does
not accept absolute paths, traversal segments or directories.
Its response includes `remaining_bytes`, a snapshot of the remaining input
files' combined size. Inputs persist across restarts: this project has no
automatic TTL or cleanup. Files disappear only through explicit deletion or
external/manual filesystem actions. Keep references unchanged until their jobs finish.

### Masked inpaint

`anima-turbo-inpaint` is the preferred Anima masked-edit route;
`moody-anima-inpaint` remains available for comparison. Both require the original image in
`parameters.input_image` and a mask in `parameters.mask_image`. Upload both
through `/upload/image`, or use names already listed by `/project/inputs`.
The red channel is used by default: white/red pixels are redrawn and black
pixels are preserved. Use `mask_channel: "alpha"` for an alpha mask. A mask is
an executable image region, not a text label such as `clothing`. For alpha masks,
transparent pixels are redrawn and opaque pixels are preserved. Source and mask
must have identical dimensions, with both dimensions multiples of 8.

`anima-turbo-inpaint` exposes the same contract on the official Anima Turbo
base and is the preferred masked-edit route when the request is already using
the Anima family. It is still generic masked sampling rather than a dedicated
inpaint-trained checkpoint; choose it with `input_mode: "inpaint"` when
resolving the route.

```json
{
  "workflow": "anima-turbo-inpaint",
  "parameters": {
    "input_image": "character.png",
    "mask_image": "clothing_mask.png",
    "mask_channel": "red",
    "prompt": "1girl, solo, a navy school uniform with a red ribbon, preserve the face, hair, pose and background",
    "negative_prompt": "duplicate, multiple views, text, watermark",
    "denoise": 0.65,
    "grow_mask_by": 6
  },
  "client_id": "remote-agent-1"
}
```

The routes use ComfyUI's core `VAEEncodeForInpaint` path, then composite the
result onto the original image through the supplied mask to preserve unmasked
pixels exactly. These are generic masked workflows, not specially trained
inpaint checkpoints. Turbo defaults to 12 steps and CFG 1; Moody defaults to
4 steps and CFG 1. For clothing replacement,
start with denoise `0.60`–`0.75`; prompts should remain short and should begin
with `1girl, solo` when the source is a character sheet or multi-view image.

### Face reference and pose control

The catalog has two SDXL control routes plus the Anima-native face-reference
route so an agent can choose the smallest matching capability:

- `sdxl-face-reference-ipadapter`: requires `face_image`, uses IPAdapter Plus
  Face with the matching ViT-H encoder, and primarily controls facial
  appearance. It does not carry the outfit, pose or background and is not
  FaceID recognition.
- `sdxl-openpose-controlnet`: requires `control_image`, runs the pinned
  `OpenposePreprocessor` internally and applies an SDXL OpenPose ControlNet.
  It primarily controls body pose and orientation; character identity and
  clothing stay in the prompt.
- `anima-turbo-face-reference-ipadapter`: requires `face_image`, uses the
  Anima-specific SigLIP2 IPAdapter path described above, and is the preferred
  face-reference route when the selected base model is Anima Turbo.

Both routes accept SFW or NSFW-capable prompts supplied by the caller. They
expose only one adapter/control strength knob in the named API, with the
remaining node choices fixed to the validated route so an agent does not have
to choose among incompatible model variants.

The first OpenPose request may take longer because the preprocessor caches its
body/hand/face detector weights under the custom-node checkout. The model
downloader installs the route's fixed SDXL adapter and ControlNet weights:

```powershell
.\scripts\download_models.ps1 -Profile ipadapter-face-sdxl
.\scripts\download_models.ps1 -Profile controlnet-openpose-sdxl
```

After route selection, the agent can submit a named workflow:

```json
{
  "workflow": "animagine-xl-3.1-nsfw",
  "parameters": {
    "prompt": "masterpiece, best quality, 1girl, solo, rapi (nikke), standing, looking at viewer",
    "negative_prompt": "worst quality, low quality, bad anatomy, bad hands, watermark",
    "width": 832,
    "height": 1216,
    "seed": 382624,
    "steps": 28,
    "cfg": 6.0,
    "loras": {
      "concept": 0.9,
      "rapi": 0.8
    },
    "filename_prefix": "agent_rapi"
  },
  "client_id": "remote-agent-1"
}
```

For a generic anime request, the agent should normally resolve to
`anima-turbo-v1-1`. Its prompt contract accepts structured natural language or
Danbooru-style tags, uses the official BF16 Turbo checkpoint with CFG 1 and
8–12 steps, and defaults to 12 steps for the comparison-quality setting. The
route is NSFW-capable but does not require NSFW wording; the caller supplies
the final prompt.

`moody-anima-mix` remains useful for direct comparison and as a lower-priority
fallback. Its prompt contract is deliberately short natural language with
optional light anime tags, and its default is a 4-step Anima render. The named
route exposes only the minimal generation core of the referenced workflow, not
its optional upscale/detailer branch.

### Official Anima Turbo and face reference

`anima-turbo-v1-1` uses the official `anima-turbo-v1.1.safetensors` BF16
checkpoint, `ModelSamplingAuraFlow` with shift `3`, the official Qwen 3 0.6B
text encoder and Qwen image VAE. The downloader pins the published file size
and SHA-256. This route is the generic high-precision comparison target and
does not include a character-specific LoRA.

The separate `anima-turbo-face-reference-ipadapter` route keeps an Anima-native
face/character-reference path available to agents. It uses the pinned Anima
IPAdapter custom node, the `ip_adapter-Character_Reference-10.safetensors`
weights and a local SigLIP2 base patch16 512 vision encoder. It requires
`parameters.face_image`, preferably a clear single-person portrait, and keeps
outfit, pose, scene and lighting in `parameters.prompt`:

```json
{
  "workflow": "anima-turbo-face-reference-ipadapter",
  "parameters": {
    "face_image": "xiaoman_portrait.png",
    "prompt": "masterpiece, best quality, 1girl, solo, an anime maid in a sunlit cafe, standing, warm afternoon light",
    "ipadapter_weight": 1.0,
    "steps": 12,
    "cfg": 1.0
  }
}
```

This is reference conditioning rather than FaceID recognition: the current
implementation encodes the supplied image as a whole and does not guarantee
pixel-locked identity. The route is marked `face_only: false` in the manifest;
the existing SDXL face-reference route remains available when its separate
architecture is explicitly preferred.

`POST /project/generate` and `POST /project/prompt` accept the same body. The
optional top-level `prompt_granularity` is advisory and echoed in the response.
`parameters.prompt` is required, with `subject` accepted as a compatibility
alias; an empty string is allowed. Suggested subject/appearance details in
`agent_hints` describe prompt writing, not additional required API fields.
Unknown fields return 400, missing required inputs return 422, and unknown
workflows return 404. Errors consistently include `message`, `route`, and
`expected`. The final positive
and negative prompt text remains caller-provided; the API only exposes the
route-specific prompting contract and does not synthesize prompt content.
Passing `character` or `outfit` to `/project/resolve` is rejected deliberately;
those fields belong in `parameters.prompt` after the route is selected.

`workflows\agent_route_simulation_10.json` contains ten natural-language user
requests, the route query an agent would extract, and the resulting prompt
input. It includes SFW and NSFW-capable NIKKE cases. It is a simulation fixture
only and is not submitted to the generation endpoint. Repeat the real route
simulation with:

```powershell
.\scripts\simulate_agent_routes.ps1
```

For a public HTTP contract audit, run the fast non-generating probe with the
ComfyUI virtual environment:

```powershell
.\ComfyUI\.venv\Scripts\python.exe .\scripts\verify_agent_api.py
```

It checks discovery, route selection, parameter validation, every ready
workflow's dry-run graph, input lifecycle and resource reporting. The optional
`--generate` flag runs the same SFW teapot probes through the named queue.

The API returns identical `prompt_id` and `job_id` plus `job_url`; poll `/project/jobs/{prompt_id}`
until it reports `succeeded` or `failed`, then use the returned `/view` URL for
the output. During execution, the job response also exposes `phase` values such
as `waiting_for_turn`, `waiting_for_comfyui`, `model_switch` and `generation`,
plus queue position where applicable. `queued_ms` measures acceptance to worker
start; `elapsed_ms` includes model switching and generation after worker start.
`dry_run: true` validates the resolved graph without queueing it.

Each workflow publishes `parameter_schema`, `control_ports`, `resource_limits`
and `agent_hints.recommended` sampling ranges. Responses include
`applied_parameters`, `ignored_parameters` (empty, since unknown fields are
rejected), `effective_parameters`, and advisory `warnings`. Values outside a
recommended sampling range are retained. Official Anima Turbo recommends
8–12 steps at CFG 1; negative prompts have no effect at CFG 1.
Omitted seeds are random; the selected seed is returned for reproducibility.
Img2img honors size and batch: two explicit dimensions resize to that size,
one preserves aspect ratio, and neither uses source size rounded down to 8.

Named `/project/prompt` requests are admitted to a FIFO project queue guarded by
one mutex. Only one named workflow is handed to ComfyUI at a time. Consecutive
jobs for the same workflow keep its model set resident; switching to another
workflow clears executor caches and releases models inside ComfyUI's serial
worker before the next job starts. Native/project transitions use the same
boundary; cleanup cannot unload a concurrently executing native job.
A failed job clears the residency assumption, so the next job also gets
a clean model boundary. The native ComfyUI `POST /prompt` and `/history`
endpoints remain available for clients that already build complete API-format
graphs. Raw native graphs bypass project size and input admission budgets.
The project queue allows 32 pending jobs (overflow returns 429), retains 1,000
job states, and detects native queue removals without waiting a day. Queue and
job state are in memory; restarting the service does not preserve pending jobs.

For PowerShell callers, the helper supports both named workflows and the
legacy raw workflow path:

```powershell
.\scripts\queue_workflow.ps1 -WorkflowName animagine-xl-3.1-nsfw `
  -Prompt 'masterpiece, best quality, 1girl, solo, rapi (nikke)' `
  -LoraConceptStrength 0.9 -LoraRapiStrength 0.8

.\scripts\queue_workflow.ps1 -WorkflowPath .\workflows\moody_desire_v30_minimal.json
```

The independent interface can use the same HTTP contract as the ComfyUI
client; it does not need to know internal node IDs until it asks for a
workflow description.

## Included workflows

| ID | Family | Purpose | Default assets |
| --- | --- | --- | --- |
| `moody-anima-mix` | Anima | Generic fast anime fallback; SFW/NSFW-capable | Moody Anima Mix v1.0, Qwen 3 0.6B encoder, Qwen image VAE |
| `moody-anima-inpaint` | Anima | Masked edits with unmasked original pixels preserved | Moody Anima Mix, Qwen encoder and VAE |
| `anima-turbo-v1-1` | Anima Turbo | Official high-precision generic anime route; SFW/NSFW-capable | Official Anima Turbo v1.1 BF16, Qwen 3 0.6B encoder, Qwen image VAE |
| `anima-turbo-inpaint` | Anima Turbo | Masked edits with unmasked original pixels preserved | Official Anima Turbo v1.1 BF16, Qwen encoder and VAE |
| `anima-turbo-face-reference-ipadapter` | Anima Turbo + IPAdapter | Anima face/character reference route; SFW/NSFW-capable | Official Anima Turbo v1.1 BF16, Anima IPAdapter Character Reference v10, SigLIP2 |
| `moody-desire-v30` | Flux.2/Klein | General anime/portrait workflow; SFW/NSFW-capable | Moody checkpoint, Qwen3 text encoder, Flux.2 VAE |
| `animagine-xl-3.1-nsfw` | SDXL | Anime/NIKKE workflow that can produce NSFW content | Animagine XL 3.1, profile LoRA, Rapi LoRA |
| `blue-archive-style-sdxl` | SDXL | Blue Archive game-CG style workflow; SFW/NSFW-capable | Animagine XL 3.1, Blue Archive Game CG Style LoCon |
| `blue-archive-character-illustrious-xl` | SDXL/Illustrious | Asuna identity and four-outfit workflow, SFW/NSFW-capable | Illustrious XL v0.1, Asuna Ichinose 4-outfit LoRA |
| `sdxl-face-reference-ipadapter` | SDXL + IPAdapter | Portrait reference conditioning | Animagine XL 3.1, IPAdapter Plus Face, ViT-H |
| `sdxl-openpose-controlnet` | SDXL + ControlNet | Body pose conditioning | Animagine XL 3.1, Xinsir OpenPose ControlNet and preprocessor |

Every workflow and every model/component profile in the registry now declares
`nsfw_capable: true` and includes `nsfw` in its content modes. This is a
capability declaration for agent routing; the existing validation probes remain
SFW-only.

The runtime probes and their scope are recorded in the linked audit. A successful probe proves that the
graph/model stack loaded and completed; it is not a benchmark of character
likeness, NSFW quality or prompt fidelity. Individual model profiles in
`workflows\models.json` retain the probe IDs and scope. The two Anima Turbo
routes were additionally re-probed after installing the official checkpoint,
Anima IPAdapter and SigLIP2 assets.

Adding another workflow is data-driven: place an API-format JSON graph in
`workflows\`, add its `model_ids`, `routing`, `capabilities`, `prompting` and
`agent_hints` alongside `required_files`, `bindings`, defaults and compatibility
notes in `workflows\catalog.json`, then add the model profiles to
`workflows\models.json` and restart ComfyUI so the custom-node state is
refreshed.

## Moody Anima Mix model notes

`moody-anima-mix` is the deliberately broad anime comparison/fallback route.
The official `anima-turbo-v1-1` route has higher generic priority now, while a
request that explicitly names a Blue Archive or NIKKE model/style is sent to
that specialized route instead. Character identity remains part of the
agent's prompt; this route does not contain a character LoRA.

The route uses the Civitai Moody Anima Mix v1.0 checkpoint with the official
Anima Qwen 3 0.6B text encoder and Qwen image VAE. Its API graph follows the
reference minimal workflow's fast 4-step core. The reference archive also
contains an optional UltimateSDUpscale branch and UI-only helper nodes; those
are not exposed as separate route choices. The downloaded checkpoint and
dependencies are pinned by byte size and SHA-256 in
`scripts\download_models.ps1`.

## Animagine/NIKKE model notes

Animagine XL 3.1 is an anime-focused SDXL checkpoint. Its model card
recommends Danbooru-style tag prompting, a quality prefix such as
`masterpiece, best quality, very aesthetic, absurdres`, CFG around 5–7 and
fewer than 30 steps. The included workflow follows those defaults and does
not put `nsfw` in the negative prompt, because this profile is deliberately
NSFW-capable.

The downloaded assets are stored under stable project filenames:

- `Animagine_XL_3.1.safetensors` — official 6.94 GB SDXL checkpoint.
- `NIKKES_GoddessOfVictory_Concept.safetensors` — the Animagine/SDXL profile
  LoRA; trigger `IncrsNikkeProfile`, default strength `0.9`.
- `NIKKES_Rapi.safetensors` — the selected current Pony-XL/SDXL-shaped Rapi
  candidate; triggers `rapidef`, `rapiswim` and `rapirnd`, disabled by default
  (`0.0`) and available through an explicit `loras.rapi` value.

The Rapi candidate is not a legacy SD 1.5 LoRA, but it was trained for Pony
family models rather than specifically for Animagine. It loaded and generated
successfully with the included Animagine graph, but visual fidelity is not
guaranteed. Enable `loras.rapi` only for an intentional experiment, or replace the aliased file
and update the catalog metadata when a native Animagine-trained Rapi LoRA is
preferred.

The third workflow uses `BlueArchive_GameCG_Style_Animagine31.safetensors`,
the Animagine3.1/SDXL version of Blue Archive Game CG Style. The published
model notes specify no trigger word and a style strength around `0.8`–`1.0`;
the catalog records those facts so an agent can choose and parameterize it.
Its default graph uses 832×1216, CFG 5, 36 steps, `dpmpp_2m_sde` and Karras.

The fourth workflow uses the compatible Illustrious XL v0.1 checkpoint and
the Asuna Ichinose four-outfit adapter. The adapter model card documents the
identity token `1chn` and the following tag groups: maid, bunny, school and
sports. Use exactly one outfit group per request. The model card describes
these as trained tags rather than separate named outfit trigger words, so the
agent should select the tags from the route's `prompting.lora_triggers`
metadata. This route is a separate workflow/model route; it does not make
character or outfit a resolver selector.

Sources used for the pinned model choices and workflow defaults:

- [Moody Anima Mix model page](https://civitai.com/models/2700077/moody-anima-mix)
- [Moody Anima Mix current metadata](https://civitai.com/api/v1/models/2700077)
- [Moody Anima Minimal Workflow metadata](https://civitai.com/api/v1/models/2700120)
- [CivArchive record for the pinned Moody Anima Mix hash and active mirror](https://civitaiarchives.com/sha256/5368c9b1318c783d7215492002cd1bca94895d1dca8c780ef24f75536fd341f8)
- [CivArchive record for the pinned minimal workflow archive](https://civitaiarchives.com/sha256/4970209212d8ff406f05b5a9c30ae6263bc11340bd3a241d730ae903a07664f9)
- [Official Anima model repository and dependencies](https://huggingface.co/circlestone-labs/Anima)
- [Official Anima Turbo v1.1 diffusion weights](https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/diffusion_models/anima-turbo-v1.1.safetensors)
- [Anima-specific IPAdapter custom node](https://github.com/LuciferTC9527/ComfyUI-Anima_IP-Adapter)
- [Anima IPAdapter Character Reference v10 weights](https://huggingface.co/LuciferTC/Anima-IP-Adapter)
- [SigLIP2 base patch16 512 vision encoder](https://huggingface.co/google/siglip2-base-patch16-512)
- [ComfyUI Anima workflow guidance](https://github.com/niknah/comfyui-docs/blob/main/tutorials/image/anima/anima.mdx)
- [Animagine XL 3.1 model card](https://huggingface.co/cagliostrolab/animagine-xl-3.1)
- [NIKKE profile LoRA mirror](https://huggingface.co/Muapi/nikke-goddess-of-victory-profile-concept-lora-animagine)
- [Selected Rapi Pony-XL/SDXL model](https://civitai.com/models/382624)
- [Blue Archive Game CG Style — Animagine3.1](https://huggingface.co/licyk/sd-lora/blob/main/sdxl/style/ag31_style_ba_v1-000016.safetensors)
- [Illustrious XL v0.1 model card](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0)
- [Asuna Ichinose 4-outfit IllustriousXL LoRA](https://huggingface.co/Muapi/asuna-ichinose-blue-archive-4-outfits-illustriousxl)
- [ComfyUI LoRA guide](https://docs.comfy.org/tutorials/basic/lora)

Models, generated images and runtime logs are ignored by the root repository;
only the deployment scripts, workflow catalog and API-format graphs are meant
to be versioned here.
