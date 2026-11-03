# Design decisions

Why the service works the way it does. Current behaviour lives in the [README](../README.md),
[`workflows/catalog.json`](../workflows/catalog.json) and the agent manifest
(`/.well-known/agent-manifest.json`); this file keeps the reasons and the evidence behind them.
The dated reports it replaces (interface audit 2026-09-17, Qwen integration 2026-09-22, Qwen
uncensored swap 2026-09-26, MOSS integration 2026-10-16) are in git history before this file.

## Runtime and resources

**One serial worker; models switch inside it** (2026-09-17, `48a8bcd`). Named jobs run through
ComfyUI's native serial prompt worker. `project_nodes/runtime_guard.py` wraps it: a different
workflow clears executor caches and unloads models before running, the same workflow keeps them,
and a failed job resets what is assumed resident. An earlier coordinator unloaded models from a
background thread while a native job was running; doing it inside the worker removed that race.

**No memory pre-checks** (2026-10-15, `3ddbe71`). A free RAM/VRAM snapshot cannot predict whether
a job fits (Windows pages, ComfyUI offloads). The old 20 GiB gate refused Qwen jobs that ran fine:
one succeeded with 15.4 GiB free. Jobs now always run; a real out-of-memory failure is reported as
`error.code: out_of_memory`, and the pixel budget only adds a warning.

**CPU VAE commit misses become clean OOM** (2026-10-25, `81b1994`). With `--cpu-vae`, torch's
mimalloc arena wrote into uncommitted pages when Windows refused a commit (error 1455), crashing
the server with an access violation instead of raising. `MIMALLOC_DISALLOW_ARENA_ALLOC=1` in the
launcher turns that into a "not enough memory" error that the API reports as `out_of_memory`
(about +0.3 s per 512² decode). The page file was also raised to 128,000 MB.

**The GPU is pinned by UUID.** CUDA's device order differs from `nvidia-smi`'s on this machine, so
the launcher exposes only the RTX 4090's UUID and refuses device-override arguments.

## Models and routing

**Qwen-Image 2.1 runs the uncensored Q4_K_M GGUF** (2026-09-26, `ad14afc`). It replaced the
official INT8 ConvRot diffusion model with the same graphs and API contract (Hugging Face revision
`40319fb15542f0ad22921e0124a191a8a935a60a`, SHA-256 `e79c8a00…41b41`). The INT8 file stays on disk
as a rollback copy and no route loads it. The publisher's uncensored LoRA was added later
(`861b066`, default strength 1.0). On the 4090 a 1024² text-to-image job takes about 52 s and a
single-reference edit about 63 s; edit output follows the source size, rounded to 32 px.

**Qwen for real-world images, MOSS for layouts** (2026-10-16, `8eff134`). Twelve shared prompts
at seed 4242, plus layout JSON for MOSS:

| Test | Winner | Deciding detail |
| --- | --- | --- |
| Photo portrait, product label, chalkboard menu, counting, anime tags, prose poster | Tie | Both met the prompt |
| Chinese poster | Qwen | MOSS invented a garbled subtitle and seal |
| 12-detail scene | MOSS | 12/12; Qwen missed one object |
| NZ landscape | Qwen | Recognisably Central Otago |
| Weather-app UI mock-up | MOSS | Qwen sparse, garbled glyph |
| Macro at 1456² | Qwen | More authentic, 87 s vs 110 s |
| "a cat" | Qwen | MOSS deformed the face on a terse prompt |
| Poster as layout JSON | MOSS only | Every box within ~10 px, two seeds |

Qwen is the default for `photo`, `product`, `landscape`, `poster` and `realistic` because it
tolerates short prompts, knows real places and does not invent text. MOSS is exclusive: only
`layout`, `ui mockup`, `moss` or its id select it, and it needs full descriptive prompts. MOSS's
explicit output is untested.

**Checkpoint versions are styles** (`f3fb305`). JANKU v5.0 stays beside v7.77 because they look
different: 'visual novel' selects v5, 'mainstream anime' selects v7.77.

**Content does not route** (2026-11-04, `6be7f30`). The model families here all make explicit
images (the owner's assessment; seen on JANKU, Anima Turbo and its character routes, Qwen-Image and
Moody Desire; MOSS is untested), so resolve chooses by subject and look, never by SFW/NSFW. A same-seed comparison showed
JANKU v7.77 (glossy doujin) and Anima Turbo (flat cel) both handle explicit anime well; choosing
between them is a style preference left to each agent's skill. `content_mode`, `input_type` and
`prompt_granularity` are accepted but ignored with a `deprecated_selector` warning, so older
callers keep working. A route's `best_for` may still name a specialty: Moody Desire (Flux.2
Klein) is a realism-first model ranked below Qwen-Image, MOSS and Anima Turbo, selected by style
'moody desire', with explicit realistic people as its stated specialty (`3cd4a6d`).

**One Anima route per AI girl** (2026-11-03, `9d6a18c`, `4aedb4c`). Character LoRAs are judged
against the public setting sheets (Lanpose's muse sheets), not only their own cards. The JANKU
LoRAs for Claude and DeepSeek drew a different design; those for ChatGPT, Qwen and Gemini were
less faithful than Anima. A same-seed NSFW check showed each Anima character route keeps its
identity, so every AI girl has exactly one route. GLM, Grok and Kimi use aiseaK's multi-character
aigirls LoRA. Doubao was skipped by choice; Llama and MiniMax have no LoRA yet.

## Agent interface

**API 2.0: a short manifest** (2026-11-04, `6be7f30`). The manifest went from 552 KB to about
31 KB: a five-step quick start, the route table and the input, pose, warning and error rules.
Full route detail comes from `/project/resolve` or `/project/workflows/{id}`. In the same 12 blind
tasks (a local LLM through tool calls, and a Haiku client that only saw the HTTP API) both agents
had every request accepted on the first try afterwards; the local model managed 10 of 12 before,
and Haiku went from eight partial manifest fetches to one full read.

**Contract rules from the 2026-09-17 audit, still in force.**
- Unknown fields return 400 and an unknown workflow 404, with one error shape: `type`, `message`,
  `route`, `expected` and optional `details`. A missing prompt returns 422.
- `job_id` is an alias of `prompt_id`.
- A dry run runs ComfyUI's native graph validation.
- Uploaded inputs persist until deleted, with no TTL.
- Inpaint composites the unmasked source pixels back exactly.
- At CFG 1 the sampler skips negative conditioning. The API warns only when the caller sent a
  negative prompt.

## Pose add-on

**One `pose` parameter for every family** (2026-11-03). It takes a library view id or an
uploaded image. A blind test chose this interface: with a mock API, a local LLM and Haiku each
handled 11 tasks, and library ids plus uploads scored 11/11 after two doc fixes, while
text-matched descriptions and agent-written keypoints lost. DWPose turns either input into a
whole-body skeleton, and each family's adapter draws it in the style that adapter was trained on.
Fidelity is PCK@0.1 with and without the pose, measured through the API with
`scripts/verify_pose_fidelity.py`. The test subject is Xiaoman, a prompt-only character: a
character LoRA (Claude 娘) held its own stance and hid the adapter's effect.

| Family | Mechanism | With pose | Without | Commit |
| --- | --- | --- | --- | --- |
| SDXL / Illustrious | Xinsir OpenPose ControlNet | 0.75 | 0.34 | `c0242b3` |
| Anima | Anima Control Pose Preview-2 (control LoRA + embedder) | 0.77 | 0.28 | `560b401` |
| Qwen-Image 2.1 text-to-image | Skeleton as a reference image of the text encoder | 0.96 | 0.44 | `b0aa46c` |
| Moody Desire (Flux.2 Klein) | Skeleton as a `ReferenceLatent` | 0.83 | 0.44 | `6d72fee` |

What was tried and rejected:
- **Kohya's Anima LLLite pose-1** gained only +0.09.
- **Anima any-test-v2** copied the sketch's mannequin shape instead of the pose.
- **Qwen's edit route** pasted the pose image into the picture instead of re-posing the character
  (4 of 4 tries).

Pose therefore applies only to generation from text, with or without a face reference. Edit,
image-to-image and inpaint requests are refused, since the pose fights the source image, and
Anima inpaint also crashed on odd latent sizes. MOSS accepts no image input, so it takes no pose.
