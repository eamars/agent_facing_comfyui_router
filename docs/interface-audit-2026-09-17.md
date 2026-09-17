# Agent interface audit — 2026-09-17

Status: **verified for the tested API and runtime paths**. Visual likeness,
NSFW output quality, and every hardware/resolution combination were not tested.
All generation probes described a ceramic teapot.

The design hypothesis was that an agent can discover a compatible route,
submit only documented parameters, obtain the requested image dimensions and
batch, and observe failures without breaking subsequent queued work. Probes
crossed the public HTTP API, ComfyUI validation, real installed model loading,
GPU execution, input/output files and the native prompt worker.

## Caller feedback

| Reported issue | Result |
| --- | --- |
| `subject` vs `prompt`, missing required fields | `prompt` is canonical; `subject` is a documented compatibility alias. Conflicting aliases fail. Suggested prompt details are explicitly advisory. Missing required fields return 422. |
| Missing `job_id`, silent parameter loss | Queue/job responses include identical `job_id` and `prompt_id`. Queue/dry-run responses include canonical `applied_parameters`, `effective_parameters`, and empty `ignored_parameters`; unknown fields return 400. OpenAPI describes response schemas. |
| Inconsistent validation/error bodies | Errors consistently include `type`, `message`, `route`, `expected`, and optional `details`. Unknown workflow returns 404. Unsupported route parameters still return a human-readable 400. |
| No model sampling ranges | Every route declares `agent_hints.recommended.steps` and `.cfg`. Out-of-range requests retain their values and receive structured warnings. Turbo's published range is **8–12 steps, CFG 1**, rather than 4–8. |
| Uneven control capabilities / no Turbo inpaint | `control_ports` and manifest `route_table` expose the available controls. Added `anima-turbo-inpaint`; it uses the official Turbo checkpoint and sampling shift, generic latent masking, and final pixel compositing. |
| Input lifecycle / deletion accounting | Documented that the project has no automatic TTL or input cleanup. Inputs persist across restarts unless explicitly or externally removed. DELETE returns `remaining_bytes`; listing returns `total_bytes`. Configured ComfyUI input directories are respected. |
| No queue or generation timing | Job responses include `queued_ms` and `elapsed_ms`. Elapsed time starts at worker entry and includes model switching, generation and failure cleanup; it excludes queue time. |

The three supplied regression scenarios are covered: subject-only input maps
to the prompt; `nonexistent_field` returns 400 naming the field; an unknown
workflow returns 404 with the same error shape. Missing prompt returns 422.

The Turbo sampling recommendation comes from the
[official Anima model card](https://huggingface.co/circlestone-labs/Anima).
At CFG 1, the installed ComfyUI sampler skips negative conditioning. The API
now warns about that behavior rather than implying a negative prompt is active.

## Other defects fixed

- Generic Blue Archive requests previously selected an Asuna identity adapter.
  Asuna is now explicit-only; generic Blue Archive requests select the style
  route. Matching no longer treats short aliases such as `BA` as arbitrary
  substrings. Exact `family` selection distinguishes SDXL and Anima controls.
- The Pony-trained Rapi adapter previously applied to every NIKKE request on
  Animagine. It defaults to zero and remains an explicit experiment. Successful
  tensor loading does not establish cross-family visual compatibility.
- Img2img previously ignored requested dimensions and batch size. Its graph
  now resizes before encoding and repeats latents for the requested batch.
  Image inputs are accepted only through a route's declared input contract.
- Inpainting now composites through the original mask after decode, preserving
  unmasked pixels exactly. Source/mask dimensions must match. RGB and inverse
  alpha semantics are documented; masks without alpha cannot silently act as
  alpha masks. These remain generic masked workflows, not inpaint-trained models.
- Dry runs now execute native graph validation. Unknown fields and sampler
  names can no longer return misleading successful readiness responses.
- Seeds omitted by the caller are freshly generated and returned. Defaults
  use the JSON/JavaScript exact integer range; explicit supported larger seeds
  remain available. Prompt granularity is advisory rather than a rejection gate.
- Readiness checks include registry eligibility, installed node classes and
  recorded file sizes. Discovery does not hash multi-gigabyte weights on each
  request; the downloader remains responsible for SHA-256 verification.
- Raw PowerShell `-WorkflowPath -DryRun` no longer submits a real job. The helper
  exposes image/control parameters. Setup checks intermediate install failures
  and creates input/output/runtime folders in the correct location.

## Resource and concurrency fixes

The former coordinator could unload models from a background thread while a
native ComfyUI job was executing. Cleanup now happens inside the pinned
ComfyUI serial worker, including native/project transitions. Executor caches
are cleared on a workflow change, releasing custom-node objects and vision
encoders that the model manager alone might not own. Consecutive jobs for one
workflow retain useful caches; failures clear the residency assumption.

Requests have visible, configurable total-pixel budgets (including batch),
input-image limits, and RAM/VRAM headroom checks. Single-frame inputs are
required. Queued input changes are detected before loading. There are at most
32 pending project jobs and 1,000 retained project states. A native queue
removal is detected promptly, and failures return actionable details.

Managed startup reserves 1 GiB in both the dynamic and standard memory
management paths and uses CPU VAE decoding. These settings operate at different
layers; they are not a promise of a combined 2 GiB reservation. Raw native
graphs bypass project admission budgets. Other processes can consume memory
after an admission check, so OOM cannot be ruled out universally.

On this Windows machine the CUDA allocator estimate overstated physically
free VRAM by roughly 6 GB in one observed snapshot. The new guard queries
NVIDIA physical free memory by **GPU UUID**, takes the lower estimate, and
reports when it has to fall back to the allocator estimate. CUDA device 0
here is the RTX 4090, while `nvidia-smi` gives it a different ordinal.

## Verification evidence

- 81 public-HTTP checks passed across 21 jobs on an isolated local instance:
  20 succeeded and one intentionally failed after its queued reference was
  deleted. The next job succeeded, establishing failure recovery.
- All 11 named routes passed native graph validation and real model execution.
  Both inpaint routes preserved all unmasked source pixels in the test.
- Six concurrent submissions completed in order alongside a native graph.
  Their measured queue delays rose from 4,485 ms to 23,921 ms.
- An additional 1024×1024, batch-two Anima Turbo probe completed in 19,812 ms
  with two outputs. The minimum sampled physical free VRAM was about 11.25 GiB;
  minimum sampled free RAM was about 26.1 GiB. These are samples, not certified
  instantaneous peak-memory measurements.
- The existing service on port 8191 subsequently reported API 1.5. All 39
  non-generating HTTP contract checks passed against that service.
- All 15 targeted unit regressions passed, covering alias and error contracts, graph bindings,
  input changes, readiness, route selection, memory boundaries, queue limits,
  removal detection and timing. PowerShell parsing and Python compilation pass.
- After the final hardening changes, all 39 contract checks passed again and
  a subject-only request generated successfully using the route defaults.
  Its queue and job responses validated against the published JSON schemas.

The main runtime report is
[`verification.json`](../ComfyUI/.codex-runtime/interface-audit/verification.json),
with separate [live contract](../ComfyUI/.codex-runtime/interface-audit/live-contract.json)
and [larger workload](../ComfyUI/.codex-runtime/interface-audit/2mp-probe.json)
reports. Runtime artifacts are intentionally ignored by Git.

Repeat the checks with:

```powershell
.\ComfyUI\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\ComfyUI\.venv\Scripts\python.exe scripts\verify_agent_api.py
.\ComfyUI\.venv\Scripts\python.exe scripts\verify_agent_api.py --generate
```

The generating probe uploads unique synthetic references, deletes only those
references, and writes a JSON report. For isolated testing, use a separate
port, input/output directories and SQLite database. An earlier shared-directory
run was interrupted when its references disappeared during concurrent workspace
activity; the complete passing run used isolated directories. No project TTL
or automatic input deletion was found, and the historical external cause of
the caller's missing inputs remains unidentified.

## Content behavior

No content filter or keyword blocklist was added. The project passes caller
prompt text through unchanged, including empty prompts and empty negatives.
The inherited `guro` suppression tag was removed from the Illustrious default
negative prompt. Content mode remains routing metadata, not a prompt rewrite.
No explicit prompts were used for generation in this audit.

The original attempt to restart the existing service was rejected by automatic
approval review with only “blocked by policy” supplied as the reason. Testing
continued on a separate instance. The existing service was subsequently
restarted by another process and was checked independently afterward.

## Deployment verification

After the user explicitly requested deployment, the idle managed service was
restarted successfully with the final code. The LAN endpoint
`http://192.168.2.13:8191` reports API 1.5.0 and all 11 workflows ready.
All 39 live contract checks passed. A real subject-only Turbo request completed
in 11,875 ms with a 896×1152 image; its submission and job responses passed the
published JSON schemas, and image retrieval succeeded through the LAN endpoint.

The [deployment contract report](../ComfyUI/.codex-runtime/interface-audit/deployment-contract.json)
and [generation report](../ComfyUI/.codex-runtime/interface-audit/deployment-generation.json)
record the results. The generation report also records SHA-256 fingerprints of
the deployed API, runtime guard, workflow definitions, and launch scripts.
