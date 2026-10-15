"""Model residency boundaries inside ComfyUI's single prompt worker.

The pinned ComfyUI worker calls PromptExecutor.execute serially for both native
and project jobs. Never unload from the HTTP/event-loop thread: a native prompt
can already be using those same models. Resetting executor caches also drops
custom-node objects (including vision encoders) not owned by model_management.
"""
from __future__ import annotations

import functools
import gc
import logging
import subprocess
import time
from typing import Any


def memory_snapshot() -> dict[str, Any]:
    import psutil
    import comfy.model_management as mm
    from comfy.cli_args import args

    device = mm.get_torch_device()
    allocator_free = int(mm.get_free_memory(device)) if device.type not in {"cpu", "mps"} else None
    physical_free = None
    # WDDM/CUDA can report a per-process budget larger than physically free VRAM.
    # Match by UUID: nvidia-smi's ordinal need not equal the CUDA ordinal.
    if device.type == "cuda":
        import torch
        device_uuid = str(getattr(torch.cuda.get_device_properties(device), "uuid", ""))
        if device_uuid:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3, check=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                for line in result.stdout.splitlines():
                    gpu_uuid, free_mib = (part.strip() for part in line.split(",", 1))
                    if gpu_uuid.removeprefix("GPU-").lower() == device_uuid.removeprefix("GPU-").lower():
                        physical_free = int(free_mib) * 1024**2
                        break
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
    return {
        "device": str(device),
        "device_name": mm.get_torch_device_name(device),
        "free_vram_bytes": min(allocator_free, physical_free) if physical_free is not None else allocator_free,
        "allocator_free_vram_bytes": allocator_free,
        "physical_free_vram_bytes": physical_free,
        "vram_measurement": "cuda-and-nvidia-smi-uuid" if physical_free is not None else "comfyui-allocator-estimate",
        "free_ram_bytes": psutil.virtual_memory().available,
        "cpu_vae": bool(args.cpu_vae),
        "vram_headroom_gib": args.vram_headroom,
        "reserve_vram_gib": args.reserve_vram,
        "scope": "Current selected-device snapshot, not a reservation or model-fit guarantee.",
    }


def release_models() -> None:
    import comfy.model_management as mm

    mm.unload_all_models()
    gc.collect()
    mm.cleanup_models_gc()
    mm.soft_empty_cache()


def install(execution_module: Any, notify_phase: Any, validate_inputs: Any) -> None:
    executor_class = execution_module.PromptExecutor
    if getattr(executor_class.execute, "_project_guard", False):
        return
    original = executor_class.execute

    @functools.wraps(original)
    def execute(executor, prompt, prompt_id, extra_data=None, execute_outputs=None):
        extra_data = extra_data or {}
        execute_outputs = execute_outputs or []
        workflow = extra_data.get("project_workflow")
        timing = extra_data.get("project_timing")
        if timing is not None:
            timing["started_monotonic"] = time.monotonic()
            timing["queued_ms"] = max(0, round((timing["started_monotonic"] - timing["submitted_monotonic"]) * 1000))
        previous = getattr(executor, "_project_previous_workflow", None)
        try:
            if workflow != previous and (workflow is not None or previous is not None):
                notify_phase(prompt_id, "model_switch")
                executor.reset()
                release_models()
                logging.info("Project worker model boundary: %s -> %s", previous, workflow)
            if workflow:
                resources = extra_data["project_resources"]
                validate_inputs(resources)
                notify_phase(prompt_id, "generation")
            original(executor, prompt, prompt_id, extra_data, execute_outputs)
        except Exception as exc:
            logging.exception("Prompt worker failed for %s", prompt_id)
            executor.success = False
            executor.history_result = {"outputs": {}}
            executor.status_messages = []
            executor.server.client_id = extra_data.get("client_id")
            executor.add_message("execution_error", {
                "prompt_id": prompt_id, "node_id": None, "node_type": "ProjectRuntimeGuard",
                "exception_type": type(exc).__name__, "exception_message": str(exc),
                "traceback": [], "executed": [], "current_inputs": {}, "current_outputs": [],
            }, broadcast=False)
        finally:
            executor._project_previous_workflow = workflow if executor.success else None
            if not executor.success:
                # reset() resets status; retain the failure for native history.
                messages, history = executor.status_messages, executor.history_result
                try:
                    executor.reset()
                    release_models()
                except Exception:
                    logging.exception("Failed to release models after prompt failure")
                executor.success, executor.status_messages, executor.history_result = False, messages, history
            if timing is not None:
                timing["elapsed_ms"] = max(0, round((time.monotonic() - timing["started_monotonic"]) * 1000))

    execute._project_guard = True
    executor_class.execute = execute
