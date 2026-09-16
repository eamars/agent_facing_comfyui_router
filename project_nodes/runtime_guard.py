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
from typing import Any


def memory_snapshot() -> dict[str, Any]:
    import psutil
    import comfy.model_management as mm
    from comfy.cli_args import args

    device = mm.get_torch_device()
    return {
        "device": str(device),
        "device_name": mm.get_torch_device_name(device),
        "free_vram_bytes": int(mm.get_free_memory(device)) if device.type != "cpu" else None,
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


def install(execution_module: Any, notify_phase: Any) -> None:
    executor_class = execution_module.PromptExecutor
    if getattr(executor_class.execute, "_project_guard", False):
        return
    original = executor_class.execute

    @functools.wraps(original)
    def execute(executor, prompt, prompt_id, extra_data=None, execute_outputs=None):
        extra_data = extra_data or {}
        execute_outputs = execute_outputs or []
        workflow = extra_data.get("project_workflow")
        previous = getattr(executor, "_project_previous_workflow", None)
        try:
            if workflow != previous and (workflow is not None or previous is not None):
                notify_phase(prompt_id, "model_switch")
                executor.reset()
                release_models()
                logging.info("Project worker model boundary: %s -> %s", previous, workflow)
            if workflow:
                resources = extra_data["project_resources"]
                snapshot = memory_snapshot()
                limits = resources["limits"]
                if snapshot["free_ram_bytes"] < limits["min_free_ram_gib"] * 1024**3:
                    raise RuntimeError("Insufficient free system RAM; close other workloads or reduce the deployment memory budget.")
                free_vram = snapshot["free_vram_bytes"]
                if free_vram is not None and free_vram < limits["min_free_vram_gib"] * 1024**3:
                    # Same-workflow caches may be reclaimable; retry only the
                    # memory check, never silently change the user's generation.
                    executor.reset()
                    release_models()
                    if memory_snapshot()["free_vram_bytes"] < limits["min_free_vram_gib"] * 1024**3:
                        raise RuntimeError("Insufficient free VRAM headroom on the selected device; release other GPU workloads before retrying.")
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

    execute._project_guard = True
    executor_class.execute = execute
