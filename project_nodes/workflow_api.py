"""Named workflow catalog and parameterized API for the project deployment.

This module is loaded by ComfyUI as a zero-node custom-node extension.  It
adds a small, unauthenticated, server-side workflow catalog while leaving
ComfyUI's native /prompt endpoint available for full workflow clients.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import secrets
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from aiohttp import web
from PIL import Image, ImageOps, UnidentifiedImageError

import execution
import folder_paths
from comfy_execution.jobs import validate_job_id
from server import PromptServer

from project_nodes import pose as pose_addon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PROJECT_ROOT / "ComfyUI"
WORKFLOW_ROOT = PROJECT_ROOT / "workflows"
CATALOG_PATH = WORKFLOW_ROOT / "catalog.json"
MAX_TEXT_LENGTH = 16_000
MAX_PROMPT_LENGTH = 8_000
MAX_INPUT_IMAGE_LENGTH = 512
MAX_INPUT_PIXELS = 16_777_216
INPUT_FIT_MODES = ("downscale", "none")
FITTED_INPUT_DIR = "_fitted"
MAX_PROJECT_QUEUE = 32
MAX_PROJECT_JOB_STATES = 1000
PROJECT_QUEUE_POLL_SECONDS = 0.2
PROJECT_JOB_TIMEOUT_SECONDS = 24 * 60 * 60
# API 1.x fields that no longer steer anything; accepted with a deprecated_selector warning.
DEPRECATED_SELECTORS = ("content_mode", "content", "safety_mode", "input_type", "input", "prompt_granularity", "granularity")
RESOLVE_SELECTORS = ["domain", "style", "input_mode", "family", "workflow"]
INPUT_MODES = ["text-to-image", "image-edit", "image-to-image", "inpaint", "face-reference", "openpose-control"]


class ApiError(Exception):
    def __init__(self, status: int, message: str, details: Any = None, *, route: str | None = None, expected: list[str] | None = None,
                 code: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.details = details
        self.code = code
        self.route = route
        self.expected = expected


def _json_error(error: ApiError) -> web.Response:
    payload: dict[str, Any] = {
        "error": {
            "type": "project_workflow_api_error",
            "code": error.code,
            "message": error.message,
            "route": error.route,
            "expected": error.expected or [],
        }
    }
    if error.details is not None:
        payload["error"]["details"] = error.details
    return web.json_response(payload, status=error.status)


def _read_catalog() -> dict[str, Any]:
    try:
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiError(500, "Workflow catalog is unavailable", str(exc)) from exc
    if not isinstance(catalog, dict) or not isinstance(catalog.get("workflows"), list):
        raise ApiError(500, "Workflow catalog has an invalid schema")
    model_catalog_file = catalog.get("model_registry_file")
    if model_catalog_file is None:
        catalog["model_registry"] = []
    elif not isinstance(model_catalog_file, str) or not model_catalog_file:
        raise ApiError(500, "Workflow catalog has an invalid model registry path")
    else:
        root = WORKFLOW_ROOT.resolve()
        model_path = (root / model_catalog_file).resolve()
        try:
            model_path.relative_to(root)
        except ValueError as exc:
            raise ApiError(500, "Model registry file escapes the workflow directory") from exc
        try:
            model_catalog = json.loads(model_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ApiError(500, "Model registry is unavailable", str(exc)) from exc
        if isinstance(model_catalog, dict):
            model_registry = model_catalog.get("models")
        else:
            model_registry = model_catalog
        if not isinstance(model_registry, list):
            raise ApiError(500, "Model registry has an invalid schema")
        catalog["model_registry"] = model_registry
    _annotate_pose_routes(catalog)
    return catalog


# input_mode values that pick routes able to take parameters.pose.
POSE_INPUT_MODES = ("openpose-control", "pose")


def _annotate_pose_routes(catalog: dict[str, Any]) -> None:
    """Give each route that generates from text its family's pose adapter; text-to-image routes also match
    input_mode openpose-control.

    Edit and inpaint routes start from an existing image whose composition a pose would fight (Qwen edit pasted
    the skeleton in instead of re-posing), so they take no pose.
    """
    for entry in catalog["workflows"]:
        if not isinstance(entry, dict):
            continue
        modes = entry.get("capabilities", {}).get("input_modes", [])
        generates = "text-to-image" in modes or "face-reference" in modes
        adapter = pose_addon.adapter_for(catalog, entry) if generates else None
        if adapter is None:
            entry["pose"] = {"supported": False}
            continue
        entry["pose"] = {"supported": True, "adapter": adapter["id"], "mechanism": adapter["mechanism"],
                         "model_ids": adapter["model_ids"], "default_strength": adapter["default_strength"],
                         "measured_fidelity": adapter.get("measured_fidelity")}
        entry.setdefault("control_ports", {})["controlnet"] = adapter["mechanism"].startswith("controlnet")
        if "text-to-image" in modes:  # face-reference routes take pose too but keep their own selector
            modes.extend(mode for mode in POSE_INPUT_MODES if mode not in modes)
            routing_modes = entry.get("routing", {}).get("input_modes")
            if isinstance(routing_modes, list):
                routing_modes.extend(mode for mode in POSE_INPUT_MODES if mode not in routing_modes)


def _is_workflow(entry: dict[str, Any], workflow_id: Any) -> bool:
    # previous_ids keep renamed routes callable by their old ids.
    return entry.get("id") == workflow_id or workflow_id in entry.get("previous_ids", [])


def _find_workflow(workflow_id: str) -> dict[str, Any]:
    if not isinstance(workflow_id, str) or not workflow_id:
        raise ApiError(400, "workflow must be a non-empty workflow id")
    catalog = _read_catalog()
    for entry in catalog["workflows"]:
        if isinstance(entry, dict) and _is_workflow(entry, workflow_id):
            return entry
    raise ApiError(404, f"Unknown workflow: {workflow_id}", route=workflow_id,
                   expected=[entry["id"] for entry in catalog["workflows"]])


def _workflow_path(entry: dict[str, Any]) -> Path:
    relative_file = entry.get("file")
    if not isinstance(relative_file, str) or not relative_file:
        raise ApiError(500, f"Workflow {entry.get('id', '<unknown>')} has no file")
    root = WORKFLOW_ROOT.resolve()
    path = (root / relative_file).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ApiError(500, "Workflow file escapes the workflow directory") from exc
    if not path.is_file():
        raise ApiError(500, f"Workflow file is missing: {relative_file}")
    return path


def _read_workflow(entry: dict[str, Any]) -> dict[str, Any]:
    path = _workflow_path(entry)
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiError(500, f"Workflow file cannot be read: {path.name}", str(exc)) from exc
    if not isinstance(graph, dict):
        raise ApiError(500, f"Workflow {path.name} must contain an API-format object")
    return graph


def _required_files(entry: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    comfy_root = COMFY_ROOT.resolve()
    for relative_file in entry.get("required_files", []):
        if not isinstance(relative_file, str):
            continue
        path = (comfy_root / relative_file).resolve()
        try:
            path.relative_to(comfy_root)
        except ValueError:
            missing.append(relative_file)
            continue
        if not path.is_file():
            missing.append(relative_file)
    return missing


def _model_missing_files(model: dict[str, Any]) -> list[str]:
    relative_files = model.get("required_files")
    if not isinstance(relative_files, list):
        relative_file = model.get("required_file")
        if relative_file is None:
            relative_file = model.get("file")
        relative_files = [relative_file] if isinstance(relative_file, str) and relative_file else []
    comfy_root = COMFY_ROOT.resolve()
    missing: list[str] = []
    for relative_file in relative_files:
        if not isinstance(relative_file, str) or not relative_file:
            continue
        path = (comfy_root / relative_file).resolve()
        try:
            path.relative_to(comfy_root)
        except ValueError:
            missing.append(relative_file)
            continue
        if not path.is_file():
            missing.append(relative_file)
    return missing


def _public_model(model: dict[str, Any]) -> dict[str, Any]:
    public_keys = (
        "id",
        "name",
        "kind",
        "status",
        "routeable",
        "family",
        "architecture",
        "base_model",
        "precision",
        "tensor_dtypes",
        "higher_precision_alternative",
        "description",
        "capabilities",
        "prompting",
        "compatibility",
        "source",
        "validation",
        "required_file",
        "required_files",
        "route_roles",
        "file_sizes",
    )
    result = {key: model[key] for key in public_keys if key in model}
    result["missing_files"] = _model_missing_files(model)
    result["invalid_files"] = []
    for name in model.get("required_files", [model.get("required_file")]):
        if not isinstance(name, str) or name in result["missing_files"]:
            continue
        size = (COMFY_ROOT / name).stat().st_size
        expected = model.get("file_sizes", {}).get(name)
        if size == 0 or (expected is not None and size != expected):
            result["invalid_files"].append({"file": name, "size_bytes": size, "expected_bytes": expected})
    result["ready"] = bool(model.get("status") == "installed" and not result["missing_files"] and not result["invalid_files"])
    return result


def _public_entry(
    entry: dict[str, Any], model_registry: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    public_keys = (
        "id",
        "previous_ids",
        "name",
        "description",
        "family",
        "resident_group",
        "tags",
        "model_ids",
        "required_files",
        "compatibility_notes",
        "capabilities",
        "input_contract",
        "selection",
        "routing",
        "agent_hints",
        "prompting",
        "validation",
        "bindings",
        "lora_bindings",
        "defaults",
        "resource_limits",
        "control_ports",
        "pose",
    )
    result = {key: entry[key] for key in public_keys if key in entry}
    result["file"] = entry.get("file")
    result["missing_files"] = _required_files(entry)
    result["ready"] = not result["missing_files"]
    if model_registry is not None:
        models_by_id = {
            model.get("id"): model
            for model in model_registry
            if isinstance(model, dict) and isinstance(model.get("id"), str)
        }
        result["models"] = [
            _public_model(models_by_id[model_id])
            for model_id in entry.get("model_ids", [])
            if isinstance(model_id, str) and model_id in models_by_id
        ]
        missing_model_ids = [
            model_id
            for model_id in entry.get("model_ids", [])
            if isinstance(model_id, str) and model_id not in models_by_id
        ]
        model_missing_files = list(
            dict.fromkeys(
                missing_file
                for model_id in entry.get("model_ids", [])
                if isinstance(model_id, str) and model_id in models_by_id
                for missing_file in _model_missing_files(models_by_id[model_id])
            )
        )
        result["missing_model_ids"] = missing_model_ids
        result["model_missing_files"] = model_missing_files
        result["unavailable_model_ids"] = [model["id"] for model in result["models"]
                                           if not model["ready"] or not model.get("routeable", False)]
        result["ready"] = not result["missing_files"] and not missing_model_ids and not model_missing_files and not result["unavailable_model_ids"]
        if result.get("pose", {}).get("supported"):
            # Pose is optional: a missing adapter model disables pose, not the route.
            pose_models = [models_by_id.get(model_id) for model_id in result["pose"]["model_ids"]]
            result["pose"] = {**result["pose"], "ready": all(model and _public_model(model)["ready"] for model in pose_models)}
    import nodes
    try:
        graph = _read_workflow(entry)
        required_nodes = {node["class_type"] for node in graph.values()}
        if entry.get("input_contract", {}).get("image", {}).get("mode") == "latent-img2img":
            required_nodes.update({"LoadImage", "ImageScale", "VAEEncode", "RepeatLatentBatch"})
        result["missing_nodes"] = sorted(required_nodes - nodes.NODE_CLASS_MAPPINGS.keys())
        result["ready"] = result["ready"] and not result["missing_nodes"]
        result["parameter_schema"] = _parameter_schema(entry, graph)
    except ApiError as exc:
        result["ready"] = False
        result["graph_error"] = exc.message
    result["resource_limits"] = _resource_limits(entry)
    result["readiness_scope"] = "Model files, registry eligibility and registered node classes; runtime fit and output fidelity are not guaranteed."
    return result


def _parameter_schema(entry: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    templates = _openapi_document()["components"]["schemas"]["WorkflowParameters"]["properties"]
    properties = {name: copy.deepcopy(templates[name]) for name in entry["bindings"] if name in templates}
    if "subject" in templates:
        properties["subject"] = copy.deepcopy(templates["subject"])
    required = []
    for contract in entry.get("input_contract", {}).values():
        if not isinstance(contract, dict):
            continue
        name = contract.get("parameter")
        if name in templates:
            properties[name] = copy.deepcopy(templates[name])
            if contract.get("required") and name != "prompt" and name not in required:
                required.append(name)
    for name, spec in properties.items():
        if name in entry["bindings"] and name not in {"prompt", "seed", "input_image", "mask_image", "face_image"}:
            spec["default"] = _binding_value(graph, entry, name)
    if entry.get("pose", {}).get("supported"):
        properties["pose"] = copy.deepcopy(templates["pose"])
        if entry["pose"]["default_strength"] is not None:  # a reference-image adapter has no strength to set
            properties["pose_strength"] = {**copy.deepcopy(templates["pose_strength"]), "default": entry["pose"]["default_strength"]}
    if "seed" in properties:
        properties["seed"]["description"] = "Random when omitted; the effective seed is returned in the response."
    if _image_inputs(entry):
        properties["input_fit"] = {
            "type": "string", "enum": list(INPUT_FIT_MODES), "default": "downscale",
            "description": "downscale: shrink oversized image inputs to what this route can run and use the first frame of animated "
                           "input; each change is reported as an input_downscaled/input_resized/input_first_frame warning. "
                           "none: keep the upload unchanged; inputs the route cannot run are refused with a stable error code.",
        }
    if entry.get("lora_bindings"):
        properties["loras"] = {"type": "object", "additionalProperties": False, "properties": {
            name: {"type": "number", "minimum": -2, "maximum": 2, "default": spec["default"]}
            for name, spec in entry["lora_bindings"].items()
        }}
    return {"type": "object", "additionalProperties": False, "required": required, "properties": properties,
            "anyOf": [{"required": ["prompt"]}, {"required": ["subject"]}]}


def _node_input(graph: dict[str, Any], spec: dict[str, Any], name: str) -> tuple[dict[str, Any], str]:
    node_id = str(spec.get("node", ""))
    input_name = spec.get("input")
    node = graph.get(node_id)
    if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict) or not isinstance(input_name, str):
        raise ApiError(500, f"Workflow binding {name} is invalid")
    return node["inputs"], input_name


def _set_binding(graph: dict[str, Any], entry: dict[str, Any], name: str, value: Any) -> None:
    bindings = entry.get("bindings", {})
    spec = bindings.get(name) if isinstance(bindings, dict) else None
    if not isinstance(spec, dict):
        raise ApiError(400, f"Workflow does not expose parameter: {name}")
    inputs, input_name = _node_input(graph, spec, name)
    inputs[input_name] = value


def _as_int(name: str, value: Any, minimum: int, maximum: int, *, multiple: int | None = None) -> int:
    if isinstance(value, bool):
        raise ApiError(400, f"{name} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ApiError(400, f"{name} must be an integer") from exc
    if converted != value or not minimum <= converted <= maximum:
        raise ApiError(400, f"{name} must be between {minimum} and {maximum}")
    if multiple and converted % multiple:
        raise ApiError(400, f"{name} must be a multiple of {multiple}")
    return converted


def _as_float(name: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ApiError(400, f"{name} must be a number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ApiError(400, f"{name} must be a number") from exc
    if not minimum <= converted <= maximum:
        raise ApiError(400, f"{name} must be between {minimum} and {maximum}")
    return converted


def _as_text(name: str, value: Any, maximum: int) -> str:
    if not isinstance(value, str):
        raise ApiError(400, f"{name} must be a string")
    if len(value) > maximum:
        raise ApiError(400, f"{name} is too long; maximum is {maximum} characters")
    return value


def _safe_filename_prefix(value: Any) -> str:
    prefix = _as_text("filename_prefix", value, 120).strip()
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("._")
    if not prefix:
        raise ApiError(400, "filename_prefix must contain a letter or number")
    return prefix


def _safe_input_image(value: Any) -> str:
    """Validate a ComfyUI input-directory image reference.

    The named API deliberately accepts a relative ComfyUI input filename, not
    an arbitrary host path or a job-output URL. Clients can use ComfyUI's
    native /upload/image endpoint to place a reference in that directory.
    """
    image_name = _as_text("input_image", value, MAX_INPUT_IMAGE_LENGTH).strip()
    if not image_name or "\x00" in image_name:
        raise ApiError(400, "input_image must be a non-empty ComfyUI input filename")
    normalised = image_name.replace("\\", "/")
    if (
        normalised.startswith("/")
        or ":" in normalised
        or re.search(r"\[(?:input|output|temp)\]\s*$", normalised)
        or any(part in {"", ".", ".."} for part in normalised.split("/"))
    ):
        raise ApiError(400, "input_image must be a relative filename inside ComfyUI's input directory")
    return normalised


def _safe_input_file(value: Any, parameter_name: str) -> str:
    """Validate a relative file name under ComfyUI's input directory."""
    file_name = _as_text(parameter_name, value, MAX_INPUT_IMAGE_LENGTH).strip()
    if not file_name or "\x00" in file_name:
        raise ApiError(400, f"{parameter_name} must be a non-empty ComfyUI input filename")
    normalised = file_name.replace("\\", "/")
    if (
        normalised.startswith("/")
        or ":" in normalised
        or re.search(r"\[(?:input|output|temp)\]\s*$", normalised)
        or any(part in {"", ".", ".."} for part in normalised.split("/"))
    ):
        raise ApiError(400, f"{parameter_name} must be a relative filename inside ComfyUI's input directory", code="input_invalid_path")
    return normalised


def _resolve_input_file(value: Any, parameter_name: str) -> tuple[str, Path]:
    file_name = _safe_input_file(value, parameter_name)
    input_root = Path(folder_paths.get_input_directory()).resolve()
    input_path = (input_root / file_name).resolve()
    try:
        input_path.relative_to(input_root)
    except ValueError as exc:
        raise ApiError(400, f"{parameter_name} must remain inside ComfyUI's input directory", code="input_invalid_path") from exc
    if not input_path.is_file():
        raise ApiError(400, f"{parameter_name} was not found in ComfyUI's input directory: {file_name}", code="input_not_found")
    return file_name, input_path


def _image_info(path: Path, parameter_name: str) -> dict[str, Any]:
    """Inspect headers before LoadImage can allocate a full float tensor."""
    try:
        with Image.open(path) as source:
            width, height = source.size
            if source.getexif().get(274) in {5, 6, 7, 8}:
                width, height = height, width
            return {"width": width, "height": height, "frames": getattr(source, "n_frames", 1),
                    "alpha": "A" in source.getbands() or "transparency" in source.info}
    except Image.DecompressionBombError as exc:
        raise ApiError(400, f"{parameter_name} is too large to decode safely", str(exc), code="input_too_large") from exc
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ApiError(400, f"{parameter_name} is not a readable image", str(exc), code="input_unreadable") from exc


def _fit_size(width: int, height: int, *, max_pixels: int, max_side: int | None = None, multiple: int = 1,
              min_side: int = 1) -> tuple[int, int]:
    """Largest size within the limits, aspect kept, each side a multiple of `multiple`."""
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    if max_side:
        scale = min(scale, max_side / max(width, height))
    scale = max(scale, min_side / min(width, height))
    return tuple(max(min_side, int(value * scale) // multiple * multiple) for value in (width, height))


def _write_fitted_input(path: Path, size: tuple[int, int], *, mask: bool) -> str:
    """Write a first-frame, orientation-applied, resized PNG copy; the original upload is never changed."""
    with Image.open(path) as source:
        source.seek(0)
        image = ImageOps.exif_transpose(source)
        if image.mode not in {"RGB", "RGBA", "L", "LA"}:
            image = image.convert("RGBA" if "transparency" in source.info or "A" in image.getbands() else "RGB")
        if image.size != size:
            image = image.resize(size, Image.Resampling.BILINEAR if mask else Image.Resampling.LANCZOS)
        name = f"{FITTED_INPUT_DIR}/{uuid.uuid4().hex}.png"
        target = Path(folder_paths.get_input_directory()) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "PNG")
    return name


def _delete_fitted_inputs(names: list[str]) -> None:
    for name in names:
        try:
            _, path = _resolve_input_file(name, "fitted input")
            path.unlink()
        except (ApiError, OSError):
            logging.warning("Could not delete fitted input %s", name)


def _fit_inputs(entry: dict[str, Any], parameters: dict[str, Any], image_info: dict[str, dict[str, Any]],
                fit_mode: str, report: list[dict[str, Any]]) -> None:
    """Bring image inputs within what the route can run, recording every change for the caller.

    downscale (default): shrink to the route's limits and take the first frame of animated input.
    none: keep the upload unchanged and refuse, with a stable code, only what the route cannot run.
    """
    budget = _resource_limits(entry)["max_total_pixels"]
    image_mode = entry.get("input_contract", {}).get("image", {}).get("mode")
    for name in ("input_image", "mask_image", "face_image"):
        if name not in image_info:
            continue
        info = image_info[name]
        size = (info["width"], info["height"])
        if image_mode == "inpaint-source" and name == "input_image":
            target = _fit_size(*size, max_pixels=budget, max_side=2048, multiple=8, min_side=64)
        elif image_mode == "inpaint-source" and name == "mask_image":
            source = image_info["input_image"]
            original = next((r["original_size"] for r in report if r["parameter"] == "input_image"), (source["width"], source["height"]))
            if abs(size[0] / size[1] - original[0] / original[1]) > 0.01:
                raise ApiError(400, "mask_image must have the same aspect ratio as input_image",
                               {"input_image": list(original), "mask_image": list(size)}, code="mask_size_mismatch")
            target = (source["width"], source["height"])
            if fit_mode == "none" and size != tuple(original):
                raise ApiError(400, "mask_image must have the same dimensions as input_image",
                               {"input_image": list(original), "mask_image": list(size)}, code="mask_size_mismatch")
        elif image_mode == "qwen-image-2.1-reference" and name == "input_image":
            target = _fit_size(*size, max_pixels=budget, multiple=32) if size[0] * size[1] > budget else size
        else:
            target = _fit_size(*size, max_pixels=MAX_INPUT_PIXELS)
        animated = info["frames"] > 1
        if target == size and not animated:
            continue
        if fit_mode == "none":
            if animated:
                raise ApiError(400, f"{name} is animated; send a still image or use input_fit=downscale to take the first frame",
                               {"frames": info["frames"]}, code="input_animated")
            if image_mode == "inpaint-source" and name == "input_image" and target != size:
                raise ApiError(400, "input_image must be 64-2048 px per side, a multiple of 8 and within the pixel budget for inpaint",
                               {"size": list(size), "fitted_size": list(target)}, code="input_size_unsupported")
            if size[0] * size[1] > MAX_INPUT_PIXELS:
                raise ApiError(400, f"{name} exceeds the input pixel limit", {"size": list(size), "max_pixels": MAX_INPUT_PIXELS},
                               code="input_too_large")
            continue
        _, path = _resolve_input_file(parameters[name], name)
        fitted = _write_fitted_input(path, target, mask=name == "mask_image")
        parameters[name] = fitted
        image_info[name] = {**info, "width": target[0], "height": target[1], "frames": 1}
        report.append({"parameter": name, "original_size": list(size), "used_size": list(target), "first_frame": animated,
                       "file": fitted})


def _resource_limits(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_total_pixels": 2_359_296,
        "max_input_pixels": MAX_INPUT_PIXELS,
        **entry.get("resource_limits", {}),
    }


def _binding_value(graph: dict[str, Any], entry: dict[str, Any], name: str) -> Any:
    inputs, input_name = _node_input(graph, entry["bindings"][name], name)
    return inputs[input_name]


def _resource_request(graph: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    if "width" in entry["bindings"]:
        width, height, batch = (_binding_value(graph, entry, name) for name in ("width", "height", "batch_size"))
    else:
        _, path = _resolve_input_file(_binding_value(graph, entry, "input_image"), "input_image")
        info = _image_info(path, "input_image")
        width, height, batch = info["width"], info["height"], 1
        if entry.get("input_contract", {}).get("image", {}).get("mode") == "qwen-image-2.1-reference":
            # Official resolution=0 rounds each reference dimension to 32.
            width, height = (max(32, round(value / 32) * 32) for value in (width, height))
    limits = _resource_limits(entry)
    total = width * height * batch
    inputs = {}
    for node in graph.values():
        if node.get("class_type") in {"LoadImage", "LoadImageMask"}:
            name, path = _resolve_input_file(node["inputs"]["image"], "image input")
            _image_info(path, "image input")
            stat = path.stat()
            inputs[name] = {"size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}
    return {"width": width, "height": height, "batch_size": batch, "total_pixels": total, "limits": limits, "input_files": inputs}


def _validate_queued_inputs(resources: dict[str, Any]) -> None:
    for name, expected in resources.get("input_files", {}).items():
        _, path = _resolve_input_file(name, "image input")
        stat = path.stat()
        if {"size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns} != expected:
            raise RuntimeError(f"Input file changed after submission: {name}; resubmit to validate the replacement")


def _attach_dynamic_image_input(graph: dict[str, Any], entry: dict[str, Any], image_name: str) -> None:
    """Attach latent img2img nodes to a graph that exposes an empty latent.

    Most catalog routes use the same graph for text-to-image and image-to-image
    requests.  Keeping the EmptyLatentImage node in place preserves the normal
    width/height bindings; the sampler is switched to a new LoadImage ->
    VAEEncode latent only when an input_image is actually supplied.
    """
    empty_nodes = [
        (node_id, node)
        for node_id, node in graph.items()
        if isinstance(node, dict)
        and node.get("class_type") in {"EmptyLatentImage", "EmptySD3LatentImage"}
    ]
    sampler_nodes = [
        node
        for node in graph.values()
        if isinstance(node, dict) and node.get("class_type") == "KSampler"
    ]
    decode_nodes = [
        node
        for node in graph.values()
        if isinstance(node, dict) and node.get("class_type") == "VAEDecode"
    ]
    if not empty_nodes or not sampler_nodes or not decode_nodes:
        raise ApiError(
            400,
            f"Workflow {entry.get('id', '<unknown>')} cannot accept input_image with its current graph",
            {"required_nodes": ["EmptyLatentImage or EmptySD3LatentImage", "KSampler", "VAEDecode"]},
        )
    vae_ref = decode_nodes[0].get("inputs", {}).get("vae")
    if not isinstance(vae_ref, list) or len(vae_ref) != 2:
        raise ApiError(500, f"Workflow {entry.get('id', '<unknown>')} has no usable VAE binding for input_image")

    numeric_ids = [int(node_id) for node_id in graph if str(node_id).isdigit()]
    next_id = max(numeric_ids, default=0) + 1
    load_id = str(next_id)
    resize_id, encode_id, repeat_id = (str(next_id + i) for i in (1, 2, 3))
    graph[load_id] = {
        "class_type": "LoadImage",
        "inputs": {"image": image_name},
    }
    graph[resize_id] = {
        "class_type": "ImageScale",
        "inputs": {"image": [load_id, 0], "upscale_method": "lanczos", "crop": "disabled",
                   "width": _binding_value(graph, entry, "width"), "height": _binding_value(graph, entry, "height")},
    }
    graph[encode_id] = {
        "class_type": "VAEEncode",
        "inputs": {"pixels": [resize_id, 0], "vae": copy.deepcopy(vae_ref)},
    }
    graph[repeat_id] = {
        "class_type": "RepeatLatentBatch",
        "inputs": {"samples": [encode_id, 0], "amount": _binding_value(graph, entry, "batch_size")},
    }
    sampler_nodes[0].setdefault("inputs", {})["latent_image"] = [repeat_id, 0]


def _request_parameters(body: dict[str, Any]) -> dict[str, Any]:
    parameters = body.get("parameters", {})
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        raise ApiError(400, "parameters must be an object")
    result: dict[str, Any] = {}
    aliases = {
        "subject": "prompt",
        "positive_prompt": "prompt",
        "negative": "negative_prompt",
        "negative_prompt_text": "negative_prompt",
        "image": "input_image",
        "reference_image": "input_image",
        "source_image": "input_image",
        "mask": "mask_image",
        "mask_file": "mask_image",
        "face_reference": "face_image",
        "reference_face": "face_image",
        "pose_image": "pose",
        "control_image": "pose",  # the retired OpenPose route's parameter
    }
    parameter_names = {
        "prompt",
        "negative_prompt",
        "seed",
        "width",
        "height",
        "batch_size",
        "steps",
        "cfg",
        "denoise",
        "sampler_name",
        "scheduler",
        "filename_prefix",
        "input_image",
        "mask_image",
        "mask_channel",
        "grow_mask_by",
        "face_image",
        "pose",
        "pose_strength",
        "ipadapter_weight",
        "input_fit",
        "image",
        "reference_image",
        "source_image",
        "mask",
        "mask_file",
        "face_reference",
        "reference_face",
        "pose_image",
        "control_image",
        "loras",
    }
    envelope = {"workflow", "workflow_id", "parameters", "prompt_granularity", "client_id", "dry_run"}
    unknown = set(body) - envelope - parameter_names - set(aliases)
    if unknown:
        raise ApiError(400, "Unknown request fields", sorted(unknown))
    for source in ({key: value for key, value in body.items() if key not in envelope}, parameters):
        for name, value in source.items():
            target = aliases.get(name, name)
            if target not in parameter_names:
                raise ApiError(400, f"Unknown parameter: {name}")
            if target in result and result[target] != value:
                raise ApiError(400, f"Conflicting values for parameter: {target}")
            result[target] = value
    return result


def _prepare_graph(entry: dict[str, Any], body: dict[str, Any], fit_report: list[dict[str, Any]] | None = None,
                   pose_request: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bind a request into the route's graph. Fitted input copies are appended to fit_report; the caller deletes them.

    pose_request comes from _resolve_pose; the skeleton is drawn for the final output size and attached last.
    """
    fit_report = [] if fit_report is None else fit_report
    graph = copy.deepcopy(_read_workflow(entry))
    parameters = _request_parameters(body)
    parameters.pop("pose", None)
    pose_strength = parameters.pop("pose_strength", None)
    allowed = set(entry.get("bindings", {}))
    # Strength bindings are internal; expose only the documented loras object.
    for spec in entry.get("lora_bindings", {}).values():
        allowed -= {spec.get("strength_model"), spec.get("strength_clip")}
    if entry.get("lora_bindings"):
        allowed.add("loras")
    image_contract = entry.get("input_contract", {}).get("image", {})
    if image_contract.get("supported"):
        allowed.add("input_image")
    if _image_inputs(entry):
        allowed.add("input_fit")
    unknown = set(parameters) - allowed
    if unknown:
        raise ApiError(400, "Workflow does not expose parameter: " + ", ".join(sorted(unknown)), sorted(unknown),
                       code="unsupported_parameter")
    fit_mode = parameters.pop("input_fit", "downscale")
    if fit_mode not in INPUT_FIT_MODES:
        raise ApiError(400, "input_fit must be downscale or none", expected=list(INPUT_FIT_MODES), code="invalid_parameter")
    if "prompt" not in parameters:
        raise ApiError(422, "parameters.prompt (or its subject alias) is required; an empty string is allowed", expected=["prompt", "subject"])
    if "seed" not in parameters:
        parameters["seed"] = secrets.randbelow(2**53)
    image_info: dict[str, dict[str, Any]] = {}
    for name in ("input_image", "mask_image", "face_image"):
        if name in parameters:
            normalized, path = _resolve_input_file(parameters[name], name)
            parameters[name] = normalized
            image_info[name] = _image_info(path, name)
    _fit_inputs(entry, parameters, image_info, fit_mode, fit_report)
    if "input_image" in image_info and image_contract.get("mode") == "latent-img2img":
        source = image_info["input_image"]
        for dimension in ("width", "height"):
            if dimension in parameters:
                parameters[dimension] = _as_int(dimension, parameters[dimension], 64, 2048, multiple=8)
        width, height = parameters.get("width"), parameters.get("height")
        if width is None and height is None:
            width, height = source["width"], source["height"]
            fitted = _fit_size(width, height, max_pixels=_resource_limits(entry)["max_total_pixels"], max_side=2048, multiple=8, min_side=64)
            if fit_mode == "downscale" and fitted != (width // 8 * 8, height // 8 * 8):
                fit_report.append({"parameter": "input_image", "original_size": [width, height], "used_size": list(fitted),
                                   "first_frame": False, "file": None, "scope": "output size"})
                width, height = fitted
            elif fit_mode == "none" and max(width, height) > 2048:
                raise ApiError(400, "input_image is larger than 2048 px per side; pass width/height or use input_fit=downscale",
                               {"size": [width, height]}, code="input_size_unsupported")
        elif width is None:
            height = _as_int("height", height, 64, 2048, multiple=8)
            width = round(source["width"] * height / source["height"])
        elif height is None:
            width = _as_int("width", width, 64, 2048, multiple=8)
            height = round(source["height"] * width / source["width"])
        parameters.setdefault("width", max(64, width // 8 * 8))
        parameters.setdefault("height", max(64, height // 8 * 8))
    elif pose_request and "width" in entry.get("bindings", {}) and "width" not in parameters and "height" not in parameters:
        # Without a requested size the output takes the pose image's aspect at the route's default pixel count.
        default_pixels = _binding_value(graph, entry, "width") * _binding_value(graph, entry, "height")
        parameters["width"], parameters["height"] = pose_addon.output_size(pose_request["skeleton"], default_pixels)

    text_limits = {"prompt": MAX_PROMPT_LENGTH, "negative_prompt": MAX_PROMPT_LENGTH}
    integer_limits = {
        "width": (64, 2048, 8),
        "height": (64, 2048, 8),
        "batch_size": (1, 4, None),
        "steps": (1, 100, None),
        "seed": (0, 2**63 - 1, None),
        "grow_mask_by": (0, 64, None),
    }
    float_limits = {
        "cfg": (0.0, 30.0),
        "denoise": (0.0, 1.0),
        "ipadapter_weight": (-1.0, 3.0),
    }
    for name, maximum in text_limits.items():
        if name in parameters:
            _set_binding(graph, entry, name, _as_text(name, parameters[name], maximum))
    if "input_image" in parameters:
        image_name, _ = _resolve_input_file(parameters["input_image"], "input_image")
        bindings = entry.get("bindings", {})
        image_binding = bindings.get("input_image") if isinstance(bindings, dict) else None
        if isinstance(image_binding, dict):
            _set_binding(graph, entry, "input_image", image_name)

    for parameter_name in ("mask_image", "face_image"):
        if parameter_name not in parameters:
            continue
        file_name, _ = _resolve_input_file(parameters[parameter_name], parameter_name)
        bindings = entry.get("bindings", {})
        binding = bindings.get(parameter_name) if isinstance(bindings, dict) else None
        if not isinstance(binding, dict):
            raise ApiError(400, f"Workflow does not expose parameter: {parameter_name}")
        _set_binding(graph, entry, parameter_name, file_name)

    if "mask_channel" in parameters:
        mask_channel = _as_text("mask_channel", parameters["mask_channel"], 16).strip().lower()
        if mask_channel not in {"alpha", "red", "green", "blue"}:
            raise ApiError(400, "mask_channel must be one of alpha, red, green or blue")
        _set_binding(graph, entry, "mask_channel", mask_channel)
    input_contract = entry.get("input_contract", {})
    if isinstance(input_contract, dict):
        required_inputs = (
            ("image", "input_image"),
            ("mask", "mask_image"),
            ("face", "face_image"),
        )
        for contract_name, parameter_name in required_inputs:
            contract = input_contract.get(contract_name)
            if isinstance(contract, dict) and contract.get("required") and parameter_name not in parameters:
                raise ApiError(
                    422,
                    f"Workflow {entry.get('id', '<unknown>')} requires parameters.{parameter_name}",
                    {"input_type": contract_name, "parameter": parameter_name},
                    expected=[parameter_name],
                )
    image_contract = input_contract.get("image") if isinstance(input_contract, dict) else None
    if "input_image" in parameters and "denoise" not in parameters and isinstance(image_contract, dict):
        default_denoise = image_contract.get("default_denoise")
        if default_denoise is not None:
            _set_binding(graph, entry, "denoise", _as_float("denoise", default_denoise, 0.0, 1.0))
    for name, (minimum, maximum, multiple) in integer_limits.items():
        if name in parameters:
            _set_binding(graph, entry, name, _as_int(name, parameters[name], minimum, maximum, multiple=multiple))
    for name, (minimum, maximum) in float_limits.items():
        if name in parameters:
            _set_binding(graph, entry, name, _as_float(name, parameters[name], minimum, maximum))
    for name in ("sampler_name", "scheduler"):
        if name in parameters:
            _set_binding(graph, entry, name, _as_text(name, parameters[name], 80))
    if "filename_prefix" in parameters:
        _set_binding(graph, entry, "filename_prefix", _safe_filename_prefix(parameters["filename_prefix"]))

    lora_values = parameters.get("loras", {})
    if lora_values is None:
        lora_values = {}
    if not isinstance(lora_values, dict):
        raise ApiError(400, "loras must be an object mapping LoRA ids to strengths")
    lora_bindings = entry.get("lora_bindings", {})
    if not isinstance(lora_bindings, dict):
        lora_bindings = {}
    for lora_id, value in lora_values.items():
        spec = lora_bindings.get(lora_id)
        if not isinstance(spec, dict):
            raise ApiError(400, f"Workflow does not expose LoRA: {lora_id}")
        strength = _as_float(f"loras.{lora_id}", value, -2.0, 2.0)
        for input_name in ("strength_model", "strength_clip"):
            binding_name = spec.get(input_name)
            if binding_name:
                binding = entry.get("bindings", {}).get(binding_name)
                if not isinstance(binding, dict):
                    raise ApiError(500, f"LoRA binding {lora_id} is invalid")
                inputs, actual_input = _node_input(graph, binding, binding_name)
                inputs[actual_input] = strength

    if "input_image" in parameters and "input_image" not in entry.get("bindings", {}):
        _attach_dynamic_image_input(graph, entry, parameters["input_image"])
    if "mask_image" in image_info:
        if _binding_value(graph, entry, "mask_channel") == "alpha" and not image_info["mask_image"]["alpha"]:
            raise ApiError(400, "mask_channel=alpha requires a mask with transparency; transparent pixels are redrawn",
                           code="mask_alpha_missing")
    if pose_request:
        if pose_strength is not None and entry["pose"]["default_strength"] is None:
            raise ApiError(400, "Workflow does not expose parameter: pose_strength; this route follows the pose without a strength "
                                "setting", ["pose_strength"], code="unsupported_parameter")
        strength = entry["pose"]["default_strength"] if pose_strength is None else _as_float("pose_strength", pose_strength, 0.0, 1.0)
        width, height = _binding_value(graph, entry, "width"), _binding_value(graph, entry, "height")
        multiple = pose_request["adapter"].get("size_multiple", 8)
        if width % multiple or height % multiple:  # Anima's control embedder patches the latent 2x2
            width, height = max(multiple, width // multiple * multiple), max(multiple, height // multiple * multiple)
            _set_binding(graph, entry, "width", width)
            _set_binding(graph, entry, "height", height)
        skeleton_image = pose_addon.render(pose_request["skeleton"], width, height, pose_request["adapter"].get("render", "openpose"))
        pose_addon.attach(graph, pose_request["adapter"], skeleton_image, strength)
        pose_request.update(strength=strength, skeleton_image=skeleton_image, size=[width, height])
    elif pose_strength is not None:
        raise ApiError(400, "pose_strength needs parameters.pose", code="invalid_parameter")
    _resource_request(graph, entry)
    return graph


def _resolve_pose(entry: dict[str, Any], body: dict[str, Any]) -> dict[str, Any] | None:
    """Turn parameters.pose (library view id or uploaded image) into the route's adapter plus a skeleton.

    Blocking (gallery fetch, DWPose on CPU): call it from a worker thread.
    """
    parameters = _request_parameters(body)
    if "pose" not in parameters:
        return None
    if "input_image" in parameters:
        raise ApiError(400, "pose works when generating from text (face_image is fine); it cannot re-pose input_image. "
                            "Leave input_image out, or leave pose out and describe the pose in the prompt",
                       code="pose_unsupported")
    if not entry.get("pose", {}).get("supported"):
        catalog = _read_catalog()
        routes = [item["id"] for item in catalog["workflows"] if isinstance(item, dict) and item.get("pose", {}).get("supported")
                  and "text-to-image" in item.get("capabilities", {}).get("input_modes", [])]
        raise ApiError(400, f"Workflow {entry['id']} does not take a pose yet; leave pose out and describe it in the prompt, "
                            "or use one of routes_with_pose if it suits the request", {"routes_with_pose": routes}, code="pose_unsupported")
    adapter = pose_addon.adapter_for(_read_catalog(), entry)
    public_pose = _public_entry(entry, _read_catalog().get("model_registry", []))["pose"]
    if not public_pose.get("ready"):
        profiles = " and ".join(f"-Profile {name}" for name in dict.fromkeys(("pose", adapter.get("download_profile", "pose"))))
        raise ApiError(409, f"Pose dependencies are not installed; run scripts/download_models.ps1 {profiles}", public_pose,
                       code="pose_unavailable")
    value = _as_text("pose", parameters["pose"], MAX_INPUT_IMAGE_LENGTH).strip()
    found = pose_addon.lookup(value)
    if found:
        library_pose, angle = found
        source_name = pose_addon.library_image(library_pose, angle)
        source = {"source": "library", "id": library_pose["views"][angle]["id"], "name": library_pose["name"],
                  "angle": pose_addon.ANGLES[angle], "people": library_pose["people"]}
    else:
        try:
            source_name, path = _resolve_input_file(value, "pose")
        except ApiError as exc:
            if exc.code == "input_not_found" and pose_addon.looks_like_library_id(value):
                raise ApiError(400, f"{value} is neither a pose library id nor an uploaded image; find library poses "
                                    "with GET /project/poses?q=", code="pose_unknown_id") from exc
            raise
        _image_info(path, "pose")
        source = {"source": "upload", "file": source_name}
    skeleton = pose_addon.skeleton(source_name)
    return {**source, "people_detected": len(skeleton["people"]), "found_by": skeleton["found_by"],
            "skeleton": skeleton, "adapter": adapter}


def _pose_summary(pose_request: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pose_request:
        return None
    folder, _, file_name = pose_request["skeleton_image"].rpartition("/")
    summary = {key: pose_request[key] for key in ("source", "id", "name", "angle", "file", "people_detected", "found_by",
                                                   "strength", "size") if key in pose_request}
    return {**summary, "adapter": pose_request["adapter"]["id"],
            "skeleton_url": "/view?" + urlencode({"filename": file_name, "subfolder": folder, "type": "input"})}


def _normalise_match_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.casefold().replace("_", " ")
    value = re.sub(r"[^\w\u3400-\u9fff]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _query_value(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return None


def _route_query(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Selectors resolve scores, and the deprecated ones the caller sent (ignored)."""
    query: dict[str, Any] = {}
    aliases = {
        "domain": ("domain", "subject_domain", "franchise", "series"),
        "style": ("style", "aesthetic"),
        "input_mode": ("input_mode", "generation_mode", "mode"),
        "family": ("family",),
    }
    for target, names in aliases.items():
        value = _query_value(payload, *names)
        if value is not None:
            if not isinstance(value, str) or len(value) > 256:
                raise ApiError(400, f"{target} must be a string of at most 256 characters")
            query[target] = value
    prompt_fields = {
        "character",
        "character_name",
        "subject",
        "outfit",
        "costume",
    }
    if prompt_fields.intersection(payload):
        raise ApiError(
            400,
            "character and outfit are prompt inputs, not route selectors",
            {
                "route_selectors": RESOLVE_SELECTORS,
                "next_step": "resolve the workflow first, then include character and outfit in parameters.prompt",
            },
        )
    allowed = {name for names in aliases.values() for name in names} | set(DEPRECATED_SELECTORS) | {
        "workflow", "workflow_id", "include_unready", "limit"
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ApiError(400, "Unknown route selectors", sorted(unknown))
    workflow = _query_value(payload, "workflow", "workflow_id")
    if workflow is not None:
        if not isinstance(workflow, str) or not workflow:
            raise ApiError(400, "workflow must be a non-empty workflow id")
        query["workflow"] = workflow
    return query, sorted(name for name in DEPRECATED_SELECTORS if name in payload)


def _deprecation_warnings(names: list[str]) -> list[dict[str, Any]]:
    return [{"code": "deprecated_selector", "parameter": name,
             "message": f"{name} is deprecated and ignored: every route makes SFW and explicit images (the prompt decides) and "
                        f"routes are chosen by {', '.join(RESOLVE_SELECTORS)}."} for name in names]


def _normalised_values(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return []
    return [normalised for item in values if (normalised := _normalise_match_text(item))]


def _route_values(entry: dict[str, Any], key: str) -> list[str]:
    routing = entry.get("routing", {})
    capabilities = entry.get("capabilities", {})
    selection = entry.get("selection", {})
    values: list[str] = []
    routing_values: list[str] = []
    if isinstance(routing, dict):
        routing_key = {
            "domain": "domains",
            "character": "characters",
            "outfit": "outfits",
            "style": "styles",
            "input_mode": "input_modes",
        }.get(key, key)
        routing_values.extend(_normalised_values(routing.get(routing_key)))
        routing_values.extend(_normalised_values(routing.get(key)))
        routing_values.extend(_normalised_values(routing.get("aliases")) if key in {"domain", "character"} else [])
    # `routing` is the explicit selector contract.  Capabilities describe what
    # a ready route can do, but must not silently broaden its match surface.
    # Use them only for legacy entries that have no corresponding routing
    # values at all.
    values.extend(routing_values)
    if not routing_values and isinstance(capabilities, dict):
        capability_key = {
            "domain": "domains",
            "character": "subjects",
            "style": "styles",
            "input_mode": "input_modes",
        }.get(key)
        if capability_key:
            values.extend(_normalised_values(capabilities.get(capability_key)))
    if key == "style" and not routing_values:
        values.extend(_normalised_values(entry.get("tags")))
        if isinstance(selection, dict):
            values.extend(_normalised_values(selection.get("priority_signals")))
    return list(dict.fromkeys(values))


def _route_value_matches(query: str, values: list[str]) -> bool:
    normalised_query = _normalise_match_text(query)
    return bool(normalised_query) and any(
        normalised_query == value or re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", normalised_query)
        for value in values
    )


def _route_score(entry: dict[str, Any], query: dict[str, Any]) -> tuple[int, list[str]] | None:
    routing = entry.get("routing", {})
    if not isinstance(routing, dict):
        routing = {}
    if query.get("workflow") and not _is_workflow(entry, query["workflow"]):
        return None
    explicit_workflow = bool(query.get("workflow")) and _is_workflow(entry, query["workflow"])
    required_selectors = routing.get("required_selectors", [])
    if isinstance(required_selectors, list) and not explicit_workflow:
        for selector in required_selectors:
            if isinstance(selector, str) and not query.get(selector):
                return None
    score = int(routing.get("priority", 0)) if isinstance(routing.get("priority", 0), int) else 0
    if routing.get("explicit_only") and not explicit_workflow:
        return None
    if query.get("family") and _normalise_match_text(query["family"]) != _normalise_match_text(entry.get("family")):
        return None
    if routing.get("exclusive") and not explicit_workflow and not any(
        query.get(selector) and _route_value_matches(query[selector], _route_values(entry, selector))
        for selector in ("domain", "style")
    ):
        return None
    reasons: list[str] = []

    requested_input = query.get("input_mode")
    if isinstance(requested_input, str) and requested_input:
        input_values = _route_values(entry, "input_mode")
        if _normalise_match_text(requested_input) in input_values:
            score += 40
            reasons.append("input mode matched")
        elif input_values:
            return None

    requested_domain = query.get("domain")
    if isinstance(requested_domain, str) and requested_domain:
        domain_values = _route_values(entry, "domain")
        if _route_value_matches(requested_domain, domain_values):
            score += 50
            reasons.append("domain matched")
        elif routing.get("exclusive", False) and not explicit_workflow:
            return None
        else:
            # A domain-specific route must outrank a generic fallback.
            score -= 100
            reasons.append("generic fallback route")

    requested_style = query.get("style")
    if isinstance(requested_style, str) and requested_style:
        style_values = _route_values(entry, "style")
        if _normalise_match_text(requested_style) in style_values:
            score += 30
            reasons.append("style matched")
        elif _route_value_matches(requested_style, style_values):
            # A generic word inside the request ("anime" in "mainstream anime") must not tie an exact match.
            score += 15
            reasons.append("style partially matched")
        elif routing.get("strict_style", False) and not explicit_workflow:
            return None
        elif routing.get("exclusive", False) and requested_domain:
            # A domain-specific route may still be useful for an unknown style,
            # but it should not outrank an explicit style match elsewhere.
            score -= 5

    if not reasons:
        reasons.append("catalog priority fallback")
    return score, reasons


def _resolve_routes(
    catalog: dict[str, Any],
    query: dict[str, Any],
    *,
    include_unready: bool = False,
    limit: int = 1,
    deprecated: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    model_registry = catalog.get("model_registry", [])
    matches: list[dict[str, Any]] = []
    for entry in catalog["workflows"]:
        if not isinstance(entry, dict):
            continue
        public = _public_entry(entry, model_registry)
        if not include_unready and not public["ready"]:
            continue
        scored = _route_score(entry, query)
        if scored is None:
            continue
        score, reasons = scored
        matches.append(
            {
                "workflow_id": entry.get("id"),
                "score": score,
                "ready": public["ready"],
                "reasons": reasons,
                "workflow": public,
            }
        )
    matches.sort(key=lambda item: (-item["score"], str(item["workflow_id"])))
    total_matches = len(matches)
    ready_matches = [match for match in matches if match["ready"]]
    top_match = ready_matches[0] if ready_matches else None
    selected = top_match
    has_specific_selector = any(
        query.get(key)
        for key in RESOLVE_SELECTORS
    )
    selection_status = "selected" if selected else "no_match"
    guidance = "Use the selected workflow's prompting and model hints."
    if not has_specific_selector:
        selected = None
        selection_status = "needs_route_selector"
        guidance = "Provide domain, style, input_mode, family or workflow before generating."
    elif (
        selected
        and not query.get("workflow")
        and not query.get("domain")
        and not query.get("family")
        and not query.get("style")
        and len(ready_matches) > 1
        and selected["score"] - ready_matches[1]["score"] < 40
    ):
        selected = None
        selection_status = "ambiguous"
        guidance = "Provide domain or an explicit workflow because multiple model/workflow routes are plausible."
    matches = matches[:limit]
    # Sibling routes of one family (e.g. two JANKU versions) differ in house style, not score; surface the hand-off.
    prefer_instead = selected["workflow"].get("selection", {}).get("prefer_instead", []) if selected else []
    return {
        "schema_version": 1,
        "request": query,
        "selected_workflow": selected["workflow_id"] if selected else None,
        "matches": matches,
        "selection": {
            "status": selection_status,
            "strategy": "highest-scoring ready workflow/model route; character and outfit remain prompt inputs",
            "fallback": bool(selected and "generic fallback route" in selected["reasons"]),
            "total_matches": total_matches,
            "guidance": guidance,
            "prefer_instead": prefer_instead,
        },
        "warnings": _deprecation_warnings(list(deprecated)),
    }


def _image_inputs(entry: dict[str, Any]) -> list[str]:
    """Image parameters _prepare_graph accepts for this route."""
    bindings = entry.get("bindings", {})
    accepted = ["input_image"] if entry.get("input_contract", {}).get("image", {}).get("supported") or "input_image" in bindings else []
    accepted += [name for name in ("mask_image", "face_image") if name in bindings]
    return accepted + (["pose"] if entry.get("pose", {}).get("supported") else [])


# Reference-image intents, keyed by the contract mode that implements them.
_REFERENCE_USES = (
    ("qwen-image-2.1-reference", "image-edit", "input_image",
     "Edit an uploaded image by instruction: change a colour, object, material, background or text and keep the rest.",
     "Output keeps the source size (rounded to multiples of 32). The prompt is the edit instruction."),
    ("latent-img2img", "image-to-image", "input_image",
     "Redraw or vary an uploaded image in the route's model style.",
     "Composition and colours carry over; character identity is not guaranteed. denoise defaults to 0.65; lower keeps more of the source."),
    ("inpaint-source", "inpaint", "input_image + mask_image",
     "Repaint only a masked region of an uploaded image.",
     "Mask must have the source's aspect ratio (it is resized with the source when fitted). denoise 1.0 replaces the region; lower keeps more of the original. Unmasked pixels are composited back unchanged."),
    ("face-reference", "face-reference", "face_image",
     "Put the same face/character into a new scene described by the prompt.",
     "IPAdapter conditioning, not FaceID. Outfit, pose and background come from the prompt at the default weight; on the Anima route weights of 0.8-1.0 copy the reference's outfit and background."),
)


def _reference_uses(workflows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uses = []
    for contract_mode, input_mode, parameter, intent, behaviour in _REFERENCE_USES:
        routes = [entry["id"] for entry in workflows
                  if any(isinstance(contract, dict) and contract.get("mode") == contract_mode
                         for contract in entry.get("input_contract", {}).values())]
        if routes:
            uses.append({"intent": intent, "input_mode": input_mode, "parameter": parameter, "routes": routes, "behaviour": behaviour})
    pose_routes = [entry["id"] for entry in workflows if entry.get("pose", {}).get("supported")]
    if pose_routes:
        uses.append({"intent": "Copy the body pose of a person in an uploaded photo, illustration or sketch.",
                     "input_mode": "openpose-control", "parameter": "pose", "routes": pose_routes,
                     "behaviour": "The service extracts the skeleton; clothing, face and scene still come from the prompt. "
                                  "It can be combined with the route's other inputs. Library poses need no upload (see poses)."})
    return uses


def _agent_manifest(catalog: dict[str, Any]) -> dict[str, Any]:
    model_registry = catalog.get("model_registry", [])
    ready_workflows = [
        entry
        for entry in catalog["workflows"]
        if isinstance(entry, dict) and _public_entry(entry, model_registry)["ready"]
    ]
    return {
        "schema_version": 1,
        "protocol": "project-comfyui-agent/v1",
        "api_version": "2.0.0",
        "service": {
            "description": "Image generation for this ComfyUI server: pick a named route, then generate from text, edit an image "
                           "by instruction, restyle, inpaint, reuse a face or copy a body pose.",
            "api_base": "/project",
            "authentication": "none",
            "openapi": "/project/openapi.json",
            "root_path": "/ is the native ComfyUI UI, not this API.",
        },
        "quick_start": [
            "1. Pick a route. Match the request to route_table[].best_for and take domain and style words from "
            "route_table[].selectors. If the requester supplies an image, choose its use in input_files.reference_uses. "
            "Then POST /project/resolve with what you know, e.g. {\"domain\": \"anime\", \"style\": \"visual novel\"} or "
            "{\"input_mode\": \"image-edit\"}; every field is optional. Character, outfit, pose and scene are prompt content, "
            "not resolve fields.",
            "2. Use selected_workflow. If selection.prefer_instead has an entry whose 'when' fits the request, use that "
            "workflow instead. selection.fallback true means no route matched your domain; the closest route by the other "
            "fields was chosen.",
            "3. Write the prompt from the selected route's agent_hints and prompting (matches[0].workflow in the resolve "
            "response, or GET /project/workflows/{id}). Use that route's dialect and trigger words, not another family's.",
            "4. POST /project/generate {\"workflow\": id, \"parameters\": {\"prompt\": ...}}, optionally with a "
            "\"client_id\" label. Send only fields in the route's parameter_schema; omitted fields take the route's defaults. "
            "Add \"dry_run\": true to validate without running.",
            "5. Poll GET /project/jobs/{prompt_id} until state is succeeded or failed, then download outputs[].url.",
        ],
        "resolve": {
            "path": "POST /project/resolve",
            "fields": {
                "domain": "subject or genre word from route_table[].selectors.domains or .aliases",
                "style": "look word from route_table[].selectors.styles (e.g. 'visual novel', 'mainstream anime', 'photo')",
                "input_mode": INPUT_MODES,
                "family": "exact route family, to stay within one model family",
                "workflow": "a route id, to inspect that one route",
            },
            "deprecated": "content_mode, input_type and prompt_granularity are accepted but ignored, with a "
                          "deprecated_selector warning.",
        },
        "content": "Resolve never routes by content: the prompt decides what an image shows, and every model family "
                   "here makes explicit images (MOSS-Image untested). Choose routes by subject and look. A route's best_for may "
                   "name a specialty (Moody Desire: explicit realistic people); picking it by workflow id or style is your "
                   "call. When the selected route has prompting.content_lines, add its 'explicit' or 'safe' line to the "
                   "prompt. Explicit images must show adults.",
        "characters": "The AI-mascot character routes (domains 'deepseek', 'claude girl', 'chatgpt', 'qwen girl', "
                      "'gemini girl', 'glm girl', 'grok girl', 'kimi girl' and aliases such as 大肥鱼, Claude娘, 千问娘, "
                      "哈基米, 智谱娘, Grok娘, Kimi娘) carry a character LoRA whose trigger draws her canonical design. Use them "
                      "only for that character, write her as an adult, and adjust loras.character to weaken the LoRA.",
        "route_table": [{"id": entry["id"], "family": entry["family"], "ready": True,
                         "required_fields": entry.get("input_contract", {}).get("required_fields", []),
                         "image_inputs": _image_inputs(entry),
                         "pose": entry.get("pose", {}).get("supported", False),
                         "recommended": entry.get("agent_hints", {}).get("recommended", {}),
                         "explicit_only": entry.get("routing", {}).get("explicit_only", False),
                         # The words /project/resolve matches, so agents choose from a menu instead of guessing.
                         "selectors": {"domains": entry.get("routing", {}).get("domains", []),
                                       "aliases": entry.get("routing", {}).get("aliases", []),
                                       "styles": entry.get("routing", {}).get("styles", [])},
                         "best_for": entry.get("selection", {}).get("best_for", [])} for entry in ready_workflows],
        "input_files": {
            "reference_uses": _reference_uses(ready_workflows),
            "text_only_routes": [entry["id"] for entry in ready_workflows if not _image_inputs(entry)],
            "find_or_upload": "GET /project/inputs lists existing files. To add one, POST multipart/form-data to /upload/image "
                              "with fields image (the file) and type=input; join the response's subfolder and name with '/' "
                              "and pass that as the image parameter. Only names inside the input directory work, never a "
                              "host path or URL. Use a unique basename per upload.",
            "input_fit": "Inputs are fitted automatically (parameters.input_fit=downscale): oversized images shrink and "
                         "animated ones use their first frame, reported as input_downscaled, input_resized or "
                         "input_first_frame warnings with original_size and used_size; tell the requester. input_fit=none "
                         "keeps the upload unchanged and refuses what the route cannot run.",
            "lifecycle": "Inputs persist until DELETE /project/inputs/{name}; delete only files you uploaded and no job "
                         "still uses.",
        },
        "poses": {
            "parameter": "parameters.pose",
            "when": "Send pose when the requester asks for a specific body pose or hand gesture, or gives a pose image. "
                    "Otherwise leave it out.",
            "value": "A pose id from the pose library, or the name of an uploaded image whose pose should be copied.",
            "search": {
                "method": "GET", "path": "/project/poses",
                "query": {"q": "words; matches English or Chinese pose names", "people": "1 or 2",
                          "angle": "front, diag, side, high or low", "limit": "1-20, default 8"},
                "returns": "Each result has views: one view id per camera angle: front, diag (three-quarter), side, "
                           "high (camera above, looking down), low (camera below, looking up). Pass the view id for the angle "
                           "you want; a bare pose id means its front view, or its first view when it has no front.",
            },
            "no_match": "If no library pose fits the request, leave pose out and describe the pose in the prompt.",
            "applies_to": "Generation from text, with or without face_image. A request that also sends input_image (edit, "
                          "image-to-image, inpaint) is refused with pose_unsupported: a pose cannot re-pose an existing image.",
            "prompt": "Also name the pose in the prompt (e.g. sitting, knee up, hand on chin): on Illustrious and character-LoRA "
                      "routes seated and kneeling poses often stay standing when only the skeleton asks for them.",
            "pose_strength": "Optional parameters.pose_strength 0-1 where the route's parameter_schema lists it: how strictly "
                             "the pose is followed. Routes without it follow the pose without a strength setting.",
            "routing": "Not every route takes a pose (route_table[].pose). When sending pose, add input_mode "
                       "'openpose-control' to /project/resolve so only routes that take one are chosen. If that drops the "
                       "route the requester asked for (selection.fallback is true), resolve without input_mode, keep that "
                       "route and describe the pose in the prompt.",
            "output_size": "Without width and height the output follows the pose image's aspect ratio.",
            "response": "The queued or dry-run response has a pose object: the pose used, people_detected and skeleton_url, "
                        "the skeleton the model follows.",
            "library_size": sum(1 for item in pose_addon.library().get("poses", {}).values() if item["views"]),
        },
        "instructions": [
            "parameters.prompt is required (an empty string is allowed). Seed is random when omitted.",
            "Width and height are multiples of 8, 64-2048. MOSS-Image floors them to multiples of 16, and Anima routes do "
            "the same when a pose is sent.",
            "For inpaint, send a source image and a mask with the same aspect ratio (normally the same size). White or red "
            "mask pixels are redrawn, or transparent ones with mask_channel=alpha. denoise 1.0 (default) replaces the "
            "masked region; lower keeps more of the original. Unmasked pixels are composited back unchanged.",
            "For face-reference, send a head-and-shoulders crop as face_image; it is IPAdapter conditioning, not FaceID. "
            "The Anima route encodes the whole reference: keep ipadapter_weight at its 0.6 default unless likeness matters "
            "more than the prompted outfit and scene, since 0.8-1.0 copies the reference's outfit and background.",
            "Warnings never block a job; see warning_codes. If a job fails with error.code out_of_memory, report it with "
            "the GET /project/resources snapshot before resubmitting.",
            "Saved PNGs keep ComfyUI's prompt metadata (input file names are replaced with [input image]); strip metadata "
            "before sending an image somewhere that should not see the prompt.",
        ],
        "warning_codes": {
            "input_downscaled": "An input was shrunk to fit the route; original_size and used_size say how.",
            "input_resized": "An input or the output size was adjusted to the route's size rules.",
            "input_first_frame": "An animated input was reduced to its first frame.",
            "over_pixel_budget": "The request exceeds the route's tested pixel budget; it still runs but may be slow or "
                                 "fail with out_of_memory.",
            "outside_recommended_range": "A sampling value is outside the route's recommended range; it was kept.",
            "negative_prompt_inactive": "You sent negative_prompt to a route running at CFG 1, where it has no effect.",
            "deprecated_selector": "An API 1.x field was sent; it is ignored.",
        },
        "error_codes": {
            "input_not_found": "The named input file is not in ComfyUI's input directory; upload it or check GET /project/inputs.",
            "input_invalid_path": "The input name is not a relative path inside the input directory.",
            "input_unreadable": "The file is not a readable image.",
            "input_too_large": "The image cannot be decoded safely, or exceeds max_input_pixels with input_fit=none.",
            "input_animated": "Animated input with input_fit=none; send a still image or use input_fit=downscale.",
            "input_size_unsupported": "With input_fit=none the image size cannot run on this route (e.g. inpaint needs 64-2048 px, multiples of 8).",
            "mask_size_mismatch": "mask_image has a different aspect ratio from input_image (or a different size with input_fit=none).",
            "mask_alpha_missing": "mask_channel=alpha was requested but the mask has no transparency.",
            "unsupported_parameter": "The route does not expose this parameter (for example an image sent to a text-only route).",
            "invalid_parameter": "A parameter value is outside its allowed set.",
            "pose_unsupported": "The route takes no pose, or the request also sends input_image; details.routes_with_pose "
                                "lists routes that do take one.",
            "pose_unknown_id": "The pose value is neither a library pose id nor an uploaded image; search GET /project/poses.",
            "pose_not_detected": "No human body pose was found in the pose image; use a clearer image of the whole body.",
            "pose_fetch_failed": "The library sketch could not be fetched from the pose gallery; retry later or pick another pose.",
            "pose_unavailable": "Pose dependencies are not installed on this server.",
            "out_of_memory": "Job error.code: the job ran out of RAM/VRAM while running.",
        },
        "result_contract": {
            "queue_response": "prompt_id (also job_id), job_url, applied_parameters, effective_parameters, warnings and, "
                              "with a pose, a pose object. A dry run returns the same fields plus validated and the "
                              "API-format graph (prompt), without prompt_id.",
            "job_states": ["queued", "running", "succeeded", "failed"],
            "job_phases": ["waiting_for_turn", "model_switch", "generation", "complete"],
            "queue": f"Named jobs run one at a time; at most {MAX_PROJECT_QUEUE} wait. queue_position is zero-based. "
                     "queued_ms is the wait before start; elapsed_ms covers model switching and generation.",
            "download_urls": "relative /view URLs in outputs[] of GET /project/jobs/{prompt_id}",
        },
        "operations": [
            {"method": "GET", "path": "/.well-known/agent-manifest.json", "description": "This guide (alias: /project/agent-manifest)."},
            {"method": "GET", "path": "/project/openapi.json", "description": "OpenAPI 3.1 contract."},
            {"method": "POST", "path": "/project/resolve", "description": "Rank ready routes for a request; the selected route's full entry is in matches[0].workflow."},
            {"method": "GET", "path": "/project/routes", "description": "Same ranking as resolve, as GET query parameters."},
            {"method": "GET", "path": "/project/workflows", "description": "All routes with capabilities, prompt guidance and readiness."},
            {"method": "GET", "path": "/project/workflows/{workflow_id}", "description": "One route: {\"workflow\": its entry with parameter_schema, prompting and agent_hints, \"prompt\": its API-format graph (not parameters.prompt)}."},
            {"method": "GET", "path": "/project/models", "description": "Model and adapter profiles (include_candidates=true adds unused ones)."},
            {"method": "GET", "path": "/project/models/{model_id}", "description": "One model or adapter profile."},
            {"method": "GET", "path": "/project/poses", "description": "Search the pose library; pass a view id as parameters.pose."},
            {"method": "GET", "path": "/project/inputs", "description": "List files in ComfyUI's input directory."},
            {"method": "POST", "path": "/upload/image", "description": "Upload a reference image (multipart: image, type=input, optional overwrite, subfolder)."},
            {"method": "DELETE", "path": "/project/inputs/{name}", "description": "Delete one input file."},
            {"method": "POST", "path": "/project/generate", "description": "Queue a named route (alias: /project/prompt), or validate it with dry_run."},
            {"method": "GET", "path": "/project/jobs/{prompt_id}", "description": "Job state and output download URLs."},
            {"method": "GET", "path": "/project/resources", "description": "Advisory device memory snapshot and queue/input limits."},
        ],
    }


def _openapi_document() -> dict[str, Any]:
    document = {
        "openapi": "3.1.0",
        "info": {
            "title": "Project ComfyUI Agent API",
            "version": "2.0.0",
            "description": "Self-describing named-workflow API layered on ComfyUI.",
        },
        "paths": {
            "/.well-known/agent-manifest.json": {
                "get": {
                    "operationId": "getPreferredAgentManifest",
                    "summary": "Get the preferred first-read agent manifest",
                    "responses": {"200": {"description": "Agent manifest"}},
                }
            },
            "/project/openapi.json": {
                "get": {
                    "operationId": "getOpenApiDocument",
                    "summary": "Get the OpenAPI contract",
                    "responses": {"200": {"description": "OpenAPI 3.1 document"}},
                }
            },
            "/project/resources": {
                "get": {
                    "operationId": "getProjectResources",
                    "summary": "Get advisory runtime resource information",
                    "description": "Return selected-device memory and configured queue/input limits. This is diagnostic information, not a reservation.",
                    "responses": {"200": {"description": "Runtime resource snapshot"}},
                }
            },
            "/project/agent-manifest": {
                "get": {
                    "operationId": "getAgentManifest",
                    "summary": "Get the agent operating guide",
                    "responses": {"200": {"description": "Agent manifest"}},
                }
            },
            "/project/poses": {
                "get": {
                    "operationId": "searchPoses",
                    "summary": "Search the pose library",
                    "description": "Rank library poses by words in their English or Chinese names. Pass a returned view id (or pose id for the front view) as parameters.pose.",
                    "parameters": [
                        {"name": "q", "in": "query", "schema": {"type": "string", "maxLength": 256}},
                        {"name": "people", "in": "query", "schema": {"type": "integer", "enum": [1, 2]}},
                        {"name": "angle", "in": "query", "schema": {"type": "string", "enum": list(pose_addon.ANGLES)}},
                        {"name": "limit", "in": "query", "schema": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8}},
                    ],
                    "responses": {"200": {"description": "Ranked poses; each has one view id per camera angle",
                                          "content": {"application/json": {"schema": {"type": "object", "properties": {
                                              "results": {"type": "array", "items": {"type": "object", "properties": {
                                                  "id": {"type": "string"}, "name": {"type": "string"}, "name_zh": {"type": "string"},
                                                  "category": {"type": "string"}, "framing": {"type": "string"},
                                                  "people": {"type": "integer"}, "match_score": {"type": "number"},
                                                  "views": {"type": "object", "additionalProperties": {"type": "string"}},
                                              }}},
                                              "angle_hint": {"type": ["string", "null"]},
                                          }}}}}},
                },
            },
            "/project/inputs": {
                "get": {
                    "operationId": "listProjectInputs",
                    "summary": "List available ComfyUI input files",
                    "description": "Return relative filenames and metadata for files currently available as image references.",
                    "responses": {
                        "200": {
                            "description": "Input directory listing",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["schema_version", "files"],
                                        "properties": {
                                            "schema_version": {"type": "integer"},
                                            "directory": {"type": "string"},
                                            "directory_exists": {"type": "boolean"},
                                            "count": {"type": "integer", "minimum": 0},
                                            "files": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "required": ["name", "size_bytes"],
                                                    "properties": {
                                                        "name": {"type": "string"},
                                                        "size_bytes": {"type": "integer", "minimum": 0},
                                                        "modified_time": {"type": "number", "description": "Unix timestamp in seconds (UTC)."},
                                                        "extension": {"type": "string"},
                                                    },
                                                },
                                            },
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/project/inputs/{name}": {
                "delete": {
                    "operationId": "deleteProjectInput",
                    "summary": "Delete one ComfyUI input file",
                    "description": "Delete exactly one regular file below ComfyUI/input. The name is relative and may contain URL-encoded subfolders.",
                    "parameters": [
                        {
                            "name": "name",
                            "in": "path",
                            "required": True,
                            "description": "Relative input filename, for example probe_frontdoor.jpg.",
                            "schema": {"type": "string", "minLength": 1, "maxLength": MAX_INPUT_IMAGE_LENGTH},
                        }
                    ],
                    "responses": {
                        "200": {"description": "Deleted input file"},
                        "400": {"description": "Unsafe or invalid relative filename"},
                        "404": {"description": "Input file not found"},
                    },
                }
            },
            "/upload/image": {
                "post": {
                    "operationId": "uploadInputImage",
                    "summary": "Upload an image reference",
                    "description": "Native ComfyUI multipart upload endpoint. Use the returned relative name as parameters.input_image.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["image", "type"],
                                    "properties": {
                                        "image": {"type": "string", "format": "binary"},
                                        "type": {"type": "string", "enum": ["input"], "default": "input"},
                                        "overwrite": {"type": "string", "enum": ["true", "false", "1", "0"]},
                                        "subfolder": {"type": "string", "description": "Optional relative input subfolder."},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Uploaded input reference",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["name", "subfolder", "type"],
                                        "properties": {
                                            "name": {"type": "string"},
                                            "subfolder": {"type": "string"},
                                            "type": {"type": "string", "enum": ["input"]},
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Invalid upload or unsafe path"},
                    },
                }
            },
            "/project/workflows": {
                "get": {
                    "operationId": "listWorkflows",
                    "summary": "List selectable workflows and capabilities",
                    "responses": {"200": {"description": "Workflow catalog"}},
                }
            },
            "/project/routes": {
                "get": {
                    "operationId": "listRoutes",
                    "summary": "List ranked workflow routes",
                    "parameters": [
                        {"$ref": "#/components/parameters/DomainQuery"},
                        {"$ref": "#/components/parameters/StyleQuery"},
                        {"$ref": "#/components/parameters/InputModeQuery"},
                        {"$ref": "#/components/parameters/FamilyQuery"},
                        {"$ref": "#/components/parameters/IncludeUnreadyQuery"},
                    ],
                    "responses": {"200": {"description": "Ranked workflow route table"}},
                }
            },
            "/project/resolve": {
                "post": {
                    "operationId": "resolveWorkflowRoute",
                    "summary": "Resolve an agent request to the best workflow",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/RouteResolutionRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Ranked workflow routes"},
                        "400": {"description": "Invalid route query"},
                    },
                }
            },
            "/project/models": {
                "get": {
                    "operationId": "listModels",
                    "summary": "List model and adapter profiles",
                    "parameters": [
                        {"$ref": "#/components/parameters/FamilyQuery"},
                        {"$ref": "#/components/parameters/StatusQuery"},
                        {"$ref": "#/components/parameters/ReadyQuery"},
                        {"$ref": "#/components/parameters/IncludeCandidatesQuery"},
                        {"$ref": "#/components/parameters/IncludeNonRouteableQuery"},
                    ],
                    "responses": {"200": {"description": "Model capability registry"}},
                }
            },
            "/project/models/{model_id}": {
                "get": {
                    "operationId": "getModel",
                    "summary": "Inspect a model or adapter profile",
                    "parameters": [{"$ref": "#/components/parameters/ModelId"}],
                    "responses": {
                        "200": {"description": "Model capability profile"},
                        "404": {"description": "Unknown model"},
                    },
                }
            },
            "/project/workflows/{workflow_id}": {
                "get": {
                    "operationId": "getWorkflow",
                    "summary": "Inspect one workflow",
                    "parameters": [{"$ref": "#/components/parameters/WorkflowId"}],
                    "responses": {
                        "200": {"description": "Workflow metadata and API graph"},
                        "404": {"description": "Unknown workflow"},
                    },
                }
            },
            "/project/prompt": {
                "post": {
                    "operationId": "queueWorkflow",
                    "summary": "Queue a named workflow",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/NamedPromptRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Queued prompt or dry-run graph"},
                        "400": {"description": "Invalid request or workflow binding"},
                        "409": {"description": "Required model files are missing"},
                    },
                }
            },
            "/project/generate": {
                "post": {
                    "operationId": "generateImage",
                    "summary": "Queue a named workflow through the agent API",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/NamedPromptRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Queued prompt or dry-run graph"},
                        "400": {"description": "Invalid request or workflow binding"},
                        "409": {"description": "Required model files are missing"},
                    },
                }
            },
            "/project/jobs/{prompt_id}": {
                "get": {
                    "operationId": "getProjectJob",
                    "summary": "Get job state and output references",
                    "parameters": [{"$ref": "#/components/parameters/PromptId"}],
                    "responses": {
                        "200": {"description": "Job state"},
                        "404": {"description": "Unknown job"},
                    },
                }
            },
            "/history/{prompt_id}": {
                "get": {
                    "operationId": "getNativeHistory",
                    "summary": "Native ComfyUI history response",
                    "parameters": [{"$ref": "#/components/parameters/PromptId"}],
                    "responses": {"200": {"description": "ComfyUI history"}},
                }
            },
            "/view": {
                "get": {
                    "operationId": "viewOutput",
                    "summary": "Native ComfyUI output download",
                    "parameters": [
                        {"name": "filename", "in": "query", "required": True, "schema": {"type": "string"}},
                        {"name": "subfolder", "in": "query", "schema": {"type": "string", "default": ""}},
                        {"name": "type", "in": "query", "schema": {"type": "string", "default": "output"}},
                    ],
                    "responses": {"200": {"description": "Image or other generated output"}},
                }
            },
        },
        "components": {
            "parameters": {
                "WorkflowId": {
                    "name": "workflow_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                "PromptId": {
                    "name": "prompt_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string", "format": "uuid"},
                },
                "ModelId": {
                    "name": "model_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                "DomainQuery": {
                    "name": "domain",
                    "in": "query",
                    "schema": {"type": "string", "maxLength": 256},
                },
                "StyleQuery": {
                    "name": "style",
                    "in": "query",
                    "schema": {"type": "string", "maxLength": 256},
                },
                "InputModeQuery": {
                    "name": "input_mode",
                    "in": "query",
                    "schema": {"type": "string", "enum": INPUT_MODES},
                },
                "IncludeUnreadyQuery": {
                    "name": "include_unready",
                    "in": "query",
                    "schema": {"type": "boolean", "default": False},
                },
                "IncludeCandidatesQuery": {
                    "name": "include_candidates",
                    "in": "query",
                    "schema": {"type": "boolean", "default": False},
                },
                "IncludeNonRouteableQuery": {
                    "name": "include_nonrouteable",
                    "in": "query",
                    "schema": {"type": "boolean", "default": False},
                },
                "FamilyQuery": {
                    "name": "family",
                    "in": "query",
                    "schema": {"type": "string"},
                },
                "StatusQuery": {
                    "name": "status",
                    "in": "query",
                    "schema": {"type": "string"},
                },
                "ReadyQuery": {
                    "name": "ready",
                    "in": "query",
                    "schema": {"type": "boolean"},
                },
            },
            "schemas": {
                "Deprecated": {
                    "description": "API 1.x field; accepted, ignored, and reported as a deprecated_selector warning.",
                    "deprecated": True,
                },
                "RouteResolutionRequest": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string", "maxLength": 256},
                        "family": {"type": "string", "maxLength": 256, "description": "Exact workflow family from the catalog, e.g. sdxl-ipadapter or anima-ipadapter."},
                        "style": {"type": "string", "maxLength": 256},
                        "input_mode": {"type": "string", "enum": INPUT_MODES},
                        "content_mode": {"$ref": "#/components/schemas/Deprecated"},
                        "input_type": {"$ref": "#/components/schemas/Deprecated"},
                        "prompt_granularity": {"$ref": "#/components/schemas/Deprecated"},
                        "workflow": {"type": "string"},
                        "include_unready": {"type": "boolean", "default": False},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 1},
                    },
                    "additionalProperties": False,
                },
                "NamedPromptRequest": {
                    "type": "object",
                    "anyOf": [{"required": ["workflow"]}, {"required": ["workflow_id"]}],
                    "properties": {
                        "workflow": {"type": "string"},
                        "workflow_id": {"type": "string"},
                        "parameters": {"$ref": "#/components/schemas/WorkflowParameters"},
                        "prompt_granularity": {"$ref": "#/components/schemas/Deprecated"},
                        "client_id": {"type": "string", "maxLength": 256},
                        "dry_run": {"type": "boolean", "default": False},
                    },
                    "additionalProperties": False,
                },
                "WorkflowParameters": {
                    "type": "object",
                    "anyOf": [{"required": ["prompt"]}, {"required": ["subject"]}],
                    "properties": {
                        "prompt": {"type": "string", "maxLength": MAX_PROMPT_LENGTH},
                        "subject": {"type": "string", "maxLength": MAX_PROMPT_LENGTH, "deprecated": True,
                                    "description": "Compatibility alias for prompt. Applied verbatim as prompt; conflicting prompt and subject values are rejected."},
                        "negative_prompt": {"type": "string", "maxLength": MAX_PROMPT_LENGTH},
                        "seed": {"type": "integer", "minimum": 0, "maximum": 9223372036854775807},
                        "width": {"type": "integer", "minimum": 64, "maximum": 2048, "multipleOf": 8},
                        "height": {"type": "integer", "minimum": 64, "maximum": 2048, "multipleOf": 8},
                        "batch_size": {"type": "integer", "minimum": 1, "maximum": 4},
                        "steps": {"type": "integer", "minimum": 1, "maximum": 100},
                        "cfg": {"type": "number", "minimum": 0, "maximum": 30},
                        "denoise": {"type": "number", "minimum": 0, "maximum": 1},
                        "sampler_name": {"type": "string", "maxLength": 80},
                        "scheduler": {"type": "string", "maxLength": 80},
                        "filename_prefix": {"type": "string", "maxLength": 120},
                        "input_image": {
                            "type": "string",
                            "maxLength": MAX_INPUT_IMAGE_LENGTH,
                            "description": "Relative filename returned by POST /upload/image or listed by GET /project/inputs; only workflows declaring image input accept it.",
                        },
                        "mask_image": {
                            "type": "string",
                            "maxLength": MAX_INPUT_IMAGE_LENGTH,
                            "description": "Relative input filename for an inpaint mask; white/red pixels are redrawn and black pixels are preserved according to the selected workflow's mask_channel.",
                        },
                        "mask_channel": {
                            "type": "string",
                            "enum": ["alpha", "red", "green", "blue"],
                        },
                        "grow_mask_by": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 64,
                        },
                        "face_image": {
                            "type": "string",
                            "maxLength": MAX_INPUT_IMAGE_LENGTH,
                            "description": "Relative input filename for a face-centric IPAdapter reference; it is not a job-output reference.",
                        },
                        "pose": {
                            "type": "string",
                            "maxLength": MAX_INPUT_IMAGE_LENGTH,
                            "description": "Body pose to follow: a pose id or view id from GET /project/poses, or the relative name of an uploaded image whose pose should be copied. Only routes with pose support accept it.",
                        },
                        "pose_strength": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "How strictly the pose is followed.",
                        },
                        "input_fit": {
                            "type": "string",
                            "enum": list(INPUT_FIT_MODES),
                            "default": "downscale",
                            "description": "Only on routes with image inputs. downscale shrinks oversized inputs and takes the first frame of animated ones, reporting each change as a warning; none keeps the upload unchanged and refuses what the route cannot run.",
                        },
                        "ipadapter_weight": {
                            "type": "number",
                            "minimum": -1,
                            "maximum": 3,
                        },
                        "loras": {
                            "type": "object",
                            "additionalProperties": {"type": "number", "minimum": -2, "maximum": 2},
                        },
                    },
                    "additionalProperties": False,
                },
            },
        },
    }


    schemas = document["components"]["schemas"]
    schemas["ApiError"] = {
        "type": "object", "required": ["error"], "properties": {"error": {
            "type": "object", "required": ["type", "message", "route", "expected"],
            "properties": {"type": {"type": "string"}, "message": {"type": "string"},
                           "code": {"type": ["string", "null"], "description": "Stable machine-readable reason; see the manifest's error_codes."},
                           "route": {"type": ["string", "null"]},
                           "expected": {"type": "array", "items": {"type": "string"}}, "details": {}},
        }},
    }
    schemas["Warning"] = {"type": "object", "required": ["code", "message"], "properties": {
        "code": {"type": "string"}, "message": {"type": "string"}, "parameter": {"type": "string"},
        "value": {"type": "number"}, "recommended": {"type": "array", "items": {"type": "number"}},
        "original_size": {"type": "array", "items": {"type": "integer"}}, "used_size": {"type": "array", "items": {"type": "integer"}},
    }}
    common = {
        "applied_parameters": {"type": "object", "description": "Canonical graph-bound values including defaults; this describes application to the graph, not model adherence."},
        "ignored_parameters": {"type": "array", "items": {"type": "string"}, "maxItems": 0},
        "effective_parameters": {"type": "object"}, "resources": {"type": "object"},
        "warnings": {"type": "array", "items": {"$ref": "#/components/schemas/Warning"}},
        "workflow": {"type": "string"},
        "pose": {"type": ["object", "null"], "description": "The pose used, people_detected and skeleton_url; null without a pose."},
    }
    schemas["QueuedJob"] = {"type": "object", "required": ["prompt_id", "job_id", "job_url", "state", *common], "properties": {
        **common, "prompt_id": {"type": "string", "format": "uuid"},
        "job_id": {"type": "string", "format": "uuid", "description": "Compatibility alias, identical to prompt_id."},
        "job_url": {"type": "string"}, "state": {"const": "queued"},
    }}
    schemas["DryRun"] = {"type": "object", "required": ["prompt", "ready", "validated", *common], "properties": {
        **common, "prompt": {"type": "object"}, "ready": {"type": "boolean"}, "validated": {"type": "boolean"},
    }}
    schemas["JobStatus"] = {"type": "object", "required": ["prompt_id", "job_id", "state", "outputs", "queued_ms", "elapsed_ms"], "properties": {
        "prompt_id": {"type": "string", "format": "uuid"}, "job_id": {"type": "string", "format": "uuid"},
        "state": {"enum": ["queued", "running", "succeeded", "failed"]},
        "queued_ms": {"type": ["integer", "null"], "minimum": 0},
        "elapsed_ms": {"type": "integer", "minimum": 0}, "outputs": {"type": "array", "items": {"type": "object"}},
    }}
    schemas["NamedPromptRequest"]["description"] = "Use the selected workflow's parameter_schema. prompt (or subject alias) and required image inputs are validated before queueing."
    for path in ("/project/prompt", "/project/generate"):
        responses = document["paths"][path]["post"]["responses"]
        responses["200"]["content"] = {"application/json": {"schema": {"oneOf": [
            {"$ref": "#/components/schemas/QueuedJob"}, {"$ref": "#/components/schemas/DryRun"},
        ]}}}
        for status, description in {"400": "Invalid or unknown parameter", "404": "Unknown workflow", "409": "Unavailable dependencies", "422": "Missing required input", "429": "Project queue full"}.items():
            responses[status] = {"description": description, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ApiError"}}}}
    document["paths"]["/project/jobs/{prompt_id}"]["get"]["responses"]["200"]["content"] = {
        "application/json": {"schema": {"$ref": "#/components/schemas/JobStatus"}}}
    return document


def _validated_prompt_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ApiError(400, "prompt_id must be a UUID")
    try:
        return validate_job_id(value)
    except ValueError as exc:
        raise ApiError(400, "prompt_id must be a valid UUID") from exc


def _redact_output_metadata(history_entry: dict[str, Any]) -> None:
    """Drop input image names from saved PNG metadata before the job is reported; prompts and settings stay."""
    from project_nodes.output_metadata import redact_input_names
    output_root = Path(folder_paths.get_output_directory()).resolve()
    for reference in _output_references(history_entry):
        if reference["type"] != "output" or not reference["filename"].lower().endswith(".png"):
            continue
        path = (output_root / reference["subfolder"] / reference["filename"]).resolve()
        try:
            path.relative_to(output_root)
            redact_input_names(path)
        except (ValueError, OSError):
            logging.warning("Could not redact input names in %s", reference["filename"])


def _output_references(history_entry: dict[str, Any]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    outputs = history_entry.get("outputs", {})
    if not isinstance(outputs, dict):
        return references
    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        for output_kind in ("images", "gifs", "audio"):
            artifacts = node_output.get(output_kind, [])
            if not isinstance(artifacts, list):
                continue
            for artifact in artifacts:
                if not isinstance(artifact, dict) or not isinstance(artifact.get("filename"), str):
                    continue
                filename = artifact["filename"]
                subfolder = artifact.get("subfolder", "")
                artifact_type = artifact.get("type", "output")
                query = urlencode({"filename": filename, "subfolder": subfolder, "type": artifact_type})
                references.append(
                    {
                        "node_id": str(node_id),
                        "kind": output_kind,
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": artifact_type,
                        "url": "/view?" + query,
                    }
                )
    return references


def _history_error(entry: dict[str, Any]) -> dict[str, Any] | None:
    for event, data in reversed(entry.get("status", {}).get("messages", [])):
        if event in {"execution_error", "execution_interrupted"}:
            result = {key: data.get(key) for key in ("node_id", "node_type", "exception_type", "exception_message")}
            result["event"] = event
            message = str(data.get("exception_message", "")).lower()
            exception_type = str(data.get("exception_type", ""))
            # GPU OOM, CPU allocator failure, Python MemoryError, Windows pagefile exhaustion (os error 1455).
            if any(text in message for text in ("out of memory", "not enough memory", "paging file is too small")) \
                    or any(name in exception_type for name in ("OutOfMemory", "MemoryError")):
                result["code"] = "out_of_memory"
                result["next_step"] = "Tell the requester the job ran out of RAM/VRAM (GET /project/resources shows current free memory). Resubmit after they free memory elsewhere or choose smaller dimensions or batch size."
            return result
    return None


def _job_response(payload: dict[str, Any], coordinator_state=None, history_entry=None) -> web.Response:
    payload["job_id"] = payload["prompt_id"]
    timing = (coordinator_state or {}).get("timing", {})
    if not timing and history_entry:
        prompt = history_entry.get("prompt", [])
        if len(prompt) > 3 and isinstance(prompt[3], dict):
            timing = prompt[3].get("project_timing", {})
    now = time.monotonic()
    queued_at = timing.get("submitted_monotonic")
    started_at = timing.get("started_monotonic")
    payload["queued_ms"] = timing.get("queued_ms", max(0, round((now - queued_at) * 1000)) if queued_at is not None else None)
    payload["elapsed_ms"] = timing.get("elapsed_ms", max(0, round((now - started_at) * 1000)) if started_at is not None else 0)
    return web.json_response(payload)


async def get_project_job(request: web.Request) -> web.Response:
    try:
        prompt_id = _validated_prompt_id(request.match_info["prompt_id"])
        server = PromptServer.instance
        coordinator = _project_coordinator(server, create=False)
        coordinator_state = coordinator.snapshot(prompt_id) if coordinator is not None else None
        history = server.prompt_queue.get_history(prompt_id=prompt_id)
        entry = history.get(prompt_id) if isinstance(history, dict) else None
        if isinstance(entry, dict):
            status = entry.get("status", {})
            if not isinstance(status, dict):
                status = {}
            status_str = status.get("status_str")
            completed = status.get("completed") is True
            state = "succeeded" if status_str == "success" and completed else "failed" if status_str == "error" or completed else "running"
            result: dict[str, Any] = {
                "prompt_id": prompt_id,
                "state": state,
                "completed": completed or status_str == "error",
                "status": status_str,
                "outputs": _output_references(entry),
            }
            if isinstance(coordinator_state, dict):
                result["workflow"] = coordinator_state.get("workflow")
                result["phase"] = "complete"
            if state == "failed":
                result["error"] = _history_error(entry)
            return _job_response(result, coordinator_state, entry)

        if isinstance(coordinator_state, dict):
            result = {
                "prompt_id": prompt_id,
                "workflow": coordinator_state.get("workflow"),
                "state": coordinator_state.get("state", "queued"),
                "completed": coordinator_state.get("completed", False),
                "status": coordinator_state.get("status"),
                "outputs": coordinator_state.get("outputs", []),
            }
            for key in ("phase", "queue_position", "queue_scope", "error"):
                if key in coordinator_state:
                    result[key] = coordinator_state[key]
            return _job_response(result, coordinator_state)

        running, pending = server.prompt_queue.get_current_queue()
        if any(isinstance(item, (list, tuple)) and len(item) > 1 and item[1] == prompt_id for item in running):
            return _job_response({"prompt_id": prompt_id, "state": "running", "completed": False, "outputs": []})
        for position, item in enumerate(pending):
            if isinstance(item, (list, tuple)) and len(item) > 1 and item[1] == prompt_id:
                return _job_response(
                    {
                        "prompt_id": prompt_id,
                        "state": "queued",
                        "completed": False,
                        "queue_position": position,
                        "outputs": [],
                    }
                )
        raise ApiError(404, f"Unknown prompt_id: {prompt_id}")
    except ApiError as error:
        return _json_error(error)


@dataclass(slots=True)
class _ProjectJob:
    number: int
    prompt_id: str
    workflow_id: str
    graph: dict[str, Any]
    extra_data: dict[str, Any]
    outputs_to_execute: list[str]
    sensitive: dict[str, Any]


class ProjectWorkflowCoordinator:
    """Serialize named workflow jobs and manage model residency between them."""

    def __init__(self, server: PromptServer):
        self.server = server
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[_ProjectJob] | None = None
        self._task: asyncio.Task[None] | None = None
        self._workflow_lock: asyncio.Lock | None = None
        self._states: dict[str, dict[str, Any]] = {}
        self._pending_ids: list[str] = []
        self._active_prompt_id: str | None = None

    def _ensure_started(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
            self._queue = asyncio.Queue()
            self._workflow_lock = asyncio.Lock()
        elif self._loop is not loop:
            raise RuntimeError("Project workflow coordinator is bound to another event loop")

        if self._task is None or self._task.done():
            self._task = loop.create_task(self._run(), name="project-workflow-coordinator")

    async def submit(self, job: _ProjectJob) -> None:
        self._ensure_started()
        if self._queue is None:
            raise RuntimeError("Project workflow coordinator queue is unavailable")
        if len(self._pending_ids) >= MAX_PROJECT_QUEUE:
            raise ApiError(429, "Project queue is full; retry after a queued job completes", {"max_pending_jobs": MAX_PROJECT_QUEUE})
        completed_ids = [key for key, state in self._states.items() if state.get("completed")]
        for key in completed_ids[:max(0, len(self._states) - MAX_PROJECT_JOB_STATES + 1)]:
            del self._states[key]
        self._states[job.prompt_id] = {
            "prompt_id": job.prompt_id,
            "workflow": job.workflow_id,
            "state": "queued",
            "completed": False,
            "status": None,
            "phase": "waiting_for_turn",
            "outputs": [],
            "queue_scope": "project_workflow_api",
            "timing": job.extra_data.get("project_timing", {}),
        }
        self._pending_ids.append(job.prompt_id)
        await self._queue.put(job)

    def snapshot(self, prompt_id: str) -> dict[str, Any] | None:
        state = self._states.get(prompt_id)
        if state is None:
            return None
        result = dict(state)
        if result.get("state") == "queued":
            try:
                result["queue_position"] = self._pending_ids.index(prompt_id)
            except ValueError:
                pass
        return result

    def _set_state(self, prompt_id: str, **updates: Any) -> None:
        state = self._states.setdefault(prompt_id, {"prompt_id": prompt_id})
        state.update(updates)
        if updates.get("completed") and state.get("timing"):
            timing = state["timing"]
            now = time.monotonic()
            timing.setdefault("queued_ms", max(0, round((now - timing["submitted_monotonic"]) * 1000)))
            timing.setdefault("elapsed_ms", max(0, round((now - timing["started_monotonic"]) * 1000)) if "started_monotonic" in timing else 0)

    def _remove_pending(self, prompt_id: str) -> None:
        try:
            self._pending_ids.remove(prompt_id)
        except ValueError:
            pass

    async def _wait_for_history(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + PROJECT_JOB_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            history = self.server.prompt_queue.get_history(prompt_id=prompt_id)
            entry = history.get(prompt_id) if isinstance(history, dict) else None
            if isinstance(entry, dict):
                return entry
            running, pending = self.server.prompt_queue.get_current_queue_volatile()
            if not any(item[1] == prompt_id for item in running + pending):
                # Re-read history after the queue snapshot: the worker can finish
                # between these two reads. A removed native queue item has no history.
                history = self.server.prompt_queue.get_history(prompt_id=prompt_id)
                if prompt_id in history:
                    return history[prompt_id]
                raise RuntimeError("Job was removed from the native queue or its history was deleted")
            await asyncio.sleep(PROJECT_QUEUE_POLL_SECONDS)
        self.server.prompt_queue.delete_queue_item(lambda item: item[1] == prompt_id)
        self.server.prompt_queue.interrupt_if_running(prompt_id)
        raise RuntimeError(f"Timed out waiting for ComfyUI job {prompt_id}")

    async def _run_one(self, job: _ProjectJob) -> None:
        if self._workflow_lock is None:
            raise RuntimeError("Project workflow coordinator lock is unavailable")
        async with self._workflow_lock:
            self._active_prompt_id = job.prompt_id
            self._remove_pending(job.prompt_id)

            self._set_state(job.prompt_id, state="queued", phase="waiting_for_comfyui")
            self.server.prompt_queue.put(
                (
                    job.number,
                    job.prompt_id,
                    job.graph,
                    job.extra_data,
                    job.outputs_to_execute,
                    job.sensitive,
                )
            )
            history_entry = await self._wait_for_history(job.prompt_id)
            _redact_output_metadata(history_entry)
            status = history_entry.get("status", {})
            if not isinstance(status, dict):
                status = {}
            status_str = status.get("status_str")
            completed = status.get("completed") is True
            succeeded = status_str == "success" and completed
            self._set_state(
                job.prompt_id,
                state="succeeded" if succeeded else "failed",
                completed=True,
                status=status_str or "error",
                phase="complete",
                outputs=_output_references(history_entry),
                error=_history_error(history_entry) if not succeeded else None,
            )
            self._active_prompt_id = None

    async def _run(self) -> None:
        if self._queue is None:
            raise RuntimeError("Project workflow coordinator queue is unavailable")
        while True:
            job = await self._queue.get()
            try:
                await self._run_one(job)
            except Exception as exc:
                self._active_prompt_id = None
                self._set_state(
                    job.prompt_id,
                    state="failed",
                    completed=True,
                    status="error",
                    phase="complete",
                    error={"code": "project_execution_failed", "message": str(exc)},
                    outputs=[],
                )
                logging.exception("Project workflow job %s failed: %s", job.prompt_id, exc)
            finally:
                _delete_fitted_inputs(job.extra_data.get("project_resources", {}).get("fitted_inputs", []))
                self._remove_pending(job.prompt_id)
                self._queue.task_done()


_PROJECT_COORDINATOR: ProjectWorkflowCoordinator | None = None


def _project_coordinator(server: PromptServer, *, create: bool) -> ProjectWorkflowCoordinator | None:
    global _PROJECT_COORDINATOR
    if _PROJECT_COORDINATOR is None or _PROJECT_COORDINATOR.server is not server:
        if not create:
            return None
        _PROJECT_COORDINATOR = ProjectWorkflowCoordinator(server)
    return _PROJECT_COORDINATOR


async def _validate_graph(server: PromptServer, graph: dict[str, Any], client_id: str, prompt_id: str) -> tuple[dict[str, Any], Any]:
    json_data = server.trigger_on_prompt({"prompt": graph, "client_id": client_id})
    if not isinstance(json_data, dict) or not isinstance(json_data.get("prompt"), dict):
        raise ApiError(500, "ComfyUI on-prompt handlers returned an invalid request")
    graph = json_data["prompt"]
    server.node_replace_manager.apply_replacements(graph)
    valid = await execution.validate_prompt(prompt_id, graph, None)
    if not valid[0]:
        raise ApiError(400, "Workflow validation failed", {"message": valid[1], "node_errors": valid[3]})
    return graph, valid


async def _enqueue(server: PromptServer, graph: dict[str, Any], client_id: str, workflow_id: str, resources: dict[str, Any], valid: Any) -> dict[str, Any]:
    number = server.number
    server.number += 1
    prompt_id = str(uuid.uuid4())

    extra_data: dict[str, Any] = {"client_id": client_id, "create_time": int(time.time() * 1000),
                                  "project_workflow": workflow_id, "project_resources": resources,
                                  "project_timing": {"submitted_monotonic": time.monotonic()}}
    sensitive: dict[str, Any] = {}
    for sensitive_key in execution.SENSITIVE_EXTRA_DATA_KEYS:
        if sensitive_key in extra_data:
            sensitive[sensitive_key] = extra_data.pop(sensitive_key)
    coordinator = _project_coordinator(server, create=True)
    if coordinator is None:
        raise ApiError(500, "Project workflow coordinator is unavailable")
    await coordinator.submit(
        _ProjectJob(
            number=number,
            prompt_id=prompt_id,
            workflow_id=workflow_id,
            graph=graph,
            extra_data=extra_data,
            outputs_to_execute=valid[2],
            sensitive=sensitive,
        )
    )
    return {"prompt_id": prompt_id, "job_id": prompt_id, "number": number, "node_errors": valid[3], "state": "queued"}


async def _read_json(request: web.Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise ApiError(400, "Request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise ApiError(400, "Request body must be a JSON object")
    return body


def _project_input_files() -> tuple[bool, list[dict[str, Any]]]:
    """Return safe, relative metadata for files in ComfyUI's input folder.

    The listing intentionally contains no host paths and follows only files
    whose resolved path remains inside the input directory. This makes the
    result directly usable as ``parameters.input_image`` without exposing
    filesystem layout to an agent.
    """
    input_root = Path(folder_paths.get_input_directory()).resolve()
    if not input_root.is_dir():
        return False, []

    files: list[dict[str, Any]] = []
    for candidate in input_root.rglob("*"):
        try:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            relative = resolved.relative_to(input_root)
            stat_result = resolved.stat()
        except (OSError, RuntimeError, ValueError):
            # Files can disappear during a listing, and symlinks outside the
            # input directory must not be exposed.
            continue
        name = relative.as_posix()
        if not name or name.startswith("../") or name.startswith(f"{pose_addon.CACHE_DIR}/"):
            continue  # the pose cache (library sketches, skeletons) is the service's, not the requester's
        files.append(
            {
                "name": name,
                "size_bytes": stat_result.st_size,
                "modified_time": stat_result.st_mtime,
                "extension": resolved.suffix.lower(),
            }
        )
    files.sort(key=lambda item: str(item["name"]).casefold())
    return True, files


async def list_project_inputs(request: web.Request) -> web.Response:
    del request
    try:
        directory_exists, files = _project_input_files()
        return web.json_response(
            {
                "schema_version": 1,
                "directory": "ComfyUI/input",
                "directory_exists": directory_exists,
                "count": len(files),
                "total_bytes": sum(item["size_bytes"] for item in files),
                "lifecycle": "Persistent across restarts; this project has no automatic TTL or cleanup. Files are removed only by explicit DELETE or external/manual filesystem actions.",
                "files": files,
            }
        )
    except Exception as exc:  # Keep unexpected details in the server log.
        logging.exception("Project input listing failed")
        return _json_error(ApiError(500, "Project input listing failed", str(exc)))


async def delete_project_input(request: web.Request) -> web.Response:
    """Delete one file from ComfyUI's input directory without path escape."""
    try:
        file_name = _safe_input_file(request.match_info.get("name", ""), "name")
        input_root = Path(folder_paths.get_input_directory()).resolve()
        input_path = (input_root / file_name).resolve()
        try:
            input_path.relative_to(input_root)
        except ValueError as exc:
            raise ApiError(400, "name must remain inside ComfyUI's input directory") from exc
        original_path = input_root / file_name
        if original_path.is_symlink():
            raise ApiError(400, "name must refer to a regular input file")
        if not input_path.is_file():
            raise ApiError(404, f"Input file was not found: {file_name}")
        size_bytes = input_path.stat().st_size
        input_path.unlink()
        _, remaining_files = _project_input_files()
        return web.json_response(
            {
                "deleted": True,
                "name": file_name,
                "size_bytes": size_bytes,
                "remaining_bytes": sum(item["size_bytes"] for item in remaining_files),
            }
        )
    except ApiError as error:
        return _json_error(error)
    except OSError as exc:
        logging.exception("Project input deletion failed")
        return _json_error(ApiError(500, "Project input deletion failed", str(exc)))


def _query_boolean(value: Any, name: str, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.casefold() in {"1", "true", "yes", "on"}:
        return True
    if isinstance(value, str) and value.casefold() in {"0", "false", "no", "off"}:
        return False
    raise ApiError(400, f"{name} must be a boolean")


def _query_limit(value: Any, default: int = 1) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ApiError(400, "limit must be an integer")
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(400, "limit must be an integer") from exc
    if str(limit) != str(value) or not 1 <= limit <= 20:
        raise ApiError(400, "limit must be between 1 and 20")
    return limit


async def list_poses(request: web.Request) -> web.Response:
    try:
        query = _as_text("q", request.query.get("q", ""), 256)
        people = request.query.get("people")
        if people not in (None, "", "1", "2"):
            raise ApiError(400, "people must be 1 or 2", code="invalid_parameter")
        angle = request.query.get("angle") or None
        if angle is not None and angle not in pose_addon.ANGLES:
            raise ApiError(400, "angle must be one of " + ", ".join(pose_addon.ANGLES), code="invalid_parameter")
        result = pose_addon.search(query, int(people) if people else None, angle,
                                   _query_limit(request.query.get("limit"), default=8))
        return web.json_response({**result, "usage": "Pass a view id from views (or a pose id for its front view) as parameters.pose."})
    except ApiError as error:
        return _json_error(error)


async def list_routes(request: web.Request) -> web.Response:
    try:
        query, deprecated = _route_query(dict(request.query))
        result = _resolve_routes(
            _read_catalog(),
            query,
            include_unready=_query_boolean(request.query.get("include_unready"), "include_unready"),
            limit=_query_limit(request.query.get("limit"), default=10),
            deprecated=deprecated,
        )
        return web.json_response(result)
    except ApiError as error:
        return _json_error(error)


async def resolve_route(request: web.Request) -> web.Response:
    try:
        body = await _read_json(request)
        query, deprecated = _route_query(body)
        include_unready = body.get("include_unready", False)
        if not isinstance(include_unready, bool):
            raise ApiError(400, "include_unready must be a boolean")
        limit = _as_int("limit", body.get("limit", 1), 1, 20)
        result = _resolve_routes(
            _read_catalog(), query, include_unready=include_unready, limit=limit, deprecated=deprecated
        )
        return web.json_response(result)
    except ApiError as error:
        return _json_error(error)


async def list_models(request: web.Request) -> web.Response:
    try:
        catalog = _read_catalog()
        models = [
            model for model in catalog.get("model_registry", []) if isinstance(model, dict)
        ]
        family = request.query.get("family")
        status = request.query.get("status")
        explicit_status = bool(status)
        include_candidates = _query_boolean(
            request.query.get("include_candidates"), "include_candidates"
        )
        include_nonrouteable = _query_boolean(
            request.query.get("include_nonrouteable"), "include_nonrouteable"
        )
        ready_value = request.query.get("ready")
        if not status and not include_candidates and ready_value is None:
            status = "installed"
        if not include_candidates and not include_nonrouteable and not explicit_status:
            models = [model for model in models if model.get("routeable") is True]
        if family:
            family_value = _normalise_match_text(family)
            models = [
                model
                for model in models
                if _normalise_match_text(model.get("family")) == family_value
            ]
        if status:
            status_value = _normalise_match_text(status)
            models = [
                model
                for model in models
                if _normalise_match_text(model.get("status")) == status_value
            ]
        if ready_value is not None:
            ready = _query_boolean(ready_value, "ready")
            models = [
                model for model in models if bool(_public_model(model)["ready"]) is ready
            ]
        return web.json_response(
            {
                "schema_version": catalog.get("schema_version", 1),
                "models": [_public_model(model) for model in models],
            }
        )
    except ApiError as error:
        return _json_error(error)


async def get_model(request: web.Request) -> web.Response:
    try:
        model_id = request.match_info["model_id"]
        catalog = _read_catalog()
        for model in catalog.get("model_registry", []):
            if isinstance(model, dict) and model.get("id") == model_id:
                return web.json_response({"model": _public_model(model)})
        raise ApiError(404, f"Unknown model: {model_id}")
    except ApiError as error:
        return _json_error(error)


async def list_workflows(request: web.Request) -> web.Response:
    try:
        catalog = _read_catalog()
        return web.json_response(
            {
                "schema_version": catalog.get("schema_version", 1),
                "workflows": [
                    _public_entry(entry, catalog.get("model_registry", []))
                    for entry in catalog["workflows"]
                    if isinstance(entry, dict)
                ],
            }
        )
    except ApiError as error:
        return _json_error(error)


async def get_agent_manifest(request: web.Request) -> web.Response:
    try:
        return web.json_response(_agent_manifest(_read_catalog()))
    except ApiError as error:
        return _json_error(error)


async def get_openapi(request: web.Request) -> web.Response:
    return web.json_response(_openapi_document())


async def get_workflow(request: web.Request) -> web.Response:
    try:
        catalog = _read_catalog()
        workflow_id = request.match_info["workflow_id"]
        entry = next(
            (
                item
                for item in catalog["workflows"]
                if isinstance(item, dict) and _is_workflow(item, workflow_id)
            ),
            None,
        )
        if entry is None:
            raise ApiError(404, f"Unknown workflow: {workflow_id}")
        return web.json_response(
            {
                "workflow": _public_entry(entry, catalog.get("model_registry", [])),
                "prompt": _read_workflow(entry),
            }
        )
    except ApiError as error:
        return _json_error(error)


async def post_project_prompt(request: web.Request) -> web.Response:
    entry = None
    workflow_id = None
    fit_report: list[dict[str, Any]] = []
    queued = False
    try:
        body = await _read_json(request)
        workflow_id = body.get("workflow", body.get("workflow_id"))
        if "workflow" in body and "workflow_id" in body and body["workflow"] != body["workflow_id"]:
            raise ApiError(400, "workflow and workflow_id disagree")
        entry = _find_workflow(workflow_id)
        pose_request = await asyncio.to_thread(_resolve_pose, entry, body)
        graph = _prepare_graph(entry, body, fit_report, pose_request)
        public = _public_entry(entry, _read_catalog().get("model_registry", []))
        client_id = body.get("client_id", "project-workflow-api")
        client_id = _as_text("client_id", client_id, 256)
        dry_run = body.get("dry_run", False)
        if not isinstance(dry_run, bool):
            raise ApiError(400, "dry_run must be a boolean")
        valid = None
        if public["ready"]:
            graph, valid = await _validate_graph(PromptServer.instance, graph, client_id, str(uuid.uuid4()))
        resources = _resource_request(graph, entry)
        resources["input_adjustments"] = fit_report
        resources["fitted_inputs"] = [record["file"] for record in fit_report if record["file"]]
        effective = {name: _binding_value(graph, entry, name) for name in entry["bindings"]
                     if name in {"seed", "width", "height", "batch_size", "steps", "cfg", "denoise", "sampler_name", "scheduler"}}
        parameter_schema = _parameter_schema(entry, graph)
        applied = {name: _binding_value(graph, entry, name) for name in entry["bindings"] if name in parameter_schema["properties"]}
        if "input_image" in _request_parameters(body) and "input_image" not in applied:
            applied["input_image"] = _request_parameters(body)["input_image"]
        if entry.get("lora_bindings"):
            applied["loras"] = {name: _binding_value(graph, entry, spec["strength_model"]) for name, spec in entry["lora_bindings"].items()}
        if pose_request:
            applied["pose"] = pose_request.get("id") or pose_request.get("file")
            if pose_request["strength"] is not None:
                applied["pose_strength"] = pose_request["strength"]
        warnings = _deprecation_warnings([name for name in ("prompt_granularity",) if name in body])
        for record in fit_report:
            original, used, name = record["original_size"], record["used_size"], record["parameter"]
            if record["first_frame"]:
                warnings.append({"code": "input_first_frame", "parameter": name,
                                 "message": f"{name} is animated; only its first frame was used."})
            if used != original:
                code = "input_downscaled" if used[0] * used[1] < original[0] * original[1] else "input_resized"
                target = "the output size" if record.get("scope") == "output size" else name
                warnings.append({"code": code, "parameter": name, "original_size": original, "used_size": used,
                                 "message": f"{name} is {original[0]}x{original[1]}; {target} was set to {used[0]}x{used[1]} to fit what "
                                            f"{workflow_id} can run. Send input_fit=none to keep the original size."})
        if resources["total_pixels"] > resources["limits"]["max_total_pixels"]:
            warnings.append({"code": "over_pixel_budget", "parameter": "total_pixels",
                             "value": resources["total_pixels"], "recommended": [0, resources["limits"]["max_total_pixels"]],
                             "message": f"{resources['total_pixels']} total pixels exceeds the tested budget of {resources['limits']['max_total_pixels']} for {workflow_id}; the job was kept but may be slow or fail with out_of_memory."})
        if effective.get("cfg") == 1 and "negative_prompt" in _request_parameters(body):
            warnings.append({"code": "negative_prompt_inactive", "parameter": "negative_prompt", "message": "At CFG 1 the sampler skips negative conditioning; negative_prompt has no effect at this setting."})
        for name, interval in entry.get("agent_hints", {}).get("recommended", {}).items():
            if name not in effective or not isinstance(interval, (list, tuple)) or len(interval) != 2:
                continue
            if not interval[0] <= effective[name] <= interval[1]:
                warnings.append({"code": "outside_recommended_range", "parameter": name,
                                 "value": effective[name], "recommended": interval,
                                 "message": f"{name}={effective[name]} is outside the recommended range {interval} for {workflow_id}; the requested value was retained."})
        if dry_run:
            return web.json_response(
                {
                    "workflow": entry["id"],
                    "ready": public["ready"],
                    "missing_files": public["missing_files"],
                    "validated": valid is not None,
                    "effective_parameters": effective,
                    "applied_parameters": applied,
                    "ignored_parameters": [],
                    "resources": resources,
                    "warnings": warnings,
                    "pose": _pose_summary(pose_request),
                    "prompt": graph,
                }
            )
        if not public["ready"]:
            raise ApiError(409, "Workflow dependencies are unavailable", public)
        result = await _enqueue(PromptServer.instance, graph, client_id, workflow_id, resources, valid)
        queued = True  # the job now owns its fitted inputs and deletes them when it ends
        result["workflow"] = entry["id"]
        result["job_url"] = f"/project/jobs/{result['prompt_id']}"
        result["effective_parameters"] = effective
        result["applied_parameters"] = applied
        result["ignored_parameters"] = []
        result["resources"] = resources
        result["warnings"] = warnings
        result["pose"] = _pose_summary(pose_request)
        return web.json_response(result)
    except pose_addon.PoseError as exc:
        return _json_error(ApiError(exc.status, exc.message, exc.details, code=exc.code,
                                    route=workflow_id if isinstance(workflow_id, str) else None))
    except ApiError as error:
        if error.route is None:
            error.route = workflow_id if isinstance(workflow_id, str) else None
        if error.expected is None and entry is not None:
            error.expected = sorted(_parameter_schema(entry, _read_workflow(entry))["properties"])
        return _json_error(error)
    except Exception as exc:  # Keep unexpected details in the server log.
        logging.exception("Project workflow API request failed")
        return _json_error(ApiError(500, "Project workflow API request failed", str(exc), route=workflow_id))
    finally:
        if not queued:
            _delete_fitted_inputs([record["file"] for record in fit_report if record["file"]])


def _notify_worker_phase(prompt_id: str, phase: str) -> None:
    coordinator = _PROJECT_COORDINATOR
    if coordinator is not None and coordinator._loop is not None:
        coordinator._loop.call_soon_threadsafe(
            lambda: coordinator._set_state(prompt_id, state="running", phase=phase)
            if prompt_id in coordinator._states and not coordinator._states[prompt_id].get("completed") else None
        )


async def get_project_resources(request: web.Request) -> web.Response:
    from project_nodes.runtime_guard import memory_snapshot
    return web.json_response({**memory_snapshot(), "max_pending_jobs": MAX_PROJECT_QUEUE,
                              "max_input_pixels": MAX_INPUT_PIXELS})


def _register_routes() -> None:
    if getattr(PromptServer, "instance", None) is None:
        logging.error("Project workflow API loaded before PromptServer was initialized")
        return
    routes = PromptServer.instance.routes
    routes.get("/.well-known/agent-manifest.json")(get_agent_manifest)
    routes.get("/project/agent-manifest")(get_agent_manifest)
    routes.get("/project/openapi.json")(get_openapi)
    routes.get("/project/resources")(get_project_resources)
    routes.get("/project/routes")(list_routes)
    routes.get("/project/inputs")(list_project_inputs)
    routes.get("/project/poses")(list_poses)
    routes.delete("/project/inputs/{name:.*}")(delete_project_input)
    routes.post("/project/resolve")(resolve_route)
    routes.get("/project/models")(list_models)
    routes.get("/project/models/{model_id}")(get_model)
    routes.get("/project/workflows")(list_workflows)
    routes.get("/project/workflows/{workflow_id}")(get_workflow)
    routes.post("/project/prompt")(post_project_prompt)
    routes.post("/project/generate")(post_project_prompt)
    routes.get("/project/jobs/{prompt_id}")(get_project_job)
    from project_nodes.runtime_guard import install
    install(execution, _notify_worker_phase, _validate_queued_inputs)
    logging.info(
        "Project workflow API enabled at /.well-known/agent-manifest.json, "
        "/project/agent-manifest, /project/openapi.json, /project/resources, /project/routes, "
        "/project/inputs, /project/inputs/{name}, /project/poses, /project/models, /project/workflows, /project/resolve, "
        "/project/generate, /upload/image and /project/jobs"
    )


_register_routes()

# ComfyUI requires this mapping for a V1 custom-node module.  This extension
# intentionally contributes routes only and does not add nodes to the graph.
NODE_CLASS_MAPPINGS: dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
WEB_DIRECTORY = None
