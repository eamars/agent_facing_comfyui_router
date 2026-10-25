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
PROMPT_GRANULARITY_LEVELS = ("minimal", "structured", "detailed", "expert")
PROMPT_GRANULARITY_ALIASES = {
    "short": "minimal",
    "basic": "minimal",
    "normal": "structured",
    "tagged": "structured",
    "full": "detailed",
    "advanced": "expert",
}


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
    return catalog


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
        "content_class",
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
        if name in entry["bindings"] and name not in {"prompt", "seed", "input_image", "mask_image", "face_image", "control_image"}:
            spec["default"] = _binding_value(graph, entry, name)
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
    for name in ("input_image", "mask_image", "face_image", "control_image"):
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
        "pose_image": "control_image",
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
        "control_image",
        "control_strength",
        "control_start_percent",
        "control_end_percent",
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


def _prepare_graph(entry: dict[str, Any], body: dict[str, Any], fit_report: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Bind a request into the route's graph. Fitted input copies are appended to fit_report; the caller deletes them."""
    fit_report = [] if fit_report is None else fit_report
    graph = copy.deepcopy(_read_workflow(entry))
    parameters = _request_parameters(body)
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
    for name in ("input_image", "mask_image", "face_image", "control_image"):
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
        "control_strength": (0.0, 10.0),
        "control_start_percent": (0.0, 1.0),
        "control_end_percent": (0.0, 1.0),
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

    for parameter_name in ("mask_image", "face_image", "control_image"):
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
            ("control_image", "control_image"),
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
    if "control_start_percent" in entry["bindings"]:
        if _binding_value(graph, entry, "control_start_percent") > _binding_value(graph, entry, "control_end_percent"):
            raise ApiError(400, "control_start_percent must not exceed control_end_percent")
    _resource_request(graph, entry)
    return graph


def _normalise_match_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.casefold().replace("_", " ")
    value = re.sub(r"[^\w\u3400-\u9fff]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _normalise_granularity(value: Any) -> str:
    if not isinstance(value, str):
        raise ApiError(400, "prompt_granularity must be one of minimal, structured, detailed or expert")
    normalised = _normalise_match_text(value)
    normalised = PROMPT_GRANULARITY_ALIASES.get(normalised, normalised)
    if normalised not in PROMPT_GRANULARITY_LEVELS:
        raise ApiError(400, "prompt_granularity must be one of minimal, structured, detailed or expert")
    return normalised


def _query_value(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return None


def _route_query(payload: dict[str, Any]) -> dict[str, Any]:
    query: dict[str, Any] = {}
    aliases = {
        "domain": ("domain", "subject_domain", "franchise", "series"),
        "style": ("style", "aesthetic"),
        "content_mode": ("content_mode", "content", "safety_mode"),
        "input_type": ("input_type", "input"),
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
                "route_selectors": ["domain", "content_mode", "style", "input_type", "input_mode", "prompt_granularity"],
                "next_step": "resolve the workflow first, then include character and outfit in parameters.prompt",
            },
        )
    allowed = {name for names in aliases.values() for name in names} | {
        "prompt_granularity", "granularity", "workflow", "workflow_id", "include_unready", "limit"
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ApiError(400, "Unknown route selectors", sorted(unknown))
    granularity = _query_value(payload, "prompt_granularity", "granularity")
    if granularity is not None:
        query["prompt_granularity"] = _normalise_granularity(granularity)
    workflow = _query_value(payload, "workflow", "workflow_id")
    if workflow is not None:
        if not isinstance(workflow, str) or not workflow:
            raise ApiError(400, "workflow must be a non-empty workflow id")
        query["workflow"] = workflow
    return query


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
            "content_mode": "content_modes",
            "input_type": "input_types",
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
            "content_mode": "content_modes",
            "input_type": "input_types",
            "input_mode": "input_modes",
        }.get(key)
        if capability_key:
            values.extend(_normalised_values(capabilities.get(capability_key)))
        if key == "content_mode":
            values.extend(_normalised_values(capabilities.get("content_levels")))
    if key == "style" and not routing_values:
        values.extend(_normalised_values(entry.get("tags")))
        if isinstance(selection, dict):
            values.extend(_normalised_values(selection.get("priority_signals")))
    if key == "content_mode":
        values.extend(_normalised_values(entry.get("content_class")))
    return list(dict.fromkeys(values))


def _canonical_content_mode(value: str) -> str:
    normalised = _normalise_match_text(value)
    if normalised in {"nsfw", "explicit", "mature", "adult", "nsfw capable"}:
        return "nsfw"
    if normalised in {"sfw", "general", "non explicit", "non explicit content", "safe"}:
        return "sfw"
    return normalised


def _route_content_modes(entry: dict[str, Any]) -> set[str]:
    return {
        _canonical_content_mode(value)
        for value in _route_values(entry, "content_mode")
        if value
    }


def _route_prompt_granularity(entry: dict[str, Any]) -> dict[str, Any]:
    routing = entry.get("routing", {})
    capabilities = entry.get("capabilities", {})
    hint: Any = routing.get("prompt_granularity") if isinstance(routing, dict) else None
    if not isinstance(hint, dict):
        hint = capabilities.get("prompt_granularity") if isinstance(capabilities, dict) else None
    if not isinstance(hint, dict):
        return {
            "minimum": "minimal",
            "recommended": "structured",
            "maximum": "expert",
            "accepted": list(PROMPT_GRANULARITY_LEVELS),
        }
    minimum = _normalise_granularity(hint.get("minimum", "minimal"))
    recommended = _normalise_granularity(hint.get("recommended", minimum))
    maximum = _normalise_granularity(hint.get("maximum", "expert"))
    minimum_index = PROMPT_GRANULARITY_LEVELS.index(minimum)
    maximum_index = PROMPT_GRANULARITY_LEVELS.index(maximum)
    if minimum_index > maximum_index:
        raise ApiError(500, f"Workflow {entry.get('id', '<unknown>')} has invalid prompt granularity")
    accepted = hint.get("accepted")
    if isinstance(accepted, list):
        accepted_values = [
            _normalise_granularity(value) for value in accepted if isinstance(value, str)
        ]
    else:
        accepted_values = list(PROMPT_GRANULARITY_LEVELS[minimum_index : maximum_index + 1])
    return {
        **hint,
        "minimum": minimum,
        "recommended": recommended,
        "maximum": maximum,
        "accepted": list(dict.fromkeys(accepted_values)),
    }


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

    requested_content = query.get("content_mode")
    if isinstance(requested_content, str) and requested_content:
        requested_content = _canonical_content_mode(requested_content)
        route_modes = _route_content_modes(entry)
        nsfw_capable = bool(entry.get("capabilities", {}).get("nsfw_capable"))
        if requested_content == "nsfw" and not nsfw_capable:
            return None
        if requested_content not in route_modes:
            if routing.get("exclusive", False) or route_modes:
                return None
        else:
            score += 60
            reasons.append(f"content_mode matched: {requested_content}")
            preferences = routing.get("content_mode_preferences", {})
            if isinstance(preferences, dict):
                preference = preferences.get(requested_content)
                if isinstance(preference, (int, float)) and not isinstance(preference, bool):
                    score += int(preference)
                    if preference:
                        reasons.append(f"content_mode preference: {requested_content}")

    for selector, score_bonus, label in (
        ("input_type", 35, "input type"),
        ("input_mode", 40, "input mode"),
    ):
        requested_input = query.get(selector)
        if not isinstance(requested_input, str) or not requested_input:
            continue
        input_values = _route_values(entry, selector)
        if _normalise_match_text(requested_input) in input_values:
            score += score_bonus
            reasons.append(f"{label} matched")
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
            # A domain-specific route should outrank a generic fallback's content preference.
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

    requested_granularity = query.get("prompt_granularity")
    if isinstance(requested_granularity, str):
        hint = _route_prompt_granularity(entry)
        accepted = hint["accepted"]
        if requested_granularity not in accepted:
            reasons.append("prompt granularity is outside this route's recommended range")
        elif requested_granularity == hint["recommended"]:
            score += 25
            reasons.append(f"prompt granularity preferred: {requested_granularity}")
        else:
            score += 10
            reasons.append(f"prompt granularity supported: {requested_granularity}")

    if not reasons:
        reasons.append("catalog priority fallback")
    return score, reasons


def _resolve_routes(
    catalog: dict[str, Any],
    query: dict[str, Any],
    *,
    include_unready: bool = False,
    limit: int = 1,
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
        for key in ("workflow", "family", "domain", "content_mode", "style", "input_type", "input_mode")
    )
    selection_status = "selected" if selected else "no_match"
    guidance = "Use the selected workflow's prompting and model hints."
    if not has_specific_selector:
        selected = None
        selection_status = "needs_route_selector"
        guidance = "Provide domain, content_mode, style, input_type, input_mode or workflow before generating."
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
    }


def _image_inputs(entry: dict[str, Any]) -> list[str]:
    """Image parameters _prepare_graph accepts for this route."""
    bindings = entry.get("bindings", {})
    accepted = ["input_image"] if entry.get("input_contract", {}).get("image", {}).get("supported") or "input_image" in bindings else []
    return accepted + [name for name in ("mask_image", "face_image", "control_image") if name in bindings]


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
    ("openpose-source-image", "openpose-control", "control_image",
     "Copy the body pose of a person in an uploaded image.",
     "The route extracts an OpenPose skeleton; clothing, face and scene come from the prompt."),
)


def _reference_uses(workflows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uses = []
    for contract_mode, input_mode, parameter, intent, behaviour in _REFERENCE_USES:
        routes = [entry["id"] for entry in workflows
                  if any(isinstance(contract, dict) and contract.get("mode") == contract_mode
                         for contract in entry.get("input_contract", {}).values())]
        if routes:
            uses.append({"intent": intent, "input_mode": input_mode, "parameter": parameter, "routes": routes, "behaviour": behaviour})
    return uses


def _agent_manifest(catalog: dict[str, Any]) -> dict[str, Any]:
    model_registry = catalog.get("model_registry", [])
    ready_workflows = [
        entry
        for entry in catalog["workflows"]
        if isinstance(entry, dict) and _public_entry(entry, model_registry)["ready"]
    ]
    ready_model_ids = {
        model_id
        for entry in ready_workflows
        for model_id in entry.get("model_ids", [])
        if isinstance(model_id, str)
    }
    return {
        "schema_version": 1,
        "protocol": "project-comfyui-agent/v1",
        "api_version": "1.5.0",
        "route_table": [{"id": entry["id"], "family": entry["family"],
                         "control_ports": entry.get("control_ports", {}),
                         "required_fields": entry.get("input_contract", {}).get("required_fields", []),
                         "image_inputs": _image_inputs(entry),
                         "recommended": entry.get("agent_hints", {}).get("recommended", {}),
                         "explicit_only": entry.get("routing", {}).get("explicit_only", False),
                         # The words /project/resolve matches, so agents choose from a menu instead of guessing.
                         "selectors": {"domains": entry.get("routing", {}).get("domains", []),
                                       "styles": entry.get("routing", {}).get("styles", [])},
                         "best_for": entry.get("selection", {}).get("best_for", [])} for entry in ready_workflows],
        "discovery": {
            "preferred_entrypoint": "/.well-known/agent-manifest.json",
            "alternate_entrypoints": ["/project/agent-manifest", "/project/openapi.json"],
            "native_comfyui_ui": "/",
            "agent_startup_rule": "Read the preferred entrypoint before treating the root path as a ComfyUI client surface.",
        },
        "service": {
            "description": "Named workflow selection, reference-image upload, and text-to-image, image edit, image-to-image, inpaint, face-reference and pose-control job API for this ComfyUI server.",
            "authentication": "none",
            "api_base": "/project",
            "native_comfyui_api": "available",
        },
        "resources": {"endpoint": "/project/resources", "max_pending_jobs": MAX_PROJECT_QUEUE,
                      "admission": "Each workflow exposes resource_limits. max_total_pixels (width * height * batch, or the reference size for edits) is advisory: exceeding it adds an over_pixel_budget warning but the job still runs. There is no free-memory pre-check: a job that actually runs out of RAM/VRAM fails with error.code out_of_memory."},
        "input_files": {
            "directory": "ComfyUI/input",
            "lifecycle": "Persistent across restarts. No project TTL or automatic cleanup; only explicit DELETE or external/manual cleanup removes inputs. Keep referenced files unchanged until jobs finish.",
            "list": {
                "method": "GET",
                "path": "/project/inputs",
                "returns": "Relative filenames, byte sizes, modification times and extensions.",
            },
            "delete": {
                "method": "DELETE",
                "path": "/project/inputs/{name}",
                "returns": "Deletes one regular file and returns its relative name and byte size.",
            },
            "upload": {
                "method": "POST",
                "path": "/upload/image",
                "content_type": "multipart/form-data",
                "file_field": "image",
                "form_fields": {
                    "type": "input (required for a reference image)",
                    "overwrite": "optional true/false; default is to avoid replacing an existing file; use a unique basename for a new upload",
                    "subfolder": "optional relative input subfolder",
                },
                "response_fields": ["name", "subfolder", "type"],
                "use_as": "Combine response subfolder and name with '/' and pass it as the selected route's image parameter (route_table[].image_inputs): input_image, mask_image, face_image or control_image.",
            },
            "reference_uses": _reference_uses(ready_workflows),
            "input_fit": {
                "parameter": "parameters.input_fit",
                "values": list(INPUT_FIT_MODES),
                "default": "downscale",
                "downscale": "Oversized inputs are shrunk to what the route can run (aspect kept) and animated inputs use their first frame. Each change comes back as a warning with original_size and used_size: input_downscaled, input_resized or input_first_frame. The uploaded file is never modified; the adjusted copy is deleted when the job ends.",
                "none": "The upload is used unchanged. Inputs the route cannot run are refused with a stable error.code; larger inputs that can run may add an over_pixel_budget warning.",
            },
            "text_only_routes": [entry["id"] for entry in ready_workflows if not _image_inputs(entry)],
            "reference_rule": "Upload or discover the reference before submitting /project/prompt; the named API accepts only a relative input-directory filename, never a host path or URL.",
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
            "out_of_memory": "Job error.code: the job ran out of RAM/VRAM while running.",
        },
        "instructions": [
            "Read /.well-known/agent-manifest.json first; the root path is the native ComfyUI UI, not the preferred agent entrypoint.",
            "Call POST /project/resolve with domain, content_mode, style, input_type, input_mode, optional family and prompt_granularity when the route is not already known.",
            "Take domain and style words from route_table[].selectors and match the request to route_table[].best_for; requests that match no selector fall back to the anime routes. Real-world photos, products, landscapes and posters: domain 'photo', 'product', 'landscape' or 'poster' (Qwen-Image). Exact layout or UI mock-ups: domain 'layout' or 'ui mockup' (MOSS-Image).",
            "Before generating, read selection.prefer_instead in the resolve response: if its 'when' matches the request, use that workflow instead. Pass a style selector (e.g. 'visual novel' or 'mainstream anime') when the requester describes a look.",
            "Choose only an entry with ready=true and use the selected workflow's models, capabilities, routing and agent_hints together.",
            "Treat compatibility_notes, selection.not_for, model compatibility and prompting as constraints, not optional suggestions.",
            "Match the prompt dialect and granularity to the selected workflow; do not copy tag triggers from another model family.",
            "Use parameter_schema for allowed fields, required inputs and actual defaults. The DeepSeek whale girl routes (domain 'deepseek', aliases such as 大肥鱼) carry character LoRAs; use them only for that character and write her as an adult. The Pony-trained Rapi adapter is disabled by default on Animagine.",
            "Supply parameters.prompt explicitly (empty string is allowed); subject is a compatibility alias. Suggested subject/appearance details are prompt-writing advice, not separate required API fields. Seed is random when omitted.",
            "Responses include identical prompt_id/job_id, applied_parameters, ignored_parameters (empty; unknown fields are rejected), effective_parameters, resource budgets and advisory warnings (sampling ranges, over_pixel_budget); warnings never block a job. dry_run validates without executing.",
            "When the requester supplies an image, first decide what it is for using input_files.reference_uses (instruction edit, restyle/vary, inpaint, face reference or pose), then resolve with that input_mode. Send only the image parameters listed in route_table[].image_inputs; text_only_routes accept none.",
            "For any image input, call GET /project/inputs to discover references or POST multipart/form-data to /upload/image with image and type=input, then pass the returned relative name as the route's image parameter: input_image for instruction edit, image-to-image or the inpaint source, mask_image for the inpaint mask, face_image for face reference, or control_image for OpenPose control.",
            "Image inputs are fitted automatically (input_fit=downscale). When the response has input_downscaled, input_resized or input_first_frame warnings, tell the requester the image was adjusted and give original_size and used_size. Send input_fit=none only when the requester needs the original size; then relay any refusal's error.code and message.",
            "Use a unique basename for each new upload. With overwrite=false, the native ComfyUI endpoint may return an existing same-name file; delete only input names that this caller owns.",
            "GET /project/resources returns an advisory selected-device memory snapshot and configured queue/input limits; it is diagnostic information, not a reservation.",
            "If a job fails with error.code out_of_memory, report it to the requester with the current GET /project/resources snapshot before resubmitting.",
            "For inpaint, upload a source image and a mask with the same aspect ratio (normally the same size). RGB white/red pixels are redrawn; alpha masks redraw transparent pixels. denoise 1.0 (default) replaces the masked region; lower keeps more of the original inside it. Unmasked original pixels are composited back after decode.",
            "For face-reference, pass face_image to a face-reference route; it is not FaceID recognition. Use a head-and-shoulders crop. The Anima route encodes the whole reference: keep ipadapter_weight at its 0.6 default unless likeness matters more than the prompted outfit and scene, since 0.8-1.0 copies the reference's outfit and background.",
            "For openpose-control, use control_image with the OpenPose route; the route preprocesses the supplied source/control image before applying the SDXL ControlNet.",
            "After confirming a file is no longer needed, DELETE /project/inputs/{name}; only a single regular input file is removed.",
            "The upload response is an input filename, not an output-job reference; automatic output-to-input bindings are not yet implemented.",
            "POST /project/generate (or /project/prompt) returns prompt_id; poll GET /project/jobs/{prompt_id} until state is succeeded or failed.",
        ],
        "workflow_selection": {
            "required_fields": [
                "ready",
                "family",
                "model_ids",
                "models",
                "capabilities",
                "routing",
                "input_contract",
                "selection",
                "prompting",
                "agent_hints",
                "defaults",
                "parameter_schema",
                "resource_limits",
            ],
            "matching_order": [
                "honor routing.required_selectors for capability-specific routes such as inpaint, face-reference and openpose-control",
                "content_mode against routing.content_modes and capabilities.nsfw_capable",
                "domain/content mode/style against the workflow's routing selectors and model capabilities",
                "input_type/input_mode against capabilities.input_types and capabilities.input_modes",
                "prompt_granularity against prompting.granularity and agent_hints",
                "model readiness and compatibility_notes",
            ],
            "recommended_operation": "POST /project/resolve",
            "routing_boundary": "Routing selects a workflow/model combination only. Character, outfit, pose and other subject details belong in the final prompt.",
        },
        "prompting": {
            "granularity_levels": [
                {
                    "id": "minimal",
                    "meaning": "Subject and broad visual intent; useful for natural-language or fast general routes.",
                },
                {
                    "id": "structured",
                    "meaning": "Subject/series/character plus appearance, pose and composition blocks.",
                },
                {
                    "id": "detailed",
                    "meaning": "Structured tags or prose with outfit, accessories, lighting, camera and environment details.",
                },
                {
                    "id": "expert",
                    "meaning": "Model-specific trigger words, weighted adapter tokens, negative prompt and tuned sampling settings.",
                },
            ],
            "selection_rule": "Use the selected workflow's prompting.granularity.recommended value unless the caller explicitly requests another accepted level.",
            "agent_supplies_final_prompt": True,
            "input_rule": "Use only the selected workflow's declared input_contract. Upload or discover the declared input first, then provide input_image, mask_image, face_image or control_image exactly as documented; text-to-image does not require an input file.",
        },
        "routing_boundary": {
            "route_selectors": ["domain", "content_mode", "style", "input_type", "input_mode", "prompt_granularity"],
            "prompt_inputs_after_selection": [
                "character",
                "outfit",
                "pose",
                "appearance",
                "accessories",
                "weapon",
                "camera",
                "scene",
                "lighting",
            ],
            "rule": "Do not create a route per character or outfit when the same workflow/model can handle it; keep those details in parameters.prompt.",
        },
        "scheduling": {
            "queue": "project_workflow_serial",
            "mutex": "At most one named project workflow is submitted to ComfyUI for execution at a time.",
            "model_lifecycle": {
                "same_workflow": "Keep the selected workflow resident for consecutive jobs.",
                "different_workflow": "Release ComfyUI's loaded models and CUDA cache before starting the next workflow.",
                "failed_job": "Treat model residency as unknown after a failed job and establish a clean boundary before the next job.",
            },
            "job_phases": ["waiting_for_turn", "model_switch", "generation", "complete"],
            "max_pending_jobs": MAX_PROJECT_QUEUE,
            "queue_scope": "Named jobs enter the native serial worker. Model cleanup happens inside that worker, so it cannot unload a concurrently executing native job. Native graph sizes bypass project admission budgets.",
        },
        "operations": [
            {"id": "resources", "method": "GET", "path": "/project/resources", "description": "Inspect the selected device, live memory estimates and launcher safeguards."},
            {
                "id": "agent_manifest",
                "method": "GET",
                "path": "/.well-known/agent-manifest.json",
                "description": "Preferred first-read entrypoint for an independent agent.",
            },
            {
                "id": "agent_manifest_legacy",
                "method": "GET",
                "path": "/project/agent-manifest",
                "description": "Backward-compatible alias for the agent operating guide.",
            },
            {
                "id": "openapi_document",
                "method": "GET",
                "path": "/project/openapi.json",
                "description": "OpenAPI 3.1 contract for importing the agent API into an independent client.",
            },
            {
                "id": "project_resources",
                "method": "GET",
                "path": "/project/resources",
                "description": "Return an advisory selected-device memory snapshot and configured project resource limits.",
            },
            {
                "id": "list_project_inputs",
                "method": "GET",
                "path": "/project/inputs",
                "description": "List files currently available in ComfyUI's input directory with relative names and byte sizes.",
            },
            {
                "id": "delete_project_input",
                "method": "DELETE",
                "path": "/project/inputs/{name}",
                "description": "Delete one regular file from ComfyUI's input directory; nested names must be URL-encoded when needed.",
            },
            {
                "id": "upload_input_image",
                "method": "POST",
                "path": "/upload/image",
                "description": "Upload a reference image through ComfyUI's native unauthenticated multipart endpoint on this same port.",
                "content_type": "multipart/form-data",
                "request_fields": {
                    "image": "binary file (required)",
                    "type": "input (required for a reference image)",
                    "overwrite": "optional true/false",
                    "subfolder": "optional relative subfolder",
                },
                "response": "Join returned subfolder and name with '/' and pass it as the selected route's image parameter (input_image, mask_image, face_image or control_image).",
            },
            {
                "id": "list_workflows",
                "method": "GET",
                "path": "/project/workflows",
                "description": "Discover available workflows, capabilities, prompt guidance and model readiness.",
            },
            {
                "id": "list_routes",
                "method": "GET",
                "path": "/project/routes",
                "description": "Return the workflow/model route table, optionally filtered by domain, content mode, style or prompt granularity.",
            },
            {
                "id": "resolve_route",
                "method": "POST",
                "path": "/project/resolve",
                "description": "Rank ready workflows for an agent request without loading models or queueing a job.",
            },
            {
                "id": "list_models",
                "method": "GET",
                "path": "/project/models",
                "description": "Discover model and adapter capability hints, compatibility and readiness.",
            },
            {
                "id": "describe_model",
                "method": "GET",
                "path": "/project/models/{model_id}",
                "description": "Inspect one model or adapter profile used by a workflow route.",
            },
            {
                "id": "describe_workflow",
                "method": "GET",
                "path": "/project/workflows/{workflow_id}",
                "description": "Return the selected API-format graph plus its agent metadata.",
            },
            {
                "id": "queue_workflow",
                "method": "POST",
                "path": "/project/generate",
                "description": "Resolve parameters into a named graph and queue it through the serial model scheduler.",
            },
            {
                "id": "get_job",
                "method": "GET",
                "path": "/project/jobs/{prompt_id}",
                "description": "Return a compact job state and downloadable output references.",
            },
        ],
        "prompt_contract": {
            "request_shape": {
                "workflow": "workflow id from /project/workflows",
                "parameters": "object containing only bindings exposed by the selected workflow",
                "prompt_granularity": "optional advisory level; does not restrict prompt content or reject otherwise valid prompts",
                "client_id": "optional caller label",
                "dry_run": "optional boolean; resolve without queueing",
            },
            "limits": {
                "prompt_characters": MAX_PROMPT_LENGTH,
                "width_min": 64,
                "height_min": 64,
                "width_max": 2048,
                "height_max": 2048,
                "dimension_multiple": 8,
                "batch_size_max": 4,
                "steps_max": 100,
            },
        },
        "result_contract": {
            "queue_response": "contains identical prompt_id and job_id, job_url, applied_parameters, ignored_parameters and warnings",
            "timing": "Job queued_ms is time from acceptance until worker start. elapsed_ms includes model switching, generation and failure cleanup after worker start; it excludes queue time. Native-only jobs may report unknown (null) queue timing.",
            "job_states": ["queued", "running", "succeeded", "failed"],
            "job_phases": ["waiting_for_turn", "model_switch", "generation", "complete"],
            "queued_position": "queue_position is scoped to the named project queue and is zero-based",
            "download_urls": "relative /view URLs returned by /project/jobs/{prompt_id}",
            "output_metadata": "Saved PNGs keep ComfyUI's prompt metadata (prompts, seeds, models, settings) for reproducibility; input image file names in it are replaced with [input image] before the job is reported. Strip all metadata yourself before sending an image somewhere that should not see the prompt.",
        },
        "openapi": "/project/openapi.json",
        "workflows": [
            _public_entry(entry, model_registry)
            for entry in catalog["workflows"]
            if isinstance(entry, dict)
        ],
        "models": [
            _public_model(model)
            for model in model_registry
            if isinstance(model, dict) and model.get("id") in ready_model_ids
        ],
        "model_registry": {
            "default_scope": "models referenced by ready workflows",
            "candidate_access": "GET /project/models?include_candidates=true",
        },
    }


def _openapi_document() -> dict[str, Any]:
    document = {
        "openapi": "3.1.0",
        "info": {
            "title": "Project ComfyUI Agent API",
            "version": "1.5.0",
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
                        {"$ref": "#/components/parameters/ContentModeQuery"},
                        {"$ref": "#/components/parameters/StyleQuery"},
                        {"$ref": "#/components/parameters/InputTypeQuery"},
                        {"$ref": "#/components/parameters/InputModeQuery"},
                        {"$ref": "#/components/parameters/PromptGranularityQuery"},
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
                "ContentModeQuery": {
                    "name": "content_mode",
                    "in": "query",
                    "schema": {"type": "string", "maxLength": 256},
                },
                "StyleQuery": {
                    "name": "style",
                    "in": "query",
                    "schema": {"type": "string", "maxLength": 256},
                },
                "InputTypeQuery": {
                    "name": "input_type",
                    "in": "query",
                    "schema": {"type": "string", "enum": ["text", "image"]},
                },
                "InputModeQuery": {
                    "name": "input_mode",
                    "in": "query",
                    "schema": {
                        "type": "string",
                        "enum": ["text-to-image", "image-edit", "image-to-image", "inpaint", "face-reference", "openpose-control"],
                    },
                },
                "PromptGranularityQuery": {
                    "name": "prompt_granularity",
                    "in": "query",
                    "schema": {"$ref": "#/components/schemas/PromptGranularity"},
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
                "PromptGranularity": {
                    "type": "string",
                    "enum": list(PROMPT_GRANULARITY_LEVELS),
                },
                "RouteResolutionRequest": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string", "maxLength": 256},
                        "family": {"type": "string", "maxLength": 256, "description": "Exact workflow family from the catalog, e.g. sdxl-ipadapter or anima-ipadapter."},
                        "style": {"type": "string", "maxLength": 256},
                        "content_mode": {"type": "string", "maxLength": 256},
                        "input_type": {"type": "string", "enum": ["text", "image"]},
                        "input_mode": {
                            "type": "string",
                            "enum": ["text-to-image", "image-edit", "image-to-image", "inpaint", "face-reference", "openpose-control"],
                        },
                        "prompt_granularity": {"$ref": "#/components/schemas/PromptGranularity"},
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
                        "prompt_granularity": {"$ref": "#/components/schemas/PromptGranularity"},
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
                        "control_image": {
                            "type": "string",
                            "maxLength": MAX_INPUT_IMAGE_LENGTH,
                            "description": "Relative input filename for a pose/control reference consumed by the selected ControlNet workflow.",
                        },
                        "control_strength": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 10,
                        },
                        "control_start_percent": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "control_end_percent": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
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
    }
    schemas["QueuedJob"] = {"type": "object", "required": ["prompt_id", "job_id", "job_url", "state", *common], "properties": {
        **common, "prompt_id": {"type": "string", "format": "uuid"},
        "job_id": {"type": "string", "format": "uuid", "description": "Compatibility alias, identical to prompt_id."},
        "job_url": {"type": "string"}, "state": {"const": "queued"}, "workflow": {"type": "string"},
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
        if not name or name.startswith("../"):
            continue
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


async def list_routes(request: web.Request) -> web.Response:
    try:
        query = _route_query(dict(request.query))
        result = _resolve_routes(
            _read_catalog(),
            query,
            include_unready=_query_boolean(request.query.get("include_unready"), "include_unready"),
            limit=_query_limit(request.query.get("limit"), default=10),
        )
        return web.json_response(result)
    except ApiError as error:
        return _json_error(error)


async def resolve_route(request: web.Request) -> web.Response:
    try:
        body = await _read_json(request)
        query = _route_query(body)
        include_unready = body.get("include_unready", False)
        if not isinstance(include_unready, bool):
            raise ApiError(400, "include_unready must be a boolean")
        limit = _as_int("limit", body.get("limit", 1), 1, 20)
        result = _resolve_routes(
            _read_catalog(), query, include_unready=include_unready, limit=limit
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


def _validate_prompt_granularity(entry: dict[str, Any], body: dict[str, Any]) -> str | None:
    requested = body.get("prompt_granularity")
    if requested is None:
        return None
    requested = _normalise_granularity(requested)
    return requested


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
        requested_granularity = _validate_prompt_granularity(entry, body)
        graph = _prepare_graph(entry, body, fit_report)
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
        warnings = []
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
        if effective.get("cfg") == 1 and "negative_prompt" in entry["bindings"]:
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
                    "workflow": public,
                    "ready": public["ready"],
                    "missing_files": public["missing_files"],
                    "validated": valid is not None,
                    "effective_parameters": effective,
                    "applied_parameters": applied,
                    "ignored_parameters": [],
                    "resources": resources,
                    "warnings": warnings,
                    "prompt_granularity": requested_granularity,
                    "prompt": graph,
                }
            )
        if not public["ready"]:
            raise ApiError(409, "Workflow dependencies are unavailable", public)
        result = await _enqueue(PromptServer.instance, graph, client_id, workflow_id, resources, valid)
        queued = True  # the job now owns its fitted inputs and deletes them when it ends
        result["workflow"] = entry["id"]
        result["prompt_granularity"] = requested_granularity
        result["job_url"] = f"/project/jobs/{result['prompt_id']}"
        result["effective_parameters"] = effective
        result["applied_parameters"] = applied
        result["ignored_parameters"] = []
        result["resources"] = resources
        result["warnings"] = warnings
        return web.json_response(result)
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
        "/project/inputs, /project/inputs/{name}, /project/models, /project/workflows, /project/resolve, "
        "/project/generate, /upload/image and /project/jobs"
    )


_register_routes()

# ComfyUI requires this mapping for a V1 custom-node module.  This extension
# intentionally contributes routes only and does not add nodes to the graph.
NODE_CLASS_MAPPINGS: dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
WEB_DIRECTORY = None
