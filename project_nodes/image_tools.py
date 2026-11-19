"""Image tools: crop and resize without a model.

Agents often have no image library, so the service offers the plain pixel operations itself. They run in the API
process with Pillow and never enter the job queue: a queued job on another route would unload the resident model
(runtime_guard), and a crop needs no GPU. Only super-resolution needs a model; that is a queued route of its own.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from PIL import Image, ImageOps

ANCHORS = ("center", "top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right")
MAX_SIDE = 16384
# Per framing: crop size as a multiple of the DWPose face box (brows to chin), and where the head's centre sits,
# as a fraction of the crop height from the top. Chosen by eye on an anime full figure, a seated figure and a photo.
FRAMINGS = {"head": (1.9, 0.5), "head-and-shoulders": (3.0, 0.4), "upper-body": (4.5, 0.3)}


class ToolError(Exception):
    def __init__(self, message: str, code: str = "invalid_parameter", details: Any = None):
        super().__init__(message)
        self.message, self.code, self.details = message, code, details


def _int(name: str, value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ToolError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _number(name: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= maximum:
        raise ToolError(f"{name} must be a number from {minimum} to {maximum}")
    return float(value)


def aspect_ratio(value: Any) -> float:
    """'16:9', '4/3' or a number (width / height)."""
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[:/x]\s*(\d+(?:\.\d+)?)\s*", value)
        if not match or float(match.group(2)) == 0:
            raise ToolError("crop.aspect must look like '16:9', '1:1' or a number such as 1.5 (width / height)")
        value = float(match.group(1)) / float(match.group(2))
    return _number("crop.aspect", value, 0.1, 10.0)


def _anchored(size: tuple[int, int], box_size: tuple[int, int], anchor: str) -> tuple[int, int, int, int]:
    width, height = size
    box_width, box_height = box_size
    horizontal = 0.0 if "left" in anchor else 1.0 if "right" in anchor else 0.5
    vertical = 0.0 if "top" in anchor else 1.0 if "bottom" in anchor else 0.5
    left, top = round((width - box_width) * horizontal), round((height - box_height) * vertical)
    return left, top, left + box_width, top + box_height


def aspect_box(size: tuple[int, int], aspect: float, anchor: str = "center") -> tuple[int, int, int, int]:
    """Largest box of the given aspect ratio inside the image, placed by anchor."""
    width, height = size
    if width / height > aspect:
        box = (max(1, round(height * aspect)), height)
    else:
        box = (width, max(1, round(width / aspect)))
    return _anchored(size, box, anchor)


def face_box_crop(size: tuple[int, int], face: tuple[float, float, float, float], framing: str, aspect: float) -> tuple[int, int, int, int]:
    """Box of the given aspect around a normalised face box (brows to chin), shrunk to fit inside the image."""
    width, height = size
    x0, y0, x1, y1 = face[0] * width, face[1] * height, face[2] * width, face[3] * height
    scale, head_at = FRAMINGS[framing]
    side = scale * max(x1 - x0, y1 - y0)  # the square the framing needs
    box_width, box_height = (side * aspect, side) if aspect >= 1 else (side, side / aspect)
    fit = min(1.0, width / box_width, height / box_height)
    box_width, box_height = max(1, round(box_width * fit)), max(1, round(box_height * fit))
    centre_x, head_y = (x0 + x1) / 2, (y0 + y1) / 2 - 0.25 * (y1 - y0)  # face points start at the brows; hair is above
    top = head_y - head_at * box_height
    left = min(max(round(centre_x - box_width / 2), 0), width - box_width)
    top = min(max(round(top), 0), height - box_height)
    return left, top, left + box_width, top + box_height


def crop_box(size: tuple[int, int], crop: Any, find_face: Callable[[], tuple[float, float, float, float] | None] | None = None) -> tuple[int, int, int, int]:
    if not isinstance(crop, dict):
        raise ToolError("crop must be an object: {x, y, width, height} in pixels, {aspect, anchor} or {around: face}")
    if "around" in crop:
        if crop["around"] != "face" or set(crop) - {"around", "framing", "aspect"}:
            raise ToolError("A subject crop is {\"around\": \"face\"} with optional framing and aspect")
        framing = crop.get("framing", "head-and-shoulders")
        if framing not in FRAMINGS:
            raise ToolError(f"crop.framing must be one of {', '.join(FRAMINGS)}")
        face = find_face() if find_face else None
        if face is None:
            raise ToolError("No face was found to crop around; crop by pixels or by aspect and anchor instead", "face_not_found")
        return face_box_crop(size, face, framing, aspect_ratio(crop.get("aspect", "1:1")))
    if "aspect" in crop:
        if set(crop) - {"aspect", "anchor"}:
            raise ToolError("crop takes either {x, y, width, height} or {aspect, anchor}, not both")
        anchor = crop.get("anchor", "center")
        if anchor not in ANCHORS:
            raise ToolError(f"crop.anchor must be one of {', '.join(ANCHORS)}")
        return aspect_box(size, aspect_ratio(crop["aspect"]), anchor)
    if set(crop) != {"x", "y", "width", "height"}:
        raise ToolError("crop needs x, y, width and height in pixels (or aspect and an optional anchor)")
    x, y = _int("crop.x", crop["x"], 0, MAX_SIDE), _int("crop.y", crop["y"], 0, MAX_SIDE)
    width, height = _int("crop.width", crop["width"], 1, MAX_SIDE), _int("crop.height", crop["height"], 1, MAX_SIDE)
    if x + width > size[0] or y + height > size[1]:
        raise ToolError(f"The crop box reaches outside the {size[0]}x{size[1]} image", "crop_out_of_bounds",
                        {"image_size": list(size), "crop": [x, y, width, height]})
    return x, y, x + width, y + height


def resize_target(size: tuple[int, int], resize: Any, aspect: float | None = None) -> tuple[tuple[int, int], tuple[int, int, int, int] | None]:
    """Target size, and a centre crop when width and height together ask for another aspect ratio (never stretch).

    aspect is the shape a crop asked for: a whole-pixel crop only approximates it, and one given side follows it.
    """
    if not isinstance(resize, dict) or not resize:
        raise ToolError("resize must be an object with width and/or height, or scale, or max_side")
    width, height = size
    keys = set(resize)
    if keys <= {"width", "height"}:
        target_width = _int("resize.width", resize["width"], 1, MAX_SIDE) if "width" in resize else None
        target_height = _int("resize.height", resize["height"], 1, MAX_SIDE) if "height" in resize else None
        if target_width and target_height:
            if abs(target_width / target_height - width / height) > 0.005:
                return (target_width, target_height), aspect_box(size, target_width / target_height)
            return (target_width, target_height), None
        shape = aspect or width / height
        if target_width:
            return (target_width, max(1, round(target_width / shape))), None
        return (max(1, round(target_height * shape)), target_height), None
    if keys == {"scale"}:
        factor = _number("resize.scale", resize["scale"], 0.01, 8.0)
    elif keys == {"max_side"}:
        factor = min(1.0, _int("resize.max_side", resize["max_side"], 1, MAX_SIDE) / max(width, height))
    else:
        raise ToolError("resize takes width and/or height, or scale, or max_side (one form only)")
    return (max(1, round(width * factor)), max(1, round(height * factor))), None


def transform(source: Image.Image, crop: Any = None, resize: Any = None, *, max_pixels: int,
              find_face: Callable[[], tuple[float, float, float, float] | None] | None = None) -> tuple[Image.Image, list[dict[str, Any]], list[dict[str, Any]]]:
    """Crop then resize an opened image. Returns the result, the operations applied and warnings."""
    if crop is None and resize is None:
        raise ToolError("Send crop, resize or both")
    frames = getattr(source, "n_frames", 1)
    source.seek(0)
    image = ImageOps.exif_transpose(source)
    if image.mode not in {"RGB", "RGBA", "L", "LA"}:
        image = image.convert("RGBA" if "transparency" in source.info or "A" in image.getbands() else "RGB")
    operations, warnings = [], []
    if frames > 1:
        warnings.append({"code": "input_first_frame", "message": "The image is animated; only its first frame was used."})
    aspect, found = None, {}
    if crop is not None:
        def finder():
            found["face"] = find_face() if find_face else None
            return found["face"]
        box = crop_box(image.size, crop, finder)
        operation = {"crop": list(box)}
        if found.get("face"):  # lets an agent that cannot see the image check the framing against the face
            operation["face_box"] = [round(value * image.size[index % 2]) for index, value in enumerate(found["face"])]
        operations.append(operation)
        image = image.crop(box)
        if isinstance(crop, dict) and ("aspect" in crop or "around" in crop):
            aspect = aspect_ratio(crop.get("aspect", "1:1"))
    if resize is not None:
        target, cover = resize_target(image.size, resize, aspect)
        if target[0] * target[1] > max_pixels:
            raise ToolError(f"{target[0]}x{target[1]} is over the {max_pixels} pixel limit", "input_too_large",
                            {"size": list(target), "max_pixels": max_pixels})
        if cover:
            image = image.crop(cover)
            operations.append({"crop": list(cover), "reason": "centre crop to the requested width:height, so nothing is stretched"})
        enlarged = target[0] * target[1] > image.width * image.height
        if target != image.size:
            image = image.resize(target, Image.Resampling.LANCZOS)
        operations.append({"resize": list(target), "method": "lanczos"})
        if enlarged:
            warnings.append({"code": "upscaled_without_model",
                             "message": "The image was enlarged by interpolation (Lanczos): it is bigger but gains no detail, "
                                        "so it looks soft. This server has no model upscaler yet."})
    if image.width * image.height > max_pixels:
        raise ToolError(f"{image.width}x{image.height} is over the {max_pixels} pixel limit", "input_too_large")
    return image, operations, warnings


def output_name(source_name: str, size: tuple[int, int], token: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", source_name.rsplit("/", 1)[-1].rsplit(".", 1)[0]).strip("-.")[:60] or "image"
    return f"{stem}_{size[0]}x{size[1]}_{token}.png"

