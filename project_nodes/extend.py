"""Extend add-on: grow an image beyond its borders on routes that edit by instruction.

The service pads the upload with flat grey, the route's model redraws the whole canvas from an instruction that
names the grey areas, and ProjectExtendPaste puts the original pixels back. The model does not keep pixel
positions (Qwen edit shrank and shifted the original by up to 21% and 190 px), so the paste first aligns the
generated canvas to the original with an affine fit over the original region, then blends the original in over a
feather band next to each new edge. When no alignment fits, the generated canvas is returned as it is and the job
reports a warning.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

SIDES = ("left", "top", "right", "bottom")
PAD_COLOUR = (128, 128, 128)
FEATHER = 64
# Mean absolute difference (0-255) over the original region after alignment. Aligned outputs measured 3.8-6.6;
# unaligned ones 41-51.
MAX_RESIDUAL = 15.0
WARNING_KEY = "project_warnings"


def write_canvas(path: Path, sides: dict[str, int], directory: Path, folder: str) -> tuple[str, tuple[int, int]]:
    """Write the first frame of an upload, orientation applied, on a grey canvas grown by `sides`."""
    with Image.open(path) as source:
        source.seek(0)
        image = ImageOps.exif_transpose(source)
        if image.mode != "RGB":
            rgba = image.convert("RGBA")
            image = Image.new("RGB", rgba.size, PAD_COLOUR)
            image.paste(rgba, mask=rgba.getchannel("A"))
        size = (image.width + sides["left"] + sides["right"], image.height + sides["top"] + sides["bottom"])
        canvas = Image.new("RGB", size, PAD_COLOUR)
        canvas.paste(image, (sides["left"], sides["top"]))
    name = f"{folder}/{uuid.uuid4().hex}.png"
    target = directory / name
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, "PNG")
    return name, size


def attach(graph: dict[str, Any], contract: dict[str, Any], canvas_name: str, region: tuple[int, int, int, int]) -> None:
    """Prefix the edit instruction and route the decoded image through ProjectExtendPaste before it is saved."""
    encoders = [node for node in graph.values() if isinstance(node, dict) and node.get("class_type") == "TextEncodeQwenImage21"]
    decoders = [node_id for node_id, node in graph.items() if isinstance(node, dict) and node.get("class_type") == "VAEDecode"]
    savers = [node for node in graph.values() if isinstance(node, dict) and str(node.get("class_type", "")).startswith("SaveImage")
              and node.get("inputs", {}).get("images") == [decoders[0], 0]] if len(decoders) == 1 else []
    if len(encoders) != 1 or len(savers) != 1:
        raise RuntimeError("Route graph does not have the single encoder, decoder and save node the extend add-on expects")
    encoder = encoders[0]["inputs"]
    encoder["prompt"] = contract["instruction"] + encoder.get("prompt", "")
    next_id = max((int(node_id) for node_id in graph if str(node_id).isdigit()), default=0) + 1
    load_id, paste_id = str(next_id), str(next_id + 1)
    left, top, width, height = region
    graph[load_id] = {"class_type": "LoadImage", "inputs": {"image": canvas_name}}
    graph[paste_id] = {"class_type": "ProjectExtendPaste",
                       "inputs": {"generated": [decoders[0], 0], "canvas": [load_id, 0], "left": left, "top": top,
                                  "width": width, "height": height, "feather": contract.get("feather", FEATHER),
                                  "max_residual": contract.get("max_residual", MAX_RESIDUAL)}}
    savers[0]["inputs"]["images"] = [paste_id, 0]


def align_and_paste(generated, canvas, region: tuple[int, int, int, int], feather: int = FEATHER,
                    max_residual: float = MAX_RESIDUAL) -> tuple[Any, dict[str, Any]]:
    """Generated and canvas are HxWx3 float arrays in 0-1; region is the original's (left, top, width, height) on the canvas.

    Returns the result at canvas size and a report: aligned (bool), residual, scale, shift and, when not aligned, reason.
    """
    import cv2
    import numpy as np

    height, width = canvas.shape[:2]
    left, top, region_width, region_height = region
    # Colour only: the Qwen-Image 2.1 VAE decodes four channels.
    generated = cv2.resize(np.ascontiguousarray(generated[..., :3], np.float32), (width, height), interpolation=cv2.INTER_LANCZOS4).clip(0, 1)
    canvas = np.ascontiguousarray(canvas[..., :3], np.float32)
    mask = np.zeros((height, width), np.uint8)
    mask[top:top + region_height, left:left + region_width] = 255
    target, moving = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in (canvas, generated))

    def residual(image) -> float:
        return float(np.abs(image[mask > 0] - canvas[mask > 0]).mean() * 255)

    # Coarse to fine (128 px, 512 px, full size): small passes find large zooms and shifts, larger ones refine them.
    warp, previous, converged = np.eye(2, 3, dtype=np.float32), None, False
    for long_side in (128, 512, max(width, height)):
        factor = min(1.0, long_side / max(width, height))
        if previous is not None and factor == previous:
            continue
        small = (max(1, round(width * factor)), max(1, round(height * factor)))
        guess = warp.copy()
        if previous is not None:
            guess[:, 2] *= factor / previous
        try:
            _, warp = cv2.findTransformECC(cv2.resize(target, small, interpolation=cv2.INTER_AREA),
                                           cv2.resize(moving, small, interpolation=cv2.INTER_AREA), guess, cv2.MOTION_AFFINE,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
                                           cv2.resize(mask, small, interpolation=cv2.INTER_NEAREST), 5)
            converged = True
        except cv2.error:
            warp = guess  # keep the coarser fit; a level too small to show the picture's structure can fail
        previous = factor
    warp[:, 2] /= previous
    reason = None if converged else "no alignment converged"
    aligned = cv2.warpAffine(generated, warp, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
    corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]], np.float32) @ warp.T
    margin = 0.01 * max(width, height)
    if reason is None and ((corners < -margin).any() or (corners[:, 0] > width + margin).any()
                           or (corners[:, 1] > height + margin).any()):
        reason = "the generated picture does not cover the canvas after alignment"
    score = residual(aligned)
    if reason is None and score > max_residual:
        reason = f"the original region still differs by {score:.1f} after alignment"
    report = {"aligned": reason is None, "residual": round(score, 1), "scale": [round(float(warp[0, 0]), 4), round(float(warp[1, 1]), 4)],
              "shift": [round(float(warp[0, 2]), 1), round(float(warp[1, 2]), 1)]}
    if reason:
        return generated, {**report, "reason": reason}
    # Original pixels inside the region, blended over `feather` px next to each side that grew.
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = np.full((height, width), np.inf, np.float32)
    for grew, edge in ((left > 0, x - left), (top > 0, y - top), (left + region_width < width, left + region_width - 1 - x),
                       (top + region_height < height, top + region_height - 1 - y)):
        if grew:
            distance = np.minimum(distance, edge)
    alpha = np.clip((distance + 1) / max(1, feather), 0, 1) * (mask > 0)
    return canvas * alpha[..., None] + aligned * (1 - alpha[..., None]), report


class ProjectExtendPaste:
    """Align the generated canvas to the original and paste the original pixels back (see module docstring)."""

    CATEGORY = "project"
    FUNCTION = "paste"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        size = {"default": 0, "min": 0, "max": 16384}
        return {"required": {"generated": ("IMAGE",), "canvas": ("IMAGE",), "left": ("INT", size), "top": ("INT", size),
                             "width": ("INT", size), "height": ("INT", size), "feather": ("INT", {"default": FEATHER, "min": 0, "max": 512}),
                             "max_residual": ("FLOAT", {"default": MAX_RESIDUAL, "min": 0.0, "max": 255.0})}}

    def paste(self, generated, canvas, left, top, width, height, feather, max_residual):
        import torch
        results, warnings = [], []
        reference = canvas[0].cpu().numpy()
        for image in generated:
            result, report = align_and_paste(image.cpu().numpy(), reference, (left, top, width, height), feather, max_residual)
            results.append(torch.from_numpy(result.astype("float32")))
            if not report["aligned"]:
                warnings.append({"code": "extend_original_not_kept", "parameter": "extend", **report,
                                 "message": f"The extended picture is returned as the model drew it: {report['reason']}, so the "
                                            "original pixels were not pasted back and may differ slightly."})
        return {"ui": {WARNING_KEY: warnings}, "result": (torch.stack(results),)}
