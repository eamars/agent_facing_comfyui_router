"""Pose add-on: one optional `pose` parameter for every route.

A pose is a library view id or an uploaded image. Either becomes a DWPose
skeleton once (cached by the source's content hash) and the route's family
adapter applies that skeleton to the route's own graph. The library index is
workflows/poses.json; its sketches are fetched from the gallery on first use
and cached under ComfyUI/input/_poses, which the input listing hides.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import threading
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATH = PROJECT_ROOT / "workflows" / "poses.json"
CACHE_DIR = "_poses"
DWPOSE_DIR = PROJECT_ROOT / "ComfyUI" / "custom_nodes" / "comfyui_controlnet_aux" / "ckpts" / "yzd-v" / "DWPose"
DWPOSE_FILES = ("yolox_l.onnx", "dw-ll_ucoco_384.onnx")
# A person counts when DWPose places at least this many of the 18 body points.
MIN_BODY_POINTS = 8
# A hand is drawn only when its points average at least this score (counting unseen points as 0).
MIN_HAND_SCORE = 0.4
# Part of every skeleton cache key: bump it when extraction changes so stale skeletons are not reused.
EXTRACTOR_VERSION = "dwpose-l384-filled-3"
# DWPose's own cut-off for a visible keypoint; whole-body points below it are not stored.
MIN_POINT_SCORE = 0.3
MAX_FETCH_BYTES = 8_000_000
FETCH_TIMEOUT_SECONDS = 20
ANGLES = {"front": "front", "diag": "three-quarter", "side": "side", "high": "from above", "low": "from below"}


class PoseError(Exception):
    def __init__(self, status: int, message: str, code: str, details: Any = None):
        super().__init__(message)
        self.status, self.message, self.code, self.details = status, message, code, details


def input_root() -> Path:
    try:
        import folder_paths
        return Path(folder_paths.get_input_directory()).resolve()
    except ImportError:  # offline scripts run outside the ComfyUI process
        return (PROJECT_ROOT / "ComfyUI" / "input").resolve()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


# ---------------------------------------------------------------- library

_library_lock = threading.Lock()
_library: dict[str, Any] = {"mtime": None}


def library() -> dict[str, Any]:
    """The pose index with lookup tables, reloaded when poses.json changes."""
    with _library_lock:
        mtime = LIBRARY_PATH.stat().st_mtime_ns if LIBRARY_PATH.is_file() else None
        if _library["mtime"] != mtime:
            data = json.loads(LIBRARY_PATH.read_text(encoding="utf-8")) if mtime else {"poses": []}
            poses = data.get("poses", [])
            views = {view["id"]: (pose, angle) for pose in poses for angle, view in pose["views"].items()}
            documents = {pose["id"]: set(_tokens(" ".join(str(pose.get(key, "")) for key in
                                                           ("name", "name_zh", "category", "category_zh", "framing"))))
                         for pose in poses}
            frequency: dict[str, int] = {}
            for tokens in documents.values():
                for token in tokens:
                    frequency[token] = frequency.get(token, 0) + 1
            _library.update(mtime=mtime, data=data, poses={pose["id"]: pose for pose in poses}, views=views,
                            documents=documents,
                            idf={token: math.log(1 + len(documents) / count) for token, count in frequency.items()})
        return _library


def lookup(value: str) -> tuple[dict[str, Any], str] | None:
    """(pose, angle) for a view id, or a pose id meaning its front view."""
    lib = library()
    if value in lib["views"]:
        return lib["views"][value]
    pose = lib["poses"].get(value)
    if pose and pose["views"]:
        return pose, "front" if "front" in pose["views"] else next(iter(pose["views"]))
    return None


def looks_like_library_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9]+_[A-Za-z0-9_]+", value))


_STOP = set("a an the with and of in on at to her his their its is are be she he one view shot angle camera from both "
            "girl woman man boy person people anime pose posing".split())
_SYNONYMS = {
    "seated": "sit", "sitting": "sit", "sits": "sit", "kneeling": "kneel", "kneels": "kneel", "crouch": "squat",
    "crouching": "squat", "squatting": "squat", "lying": "lie", "laying": "lie", "lay": "lie", "reclining": "lie",
    "standing": "stand", "stands": "stand", "hugging": "hug", "hugged": "hug", "embrace": "hug", "embracing": "hug",
    "hands": "hand", "arms": "arm", "legs": "leg", "knees": "knee", "raised": "raise", "raising": "raise",
    "lifting": "raise", "lift": "raise", "lifted": "raise", "overhead": "over head", "hip": "waist", "hips": "waist",
    "fingers": "finger", "lips": "lip", "mouth": "lip", "looking": "look", "shoulders": "shoulder",
    "cheeks": "cheek", "wariza": "w-sit",
}
_ANGLE_WORDS = (
    ("three-quarter", "diag"), ("three quarter", "diag"), ("3/4", "diag"), ("diagonal", "diag"), ("斜侧", "diag"),
    ("side view", "side"), ("from the side", "side"), ("profile", "side"), ("侧面", "side"),
    ("from above", "high"), ("high angle", "high"), ("top-down", "high"), ("bird's-eye", "high"), ("俯视", "high"),
    ("from below", "low"), ("low angle", "low"), ("worm's-eye", "low"), ("仰视", "low"),
    ("front view", "front"), ("正面", "front"),
)


def _tokens(text: str) -> list[str]:
    text = text.lower()
    tokens = []
    for word in re.findall(r"[a-z][a-z\-]*", text):
        for token in _SYNONYMS.get(word, word).split():
            if token not in _STOP and len(token) > 1:
                tokens.append(token)
    han = "".join(re.findall(r"[一-鿿]", text))
    tokens += [han[i:i + 2] for i in range(len(han) - 1)] or list(han)
    return tokens


def search(query: str = "", people: int | None = None, angle: str | None = None, limit: int = 8) -> dict[str, Any]:
    """Rank library poses for an agent to choose from; the agent, not this ranking, picks the pose."""
    lib = library()
    text = query.lower()
    angle_hint = None
    for word, word_angle in _ANGLE_WORDS:
        if word in text:
            angle_hint, text = word_angle, text.replace(word, " ")
            break
    terms = {token for token in _tokens(text) if token in lib["idf"]}
    total = sum(lib["idf"][token] for token in terms)
    ranked = []
    for pose in lib["data"].get("poses", []) if lib.get("data") else []:
        if not pose["views"] or (people and pose["people"] != people):
            continue
        if angle and angle not in pose["views"]:
            continue
        score = sum(lib["idf"][token] for token in terms & lib["documents"][pose["id"]]) / total if total else 0.0
        if query.strip() and score == 0:
            continue
        ranked.append((score, pose))
    ranked.sort(key=lambda item: -item[0])
    return {
        "query": query,
        "angle_hint": angle or angle_hint,
        "total_matches": len(ranked),
        "results": [{
            "id": pose["id"], "name": pose["name"], "name_zh": pose["name_zh"], "category": pose["category"],
            "framing": pose["framing"], "people": pose["people"],
            "views": {view_angle: view["id"] for view_angle, view in pose["views"].items()},
            "match_score": round(score, 2),
        } for score, pose in ranked[:limit]],
    }


def library_image(pose: dict[str, Any], angle: str) -> str:
    """Input-relative path of a library sketch, fetched from the gallery on first use."""
    view = pose["views"][angle]
    path = view["path"]
    if not re.fullmatch(r"img/[A-Za-z0-9_]+\.(png|jpg)", path):
        raise PoseError(500, f"Pose library path is invalid: {path}", "pose_library_invalid")
    name = f"{CACHE_DIR}/library/{view['id']}{Path(path).suffix}"
    target = input_root() / name
    if target.is_file():
        return name
    origin = library()["data"]["source"]["origin"].rstrip("/")
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{origin}/{path}", headers={"User-Agent": "Mozilla/5.0"}),
                                    timeout=FETCH_TIMEOUT_SECONDS) as response:
            data = response.read(MAX_FETCH_BYTES + 1)
    except OSError as exc:
        raise PoseError(502, f"Pose sketch {view['id']} could not be fetched from the pose gallery; retry later",
                        "pose_fetch_failed", str(exc)) from exc
    if len(data) > MAX_FETCH_BYTES:
        raise PoseError(502, f"Pose sketch {view['id']} is larger than expected", "pose_fetch_failed")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except Exception as exc:
        raise PoseError(502, f"Pose sketch {view['id']} is not a readable image", "pose_fetch_failed", str(exc)) from exc
    _write_atomic(target, data)
    return name


# ---------------------------------------------------------------- skeleton

_detector_lock = threading.Lock()
_detector = None


def dwpose_missing_files() -> list[str]:
    return [str((DWPOSE_DIR / name).relative_to(PROJECT_ROOT / "ComfyUI")).replace("\\", "/")
            for name in DWPOSE_FILES if not (DWPOSE_DIR / name).is_file()]


def _dwpose():
    """DWPose on CPU: extraction runs in the API process and must not take VRAM from a running job."""
    global _detector
    if _detector is None:
        missing = dwpose_missing_files()
        if missing:
            raise PoseError(409, "Pose extraction weights are missing; run scripts/download_models.ps1 -Profile pose",
                            "pose_unavailable", missing)
        import onnxruntime
        from custom_controlnet_aux.dwpose.wholebody import Wholebody
        from custom_controlnet_aux.dwpose.util import guess_onnx_input_shape_dtype
        body = Wholebody()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 4
        body.det = onnxruntime.InferenceSession(str(DWPOSE_DIR / DWPOSE_FILES[0]), options, providers=["CPUExecutionProvider"])
        body.pose = onnxruntime.InferenceSession(str(DWPOSE_DIR / DWPOSE_FILES[1]), options, providers=["CPUExecutionProvider"])
        body.det_filename, body.pose_filename = DWPOSE_FILES
        body.det_model_type = body.pose_model_type = "ort"
        body.pose_input_size, _ = guess_onnx_input_shape_dtype(DWPOSE_FILES[1])
        _detector = body
    return _detector


def _filled_figure(rgb):
    """Line art with the figure painted skin-tone and shaded at the lines.

    DWPose is trained on photos: on bare line art the detector misses the figure and joint estimates collapse
    onto the torso (side-view mannequin scores ~0.2). Filled, the same sketch scores ~0.6 with joints in place.
    """
    import cv2
    import numpy as np
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = cv2.dilate((gray < 200).astype(np.uint8), np.ones((3, 3), np.uint8))
    height, width = gray.shape
    outside = ((1 - ink) * 255).astype(np.uint8)
    for corner in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        if outside[corner[1], corner[0]] == 255:
            cv2.floodFill(outside, np.zeros((height + 2, width + 2), np.uint8), corner, 128)
    filled = np.full_like(rgb, 255)
    filled[outside != 128] = (224, 172, 140)
    shade = np.clip(cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 2), 0, 1)
    return (filled * (1 - 0.7 * shade)[..., None]).astype(np.uint8)


def _estimate(image: Image.Image) -> tuple[list, str, list]:
    """DWPose people for an RGB image, how they were found, and each person's raw COCO-WholeBody 133 points.

    Photos go through the person detector. When it finds no one (line art), the figure is filled in and
    detected again; if still no one, the estimator takes the whole frame as one person.
    """
    import numpy as np
    from custom_controlnet_aux.dwpose.dw_onnx.cv_ox_det import inference_detector
    from custom_controlnet_aux.dwpose.dw_onnx.cv_ox_pose import inference_pose
    from custom_controlnet_aux.dwpose.wholebody import Wholebody
    rgb = np.asarray(image)
    with _detector_lock:
        body = _dwpose()
        candidates = []
        for found, view in (("image", rgb), ("filled-figure", None)):
            view = _filled_figure(rgb) if view is None else view
            bgr = view[:, :, ::-1].copy()
            boxes = inference_detector(body.det, bgr, detect_classes=[0], dtype=np.float32)
            detected = boxes is not None and len(boxes) > 0
            keypoints, scores = inference_pose(body.pose, boxes if detected else [], bgr, body.pose_input_size,
                                               dtype=np.float32)
            candidates.append((detected, float(scores[:, :18].mean()), f"{found}-{'detector' if detected else 'full-frame'}",
                               keypoints, scores))
            if detected and found == "image":
                break  # a photo or illustration the detector sees needs no fallback
        # Prefer a detected result (it can hold several people), then the more confident whole-frame estimate.
        _, _, found, keypoints, scores = max(candidates, key=lambda item: (item[0], item[1]))
    # Same COCO-wholebody -> OpenPose reordering as Wholebody.__call__, with the neck inserted.
    info = np.concatenate((keypoints, scores[..., None]), axis=-1)
    neck = np.mean(info[:, [5, 6]], axis=1)
    neck[:, 2:4] = np.logical_and(info[:, 5, 2:4] > 0.3, info[:, 6, 2:4] > 0.3).astype(int)
    info = np.insert(info, 17, neck, axis=1)
    info[:, [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]] = info[:, [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]]
    wholebody = [np.concatenate((points, point_scores[:, None]), axis=-1) for points, point_scores in zip(keypoints, scores)]
    return Wholebody.format_result(info), found, wholebody


def _points(part, width: int, height: int, minimum_mean_score: float = 0.0) -> list[list[float] | None] | None:
    if part is None:
        return None
    if minimum_mean_score and sum(point.score for point in part if point) / len(part) < minimum_mean_score:
        return None  # a half-seen hand draws as stray sticks that steer the ControlNet wrong
    points = []
    for point in part:
        if point is None or not (0 <= point.x <= width and 0 <= point.y <= height):
            points.append(None)  # off-canvas guesses would distort the letterboxed render
        else:
            points.append([round(point.x / width, 5), round(point.y / height, 5), round(float(point.score), 3)])
    return points if any(points) else None


def _wholebody_points(raw, width: int, height: int) -> list[list[float] | None]:
    """All 133 COCO-WholeBody points (body, feet, face, hands), normalised, None where unseen or off-canvas."""
    return [[round(float(x) / width, 5), round(float(y) / height, 5), round(float(score), 3)]
            if score >= MIN_POINT_SCORE and 0 <= x <= width and 0 <= y <= height else None
            for x, y, score in raw]


def skeleton(source_name: str) -> dict[str, Any]:
    """DWPose skeleton of an input image, cached by content hash. Raises pose_not_detected without a usable body."""
    root = input_root()
    data = (root / source_name).read_bytes()
    key = hashlib.sha256(EXTRACTOR_VERSION.encode() + data).hexdigest()[:20]
    cached = root / CACHE_DIR / "skeletons" / f"{key}.json"
    if cached.is_file():
        result = json.loads(cached.read_text(encoding="utf-8"))
    else:
        with Image.open(io.BytesIO(data)) as source:
            source.seek(0)
            image = ImageOps.exif_transpose(source)
            if image.mode in {"RGBA", "LA", "P"}:  # transparent line art must sit on white, not black
                image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image.convert("RGBA"))
            image = image.convert("RGB")
        width, height = image.size
        people, found, wholebody = _estimate(image)
        result = {"version": 1, "key": key, "width": width, "height": height, "found_by": found, "people": []}
        for person, raw in zip(people, wholebody):
            body = _points(person.body.keypoints, width, height) or [None] * 18
            if sum(point is not None for point in body) < MIN_BODY_POINTS:
                continue
            result["people"].append({"body": body, "left_hand": _points(person.left_hand, width, height, MIN_HAND_SCORE),
                                     "right_hand": _points(person.right_hand, width, height, MIN_HAND_SCORE),
                                     "wholebody": _wholebody_points(raw, width, height)})
        _write_atomic(cached, json.dumps(result).encode())
    if not result["people"]:
        raise PoseError(422, "No human body pose was detected in the pose image; use a clearer full-body image",
                        "pose_not_detected", {"min_body_points": MIN_BODY_POINTS})
    return result


def letterbox(pose: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    """People with points mapped (normalised) onto a width x height canvas, whole pose kept, aspect preserved."""
    scale = min(width / pose["width"], height / pose["height"])
    offset_x, offset_y = (width - pose["width"] * scale) / 2, (height - pose["height"] * scale) / 2

    def place(points):
        if points is None:
            return None
        return [None if p is None else [(p[0] * pose["width"] * scale + offset_x) / width,
                                         (p[1] * pose["height"] * scale + offset_y) / height, p[2]] for p in points]

    return [{part: place(person.get(part)) for part in ("body", "left_hand", "right_hand", "wholebody")}
            for person in pose["people"]]


def _draw_openpose(people: list[dict[str, Any]], width: int, height: int):
    """OpenPose-18 body and hands in OpenPose colours with xinsir stick width (what the Xinsir ControlNet reads)."""
    from custom_controlnet_aux.dwpose import draw_poses
    from custom_controlnet_aux.dwpose.types import BodyResult, Keypoint, PoseResult

    def keypoints(points):
        if points is None:
            return None
        return [None if p is None else Keypoint(p[0], p[1], p[2], index) for index, p in enumerate(points)]

    results = []
    for person in people:
        body = keypoints(person["body"])
        results.append(PoseResult(BodyResult(body, sum(p.score for p in body if p), len(body)),
                                  keypoints(person["left_hand"]), keypoints(person["right_hand"]), None))
    return draw_poses(results, height, width, draw_body=True, draw_hand=True, draw_face=False, xinsr_stick_scaling=True)


# COCO-17 limbs and colours of the thin whole-body skeleton Anima Control Pose was trained on
# (its node's "R0_thin" style): 2 px limbs, white body joints, small white feet and face dots, cyan hand dots.
_COCO_LIMBS = ((5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
               (5, 6), (11, 12), (5, 11), (6, 12), (0, 5), (0, 6))
_COCO_LIMB_COLOURS = ((255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0), (0, 255, 0), (0, 255, 128), (0, 255, 255),
                      (0, 128, 255), (0, 0, 255), (128, 0, 255), (255, 0, 255), (255, 0, 128), (180, 180, 180), (220, 220, 220))


def _draw_wholebody_thin(people: list[dict[str, Any]], width: int, height: int):
    import cv2
    import numpy as np
    canvas = np.zeros((height, width, 3), np.uint8)
    for person in people:
        points = [None if p is None else (int(p[0] * width), int(p[1] * height)) for p in person.get("wholebody") or []]
        if not points:
            continue
        for (start, end), colour in zip(_COCO_LIMBS, _COCO_LIMB_COLOURS):
            if points[start] and points[end]:
                cv2.line(canvas, points[start], points[end], colour, 2)
        for index, point in enumerate(points):
            if point is None:
                continue
            if index < 17:
                cv2.circle(canvas, point, 3, (255, 255, 255), -1)
            elif index < 23:  # feet
                cv2.circle(canvas, point, 1, (255, 255, 255), -1)
            elif index < 91:  # face
                cv2.circle(canvas, point, 2, (255, 255, 255), -1)
            else:  # hands
                cv2.circle(canvas, point, 2, (0, 255, 255), -1)
    return canvas


_RENDER_STYLES = {"openpose": _draw_openpose, "wholebody-thin": _draw_wholebody_thin}


def render(pose: dict[str, Any], width: int, height: int, style: str = "openpose") -> str:
    """Input-relative path of the skeleton drawn in the adapter's style for a width x height output."""
    name = f"{CACHE_DIR}/rendered/{pose['key']}_{style}_{width}x{height}.png"
    target = input_root() / name
    if target.is_file():
        return name
    canvas = _RENDER_STYLES[style](letterbox(pose, width, height), width, height)
    buffer = io.BytesIO()
    Image.fromarray(canvas).save(buffer, "PNG")
    _write_atomic(target, buffer.getvalue())
    return name


def output_size(pose: dict[str, Any], pixels: int, multiple: int = 64) -> tuple[int, int]:
    """Output size that follows the pose image's aspect at the route's default pixel count."""
    aspect = min(2.0, max(0.5, pose["width"] / pose["height"]))
    width = max(multiple, round(math.sqrt(pixels * aspect) / multiple) * multiple)
    height = max(multiple, round(math.sqrt(pixels / aspect) / multiple) * multiple)
    return width, height


# ---------------------------------------------------------------- family adapters

def adapter_for(catalog: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any] | None:
    for adapter_id, adapter in catalog.get("pose_adapters", {}).items():
        if entry.get("family") in adapter.get("families", []):
            return {"id": adapter_id, **adapter}
    return None


def _attach_controlnet(graph: dict[str, Any], adapter: dict[str, Any], skeleton_image: str, strength: float) -> None:
    samplers = [node for node in graph.values() if isinstance(node, dict) and node.get("class_type") == "KSampler"]
    decoders = [node for node in graph.values() if isinstance(node, dict) and node.get("class_type") == "VAEDecode"]
    if len(samplers) != 1 or not decoders:
        raise PoseError(500, "Route graph does not have the single KSampler the ControlNet pose adapter expects",
                        "pose_adapter_failed")
    sampler = samplers[0]["inputs"]
    next_id = max((int(node_id) for node_id in graph if str(node_id).isdigit()), default=0) + 1
    load_id, net_id, apply_id = (str(next_id + offset) for offset in range(3))
    graph[load_id] = {"class_type": "LoadImage", "inputs": {"image": skeleton_image}}
    graph[net_id] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": adapter["controlnet"]}}
    graph[apply_id] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
        "positive": copy.deepcopy(sampler["positive"]), "negative": copy.deepcopy(sampler["negative"]),
        "control_net": [net_id, 0], "image": [load_id, 0], "strength": strength,
        "start_percent": 0.0, "end_percent": adapter.get("end_percent", 1.0),
        "vae": copy.deepcopy(decoders[0]["inputs"]["vae"]),
    }}
    sampler["positive"], sampler["negative"] = [apply_id, 0], [apply_id, 1]


def _attach_anima_control(graph: dict[str, Any], adapter: dict[str, Any], skeleton_image: str, strength: float) -> None:
    """Anima Control Pose: its LoRA plus the VAE-encoded skeleton added at the DiT patch embed, on the sampler's model."""
    samplers = [node for node in graph.values() if isinstance(node, dict) and node.get("class_type") == "KSampler"]
    decoders = [node for node in graph.values() if isinstance(node, dict) and node.get("class_type") == "VAEDecode"]
    if len(samplers) != 1 or not decoders:
        raise PoseError(500, "Route graph does not have the single KSampler the Anima pose adapter expects",
                        "pose_adapter_failed")
    sampler = samplers[0]["inputs"]
    next_id = max((int(node_id) for node_id in graph if str(node_id).isdigit()), default=0) + 1
    lora_id, load_id, encode_id, apply_id = (str(next_id + offset) for offset in range(4))
    # The LoRA and the control embedder are trained together and share one file; the LoRA stays at 1.0
    # because lowering it loses the pose (measured), so pose_strength scales the control alone.
    graph[lora_id] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": copy.deepcopy(sampler["model"]), "lora_name": adapter["lora"], "strength_model": 1.0}}
    graph[load_id] = {"class_type": "LoadImage", "inputs": {"image": skeleton_image}}
    graph[encode_id] = {"class_type": "VAEEncode", "inputs": {
        "pixels": [load_id, 0], "vae": copy.deepcopy(decoders[0]["inputs"]["vae"])}}
    graph[apply_id] = {"class_type": "AnimaControlApply", "inputs": {
        "model": [lora_id, 0], "control_latent": [encode_id, 0], "control_embedder_path": adapter["lora"],
        "strength": strength}}
    sampler["model"] = [apply_id, 0]


_MECHANISMS = {"controlnet-openpose": _attach_controlnet, "anima-control-lora": _attach_anima_control}


def attach(graph: dict[str, Any], adapter: dict[str, Any], skeleton_image: str, strength: float) -> None:
    _MECHANISMS[adapter["mechanism"]](graph, adapter, skeleton_image, strength)

