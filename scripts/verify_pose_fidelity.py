"""Measure how closely live renders follow a requested pose.

Run with the ComfyUI interpreter against a running server:
    ComfyUI\\.venv\\Scripts\\python.exe scripts\\verify_pose_fidelity.py [--set sdxl|anima] [--base http://127.0.0.1:8191] [--out DIR]

Each case renders once with the pose and once without (same seed). DWPose reads the output and the score is
PCK@0.1: the share of requested body points whose detected point lies within 10% of the requested figure's
size. The no-pose render is the baseline the pose has to beat. Writes a contact sheet and report.json to --out.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ComfyUI" / "custom_nodes" / "comfyui_controlnet_aux" / "src")]

from PIL import Image, ImageDraw  # noqa: E402

from project_nodes import pose  # noqa: E402

PROMPT = "1girl, solo, {outfit}, simple background, full body, masterpiece, best quality"
# Pose test subject: Xiaoman, a prompt-only character (no LoRA, so no pose prior of her own).
XIAOMAN = ("masterpiece, best quality, 1girl, solo, adult, original character, chinese young woman, black hair, bob cut, "
           "chin-length hair, grey eyes, calm face, subtle smirk, slim, white t-shirt, light beige overshirt, open shirt, "
           "dark straight-leg trousers, white canvas sneakers, full body, plain light grey background, simple background")
CASE_SETS = {  # set -> [(workflow, pose, prompt)]
    "sdxl": [(workflow, pose_value, PROMPT.format(outfit=outfit)) for workflow, pose_value, outfit in (
        ("janku-illustrious", "full_single_P035_side", "kimono"),
        ("janku-illustrious", "full_single_P012_front", "silver armor"),
        ("janku-illustrious", "full_single_P015_low", "office suit"),
        ("janku-illustrious", "full_single_P049_high", "pajamas"),
        ("janku-illustrious", "full_single_P066_diag", "streetwear hoodie"),
        ("janku-illustrious", "full_single_P041_front", "school uniform"),
        ("janku-illustrious", "full_single_P030_front", "maid dress"),
        ("one-obsession", "full_sexy_P013_diag", "evening dress"),
        ("janku-v5-illustrious", "upper80v2_P010_front", "sweater"),
        ("animagine-xl-3.1-nsfw", "full_single_P078_diag", "swimsuit"),
        ("janku-illustrious", "tests/fixtures/reference_character.webp", "gothic dress"),
    )],
    "anima": [("anima-turbo-v1-1", pose_value, XIAOMAN) for pose_value in (
        "full_single_P035_side", "full_single_P012_front", "full_single_P015_low", "full_single_P049_high",
        "full_single_P066_diag", "full_single_P030_front", "tests/fixtures/reference_character.webp")],
}


def http(base: str, path: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None,
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{path}: {exc.code} {exc.read().decode()[:400]}")


def upload(base: str, path: Path) -> str:
    boundary = "posefidelity"
    data = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"pose_fidelity_{path.name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + \
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n" \
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(base + "/upload/image", data, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    result = json.load(urllib.request.urlopen(request, timeout=60))
    return "/".join(part for part in (result.get("subfolder"), result["name"]) if part)


def render(base: str, workflow: str, prompt: str, seed: int, pose_value: str | None,
           extra: dict | None = None) -> tuple[Image.Image, dict]:
    parameters = {"prompt": prompt, "seed": seed, **(extra or {})}
    if pose_value:
        parameters["pose"] = pose_value
    job = http(base, "/project/generate", {"workflow": workflow, "parameters": parameters})
    while True:
        state = http(base, job["job_url"])
        if state["state"] in {"succeeded", "failed"}:
            break
        time.sleep(1)
    if state["state"] != "succeeded":
        raise SystemExit(f"{workflow} {pose_value}: {state.get('error')}")
    with urllib.request.urlopen(base + state["outputs"][0]["url"], timeout=60) as response:
        return Image.open(io.BytesIO(response.read())).convert("RGB"), job


def pck(requested: list, detected_people: list, size: tuple[int, int], threshold: float = 0.1) -> float:
    """Best-matching detected person's share of requested body points within threshold x figure size."""
    width, height = size
    points = [(p[0] * width, p[1] * height) if p else None for p in requested]
    present = [p for p in points if p]
    xs, ys = [p[0] for p in present], [p[1] for p in present]
    scale = max(max(xs) - min(xs), max(ys) - min(ys))
    best = 0.0
    for person in detected_people:
        hits = 0
        for want, got in zip(points, person.body.keypoints):
            if want and got and ((got.x - want[0]) ** 2 + (got.y - want[1]) ** 2) ** 0.5 <= threshold * scale:
                hits += 1
        best = max(best, hits / len(present))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8191")
    parser.add_argument("--out", type=Path, default=ROOT / "ComfyUI" / "output" / "pose_fidelity")
    parser.add_argument("--set", choices=sorted(CASE_SETS), default="sdxl", help="which adapter family to measure")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, tiles = [], []
    for index, (workflow, pose_value, prompt) in enumerate(CASE_SETS[args.set]):
        if pose_value.startswith("tests/"):
            pose_value = upload(args.base, ROOT / pose_value)
        seed = 4100 + index
        with_pose, job = render(args.base, workflow, prompt, seed, pose_value)
        summary = job["pose"]
        size = tuple(summary["size"])
        skeleton_url = args.base + summary["skeleton_url"]
        # The requested skeleton, mapped onto the output canvas exactly as the adapter drew it.
        source = pose.lookup(pose_value)
        name = pose.library_image(*source) if source else pose_value
        requested = pose.letterbox(pose.skeleton(name), *size)[0]["body"]
        without, _ = render(args.base, workflow, prompt, seed, None)
        scores = {}
        for label, image in (("with_pose", with_pose), ("no_pose", without)):
            people, _, _ = pose._estimate(image.resize(size))
            scores[label] = round(pck(requested, people, size), 2)
        rows.append({"workflow": workflow, "pose": summary.get("id") or summary.get("file"), "size": list(size), **scores})
        print(json.dumps(rows[-1]), flush=True)
        with urllib.request.urlopen(skeleton_url, timeout=30) as response:
            skeleton = Image.open(io.BytesIO(response.read())).convert("RGB")
        tile = Image.new("RGB", (3 * 256, 256 * size[1] // size[0] + 20), "white")
        for column, image in enumerate((skeleton, with_pose, without)):
            tile.paste(image.resize((256, 256 * size[1] // size[0])), (column * 256, 20))
        ImageDraw.Draw(tile).text((4, 4), f"{workflow} {rows[-1]['pose']}  pose PCK {scores['with_pose']}  no-pose {scores['no_pose']}", fill="black")
        tiles.append(tile)
    mean = {label: round(sum(row[label] for row in rows) / len(rows), 2) for label in ("with_pose", "no_pose")}
    report = {"metric": "PCK@0.1 of figure size, DWPose on the output", "mean": mean, "cases": rows}
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    sheet = Image.new("RGB", (3 * 256 * 2, sum(t.height for t in tiles[::2])), "white")
    y = 0
    for i in range(0, len(tiles), 2):
        sheet.paste(tiles[i], (0, y))
        if i + 1 < len(tiles):
            sheet.paste(tiles[i + 1], (3 * 256, y))
        y += max(t.height for t in tiles[i:i + 2])
    sheet.save(args.out / "contact_sheet.png")
    print(json.dumps({"mean": mean, "out": str(args.out)}))


if __name__ == "__main__":
    main()
