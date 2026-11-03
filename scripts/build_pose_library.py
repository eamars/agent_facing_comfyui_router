"""Build workflows/poses.json from the Lanpose pose catalog and keep only views DWPose can read.

Run with the ComfyUI interpreter (it needs DWPose):
    ComfyUI\\.venv\\Scripts\\python.exe scripts\\build_pose_library.py [--names names_en.json]

English names already in poses.json are kept; --names supplies names for new poses ({pose_id: {"name_en": ...}}).
Every view's sketch is fetched into the pose cache (ComfyUI/input/_poses/library) and its skeleton extracted;
a view stays in the library only when its skeleton passes _gate (people count and limbs that define the pose).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ComfyUI" / "custom_nodes" / "comfyui_controlnet_aux" / "src")]

from project_nodes import pose  # noqa: E402

UPSTREAM_COMMIT = "aec165637c985cc554de89b44fbd6e3fdfe6a851"
CATALOG_URL = f"https://raw.githubusercontent.com/wzc520pyfm/gen-ai-pose-image/{UPSTREAM_COMMIT}/data/catalog.json"
CATEGORIES = {
    "可爱系": "cute", "性感・大胆系": "sexy, bold", "挤胸・强调事业线系": "cleavage emphasis",
    "侧坐・女孩坐": "side sit / W-sit", "前倾": "leaning forward", "后仰・挺背": "leaning back, arched back",
    "坐姿": "sitting", "扭身": "twisting", "站姿": "standing", "趴卧": "lying", "跪立": "kneeling", "蹲姿": "squatting",
    "侧卧位": "side-lying couple", "后入位": "from-behind couple", "坐位": "seated couple", "坐位贴身": "seated close couple",
    "正常位（传教士）": "missionary couple", "立位": "standing couple", "立位贴身": "standing close couple",
    "背向坐位": "seated facing-away couple", "贴脸系": "cheek-to-cheek couple", "贴身": "close-embrace couple",
    "面对面坐位": "seated face-to-face couple", "骑乘位": "straddling couple", "特殊": "special",
    "跪立・四肢着地": "kneeling / all fours",
}
FRAMING = {"full": "full body", "sexy": "full body", "bust": "upper body", "pair": "two people"}
# Names the bulk translation got wrong (女孩坐 is the W-sit, 三角坐 is sitting with knees drawn up).
NAME_FIXES = {
    "full_single_P004": "Sitting with knees drawn up, both knees tilted left",
    "full_single_P041": "W-sit (wariza), both hands on legs",
    "full_single_P042": "W-sit (wariza), both hands propped in front",
    "full_single_P043": "W-sit (wariza), both hands behind head",
    "full_single_P044": "W-sit (wariza), hands braced behind",
    "full_single_P048": "W-sit (wariza), hand touching side of neck",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", type=Path)
    args = parser.parse_args()
    upstream = json.load(urllib.request.urlopen(CATALOG_URL, timeout=60))
    previous = json.loads(pose.LIBRARY_PATH.read_text(encoding="utf-8")) if pose.LIBRARY_PATH.is_file() else {}
    names = {p["id"]: p["name"] for p in previous.get("poses", [])}
    if args.names:
        names.update({key: value["name_en"] for key, value in json.loads(args.names.read_text(encoding="utf-8")).items()
                      if key not in names})
    names.update(NAME_FIXES)
    source = {
        "name": "ai-pose-gallery-zh via Lanpose (gen-ai-pose-image)",
        "catalog": CATALOG_URL,
        "origin": upstream["origin"],
        "attribution": "Pose sketches and raw pose data © ギア (giaaibusi); Chinese names from Anhao's ai-pose-gallery-zh; "
                       "index from wzc520pyfm/gen-ai-pose-image. The sketches may not be used commercially and are "
                       "fetched on first use, not stored in this repository. English names are machine translations.",
    }
    poses = []
    for item in upstream["poses"]:
        if item["id"] not in names:
            raise SystemExit(f"No English name for {item['id']}; pass --names")
        poses.append({
            "id": item["id"], "name": names[item["id"]], "name_zh": item["name"],
            "category": CATEGORIES[item["category"]], "category_zh": item["category"],
            "framing": FRAMING[item["systemId"]], "people": item["people"],
            "views": {view["angleId"]: {"id": view["id"], "path": view["full"]} for view in item["views"]},
        })
    # Write the full index first: fetching reads the gallery origin from it.
    draft = {"schema_version": 1, "source": source, "angles": pose.ANGLES, "poses": poses}
    pose.LIBRARY_PATH.write_text(json.dumps(draft, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    views = [(p, angle) for p in poses for angle in p["views"]]
    started = time.time()
    with ThreadPoolExecutor(4) as pool:
        fetched = list(pool.map(lambda item: _fetch(*item), views))
    print(f"fetched {sum(1 for f in fetched if f)} / {len(views)} sketches in {time.time() - started:.0f}s", flush=True)

    dropped: dict[str, int] = {}
    for index, ((p, angle), name) in enumerate(zip(views, fetched)):
        reason = "fetch_failed" if name is None else None
        if name:
            try:
                reason = _gate(pose.skeleton(name), p)
            except pose.PoseError as exc:
                reason = exc.code
        if reason:
            dropped[p["views"][angle]["id"]] = reason
            del p["views"][angle]
        if index % 100 == 0:
            print(f"{index}/{len(views)} checked, {len(dropped)} dropped, {time.time() - started:.0f}s", flush=True)
    # Poses with no usable view stay in the index (views {}) so their names survive a rebuild under another rule.
    kept = sum(len(p["views"]) for p in poses)
    library = {**draft, "poses": poses, "quality_gate": {
        "rule": f"{GATE_RULE} ({pose.EXTRACTOR_VERSION})",
        "views_checked": len(views), "views_kept": kept, "poses_kept": sum(1 for p in poses if p["views"]), "dropped": dropped,
    }}
    pose.LIBRARY_PATH.write_text(json.dumps(library, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"kept {kept} of {len(views)} views in {library['quality_gate']['poses_kept']} poses; {time.time() - started:.0f}s")


LIMBS = {"right arm": (2, 3, 4), "left arm": (5, 6, 7), "right leg": (8, 9, 10), "left leg": (11, 12, 13)}
GATE_RULE = ("DWPose finds as many people as the pose has, and each person has the limbs that define the pose "
             "(a limb counts with 2 of its 3 joints): both arms for upper-body poses, 3 of 4 limbs otherwise")


def _gate(skeleton: dict, entry: dict) -> str | None:
    """Why a view is unusable, or None. A point count is not enough: skeletons with 8-11 of 18 points were
    mostly a head and one limb, steering the ControlNet wrong rather than leaving limbs free."""
    if len(skeleton["people"]) != entry["people"]:
        return f"found {len(skeleton['people'])} of {entry['people']} people"
    for person in skeleton["people"]:
        limbs = {name for name, joints in LIMBS.items() if sum(person["body"][j] is not None for j in joints) >= 2}
        if entry["framing"] == "upper body" and not {"right arm", "left arm"} <= limbs:
            return "arms not found"
        if entry["framing"] != "upper body" and len(limbs) < 3:
            return "fewer than 3 limbs found"
    return None


def _fetch(p, angle):
    try:
        return pose.library_image(p, angle)
    except pose.PoseError as exc:
        print(f"fetch failed: {p['views'][angle]['id']}: {exc.details}", flush=True)
        return None


if __name__ == "__main__":
    main()
