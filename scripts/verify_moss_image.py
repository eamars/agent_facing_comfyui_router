"""Real named-API MOSS-Image-SFT probe on the 4090: plain prompt, layout JSON prompt, then a hot switch back."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
import urllib.request
import uuid

from PIL import Image

PLAIN = ('A cosy bookshop window at dusk. A hand-lettered chalkboard sign in the window reads "MOSS IMAGE" '
         'on the first line and "开放模型" on the second line. Warm interior light, rain on the glass.')

# 1024 square, centre origin, [y_min, x_min, y_max, x_max] in pixels.
LAYOUT = {
    "summary": {
        "meta": ["poster"],
        "canvas": "single",
        "caption": "A flat minimalist travel poster on a cream background with soft paper texture.",
    },
    "items": [
        {"label": "text", "caption": "Bold dark-blue sans-serif title across the top.", "text": "MOSS IMAGE",
         "bbox": [-460, -400, -330, 400]},
        {"label": "object", "caption": "A red vintage bicycle facing right.", "bbox": [-40, -470, 360, -60]},
        {"label": "object", "caption": "A tall green potted monstera plant in a terracotta pot.", "bbox": [-200, 140, 360, 440]},
        {"label": "text", "caption": "Small black Chinese caption centred near the bottom.", "text": "布局控制测试",
         "bbox": [400, -260, 470, 260]},
    ],
    "bbox_format": "yxyx",
    "bbox_coord_space": "abs_centralized",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8191")
    parser.add_argument("--report", type=Path, default=Path("ComfyUI/.codex-runtime/moss-image-probe.json"))
    parser.add_argument("--save-dir", type=Path, help="Also save each output PNG here for inspection.")
    args = parser.parse_args()
    base = args.server.rstrip("/")
    report = {"jobs": [], "resources": [], "routes": []}
    prefix = "moss_probe_" + uuid.uuid4().hex[:8]

    def http(path, body=None, method=None):
        request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                         method=method, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())

    def resources():
        snapshot = http("/project/resources")
        assert "RTX 4090" in snapshot["device_name"], snapshot
        report["resources"].append(snapshot)
        return snapshot

    try:
        resources()
        for query, expected in [
            ({"domain": "moss"}, "moss-image-sft-t2i"),
            ({"domain": "layout", "input_mode": "text-to-image"}, "moss-image-sft-t2i"),
            ({"domain": "qwen", "input_mode": "text-to-image"}, "qwen-image-2.1-t2i"),
            ({"domain": "anime"}, "anima-turbo-v1-1"),
        ]:
            result = http("/project/resolve", query)
            assert result["selected_workflow"] == expected, result
            report["routes"].append({"query": query, "selected": expected})

        jobs = [
            ("moss-image-sft-t2i", "plain", {"prompt": PLAIN, "seed": 1016}),
            ("moss-image-sft-t2i", "layout", {"prompt": json.dumps(LAYOUT, ensure_ascii=False), "seed": 1016}),
            ("anima-turbo-v1-1", "switch-back", {"prompt": "1girl, reading a book in a bookshop", "seed": 7, "width": 512, "height": 512}),
        ]
        for index, (workflow, label, params) in enumerate(jobs):
            body = {"workflow": workflow, "parameters": {**params, "filename_prefix": f"{prefix}_{label}"}}
            dry = http("/project/generate", {**body, "dry_run": True})
            assert dry["validated"], dry
            if workflow.startswith("moss"):
                effective = dry["effective_parameters"]
                assert (effective["steps"], effective["cfg"], effective["sampler_name"], effective["scheduler"]) == (50, 3, "euler", "sgm_uniform"), effective
            job = http("/project/generate", body)
            deadline = time.monotonic() + 1800
            phases = []
            next_sample = time.monotonic()
            min_free = {"vram": None, "ram": None}
            while time.monotonic() < deadline:
                state = http(job["job_url"])
                marker = state["state"], state.get("phase")
                if not phases or phases[-1] != marker:
                    phases.append(marker)
                    print(label, marker, flush=True)
                if time.monotonic() >= next_sample:
                    snapshot = resources()
                    for key, field in (("vram", "physical_free_vram_bytes"), ("ram", "free_ram_bytes")):
                        if min_free[key] is None or snapshot[field] < min_free[key]:
                            min_free[key] = snapshot[field]
                    next_sample = time.monotonic() + 5
                if state["state"] in {"succeeded", "failed"}:
                    break
                time.sleep(1)
            else:
                raise TimeoutError(job["prompt_id"])
            report["jobs"].append({"workflow": workflow, "label": label, "phases": phases, "result": state,
                                   "sampled_min_free_bytes": min_free})
            assert state["state"] == "succeeded", state
            with urllib.request.urlopen(base + state["outputs"][0]["url"], timeout=60) as response:
                data = response.read()
            output = Image.open(io.BytesIO(data))
            expected_size = (1024, 1024) if workflow.startswith("moss") else (512, 512)
            assert output.size == expected_size, (label, output.size)
            report["jobs"][-1]["size"] = list(output.size)
            if args.save_dir:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                (args.save_dir / f"{index}_{label}.png").write_bytes(data)
            print("PASS", label, output.size, state["elapsed_ms"], "ms", flush=True)
        report["passed"] = True
    except Exception as exc:
        report["failure"] = str(exc)
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
