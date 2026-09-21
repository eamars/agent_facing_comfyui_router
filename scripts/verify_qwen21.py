"""Real named-API Qwen 2.1 / existing-model hot-switch probe on the 4090."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
import urllib.request
import uuid

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8191")
    parser.add_argument("--report", type=Path, default=Path("ComfyUI/.codex-runtime/qwen21-hot-switch.json"))
    args = parser.parse_args()
    base = args.server.rstrip("/")
    report = {"jobs": [], "resources": [], "routes": []}
    prefix = "qwen21_probe_" + uuid.uuid4().hex[:8]
    reference = None
    active = False

    def http(path, body=None, method=None):
        request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                         method=method, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())

    def resources():
        snapshot = http("/project/resources")
        assert "RTX 4090" in snapshot["device_name"], snapshot
        assert snapshot["cpu_vae"], snapshot
        report["resources"].append(snapshot)
        return snapshot

    def upload(data):
        name = prefix + ".png"
        boundary = uuid.uuid4().hex
        content = (f'--{boundary}\r\nContent-Disposition: form-data; name="type"\r\n\r\ninput\r\n'
                   f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                   'Content-Type: image/png\r\n\r\n').encode() + data + f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(base + "/upload/image", data=content,
                                         headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read())
        return "/".join(x for x in (result.get("subfolder"), result["name"]) if x)

    try:
        resources()
        for query, expected in [
            ({"domain": "qwen", "input_mode": "text-to-image"}, "qwen-image-2.1-t2i"),
            ({"domain": "qwen", "input_mode": "image-to-image"}, "qwen-image-2.1-edit"),
            ({"domain": "anime"}, "anima-turbo-v1-1"),
            ({"domain": "anime", "content_mode": "nsfw"}, "janku-v5-illustrious"),
        ]:
            result = http("/project/resolve", query)
            assert result["selected_workflow"] == expected, result
            report["routes"].append({"query": query, "selected": expected})

        for index, workflow in enumerate([
            "anima-turbo-v1-1", "qwen-image-2.1-t2i", "qwen-image-2.1-edit",
            "janku-v5-illustrious", "qwen-image-2.1-t2i", "anima-turbo-v1-1",
        ]):
            params = {"prompt": 'A blue ceramic teapot on a wooden table beside a card with the text "QWEN 2.1", soft daylight.',
                      "seed": 921 + index, "filename_prefix": prefix + "_" + str(index)}
            if workflow == "qwen-image-2.1-edit":
                assert reference
                params.update(input_image=reference, prompt="Change the blue teapot to red. Keep the table, card and lighting unchanged.")
            elif not workflow.startswith("qwen"):
                params.update(width=512, height=512)
            body = {"workflow": workflow, "parameters": params}
            dry = http("/project/generate", {**body, "dry_run": True})
            assert dry["validated"], dry
            if workflow.startswith("qwen"):
                effective = dry["effective_parameters"]
                assert (effective["steps"], effective["cfg"], effective["sampler_name"], effective["scheduler"]) == (25, 1, "euler", "simple")
            job = http("/project/generate", body)
            active = True
            deadline = time.monotonic() + 1800
            phases = []
            next_sample = time.monotonic()
            while time.monotonic() < deadline:
                state = http(job["job_url"])
                marker = state["state"], state.get("phase")
                if not phases or phases[-1] != marker:
                    phases.append(marker)
                    print(workflow, marker, flush=True)
                if time.monotonic() >= next_sample:
                    resources()
                    next_sample = time.monotonic() + 10
                if state["state"] in {"succeeded", "failed"}:
                    active = False
                    break
                time.sleep(1)
            else:
                raise TimeoutError(job["prompt_id"])
            report["jobs"].append({"workflow": workflow, "phases": phases, "result": state})
            assert state["state"] == "succeeded", state
            with urllib.request.urlopen(base + state["outputs"][0]["url"], timeout=60) as response:
                data = response.read()
            output = Image.open(io.BytesIO(data))
            expected_size = (1024, 1024) if workflow.startswith("qwen") else (512, 512)
            assert output.size == expected_size, (workflow, output.size)
            report["jobs"][-1]["size"] = list(output.size)
            report["jobs"][-1]["image_mode"] = output.mode
            if workflow == "qwen-image-2.1-t2i" and reference is None:
                reference = upload(data)
            print("PASS", workflow, output.size, state["elapsed_ms"], "ms", flush=True)
        report["passed"] = True
    except Exception as exc:
        report["failure"] = str(exc)
        raise
    finally:
        if reference and not active:
            http("/project/inputs/" + reference, method="DELETE")
        elif reference:
            report["retained_input_for_active_job"] = reference
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
