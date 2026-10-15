"""Public-HTTP regression probe. All generated prompts describe a ceramic teapot.

Run with the ComfyUI venv. --generate additionally exercises every ready route,
native/project interleaving, six queued jobs, inpaint pixel preservation and an
input-disappearance failure. Reference fixtures are uploaded then deleted by name.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

from PIL import Image, ImageChops, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8191")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("ComfyUI/.codex-runtime/interface-audit/verification.json"))
    args = parser.parse_args()
    base = args.server.rstrip("/")
    report = {"server": base, "checks": [], "jobs": []}
    uploaded = []
    prefix = "interface_audit_" + uuid.uuid4().hex[:10]
    prompt = "An illustration of a blue ceramic teapot on a wooden table, soft daylight, clean background."

    def http(path, body=None, method=None, expected=200):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status, payload = response.status, json.loads(response.read())
        except urllib.error.HTTPError as response:
            status, payload = response.code, json.loads(response.read())
        assert status == expected, (path, status, payload)
        return payload

    def check(label, condition):
        assert condition, label
        report["checks"].append(label)
        print("PASS", label, flush=True)

    def upload(name, image):
        data = io.BytesIO()
        image.save(data, format="PNG")
        name = prefix + "_" + name + ".png"
        boundary = uuid.uuid4().hex
        content = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n"
                   f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                   "Content-Type: image/png\r\n\r\n").encode() + data.getvalue() + f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(base + "/upload/image", data=content, headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read())
        name = "/".join(part for part in (result.get("subfolder"), result["name"]) if part)
        uploaded.append(name)
        return name

    def wait_job(job):
        deadline = time.monotonic() + 600
        last = None
        while time.monotonic() < deadline:
            state = http(job["job_url"])
            marker = state["state"], state.get("phase")
            if marker != last:
                print(job.get("workflow", "native"), marker, flush=True)
                last = marker
            if state["state"] in {"succeeded", "failed"}:
                report["jobs"].append(state)
                return state
            time.sleep(0.3)
        raise TimeoutError(job["prompt_id"])

    def image_output(state):
        with urllib.request.urlopen(base + state["outputs"][0]["url"], timeout=30) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")

    manifest = http("/.well-known/agent-manifest.json")
    check("API 1.5 discovery", manifest.get("api_version") == "1.5.0")
    report["initial_resources"] = http("/project/resources")
    entries = http("/project/workflows")["workflows"]
    source = Image.new("RGB", (320, 384), (40, 70, 110))
    mask = Image.new("RGB", source.size, "black")
    ImageDraw.Draw(mask).rectangle((96, 96, 224, 288), fill="white")
    pending = []
    try:
        source_name, mask_name = upload("source", source), upload("mask", mask)
        references = {"input_image": source_name, "face_image": source_name, "control_image": source_name, "mask_image": mask_name}
        routes = [
            ({"domain": "blue archive"}, "blue-archive-style-sdxl"),
            ({"domain": "bathroom"}, "anima-turbo-v1-1"),
            ({"domain": "anime", "input_mode": "inpaint"}, "anima-turbo-inpaint"),
            ({"domain": "anime", "input_mode": "face-reference", "family": "sdxl-ipadapter"}, "sdxl-face-reference-ipadapter"),
            ({"workflow": "blue-archive-character-illustrious-xl"}, "blue-archive-character-illustrious-xl"),
            ({"domain": "anime", "content_mode": "nsfw"}, "janku-illustrious"),
        ]
        for query, expected in routes:
            check("route " + json.dumps(query), http("/project/resolve", query)["selected_workflow"] == expected)

        base_body = {"workflow": "anima-turbo-v1-1", "dry_run": True}
        alias = http("/project/prompt", {**base_body, "parameters": {"subject": prompt}})
        check("subject maps to prompt verbatim", alias["applied_parameters"]["prompt"] == prompt and alias["ignored_parameters"] == [])
        for body, status, token in [
            ({**base_body, "parameters": {"prompt": prompt, "nonexistent_field": 1}}, 400, "nonexistent_field"),
            ({**base_body, "parameters": {}}, 422, "prompt"),
            ({**base_body, "workflow": "no-such-workflow", "parameters": {"prompt": prompt}}, 404, "no-such-workflow"),
            ({**base_body, "parameters": {"prompt": prompt, "width": 2048, "height": 2048, "batch_size": 4}}, 400, "budget"),
            ({**base_body, "parameters": {"prompt": prompt, "sampler_name": "no-such-sampler"}}, 400, "validation"),
        ]:
            error = http("/project/prompt", body, expected=status)["error"]
            check(f"error {status}: {token}", token.lower() in error["message"].lower() and {"message", "route", "expected"} <= error.keys())
        advisory = http("/project/prompt", {**base_body, "parameters": {"prompt": prompt, "steps": 26, "cfg": 6}})
        check("sampling advice retains requested values", advisory["applied_parameters"]["steps"] == 26 and {w["parameter"] for w in advisory["warnings"]} >= {"steps", "cfg"})

        dry_runs = {}
        for entry in entries:
            check("dependencies " + entry["id"], entry["ready"])
            params = {"prompt": prompt, "seed": 917, "filename_prefix": prefix + "_" + entry["id"]}
            for name in entry["parameter_schema"]["required"]:
                if name in references:
                    params[name] = references[name]
            if "width" in entry["bindings"]:
                params.update(width=512, height=512, batch_size=1)
            params["steps"] = entry["agent_hints"]["recommended"]["steps"][0]
            body = {"workflow": entry["id"], "parameters": params}
            dry = http("/project/generate", {**body, "dry_run": True})
            check("validated graph " + entry["id"], dry["validated"] and dry["ignored_parameters"] == [])
            dry_runs[entry["id"]] = (body, dry)
            if args.generate:
                job = http("/project/generate", body)
                check("id aliases " + entry["id"], job["job_id"] == job["prompt_id"])
                state = wait_job(job)
                check("generation " + entry["id"], state["state"] == "succeeded")
                check("timing " + entry["id"], state["queued_ms"] >= 0 and state["elapsed_ms"] > 0)
                if entry["control_ports"]["inpaint"]:
                    diff = ImageChops.difference(source, image_output(state))
                    diff.paste((0, 0, 0), mask=mask.convert("L"))
                    check("unmasked pixels preserved " + entry["id"], diff.getbbox() is None)

        img = http("/project/generate", {**base_body, "parameters": {"prompt": prompt, "input_image": source_name, "width": 512, "height": 512, "batch_size": 2}})
        check("img2img effective size and batch", img["resources"]["total_pixels"] == 512 * 512 * 2)
        if args.generate:
            # Six queued named jobs, interleaved with one native graph.
            body = dry_runs["anima-turbo-v1-1"][0]
            native_graph = dry_runs["anima-turbo-v1-1"][1]["prompt"]
            native_sampler = next(n for n in native_graph.values() if n["class_type"] == "KSampler")
            native_sampler["inputs"]["seed"] = 1917
            native = http("/prompt", {"prompt": native_graph, "client_id": prefix})
            for index in range(6):
                params = {**body["parameters"], "seed": 2017 + index}
                if index == 0:
                    params.update(input_image=source_name, batch_size=2)
                    params["subject"] = params.pop("prompt")
                pending.append(http("/project/generate", {**body, "parameters": params}))
            states = [wait_job(job) for job in pending]
            check("six concurrent submissions complete", all(s["state"] == "succeeded" for s in states))
            check("queue time includes preceding jobs", states[-1]["queued_ms"] > states[0]["queued_ms"])
            check("img2img emits two correctly sized images", len(states[0]["outputs"]) == 2 and image_output(states[0]).size == (512, 512))
            native_state = wait_job({"prompt_id": native["prompt_id"], "job_url": "/project/jobs/" + native["prompt_id"]})
            check("native/project interleaving", native_state["state"] == "succeeded")
            pending.clear()

            # A missing queued reference must fail promptly and leave the next job runnable.
            blocker = http("/project/generate", {**body, "parameters": {**body["parameters"], "seed": 2917, "steps": 40}})
            pending.append(blocker)
            disposable = upload("disposable", source)
            inpaint = dry_runs["anima-turbo-inpaint"][0]
            doomed = http("/project/generate", {**inpaint, "parameters": {**inpaint["parameters"], "input_image": disposable}})
            pending.append(doomed)
            http("/project/inputs/" + disposable, method="DELETE")
            uploaded.remove(disposable)
            successor = http("/project/generate", {**body, "parameters": {**body["parameters"], "seed": 3917}})
            pending.append(successor)
            check("blocking job completes", wait_job(blocker)["state"] == "succeeded")
            failed = wait_job(doomed)
            check("queued input failure is actionable", failed["state"] == "failed" and bool(failed.get("error")))
            check("queue recovers after failure", wait_job(successor)["state"] == "succeeded")
            pending.clear()
    except Exception as exc:
        report["failure"] = str(exc)
        raise
    finally:
        # Do not delete fixtures while a known job is still consuming them.
        for job in pending:
            wait_job(job)
        for name in uploaded:
            try:
                result = http("/project/inputs/" + name, method="DELETE")
                check("delete reports remaining bytes", result["deleted"] and result["remaining_bytes"] >= 0)
            except AssertionError as exc:
                report.setdefault("cleanup_errors", []).append(str(exc))
        report["final_resources"] = http("/project/resources")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS {len(report['checks'])} checks; {len(report['jobs'])} jobs; report {args.report}", flush=True)


if __name__ == "__main__":
    main()
