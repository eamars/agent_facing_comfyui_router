"""Fast contract regressions. GPU integration is in scripts/verify_agent_api.py."""
import asyncio
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import uuid

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Only ComfyUI server registration is substituted here. Graph construction,
# catalog reads, image headers, routing and request validation are production code.
catalog = json.loads((ROOT / "workflows/catalog.json").read_text(encoding="utf-8"))
node_names = {node["class_type"] for entry in catalog["workflows"]
              for node in json.loads((ROOT / "workflows" / entry["file"]).read_text(encoding="utf-8")).values()}
node_names.update({"LoadImage", "VAEEncode", "ImageScale", "RepeatLatentBatch"})
stubs = {
    "execution": types.ModuleType("execution"),
    "folder_paths": types.SimpleNamespace(get_input_directory=lambda: str(api.COMFY_ROOT / "input")),
    "server": types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=None)),
    "comfy_execution": types.ModuleType("comfy_execution"),
    "comfy_execution.jobs": types.SimpleNamespace(validate_job_id=lambda value: str(uuid.UUID(value))),
    "nodes": types.SimpleNamespace(NODE_CLASS_MAPPINGS=dict.fromkeys(node_names)),
}
with patch.dict(sys.modules, stubs):
    api = importlib.import_module("project_nodes.workflow_api")
from project_nodes import runtime_guard


class AgentContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "input").mkdir()
        Image.new("RGB", (320, 384), "navy").save(self.root / "input/source.png")
        Image.new("RGB", (320, 384), "white").save(self.root / "input/mask.png")
        self.addCleanup(patch.stopall)
        patch.object(api, "COMFY_ROOT", self.root).start()
        patch.dict(sys.modules, {"nodes": stubs["nodes"]}).start()
        self.entry = copy.deepcopy(next(e for e in catalog["workflows"] if e["id"] == "anima-turbo-v1-1"))

    def graph(self, **params):
        return api._prepare_graph(self.entry, {"parameters": {"prompt": "A ceramic teapot.", **params}})

    def assertBad(self, **params):
        with self.assertRaises(api.ApiError) as caught:
            self.graph(**params)
        self.assertEqual(caught.exception.status, 400)

    def test_no_silent_parameter_loss(self):
        for params in ({"widht": 512}, {"depth_image": "source.png"}, {"mask_image": "mask.png"}):
            self.assertBad(**params)
        with self.assertRaises(api.ApiError):
            api._request_parameters({"prompt": "first", "parameters": {"positive_prompt": "second"}})
        self.assertEqual(api._request_parameters({"parameters": {"positive_prompt": "unchanged"}}), {"prompt": "unchanged"})
        self.assertEqual(api._request_parameters({"parameters": {"subject": "A ceramic teapot."}}), {"prompt": "A ceramic teapot."})

    def test_prompt_is_explicit_and_verbatim(self):
        with self.assertRaises(api.ApiError):
            api._prepare_graph(self.entry, {})
        for text in ("", "nsfw, explicit", "Unmodified (literal:1.2) text\nwith symbols!"):
            graph = self.graph(prompt=text, negative_prompt="", seed=42)
            self.assertEqual(api._binding_value(graph, self.entry, "prompt"), text)
            self.assertEqual(api._binding_value(graph, self.entry, "negative_prompt"), "")

    def test_seed_defaults_are_fresh_and_can_be_reproduced(self):
        first, second = self.graph(), self.graph()
        a, b = (api._binding_value(g, self.entry, "seed") for g in (first, second))
        self.assertNotEqual(a, b)
        self.assertLess(a, 2**53)
        self.assertEqual(api._binding_value(self.graph(seed=a), self.entry, "seed"), a)

    def test_error_response_shape_and_missing_status(self):
        with self.assertRaises(api.ApiError) as caught:
            api._prepare_graph(self.entry, {"parameters": {}})
        self.assertEqual(caught.exception.status, 422)
        caught.exception.route = self.entry["id"]
        error = json.loads(api._json_error(caught.exception).text)["error"]
        self.assertEqual(error["expected"], ["prompt", "subject"])
        self.assertEqual(error["route"], self.entry["id"])

    def test_img2img_size_batch_and_budget(self):
        graph = self.graph(input_image="source.png", width=512, height=512, batch_size=2)
        report = api._resource_request(graph, self.entry)
        self.assertEqual((report["width"], report["height"], report["batch_size"]), (512, 512, 2))
        sampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
        repeat = graph[sampler["inputs"]["latent_image"][0]]
        self.assertEqual((repeat["class_type"], repeat["inputs"]["amount"]), ("RepeatLatentBatch", 2))
        graph = self.graph(input_image="source.png", width=640)
        self.assertEqual(api._resource_request(graph, self.entry)["height"], 768)
        self.assertBad(width=2048, height=2048, batch_size=4)
        self.assertBad(input_image="source.png", width="oops")
        self.assertBad(width=float("inf"))

    def test_reference_contract_and_masks(self):
        self.entry = next(e for e in catalog["workflows"] if e["id"] == "sdxl-face-reference-ipadapter")
        self.assertBad(input_image="source.png", face_image="source.png")
        self.entry = next(e for e in catalog["workflows"] if e["id"] == "moody-anima-inpaint")
        self.assertBad(input_image="source.png", mask_image="mask.png", mask_channel="alpha")
        Image.new("RGB", (128, 128)).save(self.root / "input/mask.png")
        self.assertBad(input_image="source.png", mask_image="mask.png")

    def test_replaced_and_animated_inputs_fail_before_loading(self):
        graph = self.graph(input_image="source.png")
        resources = api._resource_request(graph, self.entry)
        Image.new("RGB", (128, 128)).save(self.root / "input/source.png")
        with self.assertRaises(RuntimeError):
            api._validate_queued_inputs(resources)
        frame = Image.new("RGB", (128, 128))
        frame.save(self.root / "input/movie.gif", save_all=True, append_images=[Image.new("RGB", (128, 128), "red")])
        self.assertBad(input_image="movie.gif")

    def test_route_scope_and_exact_capabilities(self):
        entries = {e["id"]: e for e in catalog["workflows"]}
        self.assertIsNone(api._route_score(entries["blue-archive-character-illustrious-xl"], {"domain": "blue archive"}))
        self.assertIsNone(api._route_score(entries["blue-archive-style-sdxl"], {"domain": "bathroom"}))
        self.assertIsNone(api._route_score(entries["blue-archive-style-sdxl"], {"input_type": "text"}))
        self.assertIsNotNone(api._route_score(entries["moody-anima-inpaint"], {"workflow": "moody-anima-inpaint"}))
        self.assertIsNone(api._route_score(self.entry, {"input_mode": "image"}))
        self.assertIsNotNone(api._route_score(self.entry, {"domain": "anime", "prompt_granularity": "expert"}))
        self.assertIsNone(api._route_score(entries["anima-turbo-face-reference-ipadapter"], {"family": "sdxl-ipadapter", "input_mode": "face-reference"}))
        with self.assertRaises(api.ApiError):
            api._route_query({"domian": "anime"})

    def test_readiness_includes_registry_eligibility_nodes_and_size(self):
        model_file = self.root / "model.bin"
        model_file.write_bytes(b"1234")
        model = {"id": "model", "status": "installed", "routeable": False, "required_file": "model.bin", "file_sizes": {"model.bin": 4}}
        self.entry.update(required_files=[], model_ids=["model"])
        self.assertFalse(api._public_entry(self.entry, [model])["ready"])
        model["routeable"] = True
        self.assertTrue(api._public_entry(self.entry, [model])["ready"])
        model_file.write_bytes(b"1")
        self.assertFalse(api._public_entry(self.entry, [model])["ready"])
        self.entry["model_ids"] = []
        with patch.dict(stubs["nodes"].NODE_CLASS_MAPPINGS, {}, clear=True):
            self.assertFalse(api._public_entry(self.entry, [])["ready"])

    def test_inspection_does_not_select_an_unready_workflow(self):
        high = copy.deepcopy(self.entry)
        high.update(id="missing-model")
        high["routing"]["priority"] = 1000
        available = copy.deepcopy(self.entry)
        with patch.object(api, "_public_entry", side_effect=lambda entry, models: {"ready": entry["id"] != "missing-model"}):
            result = api._resolve_routes({"workflows": [high, available]}, {"domain": "anime"}, include_unready=True)
        self.assertEqual(result["selected_workflow"], available["id"])

    def test_parameter_schema_uses_graph_defaults(self):
        graph = api._read_workflow(self.entry)
        schema = api._parameter_schema(self.entry, graph)
        self.assertIn({"required": ["prompt"]}, schema["anyOf"])
        self.assertIn("input_image", schema["properties"])
        self.assertNotIn("default", schema["properties"]["seed"])
        self.assertFalse(schema["additionalProperties"])


class WorkerGuardTests(unittest.TestCase):
    def test_boundaries_include_native_jobs_and_failure_recovery(self):
        events = []

        class Executor:
            success = True
            server = types.SimpleNamespace(client_id=None)
            history_result = {"outputs": {}}
            status_messages = []

            def reset(self):
                events.append("reset")
                self.success = True

            def execute(self, prompt, prompt_id, extra_data, execute_outputs):
                events.append(prompt_id)
                self.success = prompt_id != "failure"

        runtime_guard.install(types.SimpleNamespace(PromptExecutor=Executor), lambda *args: None, lambda r: None)
        snapshot = {"free_ram_bytes": 10 * 1024**3, "free_vram_bytes": 10 * 1024**3}
        limits = {"min_free_ram_gib": 4, "min_free_vram_gib": 2}
        named = {"project_workflow": "anima", "project_resources": {"limits": limits}}
        instance = Executor()
        with patch.object(runtime_guard, "memory_snapshot", return_value=snapshot), patch.object(runtime_guard, "release_models", side_effect=lambda: events.append("release")):
            for label, data in [("native-1", {}), ("named-1", named), ("named-2", named), ("native-2", {}), ("failure", named), ("recovered", named)]:
                instance.execute({}, label, data, [])
        self.assertEqual(events, ["native-1", "reset", "release", "named-1", "named-2", "reset", "release", "native-2", "reset", "release", "failure", "reset", "release", "reset", "release", "recovered"])


class CoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_never_started_failure_has_stable_timing(self):
        coordinator = api.ProjectWorkflowCoordinator(types.SimpleNamespace())
        coordinator._states["gone"] = {"prompt_id": "gone", "timing": {"submitted_monotonic": 1}}
        with patch.object(api.time, "monotonic", return_value=3):
            coordinator._set_state("gone", completed=True, state="failed")
        with patch.object(api.time, "monotonic", return_value=10):
            response = api._job_response({"prompt_id": "gone"}, coordinator.snapshot("gone"))
        self.assertEqual(json.loads(response.text)["queued_ms"], 2000)
        self.assertEqual(json.loads(response.text)["elapsed_ms"], 0)

    async def test_removed_native_job_does_not_block_the_queue(self):
        queue = types.SimpleNamespace(get_history=lambda **kw: {}, get_current_queue_volatile=lambda: ([], []))
        coordinator = api.ProjectWorkflowCoordinator(types.SimpleNamespace(prompt_queue=queue))
        with self.assertRaisesRegex(RuntimeError, "removed"):
            await asyncio.wait_for(coordinator._wait_for_history("gone"), 1)

    async def test_queue_admission_is_bounded(self):
        coordinator = api.ProjectWorkflowCoordinator(types.SimpleNamespace())
        coordinator._ensure_started()
        coordinator._task.cancel()
        await asyncio.gather(coordinator._task, return_exceptions=True)
        coordinator._pending_ids = [str(i) for i in range(api.MAX_PROJECT_QUEUE)]
        with patch.object(coordinator, "_ensure_started"):
            with self.assertRaises(api.ApiError) as caught:
                await coordinator.submit(api._ProjectJob(1, "id", "wf", {}, {}, [], {}))
        self.assertEqual(caught.exception.status, 429)


if __name__ == "__main__":
    unittest.main()
