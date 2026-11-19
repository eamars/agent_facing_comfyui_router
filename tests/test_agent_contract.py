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

import numpy  # noqa: F401  loaded once here: patch.dict(sys.modules) would drop it, and its C extension cannot load twice
from PIL import Image

ROOT =Path(__file__).resolve().parents[1]
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

    def test_qwen_sampling_and_edit_reference_size(self):
        self.entry = next(e for e in catalog["workflows"] if e["id"] == "qwen-image-2.1-t2i")
        graph = self.graph()
        self.assertEqual([api._binding_value(graph, self.entry, key) for key in
                          ("width", "height", "batch_size", "steps", "cfg", "sampler_name", "scheduler", "denoise")],
                         [1024, 1024, 1, 25, 1, "euler", "simple", 1])
        self.assertBad(input_image="source.png")
        self.entry = next(e for e in catalog["workflows"] if e["id"] == "qwen-image-2.1-edit")
        with self.assertRaises(api.ApiError) as caught:
            self.graph()
        self.assertEqual(caught.exception.status, 422)
        Image.new("RGB", (333, 349), "navy").save(self.root / "input/source.png")
        graph = self.graph(input_image="source.png")
        resource = api._resource_request(graph, self.entry)
        self.assertEqual((resource["width"], resource["height"], resource["total_pixels"]), (320, 352, 320 * 352))
        self.assertEqual(graph["452"]["inputs"]["resolution"], 0)
        self.assertEqual(graph["458"]["inputs"]["latent_image"], ["452", 2])
        self.assertEqual(graph["452"]["inputs"]["images.image_1"], ["470", 0])
        # Sampling starts from an empty latent, so denoise below 1 collapsed every edit to a flat texture.
        self.assertEqual(graph["458"]["inputs"]["denoise"], 1)
        self.assertBad(input_image="source.png", denoise=0.3)
        self.assertBad(input_image="source.png", width=1024)
        self.assertBad(input_image="source.png", batch_size=2)

    def test_qwen_selection_does_not_replace_existing_generic_routes(self):
        for entry in catalog["workflows"]:
            if not entry["id"].startswith("qwen-image-2.1-"):
                continue
            self.assertIsNone(api._route_score(entry, {"domain": "anime"}))
            self.assertIsNone(api._route_score(entry, {"domain": "deepseek"}))
            mode = "image-edit" if entry["id"].endswith("edit") else "text-to-image"
            self.assertIsNotNone(api._route_score(entry, {"domain": "qwen", "input_mode": mode}))
        edit = next(e for e in catalog["workflows"] if e["id"] == "qwen-image-2.1-edit")
        self.assertIsNotNone(api._route_score(edit, {"input_mode": "image-edit"}))
        # Edit takes an instruction and no denoise; image-to-image redraws from a description, so edit must not claim it.
        self.assertIsNone(api._route_score(edit, {"domain": "qwen", "input_mode": "image-to-image"}))

    def test_moss_is_explicit_and_binds_layout_prompts_verbatim(self):
        moss = next(e for e in catalog["workflows"] if e["id"] == "moss-image-sft-t2i")
        for query in ({"domain": "anime"}, {"domain": "qwen"}, {"input_mode": "text-to-image"}):
            self.assertIsNone(api._route_score(moss, query), query)
        for query in ({"domain": "moss"}, {"domain": "layout", "input_mode": "text-to-image"}, {"style": "MOSS-Image"}):
            self.assertIsNotNone(api._route_score(moss, query), query)
        # Benchmark-driven split of the heavy general-purpose bucket (docs/decisions.md).
        workflows = catalog["workflows"]

        def selected(query):
            scored = [(api._route_score(e, query), e["id"]) for e in workflows]
            return max((s[0], i) for s, i in scored if s is not None)[1]
        for domain in ("photo", "product", "landscape", "poster", "realistic"):
            self.assertEqual(selected({"domain": domain}), "qwen-image-2.1-t2i", domain)
        self.assertEqual(selected({"style": "photorealistic"}), "qwen-image-2.1-t2i")
        for domain in ("layout", "ui mockup"):
            self.assertEqual(selected({"domain": domain}), "moss-image-sft-t2i", domain)
        self.assertEqual(selected({"domain": "anime"}), "anima-turbo-v1-1")
        layout = json.dumps({"summary": {"caption": "poster"}, "items": [], "bbox_format": "yxyx",
                             "bbox_coord_space": "abs_centralized"})
        graph = api._prepare_graph(moss, {"parameters": {"prompt": layout, "seed": 7}})
        self.assertEqual(graph["4"]["inputs"]["text"], layout)
        self.assertEqual(graph["1"]["inputs"]["model_path"], "models/moss-image/MOSS-Image-SFT")
        self.assertEqual((graph["16"]["inputs"]["steps"], graph["16"]["inputs"]["cfg"], graph["15"]["inputs"]["shift"]), (50, 3, 5))

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
        oversized = api._resource_request(self.graph(width=2048, height=2048, batch_size=4), self.entry)
        self.assertGreater(oversized["total_pixels"], oversized["limits"]["max_total_pixels"])
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
        report = []
        graph = api._prepare_graph(self.entry, {"parameters": {"prompt": "x", "input_image": "movie.gif"}}, report)
        self.assertTrue(report[0]["first_frame"])
        with Image.open(self.root / "input" / report[0]["file"]) as fitted:
            self.assertEqual(fitted.n_frames, 1)
        with self.assertRaises(api.ApiError) as caught:
            self.graph(input_image="movie.gif", input_fit="none")
        self.assertEqual(caught.exception.code, "input_animated")

    def test_oversized_inputs_are_fitted_and_reported(self):
        entries = {e["id"]: e for e in catalog["workflows"]}
        Image.open(ROOT / "tests/fixtures/reference_character.webp").save(self.root / "input/reference.webp")
        Image.new("RGB", (1116, 2000), "white").save(self.root / "input/ref_mask.png")
        Image.new("RGB", (2000, 1116), "white").save(self.root / "input/wide_mask.png")
        Image.new("RGB", (3000, 4000), "navy").save(self.root / "input/big.png")
        Image.new("RGB", (5000, 4000), "navy").save(self.root / "input/huge.png")

        def fit(workflow, **params):
            report = []
            graph = api._prepare_graph(entries[workflow], {"parameters": {"prompt": "x", **params}}, report)
            return graph, report

        # inpaint: 1116 wide is not a multiple of 8; source and mask are fitted together
        graph, report = fit("anima-turbo-inpaint", input_image="reference.webp", mask_image="ref_mask.png")
        used = report[0]["used_size"]
        self.assertTrue(all(v % 8 == 0 for v in used) and used[0] * used[1] <= 2_359_296)
        self.assertEqual(report[1]["used_size"], used)
        with self.assertRaises(api.ApiError) as caught:
            fit("anima-turbo-inpaint", input_image="reference.webp", mask_image="wide_mask.png")
        self.assertEqual(caught.exception.code, "mask_size_mismatch")
        with self.assertRaises(api.ApiError) as caught:
            fit("anima-turbo-inpaint", input_image="reference.webp", mask_image="ref_mask.png", input_fit="none")
        self.assertEqual(caught.exception.code, "input_size_unsupported")

        # qwen edit: output follows the source, so an over-budget source is shrunk to multiples of 32
        _, report = fit("qwen-image-2.1-edit", input_image="big.png")
        w, h = report[0]["used_size"]
        self.assertTrue(w % 32 == 0 and h % 32 == 0 and w * h <= 2_359_296 and abs(w / h - 0.75) < 0.02)
        _, report = fit("qwen-image-2.1-edit", input_image="reference.webp")
        self.assertEqual(report, [])
        _, report = fit("qwen-image-2.1-edit", input_image="big.png", input_fit="none")
        self.assertEqual(report, [])

        # latent img2img: the source file is kept, the default output size is fitted
        graph, report = fit("janku-illustrious", input_image="big.png")
        self.assertEqual(report[0]["scope"], "output size")
        self.assertLessEqual(api._binding_value(graph, entries["janku-illustrious"], "width") * api._binding_value(graph, entries["janku-illustrious"], "height"), 2_359_296)

        # face reference: a full-body image is cropped to the face DWPose finds; without a face only the decode limit applies
        with patch.object(api.pose_addon, "face_box", return_value=(0.445, 0.083, 0.579, 0.152)):  # reference_character's face
            _, report = fit("sdxl-face-reference-ipadapter", face_image="reference.webp")
            x0, y0, x1, y1 = report[0]["crop_box"]
            self.assertTrue(x1 - x0 == y1 - y0 and 0 <= y0 < 0.083 * 2000 and x0 < 0.445 * 1116 and x1 > 0.579 * 1116)
            self.assertEqual(report[0]["used_size"], [x1 - x0, y1 - y0])
            _, report = fit("sdxl-face-reference-ipadapter", face_image="reference.webp", input_fit="none")
            self.assertEqual(report, [])
        with patch.object(api.pose_addon, "face_box", return_value=(0.2, 0.2, 0.8, 0.8)):  # already a head shot
            _, report = fit("sdxl-face-reference-ipadapter", face_image="reference.webp")
            self.assertEqual(report, [])
        with patch.object(api.pose_addon, "face_box", return_value=None):
            _, report = fit("anima-turbo-face-reference-ipadapter", face_image="huge.png")
        self.assertLessEqual(report[0]["used_size"][0] * report[0]["used_size"][1], api.MAX_INPUT_PIXELS)
        with self.assertRaises(api.ApiError) as caught:
            fit("anima-turbo-face-reference-ipadapter", face_image="huge.png", input_fit="none")
        self.assertEqual(caught.exception.code, "input_too_large")
        with self.assertRaises(api.ApiError) as caught:
            fit("qwen-image-2.1-t2i", input_image="big.png")
        self.assertEqual(caught.exception.code, "unsupported_parameter")

    def test_inpaint_on_a_large_picture_draws_the_mask_region_and_pastes_it_back(self):
        entries = {e["id"]: e for e in catalog["workflows"]}
        Image.new("RGB", (3000, 4000), "navy").save(self.root / "input/big.png")
        mask = Image.new("RGB", (3000, 4000), "black"); mask.paste((255, 255, 255), (1000, 1000, 1200, 1300)); mask.save(self.root / "input/small_mask.png")
        Image.new("RGB", (3000, 4000), "black").save(self.root / "input/empty_mask.png")
        report = []
        entry = entries["anima-turbo-inpaint"]
        graph = api._prepare_graph(entry, {"parameters": {"prompt": "x", "input_image": "big.png", "mask_image": "small_mask.png"}}, report)
        region = report[0]
        self.assertEqual((region["crop_reason"], region["crop_box"]), ("inpaint", [880, 880, 1320, 1420]))  # mask plus 120 px context
        self.assertTrue(all(v % 8 == 0 for v in region["used_size"]) and region["used_size"][0] * region["used_size"][1] >= 900_000)
        image_id = entry["bindings"]["input_image"]["node"]
        self.assertEqual(graph[image_id]["inputs"]["image"], "big.png")  # the full upload, not a shrunk copy
        save = next(n for n in graph.values() if n["class_type"] == "SaveImage")
        paste = graph[save["inputs"]["images"][0]]
        self.assertEqual((paste["class_type"], paste["inputs"]["destination"], paste["inputs"]["x"]), ("ImageCompositeMasked", [image_id, 0], 880))
        self.assertEqual(api._resource_request(graph, entry)["total_pixels"], region["used_size"][0] * region["used_size"][1])
        report = []
        api._prepare_graph(entry, {"parameters": {"prompt": "x", "input_image": "big.png", "mask_image": "empty_mask.png"}}, report)
        self.assertNotIn("crop_reason", report[0])  # nothing to redraw: the old whole-picture fit

    def test_img2img_crops_the_centre_instead_of_stretching(self):
        report = []
        graph = api._prepare_graph(self.entry, {"parameters": {"prompt": "x", "input_image": "source.png", "width": 512, "height": 512}}, report)
        scale = next(n for n in graph.values() if n["class_type"] == "ImageScale")
        self.assertEqual(scale["inputs"]["crop"], "center")
        self.assertEqual(report[-1]["crop_box"], [0, 32, 320, 352])  # 320x384 source, square output
        report = []
        api._prepare_graph(self.entry, {"parameters": {"prompt": "x", "input_image": "source.png", "width": 640}}, report)
        self.assertFalse(any(r.get("crop_box") for r in report))  # one side keeps the shape

    def test_extend_pads_the_source_and_pastes_the_original_back(self):
        entries = {e["id"]: e for e in catalog["workflows"]}
        report = []
        graph = api._prepare_graph(entries["qwen-image-2.1-edit"], {"parameters": {
            "prompt": "more sky", "input_image": "source.png", "extend": {"top": 100, "right": 40}}}, report)
        self.assertEqual((report[0]["scope"], report[0]["used_size"]), ("extend", [360, 484]))
        with Image.open(self.root / "input" / report[0]["file"]) as canvas:
            self.assertEqual((canvas.getpixel((0, 0)), canvas.getpixel((0, 100))), ((128, 128, 128), (0, 0, 128)))
        paste_id, paste = next((k, n) for k, n in graph.items() if n["class_type"] == "ProjectExtendPaste")
        self.assertEqual([paste["inputs"][k] for k in ("left", "top", "width", "height")], [0, 100, 320, 384])
        self.assertTrue(any(n.get("inputs", {}).get("images") == [paste_id, 0] for n in graph.values()))
        prompt = next(n for n in graph.values() if n["class_type"] == "TextEncodeQwenImage21")["inputs"]["prompt"]
        self.assertTrue(prompt.startswith("The flat grey areas") and prompt.endswith("more sky"))
        for bad in ({}, {"middle": 5}, {"left": 0}, {"left": -1}):
            with self.assertRaises(api.ApiError) as caught:
                api._prepare_graph(entries["qwen-image-2.1-edit"], {"parameters": {"prompt": "", "input_image": "source.png", "extend": bad}})
            self.assertIn(caught.exception.code, {"invalid_parameter", None})
        with self.assertRaises(api.ApiError) as caught:
            self.graph(input_image="source.png", extend={"left": 64})
        self.assertEqual(caught.exception.code, "unsupported_parameter")

    def test_extend_paste_aligns_a_zoomed_redraw(self):
        import cv2
        import numpy as np
        from project_nodes import extend
        rng = np.random.default_rng(0)
        canvas = cv2.GaussianBlur(rng.random((300, 400, 3), np.float32), (0, 0), 8)
        canvas = (canvas - canvas.min()) / (canvas.max() - canvas.min())
        region = (100, 0, 200, 300)
        # Like Qwen: the redraw shrinks the original to 0.8 and shifts it.
        zoom = np.float32([[0.8, 0, 50], [0, 0.8, 25]])
        redraw = cv2.warpAffine(canvas, zoom, (400, 300), borderMode=cv2.BORDER_REFLECT)
        redraw = np.dstack([redraw, np.ones((300, 400), np.float32)])  # the Qwen-Image 2.1 VAE decodes RGBA
        result, report = extend.align_and_paste(redraw, canvas, region)
        self.assertTrue(report["aligned"], report)
        np.testing.assert_allclose(result[:, 170:230], canvas[:, 170:230], atol=1e-6)  # inside the feather band: original pixels
        _, report = extend.align_and_paste(rng.random((300, 400, 3), np.float32), canvas, region)
        self.assertFalse(report["aligned"])

    def test_image_tools_crop_and_resize_never_stretch(self):
        from project_nodes import image_tools as tools
        image = Image.new("RGB", (1000, 500), "navy")
        self.assertEqual(tools.crop_box(image.size, {"aspect": "1:1", "anchor": "right"}), (500, 0, 1000, 500))
        with self.assertRaises(tools.ToolError) as caught:
            tools.crop_box(image.size, {"x": 900, "y": 0, "width": 200, "height": 10})
        self.assertEqual(caught.exception.code, "crop_out_of_bounds")
        result, operations, warnings = tools.transform(image, resize={"width": 300, "height": 300}, max_pixels=16_777_216)
        self.assertEqual((result.size, operations[0]["crop"]), ((300, 300), [250, 0, 750, 500]))  # centre crop, not stretch
        result, _, warnings = tools.transform(image, resize={"max_side": 4000}, max_pixels=16_777_216)
        self.assertEqual((result.size, warnings), ((1000, 500), []))  # max_side only shrinks
        face = lambda: (0.445, 0.083, 0.579, 0.152)  # reference_character's face, on a 1116x2000 picture
        x0, y0, x1, y1 = tools.crop_box((1116, 2000), {"around": "face", "framing": "head"}, face)
        self.assertTrue(x1 - x0 == y1 - y0 and y0 < 0.083 * 2000 and x0 < 0.445 * 1116 and x1 > 0.579 * 1116)
        x0, y0, x1, y1 = tools.crop_box((1116, 2000), {"around": "face", "framing": "upper-body", "aspect": "4:5"}, face)
        self.assertTrue(abs((x1 - x0) / (y1 - y0) - 0.8) < 0.01 and x1 <= 1116 and y0 >= 0)
        # A whole-pixel 4:5 face crop came out 537x672; one given side follows the requested shape, not that rounding.
        self.assertEqual(tools.resize_target((537, 672), {"width": 1080}, 0.8), ((1080, 1350), None))
        result, operations, _ = tools.transform(Image.new("RGB", (1116, 2000)), {"around": "face"}, None, max_pixels=16_777_216, find_face=face)
        self.assertEqual(operations[0]["face_box"], [497, 166, 646, 304])
        with self.assertRaises(tools.ToolError) as caught:
            tools.crop_box((1116, 2000), {"around": "face"}, lambda: None)
        self.assertEqual(caught.exception.code, "face_not_found")
        result, _, warnings = tools.transform(image, resize={"scale": 2}, max_pixels=16_777_216)
        self.assertEqual((result.size, [w["code"] for w in warnings]), ((2000, 1000), ["upscaled_without_model"]))

    def test_route_scope_and_exact_capabilities(self):
        entries = {e["id"]: e for e in catalog["workflows"]}
        self.assertIsNone(api._route_score(entries["chatgpt-girl-anima-turbo"], {"domain": "anime"}))
        self.assertIsNone(api._route_score(entries["deepseek-whale-girl-anima-turbo"], {"domain": "bathroom"}))
        self.assertIsNone(api._route_score(entries["deepseek-whale-girl-anima-turbo"], {"input_type": "text"}))
        self.assertIsNotNone(api._route_score(entries["moody-anima-inpaint"], {"workflow": "moody-anima-inpaint"}))
        self.assertIsNone(api._route_score(self.entry, {"input_mode": "image"}))
        self.assertIsNotNone(api._route_score(self.entry, {"domain": "anime", "prompt_granularity": "expert"}))
        self.assertIsNone(api._route_score(entries["anima-turbo-face-reference-ipadapter"], {"family": "sdxl-ipadapter", "input_mode": "face-reference"}))
        with self.assertRaises(api.ApiError):
            api._route_query({"domian": "anime"})

    def test_janku_and_one_obsession_registry_downloader_and_graph_contracts(self):
        entries = {entry["id"]: entry for entry in catalog["workflows"]}
        models = {model["id"]: model for model in api._read_catalog()["model_registry"]}
        expected = {
            "janku-illustrious": (
                "janku-v7-77",
                "JANKU_v7_77.safetensors",
                6938040674,
                "88177D224CE97F60CF3E908F87902F913ED4562BD79D237F84238A4354601EFB",
            ),
            "janku-v5-illustrious": (
                "janku-v5",
                "JANKU_v5_0.safetensors",
                6938040674,
                "217DAAE812DDC3E501C1E972F88E01D5F3BAE8A4C3EAD2F087E8A9CD4B4CDED5",
            ),
            "one-obsession": (
                "one-obsession-v24",
                "OneObsession_v24.safetensors",
                6938040682,
                "2FAE33D7F0D23920D145897827E6015AC098A2B90CB682D07846AE762213100B",
            ),
        }
        downloader = (ROOT / "scripts" / "download_models.ps1").read_text(encoding="utf-8")
        for workflow_id, (model_id, filename, size, sha256) in expected.items():
            entry = entries[workflow_id]
            model = models[model_id]
            self.assertEqual(entry["model_ids"], [model_id])
            self.assertEqual(Path(entry["file"]).suffix, ".json")
            self.assertEqual(Path(model["required_file"]).name, filename)
            self.assertEqual(model["file_sizes"][model["required_file"]], size)
            self.assertEqual(model["source"]["bytes"], size)
            self.assertEqual(model["source"]["sha256"], sha256)
            self.assertTrue(entry["required_files"])
            self.assertEqual(entry["required_files"], [model["required_file"]])
            self.assertIn(workflow_id, downloader)
            self.assertIn(filename, downloader)
            self.assertIn(sha256, downloader)
            graph = api._read_workflow(entry)
            self.assertEqual(graph["1"]["class_type"], "CheckpointLoaderSimple")
            self.assertEqual(graph["1"]["inputs"]["ckpt_name"], filename)
            self.assertEqual(
                {node["class_type"] for node in graph.values()},
                {"CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage"},
            )
            prepared = api._prepare_graph(entry, {"parameters": {"prompt": "1girl, solo, SFW anime portrait"}})
            self.assertEqual(api._binding_value(prepared, entry, "width"), 1024)
            self.assertEqual(api._binding_value(prepared, entry, "height"), 1536)
            self.assertEqual(api._binding_value(prepared, entry, "steps"), entry["defaults"]["steps"])
            for previous_id in entry.get("previous_ids", []):
                self.assertEqual(api._find_workflow(previous_id)["id"], workflow_id)
            with self.assertRaises(api.ApiError):
                api._prepare_graph(entry, {"parameters": {"prompt": "portrait", "unknown": True}})
        self.assertIn(
            "'janku-illustrious', 'janku-v5-illustrious', 'one-obsession', 'deepseek-whale-girl', 'ai-agent-girls', 'ipadapter-face-sdxl'",
            downloader,
        )

    def test_new_route_readiness_tracks_required_file_installation(self):
        entries = {entry["id"]: entry for entry in catalog["workflows"]}
        models = {model["id"]: model for model in api._read_catalog()["model_registry"]}
        for workflow_id, model_id in (
            ("janku-illustrious", "janku-v7-77"),
            ("one-obsession", "one-obsession-v24"),
        ):
            entry = copy.deepcopy(entries[workflow_id])
            model = copy.deepcopy(models[model_id])
            entry["required_files"] = ["model.bin"]
            model["required_file"] = "model.bin"
            model["file_sizes"] = {"model.bin": 4}
            self.assertFalse(api._public_entry(entry, [model])["ready"])
            (self.root / "model.bin").write_bytes(b"1234")
            self.assertTrue(api._public_entry(entry, [model])["ready"])
            (self.root / "model.bin").unlink()

    def test_content_does_not_route_and_old_selectors_are_ignored_with_a_warning(self):
        # Every route makes SFW and explicit images; routes differ by subject and look, so content_mode, input_type and
        # prompt_granularity (API 1.x) must change nothing and say so.
        ready = {"workflows": catalog["workflows"]}
        with patch.object(api, "_public_entry", side_effect=lambda entry, registry: {**entry, "ready": True, "missing_files": []}):
            plain = api._resolve_routes(ready, {"domain": "anime"})
            for old in ({"content_mode": "nsfw"}, {"content_mode": "sfw", "input_type": "text", "prompt_granularity": "detailed"}):
                query, deprecated = api._route_query({"domain": "anime", **old})
                self.assertEqual(query, {"domain": "anime"})
                result = api._resolve_routes(ready, query, deprecated=deprecated)
                self.assertEqual(result["selected_workflow"], plain["selected_workflow"])
                self.assertEqual(sorted(w["parameter"] for w in result["warnings"]), sorted(old))
                self.assertEqual({w["code"] for w in result["warnings"]}, {"deprecated_selector"})
        self.assertEqual(plain["selected_workflow"], "anima-turbo-v1-1")
        self.assertEqual(plain["warnings"], [])
        entries = {entry["id"]: entry for entry in catalog["workflows"]}
        self.assertIsNotNone(api._route_score(entries["janku-illustrious"], {"style": "JANKU v7.77"}))
        self.assertIsNotNone(api._route_score(entries["one-obsession"], {"style": "One Obsession v24"}))

    def test_character_lora_routes_split_by_content_and_bind_their_loras(self):
        entries = {entry["id"]: entry for entry in catalog["workflows"]}
        models = {model["id"]: model for model in api._read_catalog()["model_registry"]}
        downloader = (ROOT / "scripts" / "download_models.ps1").read_text(encoding="utf-8")
        characters = {
            "deepseek-whale-girl": ("deepseek", "大肥鱼", "DeepSeek娘", "deepseek whale girl"),
            "claude-girl": ("Claude-chan", "Claude娘", "claude girl"),
            "chatgpt-girl": ("chatgpt", "ChatGPT龙娘", "GPT娘"),
            "qwen-girl": ("qwen girl", "千问娘", "Qwen-chan"),
            "gemini-girl": ("gemini girl", "Gemini娘", "哈基米"),
            "glm-girl": ("glm girl", "智谱娘", "GLM娘"),
            "grok-girl": ("grok girl", "Grok娘", "grok-chan"),
            "kimi-girl": ("kimi girl", "Kimi娘", "月之暗面娘"),
        }
        # Every character has one Anima route for SFW and NSFW. The JANKU character routes were removed: their LoRAs
        # drew a different design (Claude, DeepSeek) or a less faithful one, and granularity alone could select them.
        anima_only = set(characters)
        for slug, aliases in characters.items():
            anima, janku = entries[f"{slug}-anima-turbo"], entries.get(f"{slug}-janku")
            self.assertEqual(janku is None, slug in anima_only, slug)
            for alias in aliases:
                query = {"domain": alias}
                self.assertIsNotNone(api._route_score(anima, query), alias)
                for other in characters:
                    if other != slug:
                        self.assertIsNone(api._route_score(entries[f"{other}-anima-turbo"], query), (alias, other))
            for entry, loader in ((janku, "LoraLoader"), (anima, "LoraLoaderModelOnly")) if janku else ((anima, "LoraLoaderModelOnly"),):
                (lora_id,) = [m for m in entry["model_ids"] if models.get(m, {}).get("kind") == "lora"]
                model = models[lora_id]
                (lora_name,) = entry["lora_bindings"]
                self.assertIn(model["required_file"], entry["required_files"])
                self.assertIn(model["source"]["sha256"], downloader)
                graph = api._read_workflow(entry)
                self.assertEqual(graph["2"]["class_type"], loader)
                self.assertEqual(graph["2"]["inputs"]["lora_name"], Path(model["required_file"]).name)
                self.assertEqual(entry["lora_bindings"][lora_name]["file"], Path(model["required_file"]).name)
                prepared = api._prepare_graph(entry, {"parameters": {"prompt": "1girl", "loras": {lora_name: 0.5}}})
                self.assertEqual(prepared["2"]["inputs"]["strength_model"], 0.5)
                default = api._prepare_graph(entry, {"parameters": {"prompt": "1girl"}})
                self.assertEqual(default["2"]["inputs"]["strength_model"], entry["defaults"]["loras"][lora_name])
        for query in ({"domain": "qwen"}, {"domain": "qwen", "input_mode": "text-to-image"}):
            self.assertIsNone(api._route_score(entries["qwen-girl-anima-turbo"], query))
        # Bare names that also belong to other characters must not select a character route.
        for slug, domain in (("qwen-girl", "qwen"), ("claude-girl", "Claude Monet style"), ("claude-girl", "Claude von Riegan"),
                             ("gemini-girl", "Gemini Saga"), ("gemini-girl", "Saint Seiya Gemini"), ("kimi-girl", "Kimi Finster"),
                             ("grok-girl", "Ani from Grok")):
            for side in ("anima-turbo", "janku"):
                if f"{slug}-{side}" in entries:
                    self.assertIsNone(api._route_score(entries[f"{slug}-{side}"], {"domain": domain}), domain)
        self.assertNotIn("blue-archive", downloader)

    def test_content_lines_only_where_rating_tags_were_seen_to_work(self):
        # JANKU v7.77 showed a visible SFW/NSFW effect from rating tags; Anima Turbo at CFG 1 did not.
        for entry in catalog["workflows"]:
            is_janku = entry["id"] == "janku-illustrious"
            self.assertEqual("content_lines" in entry.get("prompting", {}), is_janku, entry["id"])
            if is_janku:
                self.assertEqual(set(entry["prompting"]["content_lines"]), {"safe", "explicit"})
                # NSFW is the main use: the default negative must not suppress it.
                default_negative = api._binding_value(api._read_workflow(entry), entry, "negative_prompt")
                self.assertNotIn("nsfw", default_negative, entry["id"])

    def test_image_inputs_match_what_prepare_graph_accepts(self):
        workflows = api._read_catalog()["workflows"]
        uses = {use["input_mode"]: use["routes"] for use in api._reference_uses(workflows)}
        self.assertEqual(uses["image-edit"], ["qwen-image-2.1-edit"])
        self.assertEqual(set(uses["face-reference"]), {"sdxl-face-reference-ipadapter", "anima-turbo-face-reference-ipadapter"})
        pose_families = {"sdxl", "sdxl-illustrious", "sdxl-ipadapter", "anima-turbo", "anima", "anima-ipadapter", "qwen-image-2.1",
                         "flux2-klein"}
        self.assertEqual(set(uses["openpose-control"]), {e["id"] for e in workflows if e["family"] in pose_families
                                                         and {"text-to-image", "face-reference"} & set(e["capabilities"]["input_modes"])})
        self.assertEqual(set(uses["inpaint"]), {"moody-anima-inpaint", "anima-turbo-inpaint"})
        text_only = [e["id"] for e in workflows if not api._image_inputs(e)]
        self.assertEqual(text_only, ["moss-image-sft-t2i"])
        for entry in workflows:
            for name in api._image_inputs(entry):
                if name not in {"input_image", "pose"}:  # pose is attached by its family adapter, not a binding
                    self.assertIn(name, entry["bindings"], entry["id"])
            if "input_image" not in api._image_inputs(entry):
                with self.assertRaises(api.ApiError):
                    api._prepare_graph(entry, {"parameters": {"prompt": "x", "input_image": "missing.png"}})

    def test_pose_routes_only_and_controlnet_feeds_the_sampler(self):
        # A pose sent to a route without an adapter would be silently lost, so routing must never pick one.
        annotated = api._read_catalog()
        with patch.object(api, "_public_entry", side_effect=lambda entry, registry: {**entry, "ready": True, "missing_files": []}):
            result = api._resolve_routes(annotated, {"domain": "anime", "input_mode": "openpose-control"}, limit=20)
            self.assertTrue(result["matches"])
            for match in result["matches"]:
                self.assertTrue(match["workflow"]["pose"]["supported"], match["workflow_id"])
            self.assertEqual(result["selected_workflow"], "anima-turbo-v1-1")
        entry = next(e for e in annotated["workflows"] if e["id"] == "janku-illustrious")
        moss = next(e for e in annotated["workflows"] if e["id"] == "moss-image-sft-t2i")
        with self.assertRaises(api.ApiError) as raised:
            api._resolve_pose(moss, {"parameters": {"prompt": "x", "pose": "full_single_P001_front"}})
        self.assertEqual(raised.exception.code, "pose_unsupported")
        skeleton = {"key": "test", "width": 900, "height": 1200, "people": [{"body": [[0.5, 0.1 + i * 0.04, 0.9] for i in range(18)],
                                                                             "left_hand": None, "right_hand": None}]}
        request = {"skeleton": skeleton, "adapter": api.pose_addon.adapter_for(annotated, entry)}
        with patch.object(api.pose_addon, "render", return_value="_poses/rendered/test.png"), \
                patch.object(api, "_resource_request"):
            graph = api._prepare_graph(entry, {"parameters": {"prompt": "x", "pose_strength": 0.6}}, None, request)
        sampler = next(node for node in graph.values() if node["class_type"] == "KSampler")
        apply = graph[sampler["inputs"]["positive"][0]]
        self.assertEqual((apply["class_type"], sampler["inputs"]["negative"]), ("ControlNetApplyAdvanced", [sampler["inputs"]["positive"][0], 1]))
        self.assertEqual(apply["inputs"]["strength"], 0.6)
        self.assertEqual(graph[apply["inputs"]["image"][0]]["inputs"]["image"], "_poses/rendered/test.png")
        # No size requested: the output takes the 3:4 pose aspect at JANKU's default pixel count.
        default_graph = api._read_workflow(entry)
        default_pixels = api._binding_value(default_graph, entry, "width") * api._binding_value(default_graph, entry, "height")
        width, height = request["size"]
        self.assertAlmostEqual(width / height, 0.75, delta=0.03)
        self.assertAlmostEqual(width * height / default_pixels, 1.0, delta=0.08)
        # Anima: the control LoRA and AnimaControlApply sit on the sampler's model input, after the character LoRA.
        claude = next(e for e in annotated["workflows"] if e["id"] == "claude-girl-anima-turbo")
        request = {"skeleton": skeleton, "adapter": api.pose_addon.adapter_for(annotated, claude)}
        with patch.object(api.pose_addon, "render", return_value="_poses/rendered/test.png") as render,                 patch.object(api, "_resource_request"):
            graph = api._prepare_graph(claude, {"parameters": {"prompt": "x"}}, None, request)
        self.assertEqual(render.call_args.args[3], "wholebody-thin")
        sampler = next(node for node in graph.values() if node["class_type"] == "KSampler")
        apply = graph[sampler["inputs"]["model"][0]]
        lora = graph[apply["inputs"]["model"][0]]
        encode = graph[apply["inputs"]["control_latent"][0]]
        self.assertEqual((apply["class_type"], apply["inputs"]["strength"]), ("AnimaControlApply", 1.0))
        self.assertEqual((lora["class_type"], lora["inputs"]["lora_name"], lora["inputs"]["strength_model"]),
                         ("LoraLoaderModelOnly", apply["inputs"]["control_embedder_path"], 1.0))
        self.assertEqual(graph[lora["inputs"]["model"][0]]["class_type"], "ModelSamplingAuraFlow")
        self.assertEqual(graph[encode["inputs"]["pixels"][0]]["inputs"]["image"], "_poses/rendered/test.png")
        # Qwen: the skeleton becomes reference image 1 and the prompt names it.
        for route_id, slot in (("qwen-image-2.1-t2i", 1),):
            qwen = next(e for e in annotated["workflows"] if e["id"] == route_id)
            graph = api._read_workflow(qwen)
            encoder = next(node for node in graph.values() if node["class_type"] == "TextEncodeQwenImage21")["inputs"]
            api.pose_addon.attach(graph, api.pose_addon.adapter_for(annotated, qwen), "_poses/rendered/test.png", None)
            self.assertEqual(graph[encoder[f"images.image_{slot}"][0]]["inputs"]["image"], "_poses/rendered/test.png", route_id)
            self.assertTrue(encoder["prompt"].startswith(f"Use image {slot} only as a pose reference"), route_id)
            self.assertIn("vae", encoder)
            self.assertNotIn("pose_strength", api._parameter_schema(qwen, api._read_workflow(qwen))["properties"])
        # Klein: the encoded skeleton is a reference latent on the sampler's positive, and the prompt names it.
        klein = next(e for e in annotated["workflows"] if e["id"] == "moody-desire-v30")
        graph = api._read_workflow(klein)
        sampler = next(node for node in graph.values() if node["class_type"] == "KSampler")["inputs"]
        prompt_id = sampler["positive"][0]
        api.pose_addon.attach(graph, api.pose_addon.adapter_for(annotated, klein), "_poses/rendered/test.png", None)
        reference = graph[sampler["positive"][0]]
        self.assertEqual((reference["class_type"], reference["inputs"]["conditioning"]), ("ReferenceLatent", [prompt_id, 0]))
        encode = graph[reference["inputs"]["latent"][0]]
        self.assertEqual(graph[encode["inputs"]["pixels"][0]]["inputs"]["image"], "_poses/rendered/test.png")
        self.assertTrue(graph[prompt_id]["inputs"]["text"].startswith("Use the reference image only as a pose reference"))
        # Edit and inpaint routes take no pose, and a pose never rides along with input_image.
        for route_id in ("qwen-image-2.1-edit", "anima-turbo-inpaint", "moody-anima-inpaint"):
            self.assertFalse(next(e for e in annotated["workflows"] if e["id"] == route_id)["pose"]["supported"], route_id)
        with self.assertRaises(api.ApiError) as raised:
            api._resolve_pose(claude, {"parameters": {"prompt": "x", "pose": "full_single_P001_front", "input_image": "a.png"}})
        self.assertEqual(raised.exception.code, "pose_unsupported")
        # Anima's control embedder needs sizes in multiples of 16.
        request = {"skeleton": skeleton, "adapter": api.pose_addon.adapter_for(annotated, claude)}
        with patch.object(api.pose_addon, "render", return_value="_poses/rendered/test.png") as render,                 patch.object(api, "_resource_request"):
            api._prepare_graph(claude, {"parameters": {"prompt": "x", "width": 1000, "height": 1208}}, None, request)
        self.assertEqual(render.call_args.args[1:3], (992, 1200))

    def test_moody_desire_ranks_behind_qwen_and_anima_and_is_picked_by_style(self):
        # Realism first but behind Qwen-Image (photos) and Anima Turbo (anime); its explicit-realism specialty is the
        # agent's call through the style or workflow id, never a content route.
        ready = {"workflows": catalog["workflows"]}
        with patch.object(api, "_public_entry", side_effect=lambda entry, registry: {**entry, "ready": True, "missing_files": []}):
            def selected(query):
                return api._resolve_routes(ready, query)["selected_workflow"]
            for query in ({"domain": "photo"}, {"domain": "realistic person"}, {"domain": "photo", "style": "realistic"}):
                self.assertEqual(selected(query), "qwen-image-2.1-t2i", query)
            self.assertEqual(selected({"domain": "anime"}), "anima-turbo-v1-1")
            self.assertEqual(selected({"domain": "layout"}), "moss-image-sft-t2i")
            for query in ({"domain": "photo", "style": "moody desire"}, {"style": "moody desire"},
                          {"domain": "realistic person", "style": "klein"}, {"domain": "写实人像"}):
                self.assertEqual(selected(query), "moody-desire-v30", query)

    def test_janku_versions_split_by_style_and_point_at_each_other(self):
        ready = {"workflows": catalog["workflows"]}
        with patch.object(api, "_public_entry", side_effect=lambda entry, registry: {**entry, "ready": True, "missing_files": []}):
            novel = api._resolve_routes(ready, {"domain": "anime", "style": "visual novel"})
            mainstream = api._resolve_routes(ready, {"domain": "anime", "style": "mainstream anime"})
        self.assertEqual(novel["selected_workflow"], "janku-v5-illustrious")
        self.assertEqual(mainstream["selected_workflow"], "janku-illustrious")
        self.assertEqual([p["workflow"] for p in mainstream["selection"]["prefer_instead"]], ["janku-v5-illustrious"])
        self.assertEqual([p["workflow"] for p in novel["selection"]["prefer_instead"]], ["janku-illustrious"])

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
        named = {"project_workflow": "anima", "project_resources": {}}
        instance = Executor()
        with patch.object(runtime_guard, "release_models", side_effect=lambda: events.append("release")):
            for label, data in [("native-1", {}), ("named-1", named), ("named-2", named), ("native-2", {}), ("failure", named), ("recovered", named)]:
                instance.execute({}, label, data, [])
        self.assertEqual(events, ["native-1", "reset", "release", "named-1", "named-2", "reset", "release", "native-2", "reset", "release", "failure", "reset", "release", "reset", "release", "recovered"])
        # A light job (the upscaler) keeps the resident model on both sides of it.
        events.clear()
        light = {"project_workflow": "image-upscale", "project_light": True, "project_resources": {}}
        with patch.object(runtime_guard, "release_models", side_effect=lambda: events.append("release")):
            for label, data in [("named-3", named), ("upscale", light), ("named-4", named)]:
                instance.execute({}, label, data, [])
        self.assertEqual(events, ["named-3", "upscale", "named-4"])

    def test_real_memory_failures_are_reported_as_out_of_memory(self):
        failures = [
            ("torch.OutOfMemoryError", "CUDA out of memory. Tried to allocate 2.00 GiB"),
            ("RuntimeError", "[enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: not enough memory: you tried to allocate 17592186044416 bytes."),
            ("OSError", "The paging file is too small for this operation to complete. (os error 1455)"),
            ("MemoryError", ""),
        ]
        for exception_type, message in failures:
            error = api._history_error({"status": {"messages": [("execution_error", {"exception_type": exception_type, "exception_message": message})]}})
            self.assertEqual(error["code"], "out_of_memory", exception_type)
        error = api._history_error({"status": {"messages": [("execution_error", {"exception_type": "ValueError", "exception_message": "bad input"})]}})
        self.assertNotIn("code", error)


class OutputMetadataTests(unittest.TestCase):
    def test_input_names_are_redacted_and_everything_else_is_kept(self):
        from PIL.PngImagePlugin import PngInfo
        from project_nodes.output_metadata import REDACTED_INPUT, redact_input_names
        with tempfile.TemporaryDirectory() as root:
            for prompt_text in ("Change the teapot to blue.", "把茶壶改成蓝色。"):  # tEXt and iTXt chunks
                graph = {"1": {"class_type": "LoadImage", "inputs": {"image": "asuna-edit-0123456789ab-x1y2z3.png"}},
                         "2": {"class_type": "LoadImageMask", "inputs": {"image": "_fitted/abc.png", "channel": "red"}},
                         "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_text, "clip": ["4", 0]}},
                         "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0]}}}
                info = PngInfo(); info.add_text("prompt", json.dumps(graph, ensure_ascii=False))
                path = Path(root) / "out.png"
                Image.new("RGB", (64, 48), "navy").save(path, pnginfo=info)
                with Image.open(path) as before:
                    pixels = before.tobytes()
                self.assertTrue(redact_input_names(path))
                self.assertFalse(redact_input_names(path))
                with Image.open(path) as after:
                    saved = json.loads(after.info["prompt"])
                    self.assertEqual(after.tobytes(), pixels)
                self.assertEqual(saved["1"]["inputs"]["image"], REDACTED_INPUT)
                self.assertEqual(saved["2"]["inputs"]["image"], REDACTED_INPUT)
                self.assertEqual(saved["3"]["inputs"]["text"], prompt_text)
                self.assertEqual(saved["5"]["inputs"]["pixels"], ["1", 0])
                self.assertNotIn(b"asuna-edit", path.read_bytes())


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
