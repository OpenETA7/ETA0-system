"""Synthetic checks for candidate-card image binding and model view projection."""
import asyncio
import base64
import copy
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from embodied_harness.adapters.visual_cards import candidate_gallery
from embodied_harness.agents.model import ModelClient


def test_candidate_gallery_binds_large_ids_to_their_exact_rgb_regions():
    rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    rgb[50:110, 50:110] = (230, 20, 30)
    rgb[200:260, 300:360] = (15, 210, 40)
    candidates = [
        {"candidate_id": "c7", "bbox_xyxy": [50, 50, 110, 110]},
        {"candidate_id": "c12", "bbox_xyxy": [300, 200, 360, 260]},
    ]

    gallery = candidate_gallery(rgb, candidates)

    assert gallery.size == (352, 176)
    assert gallery.getpixel((88, 102)) == (230, 20, 30)
    assert gallery.getpixel((176 + 88, 102)) == (15, 210, 40)

    def expected_header(label):
        header = Image.new("RGB", (176, 30), "#182633")
        ImageDraw.Draw(header).text((8, 2), label,
                                    font=ImageFont.load_default(size=23), fill="white")
        return header

    assert gallery.crop((0, 0, 176, 30)).tobytes() == expected_header("c7").tobytes()
    assert gallery.crop((176, 0, 352, 30)).tobytes() == expected_header("c12").tobytes()


class MockResponse:
    status_code = 200

    def json(self):
        return {"choices": [{"message": {"content": '{"kind":"noop"}'},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}


class MockClient:
    def __init__(self):
        self.request = None
        self.closed = False

    async def post(self, path, json):
        self.request = {"path": path, "json": json}
        return MockResponse()

    async def aclose(self):
        self.closed = True


def test_model_client_attaches_two_images_and_projects_only_the_matching_views(tmp_path):
    async def run():
        image_paths = []
        image_bytes = [b"canonical-agentview", b"candidate-gallery", b"wrist-camera"]
        for index, content in enumerate(image_bytes):
            path = tmp_path / f"image-{index}.bin"
            path.write_bytes(content)
            image_paths.append(str(path))

        views = [
            {"camera_frame_id": "agentview", "source_packet_id": "packet-1"},
            {"camera_frame_id": "candidate_gallery", "source_packet_id": "packet-1"},
            {"camera_frame_id": "wrist", "source_packet_id": "packet-1"},
        ]
        context = {"milestone": {"mission": "pick the marked object"},
                   "observation": {"state": {"views": views, "feedback": {"status": "visible"}}}}
        original_context = copy.deepcopy(context)

        model = ModelClient("http://mock.invalid", "test-key", "test-model", "fast", tmp_path)
        mock_client = MockClient()
        model.client = mock_client
        try:
            decision = await model.decide("test system", context, image_paths)
        finally:
            await model.close()

        assert decision == {"kind": "noop"}
        assert mock_client.closed
        request = mock_client.request["json"]
        user_content = request["messages"][-1]["content"]
        image_items = [item for item in user_content if item["type"] == "image_url"]
        assert len(image_items) == 2
        attached = [base64.b64decode(item["image_url"]["url"].split(",", 1)[1])
                    for item in image_items]
        assert attached == image_bytes[:2]

        serialized_context = next(item["text"] for item in user_content if item["type"] == "text")
        sent_context = json.loads(serialized_context[len(model.input_instruction):])
        sent_state = sent_context["observation"]["state"]
        assert [view["camera_frame_id"] for view in sent_state["views"]] == [
            "agentview", "candidate_gallery"]
        assert sent_state["omitted_views"] == ["wrist"]
        assert "Only the listed views are attached" in sent_state["image_note"]
        assert context == original_context
        assert len(context["observation"]["state"]["views"]) == 3
        assert "omitted_views" not in context["observation"]["state"]

    asyncio.run(run())
