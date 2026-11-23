"""Offline transport regressions for checkpoint image selection."""

import asyncio
import base64
import copy
import json

import httpx
import pytest

from embodied_harness.agents.model import ModelClient


CAMERAS = ("agentview", "candidate_gallery", "robot0_eye_in_hand")
IMAGE_BYTES = (b"agent-scene-image", b"candidate-gallery-image", b"wrist-gripper-image")


def views():
    return [{"camera_frame_id": camera, "source_packet_id": f"packet-{index}"}
            for index, camera in enumerate(CAMERAS)]


def make_images(tmp_path, count=3):
    paths = []
    for index, payload in enumerate(IMAGE_BYTES[:count]):
        path = tmp_path / f"view-{index}.png"
        path.write_bytes(payload)
        paths.append(str(path))
    return tuple(paths)


def request_context(payload):
    text = payload["messages"][-1]["content"][0]["text"]
    return json.loads(text[text.index("{"):])


def attached_image_bytes(payload):
    parts = payload["messages"][-1]["content"][1:]
    return [base64.b64decode(part["image_url"]["url"].split(",", 1)[1]) for part in parts]


async def send_once(tmp_path, *, role, purpose, images, attached_visual=False):
    captured = []

    async def transport(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"kind":"wait"}'},
                                                        "finish_reason": "stop"}]})

    state = {"robot": {"gripper_open": True}, "views": views()}
    context = {"purpose": purpose, "observation": {"id": "obs-current", "state": state}}
    if attached_visual:
        context["attached_visual"] = {"observation_id": "visual-current", "views": views()}
    original_context = copy.deepcopy(context)

    model = ModelClient("http://mock.invalid/v1", "synthetic-test-key", "test-model", role, tmp_path)
    await model.client.aclose()
    model.client = httpx.AsyncClient(base_url="http://mock.invalid/v1/",
                                     transport=httpx.MockTransport(transport), trust_env=False)
    try:
        await model.decide("offline test prompt", context, images)
    finally:
        await model.close()

    trace = json.loads((tmp_path / f"{role}-model.jsonl").read_text().splitlines()[0])
    return captured[0], trace, context, original_context


def test_slow_checkpoint_sends_scene_and_wrist_images_with_matching_attached_visual_metadata(tmp_path):
    async def run():
        images = make_images(tmp_path)
        request, trace, context, original = await send_once(
            tmp_path, role="slow", purpose="checkpoint", images=images, attached_visual=True)

        sent_context = request_context(request)
        attached = sent_context["attached_visual"]
        assert [view["camera_frame_id"] for view in attached["views"]] == [
            "agentview", "robot0_eye_in_hand"]
        assert attached["omitted_views"] == ["candidate_gallery"]
        assert attached_image_bytes(request) == [IMAGE_BYTES[0], IMAGE_BYTES[2]]

        # Request tracing retains the complete input list and separately records
        # which files were actually transmitted, including their source indices.
        assert trace["images"] == list(images)
        assert trace["transmitted_images"] == [images[0], images[2]]
        assert trace["transmitted_image_indices"] == [0, 2]
        assert context == original  # Trimming metadata only affects the request copy.
        assert [v["camera_frame_id"] for v in context["attached_visual"]["views"]] == list(CAMERAS)

    asyncio.run(run())


@pytest.mark.parametrize(("role", "purpose"), [("slow", "initial_plan"), ("fast", "checkpoint")])
def test_non_checkpoint_slow_and_fast_checkpoint_keep_the_first_two_images(tmp_path, role, purpose):
    async def run():
        images = make_images(tmp_path)
        request, trace, _context, _original = await send_once(
            tmp_path, role=role, purpose=purpose, images=images)

        sent_context = request_context(request)
        sent_views = sent_context["observation"]["state"]
        assert [view["camera_frame_id"] for view in sent_views["views"]] == [
            "agentview", "candidate_gallery"]
        assert sent_views["omitted_views"] == ["robot0_eye_in_hand"]
        assert attached_image_bytes(request) == [IMAGE_BYTES[0], IMAGE_BYTES[1]]
        assert trace["images"] == list(images)
        assert trace["transmitted_images"] == list(images[:2])
        assert trace["transmitted_image_indices"] == [0, 1]

    asyncio.run(run())


def test_slow_checkpoint_falls_back_when_wrist_image_index_is_unavailable(tmp_path):
    async def run():
        images = make_images(tmp_path, count=2)
        request, trace, _context, _original = await send_once(
            tmp_path, role="slow", purpose="checkpoint", images=images)

        sent_state = request_context(request)["observation"]["state"]
        assert [view["camera_frame_id"] for view in sent_state["views"]] == [
            "agentview", "candidate_gallery"]
        assert sent_state["omitted_views"] == ["robot0_eye_in_hand"]
        assert attached_image_bytes(request) == [IMAGE_BYTES[0], IMAGE_BYTES[1]]
        assert trace["images"] == list(images)
        assert trace["transmitted_images"] == list(images)
        assert trace["transmitted_image_indices"] == [0, 1]

    asyncio.run(run())
