"""Check actual model requests keep supervision separate from motor interfaces."""
import asyncio
import json
import time

import httpx
import pytest
from PIL import Image

from embodied_harness.agents.milestone_prompts import CHECKPOINT, PLAN
from embodied_harness.agents.model import ModelClient
from embodied_harness.agents.prompts import FAST, SLOW_DELEGATE, SLOW_REVIEW
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context


class Store:
    def emit(self, *args, **kwargs):
        pass


SCHEMAS = {"move_to": {
    "description": "EXECUTOR_CATALOG_SENTINEL",
    "parameters": {"type": "object", "properties": {"executor_only_arg": {"type": "number"}}},
}}


def populated_context():
    context = Context(Store())
    context.milestone = Milestone("stage-1", 1, "put the cup in the bowl", "hold the correct cup",
                                  capabilities=("move_to",), constraints=("preserve cup identity",),
                                  guards=({"tools": ["move_to"], "predicate": "bilateral_contact"},))
    context.mission_plan = {"milestones": [{"goal": "hold the correct cup", "allowed_tools": ["move_to"]}]}
    context.ingest(Observation("obs-1", 1, 1, time.monotonic(), 0.0, 1, {
        "tool_schemas": SCHEMAS,
        "robot": {"gripper": {"openness": 0.2}},
        "tactile": {"left_fingerpad_contact": True, "right_fingerpad_contact": False},
        "feedback": {"motion": {"execution_state": "partial"}},
        "views": [{"camera_frame_id": camera} for camera in ("agentview", "candidate_gallery", "wrist")],
    }))
    context.remember("action.accepted", {"request": {
        "capability": "move_to", "args": {"executor_only_arg": 0.123},
    }})
    context.remember("action.receipt", {"status": "completed", "detail": {
        "feedback": {"motion": {"execution_state": "partial"}},
    }})
    return context


@pytest.mark.parametrize("prompt", [SLOW_DELEGATE, SLOW_REVIEW, PLAN, CHECKPOINT])
def test_slow_wire_request_has_evidence_without_execution_interface(tmp_path, prompt):
    image = tmp_path / "view.png"
    Image.new("RGB", (2, 2)).save(image)
    context = populated_context()
    captured = []

    async def transport(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"kind":"noop"}'}}]})

    async def run():
        model = ModelClient("https://offline.invalid/v1", "test-key", "slow-model", "slow", tmp_path)
        await model.client.aclose()
        model.client = httpx.AsyncClient(base_url="https://offline.invalid/v1/", transport=httpx.MockTransport(transport))
        try:
            # Include history in the second call as well as the current projection.
            for _ in range(2):
                await model.decide(prompt, context.projection("slow"), (str(image),) * 3)
        finally:
            await model.close()

    asyncio.run(run())
    for request in captured:
        text = json.dumps(request)
        for forbidden in ("EXECUTOR_CATALOG_SENTINEL", "executor_only_arg", "Available tools",
                          "observe(camera", "select_candidate(", "mark_point(", 'allowed_tools'):
            assert forbidden not in text
        assert "tools" not in request
        assert "preserve cup identity" in text
        assert "left_fingerpad_contact" in text
        assert "partial" in text
        assert "unseen view" in text
    # Projection does not corrupt Fast's shared state or permission checks.
    fast_view = context.projection("fast")
    assert fast_view["milestone"]["capabilities"] == ("move_to",)
    assert fast_view["local_state"]["last_action"]["args"] == {"executor_only_arg": 0.123}


def test_slow_cannot_accidentally_bind_executor_catalog(tmp_path):
    async def run():
        model = ModelClient("https://offline.invalid/v1", "test-key", "slow-model", "slow", tmp_path)
        try:
            with pytest.raises(ValueError, match="not execution tool interfaces"):
                model.bind_tools(SCHEMAS)
        finally:
            await model.close()
    asyncio.run(run())


def test_strong_executor_still_receives_tools(tmp_path):
    captured = []

    async def transport(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"kind":"wait"}'}}]})

    async def run():
        # The all-Slow baseline uses the strong model in the executor role.
        model = ModelClient("https://offline.invalid/v1", "test-key", "slow-model", "fast", tmp_path)
        await model.client.aclose()
        model.client = httpx.AsyncClient(base_url="https://offline.invalid/v1/", transport=httpx.MockTransport(transport))
        model.bind_tools(SCHEMAS)
        try:
            await model.decide(FAST, populated_context().projection("fast"))
        finally:
            await model.close()
    asyncio.run(run())
    assert "EXECUTOR_CATALOG_SENTINEL" in captured[0]["messages"][0]["content"]
    assert "executor_only_arg" in json.dumps(captured[0])
