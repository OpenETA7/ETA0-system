"""Offline regressions for candidate-gallery binding and scoped identity evidence."""
import asyncio
import base64
import io
import json
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from embodied_harness.adapters.libero_direct import LiberoHost
from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.grounding import IdentityChecker
from embodied_harness.agents.model import ModelClient
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator


class Store:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": payload})

    def since(self, cursor):
        return self.events[cursor:]


def identity_point(point_id, candidate_id, xyz, bbox, *, packet="direct-7"):
    return {
        "point_id": point_id,
        "source_packet_id": packet,
        "camera_frame_id": "agentview",
        "pixel_xy": [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2],
        "xyz_m": list(xyz),
        "visual_candidate": {"candidate_id": candidate_id, "bbox_xyxy": list(bbox)},
    }


class Inspector:
    generation = 1
    terminal = False

    def __init__(self, points, scope="direct-7/objects"):
        self.points = points
        self.scope = scope
        self.calls = []
        self.version = 1
        self.schemas = {"grasp_point": {"parameters": {"properties": {
            "point_id": {"type": "string"}}}}}

    async def inspect(self, tool, args, action_seq):
        self.calls.append((tool, args, action_seq))
        crop = list(args["crop_xyxy"])
        views = [
            {"source_packet_id": "direct-7", "camera_frame_id": "agentview",
             "is_current_observation": True},
            {"source_packet_id": "direct-7", "camera_frame_id": "zoom_preview",
             "source_camera": args["camera"], "original_crop_xyxy": crop},
        ]
        self.version += 1
        return Observation(
            f"crop-{self.version}", self.version, self.generation, time.monotonic(), 0.0,
            action_seq, {"candidate_scope": self.scope, "views": views},
            ("canonical.png", "selected-crop.png"),
        )

    def validate_args(self, _capability, _args):
        pass

    async def execute(self, *_args):
        raise AssertionError("a negative identity judgment must prevent physical dispatch")

    async def verify(self, _observation_id):
        return {"status": "not_verified"}


class Judge:
    def __init__(self, verdict="mismatch"):
        self.verdict = verdict
        self.calls = []
        self.history = deque([{"role": "assistant", "content": "irrelevant actor history"}], maxlen=6)

    async def decide(self, system, context, images):
        self.calls.append((system, context, tuple(images)))
        return {
            "reference_appearance": "blue jar",
            "selected_appearance": "yellow can",
            "verdict": self.verdict,
            "reason": "visible body color and shape differ",
        }


def checker_setup(points, *, verdict="mismatch", scope="direct-7/objects"):
    store = Store()
    context = Context(store)
    context.milestone = Milestone("mission-id", 1, "choose the jar", "choose the jar",
                                  capabilities=("grasp_point",))
    context.ingest(Observation(
        "initial", 1, 1, time.monotonic(), 0.0, 0,
        {"candidate_scope": scope, "views": []},
    ))
    adapter = Inspector(points, scope=scope)
    coordinator = SimpleNamespace(adapter=adapter, lock=asyncio.Lock())
    judge = Judge(verdict)
    checker = IdentityChecker(judge, context, coordinator, store)
    return store, context, adapter, judge, checker


def test_rejected_candidate_checks_accumulate_by_id_and_cached_retry_skips_model(tmp_path):
    async def run():
        points = {
            "p1": identity_point("p1", "p7o-c1", (0.10, 0.20, 0.30), [20, 30, 70, 90]),
            "p2": identity_point("p2", "p7o-c2", (0.20, 0.20, 0.30), [120, 35, 180, 95]),
        }
        store, context, adapter, judge, checker = checker_setup(points)

        assert not await checker.check("p1")
        assert not await checker.check("p2")
        assert not await checker.check("p1")  # same measured point reuses its frame-local verdict

        ledger = context.local_state["identity_candidates"]
        assert ledger["visual_scope"] == "direct-7/objects"
        assert set(ledger["checks"]) == {"p7o-c1", "p7o-c2"}
        assert ledger["checks"]["p7o-c1"] == {
            "verdict": "mismatch",
            "selected_appearance": "yellow can",
            "reason": "visible body color and shape differ",
        }
        assert len(judge.calls) == 2
        assert len(adapter.calls) == 2
        assert sum(event["kind"] == "identity.cached" for event in store.events) == 1

    asyncio.run(run())


@pytest.mark.parametrize("next_scope", ["direct-8/objects", "direct-7/horizontal_bars"])
def test_identity_candidate_ledger_survives_same_scope_observation_and_expires_on_scope_change(next_scope):
    store = Store()
    context = Context(store)
    context.ingest(Observation(
        "initial", 1, 1, time.monotonic(), 0.0, 0,
        {"candidate_scope": "direct-7/objects"},
    ))
    context.local_state["identity_candidates"] = {
        "visual_scope": "direct-7/objects", "checks": {"p7o-c1": {"verdict": "mismatch"}},
    }

    assert context.ingest(Observation(
        "same-scope-new-view", 2, 1, time.monotonic(), 0.0, 0,
        {"candidate_scope": "direct-7/objects"},
    ))
    assert "identity_candidates" in context.local_state

    assert context.ingest(Observation(
        "changed-scope", 3, 1, time.monotonic(), 1.0, 1,
        {"candidate_scope": next_scope},
    ))
    assert "identity_candidates" not in context.local_state


def test_fast_history_reset_after_rejection_preserves_structured_candidate_ledger():
    async def run():
        points = {"p1": identity_point("p1", "p7o-c1", (0.10, 0.20, 0.30), [20, 30, 70, 90])}
        store, context, adapter, judge, checker = checker_setup(points)
        coordinator = Coordinator(adapter, context, store)

        class Actor:
            def __init__(self):
                self.history = deque([{"role": "assistant", "content": "p1 is the jar"}], maxlen=6)

            async def decide(self, _system, _projection, _images):
                return {"kind": "tool", "tool": "grasp_point", "args": {"point_id": "p1"}}

        actor = Actor()
        fast = FastAgent(actor, context, coordinator, store)
        fast.identity_checker = checker

        assert await fast.run(1) == "decision_budget"
        assert not actor.history
        assert context.local_state["identity_candidates"]["checks"]["p7o-c1"]["verdict"] == "mismatch"
        assert context.action_seq == 0
        assert coordinator.running is None
        assert not any(event["kind"] == "action.accepted" for event in store.events)
        assert sum(event["kind"] == "fast.history_reset" for event in store.events) == 1
        assert judge.calls and len(adapter.calls) == 1

    asyncio.run(run())


class MockResponse:
    status_code = 200

    def json(self):
        return {"choices": [{"message": {"content": '{"kind":"noop"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class MockClient:
    def __init__(self):
        self.request = None

    async def post(self, path, json):
        self.request = {"path": path, "json": json}
        return MockResponse()

    async def aclose(self):
        pass


def host_with_bilateral_hold():
    host = object.__new__(LiberoHost)
    host.state = lambda: {"grip_xyz_m": [0.0, 0.0, 0.8]}
    host.contact = lambda: {
        "left_fingerpad_contact": True,
        "right_fingerpad_contact": True,
        "contact_pattern": "bilateral_pads",
        "retention_proven": False,
    }
    host.sim = SimpleNamespace(data=SimpleNamespace(time=4.25))
    host.execution_accounting = {"control_steps": 18}
    host.feedback = {"motion": {"execution_state": "completed"}}
    host.binding = True
    host.generation = 3
    host.packet = 27
    host.proposal_mode = "objects"
    host.obs = {
        "agentview_image": np.full((512, 512, 3), [230, 20, 30], dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((512, 512, 3), [15, 20, 220], dtype=np.uint8),
    }
    candidates = [
        {"candidate_id": "p27o-c1", "component": 1, "bbox_xyxy": [50, 60, 110, 120]},
        {"candidate_id": "p27o-c2", "component": 2, "bbox_xyxy": [300, 200, 360, 260]},
    ]
    host.candidates = lambda: (candidates, None, None)
    return host


async def model_request_with_images(tmp_path, body, image_content):
    paths = []
    for index, block in enumerate(image_content):
        path = tmp_path / f"host-image-{index}.png"
        path.write_bytes(base64.b64decode(block.data))
        paths.append(str(path))

    model = ModelClient("http://mock.invalid", "offline-key", "test-model", "fast", tmp_path)
    await model.client.aclose()
    client = MockClient()
    model.client = client
    try:
        decision = await model.decide(
            "test system", {"observation": {"id": "host-observation", "state": {"views": body["views"]}}},
            paths,
        )
    finally:
        await model.close()
    return decision, client.request, paths


def test_bilateral_hold_keeps_gallery_second_and_model_images_stay_aligned_with_views(tmp_path):
    async def run():
        host = host_with_bilateral_hold()
        response = host.result()
        body = json.loads(response.content[0].text)
        image_blocks = response.content[1:]

        assert body["tactile"]["contact_pattern"] == "bilateral_pads"
        assert body["candidate_scope"] == "direct-27/objects"
        assert [view["camera_frame_id"] for view in body["views"]] == [
            "agentview", "candidate_gallery", "robot0_eye_in_hand"]
        assert body["views"][1]["source_camera"] == "agentview"
        assert body["views"][1]["measurement_allowed"] is False
        assert [candidate["candidate_id"] for candidate in body["visible_candidates"]] == [
            "p27o-c1", "p27o-c2"]
        assert len(image_blocks) == 3
        gallery = Image.open(io.BytesIO(base64.b64decode(image_blocks[1].data)))
        assert gallery.size == (352, 176)
        assert Image.open(io.BytesIO(base64.b64decode(image_blocks[0].data))).size == (512, 512)
        assert Image.open(io.BytesIO(base64.b64decode(image_blocks[2].data))).size == (512, 512)

        decision, request, _paths = await model_request_with_images(tmp_path, body, image_blocks)
        assert decision == {"kind": "noop"}
        user_content = request["json"]["messages"][-1]["content"]
        image_items = [item for item in user_content if item["type"] == "image_url"]
        attached = [base64.b64decode(item["image_url"]["url"].split(",", 1)[1]) for item in image_items]
        emitted = [base64.b64decode(block.data) for block in image_blocks]
        assert attached == emitted[:2]
        serialized = next(item["text"] for item in user_content if item["type"] == "text")
        sent = json.loads(serialized[len("Continue from the executed previous decision. This is the NEW tool result/current snapshot, not a new episode. Choose the NEXT decision.\n"):])
        sent_views = sent["observation"]["state"]["views"]
        assert [view["camera_frame_id"] for view in sent_views] == ["agentview", "candidate_gallery"]
        assert sent["observation"]["state"]["omitted_views"] == ["robot0_eye_in_hand"]

        wrist_response = host.result(camera="wrist")
        wrist_body = json.loads(wrist_response.content[0].text)
        assert [view["camera_frame_id"] for view in wrist_body["views"]] == ["robot0_eye_in_hand"]
        assert len(wrist_response.content[1:]) == 1

    asyncio.run(run())

