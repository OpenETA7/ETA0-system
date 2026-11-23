"""Integration boundaries for visual scope and epistemically unknown receipts."""
import asyncio
import base64
import io
import json
import time
from types import SimpleNamespace

from PIL import Image

from embodied_harness.adapters.openeta import OpenETAAdapter
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.agents.slow import SlowAgent
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator


class MemoryStore:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": payload})

    def since(self, cursor):
        return self.events[cursor:]


def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (16, 12), "#345678").save(stream, format="PNG")
    return stream.getvalue()


def host_result(scope=None, include_scope=True, packet="direct-7"):
    body = {
        "robot": {"grip_xyz_m": [0.0, 0.0, 0.5]},
        "feedback": {"retention_proven": False, "placement_proven": False,
                     "status": "observed tool result"},
        "views": [{"source_packet_id": packet, "camera_frame_id": "agentview"}],
        "visible_candidates": [{"candidate_id": "p7o-c2", "bbox_xyxy": [20, 30, 80, 90]}],
        "candidate_semantics": "geometry only",
        "reward": 1.0,
        "official_success": True,
        "hidden_objects": [{"name": "privileged target", "xyz": [9, 8, 7]}],
        "hidden_target_pose": [9, 8, 7],
    }
    if include_scope:
        body["candidate_scope"] = scope
    encoded = base64.b64encode(image_bytes()).decode("ascii")
    return SimpleNamespace(content=[
        SimpleNamespace(type="text", text=json.dumps(body)),
        SimpleNamespace(type="image", data=encoded),
    ])


def adapter_for(tmp_path):
    adapter = object.__new__(OpenETAAdapter)
    adapter.images = tmp_path / "frames"
    adapter.images.mkdir()
    adapter.version = 0
    adapter.generation = 1
    adapter.action_seq = 0
    adapter.schemas = {"observe": {"parameters": {}}}
    adapter.points = {}
    adapter.cached = None
    adapter.last = None
    return adapter


def prepare(adapter, store, scope="direct-7/objects", mission="inspect the current target"):
    context = Context(store)
    coordinator = Coordinator(adapter, context, store)
    coordinator.delegate(Milestone("mission", 1, mission, mission, capabilities=()))
    first = adapter._unpack(host_result(scope=scope, packet="direct-7"))
    assert context.ingest(first)
    return context, coordinator


def test_host_scope_flows_through_adapter_context_and_expires_visual_advice(tmp_path):
    store = MemoryStore()
    adapter = adapter_for(tmp_path)
    context, coordinator = prepare(adapter, store)
    captured = []

    class PixelReview:
        async def decide(self, system, projection, images):
            captured.append(projection)
            return {"kind": "advice", "text": "Use pixel x=120, y=220 near p7o-c2",
                    "action_sensitive": False}

    slow = SlowAgent(PixelReview(), context, coordinator, store)
    asyncio.run(slow.review())

    projected_state = captured[0]["observation"]["state"]
    assert projected_state["candidate_scope"] == "direct-7/objects"
    assert not {"reward", "official_success", "hidden_objects", "hidden_target_pose"} & set(projected_state)
    assert "privileged target" not in repr(captured[0])
    assert "reward" not in repr(captured[0])
    assert projected_state["feedback"]["retention_verification"].startswith("unverified;")
    assert projected_state["feedback"]["placement_verification"].startswith("unverified;")
    assert context.latest.state["feedback"]["retention_proven"] is False
    assert context.latest.state["feedback"]["placement_proven"] is False
    assert len(context.advice) == 1
    assert context.advice[0]["expected_visual_scope"] == "direct-7/objects"

    next_observation = adapter._unpack(host_result(scope="direct-8/objects", packet="direct-8"))
    assert context.ingest(next_observation)
    current = context.projection("fast")
    assert current["advice"] == []
    assert current["target_plan"]["grounding_expired"] is True


def test_free_text_pixel_advice_without_scope_fails_closed(tmp_path):
    store = MemoryStore()
    adapter = adapter_for(tmp_path)
    context = Context(store)
    coordinator = Coordinator(adapter, context, store)
    coordinator.delegate(Milestone("mission", 1, "inspect the current target", "inspect the current target",
                                    capabilities=()))
    unscoped = adapter._unpack(host_result(include_scope=False))
    assert "candidate_scope" not in unscoped.state
    assert context.ingest(unscoped)

    class PixelReview:
        async def decide(self, system, projection, images):
            return {"kind": "advice", "text": "Mark pixel x=120, y=220", "action_sensitive": False}

    slow = SlowAgent(PixelReview(), context, coordinator, store)
    asyncio.run(slow.review())

    assert slow.errors == 1
    assert context.review_state["last_outcome"] == "error"
    assert not context.advice
    assert not any(event["kind"] == "advice.accepted" for event in store.events)
    assert not any(event["kind"] == "advice.stale" for event in store.events)
    model_errors = [event["payload"] for event in store.events if event["kind"] == "slow.model_error"]
    assert "lacks candidate_scope" in model_errors[0]["message"]


def test_unverified_flags_are_reworded_only_in_model_view_not_raw_receipt():
    store = MemoryStore()
    context = Context(store)
    raw_receipt = {
        "status": "completed",
        "detail": {"macro_status": "lift_completed_verify_retention_visually",
                   "feedback": {"retention_proven": False, "placement_proven": False}},
    }

    context.remember("action.receipt", raw_receipt)

    assert store.events[-1]["payload"] == raw_receipt
    assert raw_receipt["detail"]["feedback"] == {
        "retention_proven": False, "placement_proven": False}
    projected_receipt = context.projection("fast")["recent_events"][-1]["detail"]
    projected_feedback = projected_receipt["detail"]["feedback"]
    assert "retention_proven" not in projected_feedback
    assert "placement_proven" not in projected_feedback
    assert projected_feedback["retention_verification"].startswith("unverified;")
    assert projected_feedback["placement_verification"].startswith("unverified;")
