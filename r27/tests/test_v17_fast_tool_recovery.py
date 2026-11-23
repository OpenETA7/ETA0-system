"""Offline regressions for equivalent tool envelopes and identity recovery views."""
import asyncio
import json
import time
from collections import deque
from types import SimpleNamespace

import jsonschema
import pytest

from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.grounding import IdentityChecker
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


class FakeAdapter:
    generation = 1
    terminal = False

    schemas = {
        "select_candidate": {
            "parameters": {
                "type": "object", "additionalProperties": False,
                "required": ["candidate_id", "part"],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "part": {"type": "string", "enum": ["center", "top_surface", "top_rim"]},
                },
            },
        },
        "grasp_point": {
            "parameters": {
                "type": "object", "additionalProperties": False,
                "required": ["point_id"], "properties": {"point_id": {"type": "string"}},
            },
        },
    }

    def __init__(self):
        self.validation_calls = []
        self.inspections = []
        self.observe_calls = []
        self.execute_calls = []
        self.verify_calls = []
        self.version = 1
        self.points = {
            "p1": {
                "point_id": "p1", "source_packet_id": "packet-1",
                "camera_frame_id": "agentview", "pixel_xy": [100, 120],
                "xyz_m": [0.1, 0.2, 0.3],
                "visual_candidate": {"candidate_id": "p1o-c1", "bbox_xyxy": [70, 90, 130, 150]},
            },
        }

    def validate_args(self, capability, args):
        self.validation_calls.append((capability, dict(args)))
        jsonschema.validate(args, self.schemas[capability]["parameters"])

    async def inspect(self, tool, args, action_seq):
        self.inspections.append((tool, dict(args), action_seq))
        self.version += 1
        views = [{"source_packet_id": "packet-1", "camera_frame_id": "agentview",
                  "is_current_observation": True}]
        state = {"candidate_scope": "packet-1/objects", "views": views}
        images = ("agentview.png",)
        if tool == "select_candidate":
            state["feedback"] = {"point": self.points["p1"]}
        elif tool == "observe":
            crop = list(args["crop_xyxy"])
            views.append({"source_packet_id": "packet-1", "camera_frame_id": "zoom_preview",
                          "source_camera": args["camera"], "original_crop_xyxy": crop})
            images = ("agentview.png", "selected-crop.png")
        else:
            raise AssertionError(f"unexpected read-only tool: {tool}")
        return Observation(f"inspect-{self.version}", self.version, 1, time.monotonic(), 0.0,
                           action_seq, state, images)

    async def observe(self, action_seq):
        self.observe_calls.append(action_seq)
        self.version += 1
        views = [
            {"source_packet_id": "packet-1", "camera_frame_id": "agentview"},
            {"source_packet_id": "packet-1", "camera_frame_id": "candidate_gallery",
             "source_camera": "agentview", "measurement_allowed": False},
            {"source_packet_id": "packet-1", "camera_frame_id": "robot0_eye_in_hand"},
        ]
        return Observation(f"full-{self.version}", self.version, 1, time.monotonic(), 0.0,
                           action_seq, {"candidate_scope": "packet-1/objects", "views": views},
                           ("agentview.png", "candidate-gallery.png", "wrist.png"))

    async def execute(self, *args):
        self.execute_calls.append(args)
        raise AssertionError("identity rejection or read-only inspection must not dispatch physics")

    async def verify(self, _observation_id):
        self.verify_calls.append(_observation_id)
        return {"status": "not_verified", "evidence": {}}


class Actor:
    def __init__(self, decision):
        self.decision = decision
        self.history = deque([{"role": "assistant", "content": "old actor claim"}], maxlen=6)

    async def decide(self, _system, _projection, _images):
        return self.decision


class MismatchJudge:
    def __init__(self):
        self.history = deque([{"role": "assistant", "content": "actor rationale"}], maxlen=6)
        self.calls = []

    async def decide(self, _system, projection, images):
        self.calls.append((projection, tuple(images)))
        return {"reference_appearance": "blue jar", "selected_appearance": "red bottle",
                "verdict": "mismatch", "reason": "color and shape differ"}


def agent_setup(tmp_path, decision):
    store = Store()
    context = Context(store)
    context.milestone = Milestone("mission", 1, "select or grasp the jar", "select or grasp the jar",
                                  capabilities=("select_candidate", "grasp_point"))
    initial_views = [{"source_packet_id": "packet-1", "camera_frame_id": "agentview"}]
    context.ingest(Observation("initial", 1, 1, time.monotonic(), 0.0, 0,
                               {"candidate_scope": "packet-1/objects", "views": initial_views},
                               ("agentview.png",)))
    adapter = FakeAdapter()
    coordinator = Coordinator(adapter, context, store)
    actor = Actor(decision)
    agent = FastAgent(actor, context, coordinator, store)
    return store, context, adapter, coordinator, actor, agent


@pytest.mark.parametrize(
    ("decision", "recovery"),
    [
        ({"kind": "tool", "tool": "select_candidate", "candidate_id": "p4o-c2", "part": "top_rim"},
         "schema_named_top_level_args"),
        ({"kind": "tool", "tool": "select_candidate",
          "args": '{"candidate_id":"p4o-c2","part":"top_rim"}'},
         "json_encoded_args"),
    ],
)
def test_equivalent_tool_envelopes_reach_normal_schema_validation_and_inspection(
    tmp_path, decision, recovery,
):
    async def run():
        store, context, adapter, coordinator, _actor, agent = agent_setup(tmp_path, decision)

        assert await agent.run(1) == "decision_budget"

        expected = {"candidate_id": "p4o-c2", "part": "top_rim"}
        assert adapter.validation_calls == [("select_candidate", expected)]
        assert adapter.inspections == [("select_candidate", expected, 0)]
        assert context.latest.id == "inspect-2"
        assert adapter.verify_calls == ["inspect-2"]
        recovered = [event["payload"] for event in store.events if event["kind"] == "fast.format_recovered"]
        assert recovered == [{"tool": "select_candidate", "recovery": recovery, "args": expected}]
        assert coordinator.receipts == {}
        assert adapter.execute_calls == []

    asyncio.run(run())


@pytest.mark.parametrize(
    "decision",
    [
        # Unknown flat field must not be ignored.
        {"kind": "tool", "tool": "select_candidate", "candidate_id": "p4o-c2",
         "part": "top_rim", "guess": "top rim"},
        # Nested args plus any schema-named flat field is ambiguous, even if values agree.
        {"kind": "tool", "tool": "select_candidate",
         "args": {"candidate_id": "p4o-c2", "part": "top_rim"}, "candidate_id": "p4o-c2"},
        {"kind": "tool", "tool": "select_candidate", "args": ["p4o-c2", "top_rim"]},
        {"kind": "tool", "tool": "select_candidate", "args": '{"candidate_id":'},
        # Recovery preserves omissions; it never invents the required part.
        {"kind": "tool", "tool": "select_candidate", "candidate_id": "p4o-c2"},
        {"kind": "tool", "tool": "not-a-known-tool", "args": {}},
    ],
)
def test_malformed_unknown_or_ambiguous_tool_envelopes_fail_before_inspection(tmp_path, decision):
    async def run():
        _store, _context, adapter, coordinator, _actor, agent = agent_setup(tmp_path, decision)

        assert await agent.run(1) == "decision_budget"

        assert adapter.inspections == []
        assert adapter.execute_calls == []
        assert coordinator.receipts == {}
        assert agent.errors == 1
        recoveries = [event["payload"] for event in _store.events
                      if event["kind"] == "fast.format_recovered"]
        if decision.get("args") == ["p4o-c2", "top_rim"] or isinstance(decision.get("args"), str):
            assert adapter.validation_calls == []
            assert recoveries == []
        if decision.get("args") is None and set(decision) == {"kind", "tool", "candidate_id"}:
            # Incomplete flat arguments retain exactly what was supplied and are rejected by
            # the existing adapter schema validator, without filling in a guessed part.
            assert adapter.validation_calls == [("select_candidate", {"candidate_id": "p4o-c2"})]
            assert recoveries == [{"tool": "select_candidate", "recovery": "schema_named_top_level_args",
                                  "args": {"candidate_id": "p4o-c2"}}]
        if decision.get("tool") in {"not-a-known-tool"} or "guess" in decision or "candidate_id" in decision and "args" in decision:
            assert recoveries == []

    asyncio.run(run())


@pytest.mark.parametrize("tool", ["select_candidate", "grasp_point"])
def test_identity_rejection_restores_full_current_scope_view_without_physical_action(tmp_path, tool):
    async def run():
        decision = ({"kind": "tool", "tool": "select_candidate",
                     "args": {"candidate_id": "p1o-c1", "part": "top_surface"}}
                    if tool == "select_candidate" else
                    {"kind": "tool", "tool": "grasp_point", "args": {"point_id": "p1"}})
        store, context, adapter, coordinator, actor, agent = agent_setup(tmp_path, decision)
        judge = MismatchJudge()
        agent.identity_checker = IdentityChecker(judge, context, coordinator, store)

        assert await agent.run(1) == "decision_budget"

        assert not actor.history
        assert adapter.observe_calls == [0]
        assert context.action_seq == 0
        assert coordinator.receipts == {}
        assert coordinator.running is None
        assert adapter.execute_calls == []
        assert context.latest.state["candidate_scope"] == "packet-1/objects"
        assert [view["camera_frame_id"] for view in context.latest.state["views"]] == [
            "agentview", "candidate_gallery", "robot0_eye_in_hand"]
        assert context.latest.images == ("agentview.png", "candidate-gallery.png", "wrist.png")
        assert context.local_state["identity_candidates"]["visual_scope"] == "packet-1/objects"
        assert context.local_state["identity_candidates"]["checks"]["p1o-c1"]["verdict"] == "mismatch"
        assert any(event["kind"] == "fast.history_reset" for event in store.events)
        assert any(event["kind"] == "identity.exploration_view_restored" for event in store.events)
        assert judge.calls and adapter.inspections
        assert not any(event["kind"] == "action.accepted" for event in store.events)

    asyncio.run(run())

