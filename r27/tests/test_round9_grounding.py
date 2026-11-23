"""Synthetic regressions for memory-grounded identity checks and review recovery."""
import asyncio
import time
from collections import deque
from types import SimpleNamespace

import pytest

from embodied_harness.agents.grounding import IdentityChecker
from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.slow import SlowAgent
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


def observation(obs_id, version, state=None, images=()):
    return Observation(obs_id, version, 1, time.monotonic(), 0.0, 0,
                       state or {}, tuple(images))


def point(point_id, packet="frame-1", camera="agentview", xyz=(0.1, 0.2, 0.3)):
    return {"point_id": point_id, "source_packet_id": packet,
            "camera_frame_id": camera, "pixel_xy": [100, 120], "xyz_m": list(xyz)}


class Judge:
    def __init__(self, verdict="match"):
        self.history = deque([{"role": "assistant", "content": "actor says this is the red can"}], maxlen=6)
        self.verdict = verdict
        self.calls = []

    async def decide(self, system, context, images):
        self.calls.append((system, context, tuple(images)))
        return {"reference_appearance": "blue jar", "selected_appearance": "blue jar",
                "verdict": self.verdict, "reason": "visible color and shape agree"}


class Inspector:
    generation = 1
    terminal = False

    def __init__(self, points, crop_delta=(0, 0, 0, 0)):
        self.points = points
        self.schemas = {
            "grasp_point": {"parameters": {"properties": {"point_id": {"type": "string"}}}},
            "select_candidate": {"parameters": {"properties": {
                "candidate_id": {"type": "string"}, "part": {"type": "string"}}}},
        }
        self.crop_delta = crop_delta
        self.calls = []
        self.observe_calls = []
        self.packet = "frame-1"
        self.scope = "frame-1/objects"
        self.observation_version = 1

    async def inspect(self, tool, args, action_seq):
        self.calls.append((tool, args, action_seq))
        views = [{"source_packet_id": self.packet, "camera_frame_id": "agentview",
                  "is_current_observation": True}]
        state = {"candidate_scope": self.scope, "views": views}
        images = ("canonical.png",)
        if tool == "select_candidate":
            state["feedback"] = {"point": self.points["p1"]}
        elif tool == "observe":
            crop = list(args["crop_xyxy"])
            crop = [value + delta for value, delta in zip(crop, self.crop_delta)]
            views.append({"source_packet_id": self.packet, "camera_frame_id": "zoom_preview",
                          "source_camera": args["camera"], "original_crop_xyxy": crop})
            images = ("canonical.png", "crop.png")
        else:
            raise AssertionError(f"unexpected inspect tool: {tool}")
        self.observation_version += 1
        return observation(f"observation-{self.observation_version}",
                           self.observation_version, state, images)

    async def observe(self, action_seq):
        self.observe_calls.append(action_seq)
        self.observation_version += 1
        views = [
            {"source_packet_id": self.packet, "camera_frame_id": "agentview"},
            {"source_packet_id": self.packet, "camera_frame_id": "candidate_gallery",
             "source_camera": "agentview", "measurement_allowed": False},
            {"source_packet_id": self.packet, "camera_frame_id": "robot0_eye_in_hand"},
        ]
        return observation(f"full-view-{self.observation_version}", self.observation_version,
                           {"candidate_scope": self.scope, "views": views},
                           ("agentview.png", "candidate-gallery.png", "wrist.png"))

    def validate_args(self, capability, args):
        pass

    async def execute(self, *args):
        raise AssertionError("grounding rejection must happen before physical dispatch")

    async def verify(self, observation_id):
        return {"status": "not_verified"}


def make_checker(points, judge=None, crop_delta=(0, 0, 0, 0)):
    store = Store()
    context = Context(store)
    context.milestone = Milestone("mission-id", 1, "pick up the blue jar", "pick up the blue jar",
                                  capabilities=("grasp_point",))
    adapter = Inspector(points, crop_delta=crop_delta)
    context.ingest(observation("initial", 1,
                               {"candidate_scope": adapter.scope, "views": []}, ("initial.png",)))
    coordinator = SimpleNamespace(adapter=adapter, lock=asyncio.Lock())
    judge = judge or Judge()
    checker = IdentityChecker(judge, context, coordinator, store)
    return checker, judge, adapter, context, store


def test_mismatching_memory_identity_blocks_fast_physical_grasp():
    async def run():
        selected = point("p1")
        checker, judge, adapter, context, store = make_checker({"p1": selected}, Judge("mismatch"))
        coordinator = Coordinator(adapter, context, store)
        class FastModel:
            def __init__(self):
                self.history = deque([{"role": "assistant", "content": "actor claims p1 is the jar"}], maxlen=6)

            async def decide(self, system, projection, images):
                return {"kind": "tool", "tool": "grasp_point", "args": {"point_id": "p1"}}

        fast_model = FastModel()
        fast = FastAgent(fast_model, context, coordinator, store)
        fast.identity_checker = checker
        status = await fast.run(1)

        assert status == "decision_budget"
        assert fast.errors == 0
        assert not fast_model.history
        assert context.local_state["identity_check"]["verdict"] == "mismatch"
        assert context.action_seq == 0
        assert coordinator.running is None
        assert not any(event["kind"] == "action.accepted" for event in store.events)
        assert adapter.observe_calls == [0]
        assert [view["camera_frame_id"] for view in context.latest.state["views"]] == [
            "agentview", "candidate_gallery", "robot0_eye_in_hand"]
        assert judge.calls and judge.calls[0][1] == {"mission": "pick up the blue jar"}

    asyncio.run(run())


def test_wrong_or_stale_observed_crop_does_not_authorize_identity_match():
    async def run():
        checker, judge, adapter, context, _ = make_checker(
            {"p1": point("p1")}, crop_delta=(1, 0, 0, 0))

        with pytest.raises(ValueError):
            await checker.check("p1")

        assert len(adapter.calls) == 1
        assert judge.calls == [], "the judge must not see a crop whose provenance does not match the request"

    asyncio.run(run())


def test_identity_cache_is_scoped_to_camera_frame_and_position():
    async def run():
        points = {
            "same": point("same"),
            "same_again": point("same_again"),
            "other_camera": point("other_camera", camera="robot0_eye_in_hand"),
            "other_position": point("other_position", xyz=(0.11, 0.2, 0.3)),
            "new_frame": point("new_frame", packet="frame-2"),
        }
        checker, judge, adapter, _, _ = make_checker(points)

        assert await checker.check("same")
        assert await checker.check("same_again")  # same frame and position may reuse the verdict
        assert await checker.check("other_camera")
        assert await checker.check("other_position")
        adapter.packet = "frame-2"  # read-only observe does not advance the selected source frame
        adapter.scope = "frame-2/objects"
        assert await checker.check("new_frame")

        assert len(judge.calls) == 4
        assert len(adapter.calls) == 4

    asyncio.run(run())


def test_top_surface_selection_checks_identity_before_any_physical_action():
    async def run():
        selected = point("p1")
        checker, judge, adapter, context, store = make_checker({"p1": selected}, Judge("mismatch"))
        coordinator = Coordinator(adapter, context, store)

        class FastModel:
            def __init__(self):
                self.history = deque([{"role": "assistant", "content": "p1 is the jar"}], maxlen=6)

            async def decide(self, system, projection, images):
                return {"kind": "tool", "tool": "select_candidate",
                        "args": {"candidate_id": "p1o-c1", "part": "top_surface"}}

        fast_model = FastModel()
        fast = FastAgent(fast_model, context, coordinator, store)
        fast.identity_checker = checker

        status = await fast.run(1)

        assert status == "decision_budget"
        assert fast.errors == 0
        assert [call[0] for call in adapter.calls] == ["select_candidate", "observe"]
        assert judge.calls and judge.calls[0][1] == {"mission": "pick up the blue jar"}
        assert not fast_model.history
        assert context.local_state["identity_check"]["verdict"] == "mismatch"
        assert context.action_seq == 0
        assert not any(event["kind"] == "action.accepted" for event in store.events)
        assert adapter.observe_calls == [0]
        assert [view["camera_frame_id"] for view in context.latest.state["views"]] == [
            "agentview", "candidate_gallery", "robot0_eye_in_hand"]

    asyncio.run(run())


def test_identity_judge_receives_no_actor_history():
    async def run():
        checker, judge, _, _, _ = make_checker({"p1": point("p1")})

        assert await checker.check("p1")

        assert judge.history.maxlen == 0
        assert not judge.history
        system, projection, images = judge.calls[0]
        assert "no action history" in system
        assert projection == {"mission": "pick up the blue jar"}
        assert images == ("crop.png",)

    asyncio.run(run())


def test_slow_review_delegate_format_recovery_preserves_authorized_mission():
    async def run():
        store = Store()
        context = Context(store)
        adapter = SimpleNamespace(generation=1)
        coordinator = Coordinator(adapter, context, store)
        mission = "move the blue jar into the left basket"
        original = Milestone("mission-id", 1, mission, mission, capabilities=())
        coordinator.delegate(original)
        context.ingest(observation("review-frame", 1, {"views": []}))

        class MalformedReviewModel:
            async def decide(self, system, projection, images):
                return {"kind": "delegate", "goal": "pick up the red can instead"}

        slow = SlowAgent(MalformedReviewModel(), context, coordinator, store)
        await slow.review()

        assert context.milestone is original
        assert context.milestone.mission == mission
        assert context.target_plan["strategy"] == "pick up the red can instead"
        recovered = [event["payload"] for event in store.events if event["kind"] == "slow.format_recovered"]
        assert recovered == [{"from": "delegate", "to": "advice", "mission_unchanged": True}]
        assert slow.reviews == 1

    asyncio.run(run())
