"""Regressions for frame-scoped visual advice, candidate IDs, and bar proposals."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_harness.adapters.libero_direct import LiberoHost
from embodied_harness.adapters import vision_geometry
from embodied_harness.agents.slow import SlowAgent
from embodied_harness.domain.models import Advice, Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator


class MemoryStore:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": payload})

    def since(self, cursor):
        return self.events[cursor:]


def observation(obs_id, version, scope):
    return Observation(obs_id, version, 1, time.monotonic(), 0.0, 0,
                       {"candidate_scope": scope})


def scoped_setup(scope="direct-7/objects"):
    store = MemoryStore()
    context = Context(store)
    adapter = SimpleNamespace(generation=1, schemas={})
    coordinator = Coordinator(adapter, context, store)
    milestone = Milestone("mission", 1, "pick up the handle", "pick up the handle", capabilities=())
    coordinator.delegate(milestone)
    context.ingest(observation("observation-1", 1, scope))
    return store, context, adapter, coordinator, milestone


def test_visual_advice_and_target_plan_expire_when_candidate_scope_changes():
    store, context, _, coordinator, milestone = scoped_setup()
    old_scope = "direct-7/objects"
    context.target_plan = {"visual_scope": old_scope, "strategy": "pick p7o-c2",
                           "targets": [{"candidate_id": "p7o-c2"}]}
    advice = Advice(milestone.id, milestone.revision, 1, "advice", "Pick p7o-c2",
                    time.monotonic() + 60,
                    expected_visual_scope=old_scope)

    assert coordinator.apply_advice(advice)
    assert context.projection("fast")["advice"]

    assert context.ingest(observation("observation-2", 2, "direct-8/objects"))
    projection = context.projection("fast")

    assert projection["advice"] == []
    assert projection["target_plan"] == {
        "strategy": "Prior frame-specific grounding expired; use current visual evidence. Original proposal is retained in the event log.",
        "targets": [], "grounding_expired": True,
    }
    assert context.local_state["visual_binding_note"]
    stale = Advice(milestone.id, milestone.revision, 1, "advice", "Still pick p7o-c2",
                   time.monotonic() + 60,
                   expected_visual_scope=old_scope)
    assert not coordinator.apply_advice(stale)
    assert any(event["kind"] == "advice.stale" for event in store.events)


def test_slow_visual_review_is_rejected_if_frame_changes_during_inference():
    store, context, _, coordinator, _ = scoped_setup()

    class SlowModel:
        async def decide(self, system, projection, images):
            assert projection["observation"]["state"]["candidate_scope"] == "direct-7/objects"
            context.ingest(observation("observation-2", 2, "direct-8/objects"))
            return {"kind": "advice", "text": "Select p7o-c2", "action_sensitive": False}

    slow = SlowAgent(SlowModel(), context, coordinator, store)
    import asyncio
    asyncio.run(slow.review())

    assert context.review_state["last_outcome"] == "stale"
    assert not context.advice
    assert not any(event["kind"] == "advice.accepted" for event in store.events)
    assert any(event["kind"] == "advice.stale" for event in store.events)


def test_semantic_slow_advice_clears_previous_visual_targets():
    store, context, _, coordinator, _ = scoped_setup()
    context.target_plan = {"visual_scope": "direct-7/objects", "strategy": "pick p7o-c2",
                           "targets": [{"candidate_id": "p7o-c2"}]}

    class SlowModel:
        async def decide(self, system, projection, images):
            return {"kind": "advice", "text": "Approach from the side", "action_sensitive": False}

    slow = SlowAgent(SlowModel(), context, coordinator, store)
    import asyncio
    asyncio.run(slow.review())

    assert context.review_state["last_outcome"] == "advice"
    assert context.target_plan["visual_scope"] is None
    assert context.target_plan["strategy"] == "Approach from the side"
    assert context.target_plan["targets"] == []


def test_candidate_ids_are_bound_to_packet_and_proposal_mode(monkeypatch):
    host = object.__new__(LiberoHost)
    host.packet = 7
    host.proposal_mode = "objects"
    host.candidate_cache = None
    host.depth = lambda name: np.ones((32, 32))
    host.reference_robot_position = np.array([0.0, 0.0, 0.6])
    host.sim = SimpleNamespace(
        model=SimpleNamespace(camera_name2id=lambda name: 0, cam_fovy=[60.0]),
        data=SimpleNamespace(cam_xmat=[np.eye(3)], cam_xpos=[[0.0, 0.0, 1.0]]),
    )
    calls = []

    def proposal(mode):
        def generate(*args):
            calls.append(mode)
            return ([{"candidate_id": "c1", "bbox_xyxy": [1, 2, 10, 11]}],
                    np.zeros((32, 32), dtype=int), np.zeros((32, 32, 3)))
        return generate

    monkeypatch.setattr("embodied_harness.adapters.libero_direct.support_height",
                        lambda *args: 0.0)
    monkeypatch.setattr(vision_geometry, "propose_surfaces", proposal("objects"))
    monkeypatch.setattr(vision_geometry, "propose_horizontal_bars", proposal("horizontal_bars"))

    first = host.candidates()[0][0]["candidate_id"]
    cached = host.candidates()[0][0]["candidate_id"]
    host.packet = 8
    second = host.candidates()[0][0]["candidate_id"]
    host.proposal_mode = "horizontal_bars"
    host.candidate_cache = None  # observe(candidate_mode=...) retires the old proposal cache
    third = host.candidates()[0][0]["candidate_id"]

    assert (first, cached, second, third) == ("p7o-c1", "p7o-c1", "p8o-c1", "p8b-c1")
    assert calls == ["objects", "objects", "horizontal_bars"]


def test_horizontal_bar_prominence_filter_rejects_flush_edges_and_keeps_protruding_bars(monkeypatch):
    yy, xx = np.mgrid[:64, :64]
    cloud = np.stack(((xx - 32) * 0.002, (yy - 32) * 0.002,
                      np.zeros((64, 64), dtype=float)), axis=-1)
    bar = (xx >= 8) & (xx <= 55) & (yy >= 29) & (yy <= 33)
    cloud[..., 2][bar] = 0.30
    monkeypatch.setattr(vision_geometry, "world_cloud", lambda *args: cloud)

    def proposals(prominence):
        depth = np.full((64, 64), 0.8)
        depth[28:35, 28:36] -= prominence
        return vision_geometry.propose_horizontal_bars(
            depth, 60.0, np.eye(3), [0.0, 0.0, 1.0], 0.0, [0.0, 0.0, 0.6]
        )

    flush, _, _ = proposals(0.004)
    protruding, _, _ = proposals(0.006)

    assert flush == []
    assert len(protruding) == 1
    assert protruding[0]["local_depth_prominence_m"] == pytest.approx(0.006)
