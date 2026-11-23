"""Offline regressions for task-start references, temporal sensor evidence, and prep rollback."""
import asyncio
import base64
import copy
import hashlib
import json
import time
from collections import deque
from types import SimpleNamespace

import httpx
import jsonschema
import pytest
from PIL import Image

from embodied_harness.agents.model import ModelClient
from embodied_harness.agents.slow import SlowAgent
from embodied_harness.domain.models import Milestone, Observation
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator
from embodied_harness.runtime.milestones import MilestoneManager, PLAN_SCHEMA
from embodied_harness.runtime.pipeline import PipelinedMilestoneManager
from embodied_harness.runtime.predicates import evaluate, evaluate_condition


class Store:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": copy.deepcopy(payload)})

    def since(self, cursor):
        return self.events[cursor:]


def observation(observation_id, version, *, left=None, right=None, images=()):
    return Observation(
        observation_id, version, 1, time.monotonic(), float(version), 0,
        {"tactile": {"left_fingerpad_contact": left, "right_fingerpad_contact": right},
         "views": [{"camera_frame_id": name} for name in
                   ("agentview", "candidate_gallery", "robot0_eye_in_hand")[:len(images)]]},
        tuple(images),
    )


def test_mission_start_reference_is_fixed_separate_and_every_role_keeps_current_checkpoint_views(tmp_path):
    start = tmp_path / "mission-start.png"
    later_start = tmp_path / "later-start.png"
    current_agent = tmp_path / "current-agent.png"
    current_gallery = tmp_path / "current-gallery.png"
    current_wrist = tmp_path / "current-wrist.png"
    experience = tmp_path / "prior-experience.png"
    for path, color in ((start, "red"), (later_start, "green"),
                        (current_agent, "blue"), (current_gallery, "yellow"),
                        (current_wrist, "purple"), (experience, "orange")):
        Image.new("RGB", (2, 2), color).save(path)

    mission_start = observation("mission-start", 1, images=(str(start),))
    later_observation = observation("later-frame", 2, images=(str(later_start),))
    current = observation("current-frame", 3,
                          images=(str(current_agent), str(current_gallery), str(current_wrist)))
    captured = []

    async def transport(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"kind":"wait"}'}}]})

    async def run_role(role):
        model = ModelClient("https://offline.invalid/v1", "offline-key", "local-test", role, tmp_path)
        await model.client.aclose()
        model.client = httpx.AsyncClient(base_url="https://offline.invalid/v1/",
                                         transport=httpx.MockTransport(transport))
        model.bind_mission_reference(mission_start)
        # Binding is idempotent: a later observation must not replace the start reference.
        model.bind_mission_reference(later_observation)
        model.bind_memory([{"image": str(experience),
                            "image_sha256": hashlib.sha256(experience.read_bytes()).hexdigest(),
                            "caption": "prior experience only", "manifest_sha256": "offline-manifest"}])
        context = {"purpose": "checkpoint" if role == "slow" else "decision",
                   "observation": {"id": current.id, "state": current.state},
                   "attached_visual": {"views": copy.deepcopy(current.state["views"])}}
        try:
            await model.decide("offline prompt", context, current.images)
            assert model.mission_reference_metadata["observation_id"] == "mission-start"
            assert model.mission_reference_metadata["image_sha256"]
        finally:
            await model.close()

    async def run():
        for role in ("fast", "slow", "grounding"):
            await run_role(role)

    asyncio.run(run())

    assert len(captured) == 3
    expected_reference = "data:image/png;base64," + base64.b64encode(start.read_bytes()).decode()
    for request, role in zip(captured, ("fast", "slow", "grounding")):
        reference_message = next(message for message in request["messages"]
                                 if isinstance(message.get("content"), list)
                                 and any(block.get("type") == "text"
                                         and "THIS TASK'S START FRAME" in block.get("text", "")
                                         for block in message["content"]))
        reference_text = next(block["text"] for block in reference_message["content"]
                              if block["type"] == "text")
        reference_image = next(block["image_url"]["url"] for block in reference_message["content"]
                               if block["type"] == "image_url")
        assert "never proves current position, possession or completion" in reference_text
        assert "task_start_reference_not_current_pose_or_retention" in reference_text
        assert reference_image == expected_reference

        current_message = request["messages"][-1]
        current_text = next(block["text"] for block in current_message["content"]
                            if block["type"] == "text")
        assert '"id": "current-frame"' in current_text
        current_images = [block["image_url"]["url"] for block in current_message["content"]
                          if block["type"] == "image_url"]
        if role == "slow":
            assert current_images == [
                "data:image/png;base64," + base64.b64encode(current_agent.read_bytes()).decode(),
                "data:image/png;base64," + base64.b64encode(current_wrist.read_bytes()).decode(),
            ]
            assert '"camera_frame_id": "agentview"' in current_text
            assert '"camera_frame_id": "robot0_eye_in_hand"' in current_text
            assert '"candidate_gallery"' in current_text
        else:
            assert current_images == [
                "data:image/png;base64," + base64.b64encode(current_agent.read_bytes()).decode(),
                "data:image/png;base64," + base64.b64encode(current_gallery.read_bytes()).decode(),
            ]

        trace = json.loads((tmp_path / f"{role}-model.jsonl").read_text().splitlines()[-1])
        manifest = trace["transmitted_image_manifest"]
        assert [entry["kind"] for entry in manifest] == [
            "task_start_reference", "prior_experience_reference",
            "current_or_declared_visual", "current_or_declared_visual",
        ]
        assert manifest[0]["image"] == str(start)
        assert manifest[0]["image_sha256"] == hashlib.sha256(start.read_bytes()).hexdigest()
        assert manifest[1]["image"] == str(experience)
        assert trace["transmitted_total_image_count"] == len(manifest) == 4
        assert [entry["image"] for entry in manifest[2:]] == (
            [str(current_agent), str(current_wrist)] if role == "slow"
            else [str(current_agent), str(current_gallery)])


def test_start_reference_is_not_replaced_with_wrist_when_agentview_is_missing(tmp_path):
    wrist = tmp_path / "wrist-only.png"
    Image.new("RGB", (2, 2), "red").save(wrist)

    async def run():
        model = ModelClient("https://offline.invalid/v1", "offline-key", "local-test", "fast", tmp_path)
        try:
            wrist_only = Observation("wrist-only", 1, 1, time.monotonic(), 0.0, 0,
                                     {"views": [{"camera_frame_id": "robot0_eye_in_hand"}]},
                                     (str(wrist),))
            model.bind_mission_reference(wrist_only)
            assert model.mission_reference_metadata is None
            assert model.mission_reference_messages == []
        finally:
            await model.close()

    asyncio.run(run())


def test_current_sensor_is_live_while_during_stage_keeps_only_this_stages_witness():
    store = Store()
    context = Context(store)
    context.milestone = Milestone("stage-1", 1, "mission", "lift target")
    context.begin_stage_evidence(context.milestone.id)

    first = observation("contact-true", 1, left=True, right=True)
    assert context.ingest(first)
    current_condition = {"predicate": "bilateral_contact"}
    during_condition = {"predicate": "bilateral_contact", "when": "during_stage"}
    assert evaluate_condition(current_condition, context) == evaluate("bilateral_contact", context)
    assert evaluate_condition(current_condition, context)["status"] == "satisfied"
    assert evaluate_condition(during_condition, context)["status"] == "satisfied"

    second = observation("contact-false", 2, left=True, right=False)
    assert context.ingest(second)
    assert evaluate_condition(current_condition, context)["status"] == "contradicted"
    historical = evaluate_condition(during_condition, context)
    assert historical["status"] == "satisfied"
    assert historical["observation_id"] == first.id
    assert historical["milestone_id"] == "stage-1"
    assert historical["semantics"] == "occurred_during_stage_not_a_current_state_claim"
    assert historical["semantic_identity_verified"] is False

    # Real manager activation starts an empty evidence window.
    class ActivationCoordinator:
        mission_authority = {"mission": "mission", "constraints": set(), "capabilities": ("move_to",)}
        adapter = SimpleNamespace(generation=1)
        contract_review_pending = True

        def delegate(self, milestone, expected_revision=None):
            context.milestone = milestone

    manager = object.__new__(MilestoneManager)
    manager.remaining = 8
    manager.stages = [{"goal": "lift target", "success_conditions": [
        {"predicate": "visual", "description": "target lifted"}]},
        {"goal": "place target", "success_conditions": [
            {"predicate": "visual", "description": "target placed"}]}]
    manager.index = 0
    manager.stage_decisions = 3
    manager.pending_reason = manager.pending_milestone = None
    manager.coordinator = ActivationCoordinator()
    manager.context = context
    manager.plan_id = "plan-1"
    manager.fundamental_scopes = {}
    manager.fundamental = []
    manager.user_constraints = frozenset()
    manager.slow = SimpleNamespace(last_review_started_at=None)
    manager.store = store
    manager.publish = lambda: None
    assert manager.activate(1)
    missing = evaluate_condition(during_condition, context)
    assert missing["status"] == "unknown"
    assert missing["milestone_id"] == context.milestone.id
    assert context.milestone.stage_index == 1
    assert not context.stage_evidence


def test_during_stage_grasp_lift_requires_an_actual_receipt_and_does_not_claim_retention():
    context = Context(Store())
    context.milestone = Milestone("stage-1", 1, "mission", "lift target")
    context.begin_stage_evidence(context.milestone.id)
    during_receipt = {"predicate": "grasp_lift_receipt", "when": "during_stage"}

    # Even a stale-looking last_action projection or a fresh sensor frame is not a receipt event.
    context.local_state["last_action"] = {
        "tool": "grasp_point", "status": "lift_completed_verify_retention_visually"}
    assert context.ingest(observation("after-old-claim", 1, left=True, right=True))
    assert evaluate_condition(during_receipt, context)["status"] == "unknown"

    context.remember("action.accepted", {"request": {
        "capability": "grasp_point", "args": {"point_id": "current-point"}}})
    context.remember("action.receipt", {"status": "completed", "detail": {
        "macro_status": "lift_completed_verify_retention_visually", "host_error": None,
        "observation_id": "receipt-observation"}})
    witnessed = evaluate_condition(during_receipt, context)
    assert witnessed["status"] == "satisfied"
    assert witnessed["source"] == "public_sensor_or_receipt"
    assert witnessed["semantic_identity_verified"] is False
    assert witnessed["semantics"] == "occurred_during_stage_not_a_current_state_claim"

    context.milestone = Milestone("stage-2", 1, "mission", "place target")
    context.begin_stage_evidence(context.milestone.id)
    assert evaluate_condition(during_receipt, context)["status"] == "unknown"


class PipelineModel:
    def __init__(self, checkpoint_reply):
        self.responses = deque([pipeline_plan(), checkpoint_reply])
        self.history = ["old fast reasoning"]
        self.checkpoint_entered = asyncio.Event()
        self.release_checkpoint = asyncio.Event()

    async def decide(self, _prompt, projection, _images):
        response = self.responses.popleft()
        if projection.get("purpose") == "checkpoint":
            self.checkpoint_entered.set()
            await self.release_checkpoint.wait()
        return copy.deepcopy(response)


class PipelineAdapter:
    generation = 1
    terminal = False

    def __init__(self):
        self.version = 0
        self.points = {"current-point": {"point_id": "current-point", "source": "current_contract",
                                         "xyz_m": [0.1, 0.2, 0.3]}}
        self.schemas = {"observe": {"parameters": {"properties": {}}},
                        "mark_point": {"parameters": {"properties": {}}}}

    async def observe(self, action_seq):
        self.version += 1
        return Observation(f"pipeline-observation-{self.version}", self.version, self.generation,
                           time.monotonic(), float(self.version), action_seq, {"sensor": {"sim_time": self.version,
                           "control_steps": self.version}})

    def validate_args(self, capability, _args):
        if capability not in self.schemas:
            raise ValueError("unsupported fake tool")

    async def execute(self, *_args):
        raise AssertionError("offline regression must not execute physics")

    async def verify(self, observation_id):
        return {"status": "not_verified", "observation_id": observation_id}


def pipeline_plan():
    return {"kind": "plan", "fundamental_constraints": [], "milestones": [
        {"goal": "lift the cup", "success_conditions": [
            {"predicate": "visual", "description": "the cup is visibly lifted"}]},
        {"goal": "place the cup in the bowl", "success_conditions": [
            {"predicate": "visual", "description": "the cup is inside the bowl"}]},
    ]}


def pipeline_checkpoint(decision):
    return {"kind": "checkpoint", "decision": decision, "reason": "offline evidence check",
            "assessments": [{"condition_id": "success-0", "status": "satisfied",
                             "evidence": "synthetic current visual evidence"}],
            "invariants": []}


async def pipeline_harness(decision):
    store = Store()
    context = Context(store)
    adapter = PipelineAdapter()
    coordinator = Coordinator(adapter, context, store)
    coordinator.authorize_mission("put the cup inside the bowl", capabilities=("observe", "mark_point", "move_to"))
    await coordinator.observe()
    model = PipelineModel(pipeline_checkpoint(decision))
    slow = SlowAgent(model, context, coordinator, store)
    manager = PipelinedMilestoneManager(slow, context, coordinator, store, max_decisions=8,
                                       deadline=time.monotonic() + 20)
    assert await manager.initialize() == "planned"
    manager.fast_model = model
    return store, context, adapter, model, manager


@pytest.mark.parametrize(("decision", "expected"), [
    ("continue", "continue"), ("replan", "replan"), ("advance", "advanced"),
    ("cancel", "cancelled"),
])
def test_checkpoint_preparation_points_are_retired_unless_matching_stage_is_approved(decision, expected):
    async def run():
        requested = "advance" if decision == "advance" else decision
        store, context, adapter, model, manager = await pipeline_harness(requested)
        assert await manager.checkpoint("fast_completion_proposal") == "checkpoint_pending"
        await asyncio.wait_for(model.checkpoint_entered.wait(), timeout=1)
        allowed, terminal = await manager.guard_proposal("tool", "mark_point")
        assert allowed and terminal is None

        prior_ids = set(adapter.points)
        adapter.points["prep-point"] = {"point_id": "prep-point", "source": "checkpoint_preview",
                                        "xyz_m": [0.4, 0.5, 0.6]}
        manager.note_preparation_points(prior_ids)
        prepared = [event["payload"] for event in store.events
                    if event["kind"] == "pipeline.preparation_measurements"]
        assert prepared == [{"point_ids": ["prep-point"], "preview_goal": "place the cup in the bowl",
                             "scope": "unapproved_next_goal_preparation"}]

        if decision == "cancel":
            await manager.cancel_checkpoint("hard_control")
            outcome = "cancelled"
        else:
            model.release_checkpoint.set()
            outcome = await manager.wait_checkpoint()
        assert outcome == expected

        assert "current-point" in adapter.points
        if decision == "advance":
            assert context.milestone.goal == "place the cup in the bowl"
            assert "prep-point" in adapter.points
            assert model.history == ["old fast reasoning"]
            assert not any(event["kind"] == "pipeline.preparation_invalidated" for event in store.events)
        else:
            assert "prep-point" not in adapter.points
            assert model.history == []
            invalidated = [event["payload"] for event in store.events
                           if event["kind"] == "pipeline.preparation_invalidated"]
            assert len(invalidated) == 1
            assert invalidated[0]["point_ids"] == ["prep-point"]
        await manager.close()

    asyncio.run(run())


def test_temporal_condition_schema_defaults_to_current_and_rejects_temporal_visual_evidence():
    plan = pipeline_plan()
    jsonschema.validate(plan, PLAN_SCHEMA)
    plan["milestones"][0]["success_conditions"].append(
        {"predicate": "bilateral_contact", "when": "during_stage"})
    jsonschema.validate(plan, PLAN_SCHEMA)
    plan["milestones"][0]["success_conditions"][0]["when"] = "during_stage"
    with pytest.raises(ValueError, match="visual outcomes need current semantic assessment"):
        from embodied_harness.runtime.milestones import MilestoneManager

        manager = object.__new__(MilestoneManager)
        manager.user_constraints = frozenset()
        manager.fundamental_scopes = {}
        manager.context = SimpleNamespace(latest=None)
        manager.validate_plan(plan)


def test_retired_preparation_point_is_explained_in_projection_but_kept_in_raw_observation():
    context = Context(Store())
    raw_point = {"point_id": "preview-point", "xyz_m": [0.4, 0.5, 0.6]}
    context.latest = Observation("latest-with-retired-point", 1, 1, time.monotonic(), 0.0, 0,
                                 {"feedback": {"point": raw_point}}, ())
    context.local_state["preparation_invalidated"] = {
        "point_ids": ["preview-point"], "outcome": "continue"}

    projection = context.projection("fast")
    projected_feedback = projection["observation"]["state"]["feedback"]
    assert "point" not in projected_feedback
    assert projected_feedback["retired_measurement"]["point_id"] == "preview-point"
    assert projected_feedback["retired_measurement"]["reason"] == "unapproved_next_goal_preparation"
    assert context.latest.state["feedback"]["point"] == raw_point
