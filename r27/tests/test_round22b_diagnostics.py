"""Offline regressions for the typed historical diagnostic channel."""
import asyncio
import time

import pytest

from test_round20_regressions import FakeAdapter, MemoryStore, QueuedModel, MISSION
from embodied_harness.agents.fast import FastAgent
from embodied_harness.agents.slow import SlowAgent
from embodied_harness.domain.models import ActionRequest, Advice, Milestone, Observation, uid
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.coordinator import Coordinator, GateError


def make_harness(*, action_seq=0, retain_diagnostics=True, milestone=None):
    store = MemoryStore()
    context = Context(store)
    context.action_seq = action_seq
    context.milestone = milestone or Milestone(
        "stage-1", 1, MISSION, MISSION,
        capabilities=("move_to", "grasp_point", "place_point"),
    )
    context.ingest(Observation(
        "obs-current", 1, 1, time.monotonic(), 0.0, action_seq, {},
    ))
    adapter = FakeAdapter()
    coordinator = Coordinator(adapter, context, store)
    model = QueuedModel([])
    slow = SlowAgent(model, context, coordinator, store,
                     retain_diagnostics=retain_diagnostics)
    return store, context, adapter, coordinator, slow


def diagnostic(*, subject="the red cup", hypothesis="The last grasp may have contacted the bowl rim.",
               category="failure_cause", recheck="Check current wrist and full-scene evidence for cup retention."):
    return {"category": category, "subject": subject, "hypothesis": hypothesis, "recheck": recheck}


def publish(slow, context, hypothesis=None, *, observed_action_seq=None, observed_at=None,
            observation_id="obs-reviewed", decision_fields=None):
    decision = {"kind": "advice", "text": "Do not repeat the same grasp.",
                "diagnostic_hypotheses": [hypothesis or diagnostic()]
                if hypothesis is not None or decision_fields is None else []}
    if decision_fields:
        decision.update(decision_fields)
    return slow.publish_diagnostics(
        decision, context.milestone,
        context.action_seq if observed_action_seq is None else observed_action_seq,
        observation_id, observed_at,
    )


def test_old_action_advice_stays_stale_while_typed_diagnosis_is_visible_and_non_authoritative():
    store, context, _adapter, coordinator, slow = make_harness(action_seq=5)
    milestone = context.milestone
    version_before = context.control_version
    evidence_before = dict(context.stage_evidence)

    stale = Advice(
        milestone.id, milestone.revision, milestone.generation, "advice",
        "Repeat the old action-specific target.", time.monotonic() + 120,
        expected_action_seq=4,
    )
    assert coordinator.apply_advice(stale) is False
    assert publish(slow, context, observed_action_seq=4) == 1

    projection = context.projection("fast")
    assert projection["advice"] == []
    assert len(projection["historical_diagnostics"]) == 1
    archived = projection["historical_diagnostics"][0]
    assert archived["status"] == "historical_unverified"
    assert archived["authority"] == "diagnostic_only"
    assert archived["age_actions"] == 1
    assert archived["hypothesis"] == diagnostic()["hypothesis"]
    assert all("Repeat the old action-specific target." not in str(item)
               for item in projection["historical_diagnostics"])
    assert context.control_version == version_before
    assert coordinator.paused is False
    assert context.stage_evidence == evidence_before
    assert store.events[-1]["kind"] == "diagnostic.published"


def test_diagnostics_are_bounded_to_four_and_only_structured_hypotheses_are_copied():
    _store, context, _adapter, _coordinator, slow = make_harness(action_seq=3)
    for i in range(6):
        result = publish(slow, context, diagnostic(
            subject=f"the red cup variant {i}",
            hypothesis=f"Variant {i} may have slipped during the lift.",
        ))
        assert result == 1

    assert len(context.diagnostic_hypotheses) == 4
    projection = context.projection("fast")
    assert len(projection["historical_diagnostics"]) == 4
    assert [item["subject"] for item in projection["historical_diagnostics"]] == [
        "the red cup variant 2", "the red cup variant 3",
        "the red cup variant 4", "the red cup variant 5",
    ]
    assert all("text" not in item for item in projection["historical_diagnostics"])
    assert all("Do not repeat the same grasp." not in str(item)
               for item in projection["historical_diagnostics"])


@pytest.mark.parametrize("changed", ["id", "revision", "generation"])
def test_projection_purges_diagnostics_when_contract_identity_changes(changed):
    _store, context, _adapter, _coordinator, slow = make_harness()
    assert publish(slow, context) == 1
    original = context.milestone
    updates = {"id": "stage-2", "revision": 2, "generation": 2}
    context.milestone = Milestone(
        updates[changed] if changed == "id" else original.id,
        updates[changed] if changed == "revision" else original.revision,
        original.mission, original.goal, capabilities=original.capabilities,
        generation=updates[changed] if changed == "generation" else original.generation,
    )

    assert context.projection("fast").get("historical_diagnostics", []) == []
    assert list(context.diagnostic_hypotheses) == []


def test_four_action_age_is_inclusive_then_diagnostic_expires_after_next_action():
    _store, context, _adapter, _coordinator, slow = make_harness(action_seq=8)
    assert publish(slow, context, observed_action_seq=3) == 0
    assert publish(slow, context, observed_action_seq=4) == 1
    at_limit = context.projection("fast")["historical_diagnostics"]
    assert len(at_limit) == 1
    assert at_limit[0]["age_actions"] == 4

    context.action_seq = 9
    assert context.projection("fast").get("historical_diagnostics", []) == []
    assert list(context.diagnostic_hypotheses) == []


def test_expired_review_is_rejected_and_projection_also_enforces_entry_ttl(monkeypatch):
    _store, context, _adapter, _coordinator, slow = make_harness()
    now = time.monotonic()
    assert publish(slow, context, observed_at=now - 181) == 0
    assert list(context.diagnostic_hypotheses) == []

    assert publish(slow, context, observed_at=now) == 1
    context.diagnostic_hypotheses[0]["expires_at"] = now - 1
    assert context.projection("fast").get("historical_diagnostics", []) == []


@pytest.mark.parametrize("hypothesis", [
    {**diagnostic(), "frame_id": "frame-9"},
    {**diagnostic(), "point_id": "point-18"},
    {**diagnostic(), "pixel_xy": [31, 42]},
    {**diagnostic(), "coordinates": [0.1, 0.2, 0.3]},
    {**diagnostic(), "tools": ["grasp_point"]},
    diagnostic(hypothesis="The old point-18 at x=31 y=42 may have missed."),
])
def test_frame_bound_or_tool_like_hypotheses_are_rejected(hypothesis):
    _store, context, _adapter, _coordinator, slow = make_harness()
    assert publish(slow, context, hypothesis) == 0
    assert list(context.diagnostic_hypotheses) == []


def test_invalid_hypothesis_list_shapes_are_rejected_without_partial_archive():
    _store, context, _adapter, _coordinator, slow = make_harness()
    too_many = {"kind": "advice", "diagnostic_hypotheses": [
        diagnostic(subject="cup one"), diagnostic(subject="cup two"),
        diagnostic(subject="cup three"),
    ]}
    assert slow.publish_diagnostics(too_many, context.milestone, 0, "obs-reviewed") == 0

    mixed = {"kind": "advice", "diagnostic_hypotheses": [
        diagnostic(subject="valid cup"), diagnostic(category="tool_command"),
    ]}
    assert slow.publish_diagnostics(mixed, context.milestone, 0, "obs-reviewed") == 1
    assert [item["subject"] for item in context.diagnostic_hypotheses] == ["valid cup"]


def test_unhashable_category_value_is_rejected_as_bad_input():
    _store, context, _adapter, _coordinator, slow = make_harness()
    malformed = diagnostic()
    malformed["category"] = ["failure_cause"]
    assert publish(slow, context, malformed) == 0
    assert list(context.diagnostic_hypotheses) == []


def test_diagnostics_default_off_preserves_baseline_projection_shape():
    _store, context, _adapter, _coordinator, slow = make_harness(retain_diagnostics=False)
    assert slow.retain_diagnostics is False
    assert publish(slow, context) == 0
    assert list(context.diagnostic_hypotheses) == []
    assert "historical_diagnostics" not in context.projection("fast")


@pytest.mark.parametrize(("visual_age_s", "visual_action_seq", "expected_count"), [
    (181, 5, 0),  # The review is recent, but the attached visual has expired.
    (1, 0, 0),    # The visual is recent, but is more than four actions old.
    (1, 4, 1),    # The attached visual's own stamps are within both limits.
])
def test_review_diagnostic_uses_attached_older_visual_stamps(
        visual_age_s, visual_action_seq, expected_count):
    async def run():
        store, context, _adapter, _coordinator, slow = make_harness(action_seq=5)
        captured_at = time.monotonic() - visual_age_s
        old_visual = Observation(
            "obs-old-visual", 2, 1, captured_at, 0.0, visual_action_seq, {},
            images=("old-frame.png",),
        )
        current = Observation(
            "obs-current-latest", 3, 1, time.monotonic(), 0.0, 5, {},
        )
        assert context.ingest(old_visual)
        assert context.ingest(current)
        assert context.latest.id == "obs-current-latest"
        assert context.latest_visual.id == "obs-old-visual"
        slow.model = QueuedModel([{
            "kind": "advice", "text": "Recheck the current scene before acting.",
            "diagnostic_hypotheses": [diagnostic()],
        }])

        await slow._review()
        archived = context.projection("fast").get("historical_diagnostics", [])
        assert len(archived) == expected_count
        if expected_count:
            assert archived[0]["observation_id"] == "obs-old-visual"
            assert archived[0]["observed_action_seq"] == visual_action_seq
            assert archived[0]["observed_at"] == captured_at

    asyncio.run(run())


def test_fast_prompt_and_response_are_conditional_and_report_only_a_model_assessment(monkeypatch):
    async def run():
        _store, context, _adapter, coordinator, slow = make_harness()
        assert publish(slow, context) == 1
        diagnosis_id = context.diagnostic_hypotheses[0]["id"]
        response = {"kind": "wait", "diagnostic_response": {
            "id": diagnosis_id, "status": "corroborated",
            "evidence": "The current wrist image shows the cup still in the gripper.",
        }}
        model = QueuedModel([response])
        fast = FastAgent(model, context, coordinator, _store)

        async def no_wait(_delay):
            return None

        monkeypatch.setattr("embodied_harness.agents.fast.asyncio.sleep", no_wait)
        assert await fast.run(1) == "decision_budget"
        call = model.calls[0]
        assert "historical_diagnostics" in call["projection"]
        assert "diagnostic_response" in call["prompt"]
        assessed = [event["payload"] for event in _store.events
                    if event["kind"] == "diagnostic.assessed"]
        assert len(assessed) == 1
        assert assessed[0]["id"] == diagnosis_id
        assert assessed[0]["semantics"] == "executor model claim, not independent verification"

        # Without a retained hypothesis, neither the projection nor the extra
        # diagnostic instruction is sent to the same executor.
        _s2, empty_context, _a2, empty_coordinator, _slow2 = make_harness(
            retain_diagnostics=False)
        baseline_model = QueuedModel([{"kind": "wait"}])
        baseline_fast = FastAgent(baseline_model, empty_context, empty_coordinator, _s2)
        assert await baseline_fast.run(1) == "decision_budget"
        assert "historical_diagnostics" not in baseline_model.calls[0]["projection"]
        assert "diagnostic_response" not in baseline_model.calls[0]["prompt"]

    asyncio.run(run())


def test_diagnostic_response_is_an_allowed_envelope_field_but_does_not_bypass_action_gates():
    _store, context, adapter, coordinator, _slow = make_harness()
    context.milestone = Milestone(
        "guarded-stage", 1, MISSION, MISSION,
        capabilities=("grasp_point",),
        guards=({"tools": ["grasp_point"], "predicate": "gripper_open"},),
    )
    fast = FastAgent(QueuedModel([]), context, coordinator, _store)
    decision = {
        "kind": "tool", "tool": "grasp_point", "args": {"point_id": "fresh-point"},
        "diagnostic_response": {"id": "diag-1", "status": "unresolved", "evidence": "Need a current check."},
    }
    assert fast.tool_arguments(decision) == {"point_id": "fresh-point"}

    # A model's historical assessment cannot satisfy a live milestone guard.
    request = ActionRequest(
        uid(), context.milestone.id, context.milestone.revision,
        context.milestone.generation, context.latest.id,
        "grasp_point", {"point_id": "fresh-point"},
    )
    with pytest.raises(GateError, match="milestone_guard_not_satisfied:gripper_open"):
        coordinator.validate(request)

    with pytest.raises(ValueError, match="Ambiguous or unknown top-level"):
        fast.tool_arguments({**decision, "unrelated": "not an envelope field"})
