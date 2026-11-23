"""Bounded historical grasp-phase evidence stays separate from action authority."""
import asyncio
import copy
from types import SimpleNamespace

import pytest

from embodied_harness.adapters.skills import grasp_point
from embodied_harness.domain.models import Milestone
from embodied_harness.runtime.context import Context
from embodied_harness.runtime.phase_feedback import grasp_phase_evidence


class Store:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload, event_id=None):
        self.events.append({"kind": kind, "payload": copy.deepcopy(payload)})

    def since(self, cursor):
        return self.events[cursor:]


def phase_step(phase, *, state="completed", reason=None, residual=None, tool=None,
               physics=True, bounded=False, args=None, observation_id=None):
    tools = {"open": "gripper_control", "remeasure": "mark_point", "approach": "move_to",
             "contact": "move_to", "close": "gripper_control", "lift": "move_to"}
    motion = {"execution_state": state, "physics_executed": physics,
              "motion_summary": {"gripper_contact": {"left_fingerpad_contact": False,
                                                       "right_fingerpad_contact": False}}}
    if reason is not None:
        motion["reason_code"] = reason
    if residual is not None:
        motion["motion_summary"].update(position_error_m=residual, orientation_error_rad=0.04)
    step = {"phase": phase, "tool": tool or tools.get(phase, "move_to"),
            "args": args or {"point_id": "private-point"},
            "feedback": {"motion": motion}, "observation_id": observation_id or f"obs-{phase}"}
    if bounded:
        step["bounded_close_probe"] = {"allowed": True, "max_residual_m": 0.018,
                                       "private_note": "must not be copied"}
    return step


def failed_contact_detail(*, residual=0.0536):
    return {"skill": "grasp_point", "macro_status": "contact_failed", "state_unknown": False,
            "feedback": {"substeps": [
                phase_step("open", args={"action": "open"}),
                phase_step("remeasure", args={"x": 100, "y": 120, "point_id": "private-point"}),
                phase_step("approach", args={"point_id": "private-point", "offset_m": [0, 0, -0.1]}),
                phase_step("contact", state="partial", reason="servo_horizon", residual=residual,
                           args={"point_id": "private-point", "offset_m": [0, 0, -0.02]},
                           observation_id="contact-observation-4"),
            ], "retention_proven": False}}


class GraspAdapter:
    """A fake host returns a dispatched partial contact motion after three safe phases."""
    def __init__(self):
        self.points = {"p0": {"point_id": "p0", "source_packet_id": "packet-1",
            "camera_frame_id": "agentview", "pixel_xy": [100, 120], "xyz_m": [0.1, 0.2, 0.3],
            "measurement": {"depth_edge": False}}}
        self.last = {"views": [{"source_packet_id": "packet-1", "camera_frame_id": "agentview"}]}
        self.cached = SimpleNamespace(id="initial")
        self.possible_hold = False
        self.calls = []
        self.host = SimpleNamespace(call=self.call)

    def call(self, tool, args):
        self.calls.append((tool, copy.deepcopy(args)))
        if tool == "gripper_control":
            motion = {"execution_state": "completed", "physics_executed": True,
                      "motion_summary": {"gripper_contact": {"contact_pattern": "no_contact"}}}
            state = {"robot": {"gripper": {"openness": 1.0}},
                     "views": [{"source_packet_id": "packet-2", "camera_frame_id": "agentview"}],
                     "contact_state": {"close_binding_available": False}, "feedback": {"motion": motion}}
        elif tool == "mark_point":
            state = {"robot": {"gripper": {"openness": 1.0}},
                     "views": [{"source_packet_id": "packet-2", "camera_frame_id": "agentview"}],
                     "feedback": {"point": {**self.points["p0"], "source_packet_id": "packet-2"}}}
        elif "contact_point_id" in args:
            motion = {"execution_state": "partial", "reason_code": "servo_horizon",
                      "physics_executed": True,
                      "motion_summary": {"position_error_m": 0.0536,
                          "orientation_error_rad": 0.04,
                          "gripper_contact": {"left_fingerpad_contact": False,
                                               "right_fingerpad_contact": False}}}
            state = {"robot": {"gripper": {"openness": 1.0}},
                     "views": [{"source_packet_id": "packet-2", "camera_frame_id": "agentview"}],
                     "contact_state": {"close_binding_available": True}, "feedback": {"motion": motion}}
        else:
            motion = {"execution_state": "reached", "physics_executed": True,
                      "motion_summary": {}}
            state = {"robot": {"gripper": {"openness": 1.0}},
                     "views": [{"source_packet_id": "packet-2", "camera_frame_id": "agentview"}],
                     "feedback": {"motion": motion}}
        return state

    async def _thread(self, fn, *args):
        return fn(*args)

    def _unpack(self, state):
        self.last = state
        self.cached = SimpleNamespace(id=f"grasp-frame-{len(self.calls)}")
        return self.cached


def context_with_phase_feedback(enabled=False):
    store = Store()
    context = Context(store, phase_aware_grasp_feedback=enabled)
    context.milestone = Milestone("mission", 1, "pick up the blue jar", "lift to clear the rim",
                                  capabilities=("grasp_point",))
    return store, context


def receipt_context(context, result):
    context.remember("action.accepted", {"request": {"capability": "grasp_point",
        "args": {"point_id": "private-point", "approach_world": [0.0, 0.0, -1.0],
                 "clearance_m": 0.1, "lift_m": 0.12}}, "receipt": {}})
    return context.remember("action.receipt", {"id": "receipt-1", "status": "completed",
                                                 "detail": copy.deepcopy(result)})


def test_real_mocked_grasp_tags_phases_stops_before_close_and_exposes_residual_only(tmp_path):
    async def run():
        adapter = GraspAdapter()
        published = []
        result = await grasp_point(adapter, {"point_id": "p0"}, lambda: False, published.append)
        raw_before = copy.deepcopy(result)
        assert result["macro_status"] == "contact_failed"
        assert [step["phase"] for step in result["feedback"]["substeps"]] == [
            "open", "remeasure", "approach", "contact"]
        assert [name for name, _ in adapter.calls] == [
            "gripper_control", "mark_point", "move_to", "move_to"]
        assert len(published) == len(adapter.calls)

        store, context = context_with_phase_feedback(True)
        receipt_context(context, result)
        phase = context.projection("fast")["local_state"]["last_action"]["feedback"]["grasp_phase_evidence"]
        slow_projection = context.projection("slow")
        slow_phase = slow_projection["local_state"]["last_action"]["feedback"]["grasp_phase_evidence"]
        assert phase == slow_phase
        assert phase["close_attempted"] is False and phase["lift_attempted"] is False
        assert phase["phases"][-1]["phase"] == "contact"
        assert phase["phases"][-1]["execution_state"] == "partial"
        assert phase["phases"][-1]["reason_code"] == "servo_horizon"
        assert phase["phases"][-1]["position_error_m"] == pytest.approx(0.0536)
        assert "changing lift height cannot repair this failed phase" in phase["interpretation"]
        assert "permission to act" in phase["semantics"]
        assert not any(key in str(phase) for key in ("private-point", "point_id", "approach_world", "offset_m"))
        assert "current hold" not in phase["semantics"]
        assert "private-point" not in repr(slow_projection["recent_events"])
        assert "args" not in slow_projection["local_state"]["last_action"]
        assert result == raw_before  # Projection never rewrites the host receipt input.
        durable = [row["payload"] for row in store.events if row["kind"] == "action.receipt"][-1]
        assert durable["detail"] == raw_before
        assert "grasp_phase_evidence" not in durable["detail"]["feedback"]

    asyncio.run(run())


def test_flag_off_and_default_context_have_identical_projection():
    result = failed_contact_detail()
    _s1, default_context = context_with_phase_feedback()
    _s2, disabled_context = context_with_phase_feedback(False)
    receipt_context(default_context, result)
    receipt_context(disabled_context, result)
    assert default_context.projection("fast") == disabled_context.projection("fast")
    assert "grasp_phase_evidence" not in default_context.projection("fast")["local_state"]["last_action"]["feedback"]


@pytest.mark.parametrize("result", [
    {"skill": "grasp_point", "macro_status": "lift_completed_verify_retention_visually",
     "feedback": {"substeps": [phase_step("open")] }},
    {"skill": "grasp_point", "macro_status": "cancelled_before_close",
     "feedback": {"substeps": [phase_step("open")] }},
    {"skill": "place_point", "macro_status": "contact_failed",
     "feedback": {"substeps": [phase_step("contact", state="partial")] }},
])
def test_success_cancel_and_non_grasp_receipts_keep_legacy_projection(result):
    _store, enabled = context_with_phase_feedback(True)
    _store2, legacy = context_with_phase_feedback(False)
    receipt_context(enabled, result)
    receipt_context(legacy, result)
    assert enabled.projection("fast") == legacy.projection("fast")
    assert "grasp_phase_evidence" not in enabled.projection("fast")["local_state"]["last_action"]["feedback"]


def test_near_contact_probe_and_close_failure_are_distinguished_from_preclose_failure():
    near_contact = {"skill": "grasp_point", "macro_status": "grasp_contact_not_established",
        "feedback": {"substeps": [phase_step("open"), phase_step("remeasure"),
            phase_step("approach"),
            phase_step("contact", state="partial", residual=0.014, bounded=True),
            phase_step("close", state="completed")]}}
    evidence = grasp_phase_evidence(near_contact)
    assert evidence["close_attempted"] is True and evidence["lift_attempted"] is False
    assert next(row for row in evidence["phases"] if row["phase"] == "contact")["bounded_close_probe_authorized"] is True
    assert "Closure was attempted" in evidence["interpretation"]
    assert "No lift or successful retention is implied" in evidence["interpretation"]

    close_failed = {"skill": "grasp_point", "macro_status": "close_failed",
        "feedback": {"substeps": [phase_step("open"), phase_step("remeasure"),
            phase_step("approach"), phase_step("contact"),
            phase_step("close", state="partial", reason="servo_horizon")]}}
    evidence2 = grasp_phase_evidence(close_failed)
    assert evidence2["close_attempted"] is True and evidence2["lift_attempted"] is False


@pytest.mark.parametrize(("status", "last_phase"), [
    ("open_failed", "open"), ("open_not_measured", "open"),
    ("remeasure_failed", "remeasure"), ("surface_changed_reobserve", "remeasure"),
    ("approach_failed", "approach"), ("contact_failed", "contact"),
    ("close_binding_unavailable", "contact"), ("close_failed", "close"),
    ("grasp_contact_not_established", "close"), ("lift_failed", "lift"),
])
def test_each_failure_status_requires_its_exact_contiguous_stop_phase(status, last_phase):
    phases = ["open", "remeasure", "approach", "contact", "close", "lift"]
    stop_index = phases.index(last_phase)
    detail = {"skill": "grasp_point", "macro_status": status,
              "feedback": {"substeps": [phase_step(name) for name in phases[:stop_index + 1]]}}
    evidence = grasp_phase_evidence(detail)
    assert evidence is not None
    assert evidence["phases"][-1]["phase"] == last_phase
    assert evidence["close_attempted"] is (stop_index >= phases.index("close"))
    assert evidence["lift_attempted"] is (stop_index >= phases.index("lift"))


@pytest.mark.parametrize(("status", "prefix"), [
    ("cancelled_before_open", []), ("cancelled_after_open", ["open"]),
    ("cancelled_before_remeasure", ["open"]),
    ("cancelled_before_approach", ["open", "remeasure"]),
    ("cancelled_before_contact", ["open", "remeasure", "approach"]),
    ("cancelled_before_close", ["open", "remeasure", "approach", "contact"]),
    ("cancelled_before_lift", ["open", "remeasure", "approach", "contact", "close"]),
])
def test_cancelled_prefixes_keep_legacy_projection(status, prefix):
    detail = {"skill": "grasp_point", "macro_status": status,
              "feedback": {"substeps": [phase_step(name) for name in prefix]}}
    assert grasp_phase_evidence(detail) is None


def test_lift_failure_preserves_phase_distinction_without_claiming_retention():
    detail = {"skill": "grasp_point", "macro_status": "lift_failed", "feedback": {"substeps": [
        phase_step("open"), phase_step("remeasure"), phase_step("approach"),
        phase_step("contact"), phase_step("close"),
        phase_step("lift", state="partial", residual=0.021)]}}
    evidence = grasp_phase_evidence(detail)
    assert evidence["close_attempted"] and evidence["lift_attempted"]
    assert "failure occurred during lift" in evidence["interpretation"]
    assert "do not assume" in evidence["interpretation"]


@pytest.mark.parametrize("bad_trace", [
    [phase_step("mystery")],
    [phase_step("contact", tool="mark_point")],
    [phase_step("open"), phase_step("contact"), phase_step("approach")],
    [phase_step("open"), phase_step("open")],
    [{"phase": [], "tool": "move_to", "feedback": {"motion": {}}}],
    ["not-a-step"],
])
def test_unknown_phase_wrong_tool_and_reordered_or_nonsensical_trace_are_rejected(bad_trace):
    detail = {"skill": "grasp_point", "macro_status": "contact_failed",
              "feedback": {"substeps": bad_trace}}
    assert grasp_phase_evidence(detail) is None


def test_nonfinite_and_negative_residuals_are_omitted_without_breaking_projection():
    detail = failed_contact_detail(residual=float("nan"))
    detail["feedback"]["substeps"][-1]["feedback"]["motion"]["motion_summary"]["orientation_error_rad"] = float("inf")
    evidence = grasp_phase_evidence(detail)
    assert "position_error_m" not in evidence["phases"][-1]
    assert "orientation_error_rad" not in evidence["phases"][-1]
    detail["feedback"]["substeps"][-1]["feedback"]["motion"]["motion_summary"]["position_error_m"] = -0.1
    assert "position_error_m" not in grasp_phase_evidence(detail)["phases"][-1]


def test_schema_limits_count_and_status_fail_closed():
    base = failed_contact_detail()
    assert grasp_phase_evidence({**base, "feedback": {"substeps": base["feedback"]["substeps"] * 2}}) is None
    assert grasp_phase_evidence({**base, "macro_status": "unknown_status"}) is None
    assert grasp_phase_evidence({"skill": "grasp_point", "macro_status": "contact_failed",
                                 "feedback": {"substeps": []}}) is None


def test_phase_evidence_contains_only_bounded_whitelisted_fields():
    detail = failed_contact_detail()
    step = detail["feedback"]["substeps"][-1]
    step["tool_schemas"] = {"secret": True}
    step["unbounded_error"] = "private argument point-xyz-1"
    evidence = grasp_phase_evidence(detail)
    serialized = repr(evidence)
    for hidden in ("private-point", "tool_schemas", "unbounded_error", "point_id", "xyz_m", "offset_m"):
        assert hidden not in serialized
    assert "historical_observation_id" in evidence["phases"][-1]
    assert "semantics" in evidence


def test_runner_flag_defaults_off_and_can_be_enabled_for_direct_trial():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_round22_targeted as runner
    from test_round22d_target_judge import _strong_target_runner_job
    from test_round22_runner import load_fixture_catalog

    _runner, config, _case, default_job = _strong_target_runner_job()
    assert not default_job["feature_flags"].get("phase_aware_grasp_feedback", False)
    assert "--phase-aware-grasp-feedback" not in runner.app_arguments(default_job, Path("/tmp/memory.json"))
    flags = copy.deepcopy(config["variants"]["patched22"]["feature_flags_by_mode"])
    flags["dual_async"]["phase_aware_grasp_feedback"] = True
    config["variants"]["patched22"]["feature_flags_by_mode"] = flags
    runner.validate_config(config)
    catalog, _, _ = load_fixture_catalog(config)
    job = runner.make_jobs(config, catalog, Path(config["output_root"]))[0]
    assert job["feature_flags"]["phase_aware_grasp_feedback"] is True
    assert "--phase-aware-grasp-feedback" in runner.app_arguments(job, Path("/tmp/memory.json"))


def test_app_rejects_phase_feedback_for_non_direct_backend(tmp_path):
    from embodied_harness import app

    args = SimpleNamespace(phase_aware_grasp_feedback=True, backend="eta")
    with pytest.raises(ValueError, match="phase_aware_grasp_feedback requires direct_libero primitive receipts"):
        asyncio.run(app.episode(args))


@pytest.mark.parametrize("enabled", [False, True])
def test_app_propagates_phase_feedback_flag_to_context_without_changing_other_roles(monkeypatch, tmp_path, enabled):
    from embodied_harness import app
    from test_round20c_model_role_routing import _episode_args, _install_offline_app

    endpoints = {"fast": "http://fast.invalid/v1/", "slow": "http://slow.invalid/v1/"}
    constructed = _install_offline_app(monkeypatch, endpoints)
    base_context = app.Context
    contexts, agents = [], []

    class CapturingContext(base_context):
        def __init__(self, store):
            super().__init__(store)
            contexts.append(self)

    class CapturingFast:
        def __init__(self, _model, context, *_args, **_kwargs):
            self.steps = self.false_claims = self.errors = 0
            self.identity_checker = None
            agents.append((self, context))

        async def run(self, *_args, **_kwargs):
            return "decision_limit"

    monkeypatch.setattr(app, "Context", CapturingContext)
    monkeypatch.setattr(app, "FastAgent", CapturingFast)
    args = _episode_args(tmp_path, "dual_async")
    args.phase_aware_grasp_feedback = enabled
    result = asyncio.run(app.episode(args))

    assert contexts[0].phase_aware_grasp_feedback is enabled
    assert agents[0][1] is contexts[0]
    assert result["phase_aware_grasp_feedback"] is enabled
    assert [model.role for model in constructed] == ["fast", "slow", "grounding"]


def test_cli_parser_recognizes_phase_feedback_flag(monkeypatch, tmp_path):
    import sys
    from embodied_harness import app

    captured = {}

    async def fake_episode(args):
        captured["enabled"] = args.phase_aware_grasp_feedback

    monkeypatch.setattr(app, "episode", fake_episode)
    monkeypatch.setattr(sys, "argv", ["app", "--mission", "pick up a jar", "--output",
                                       str(tmp_path / "cli"), "--backend", "direct_libero",
                                       "--phase-aware-grasp-feedback"])
    app.main()
    assert captured["enabled"] is True
