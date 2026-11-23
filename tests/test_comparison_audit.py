import json

import pytest

from deploy.audit_comparison import audit_result


@pytest.mark.parametrize(
    ("executed", "started", "scored"),
    [
        (False, False, "native_initial_state"),
        (True, True, "last_native_action"),
        (False, True, "unknown"),
    ],
)
def test_interrupt_native_checkpoint_requires_completed_action(tmp_path, executed, started, scored):
    arm = tmp_path / "initial-00/upper-2/libero_object/task-00"
    simulation = arm / "teacher/simulation"
    rollout = arm / "teacher/rollout"
    simulation.mkdir(parents=True)
    rollout.mkdir()
    result = arm / "teacher.result.json"
    result.write_text(json.dumps(dict(mode="teacher", success=None)))
    (arm / "teacher.log").write_text(
        "RuntimeError: Model flash returned HTTP 500 (new_api_error)\n"
    )
    (simulation / "task.json").write_text(
        json.dumps(dict(task_name="case", instruction="place", initial_success=False))
    )
    events = [{"event": "action_started"}] if started else []
    if executed:
        (simulation / "latest.json").write_text(
            json.dumps(dict(feedback={"x": 0}, native_success=True))
        )
        events.append({"event": "execution", "feedback": {"x": 0}})
    events.append({"event": "infrastructure_error", "actions_completed": int(executed)})
    (rollout / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
    row = audit_result(result)
    assert row["failure_type"] == "transport_http_500"
    assert row["score_source"] == scored
    assert row["recorded_success"] is None
    assert row["success"] is (None if scored == "unknown" else executed)
