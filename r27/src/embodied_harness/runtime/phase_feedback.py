"""Bounded historical motor-phase evidence, never an action or current-state claim."""
import math
from embodied_harness.domain.tool_errors import orientation_receipt_error


PHASE_TO_TOOL = {"open": "gripper_control", "remeasure": "mark_point",
                 "approach": "move_to", "contact": "move_to",
                 "close": "gripper_control", "lift": "move_to"}
STOP_PHASE = {"open_failed": "open", "open_not_measured": "open",
              "remeasure_failed": "remeasure", "surface_changed_reobserve": "remeasure",
              "approach_failed": "approach", "contact_failed": "contact",
              "close_binding_unavailable": "contact", "close_failed": "close",
              "grasp_contact_not_established": "close", "lift_failed": "lift"}


def grasp_phase_evidence(detail):
    """Project explicitly tagged public receipts without guessing a missing phase.

    The original trace remains authoritative. No tool arguments, point bindings,
    environment identities or free-form errors are copied into the model view.
    """
    if not isinstance(detail, dict) or detail.get("skill") != "grasp_point":
        return None
    status = detail.get("macro_status")
    # Successful/cancelled macros keep the existing projection. This experiment
    # changes failure interpretation only, not positive or interrupted evidence.
    if not isinstance(status, str) or status not in STOP_PHASE:
        return None
    feedback = detail.get("feedback") or {}
    if not isinstance(feedback, dict):
        return None
    trace = feedback.get("substeps")
    if not isinstance(trace, list) or not 1 <= len(trace) <= 6:
        return None
    phases = []
    last_index = -1
    for step in trace:
        if not isinstance(step, dict):
            return None
        phase = step.get("phase")
        if (not isinstance(phase, str) or phase not in PHASE_TO_TOOL
                or step.get("tool") != PHASE_TO_TOOL[phase]):
            return None
        index = list(PHASE_TO_TOOL).index(phase)
        if index != last_index + 1:
            return None
        last_index = index
        feedback = step.get("feedback") or {}
        if not isinstance(feedback, dict):
            return None
        motion = feedback.get("motion") or {}
        if not isinstance(motion, dict):
            return None
        summary = motion.get("motion_summary") or {}
        if not isinstance(summary, dict):
            return None
        row = {"phase": phase}
        for field in ("execution_state", "reason_code"):
            if isinstance(motion.get(field), str):
                row[field] = motion[field][:100]
        if isinstance(motion.get("physics_executed"), bool):
            row["physics_executed"] = motion["physics_executed"]
        for field in ("position_error_m", "orientation_error_rad"):
            value = summary.get(field)
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                row[field] = value
        contact = summary.get("gripper_contact") or {}
        if isinstance(contact, dict):
            measured = {key: contact[key] for key in (
                "left_fingerpad_contact", "right_fingerpad_contact")
                if isinstance(contact.get(key), bool)}
            if measured:
                row["measured_contact"] = measured
        error = step.get("error") or {}
        if not isinstance(error, dict):
            return None
        if error:
            row["error_present"] = True
        if isinstance(error.get("code"), str):
            row["error_code"] = error["code"][:100]
        diagnosis = orientation_receipt_error(error)
        if diagnosis:
            row["error_cause"] = diagnosis["code"]
        observation_id = step.get("observation_id")
        if isinstance(observation_id, str):
            row["historical_observation_id"] = observation_id[:100]
        probe = step.get("bounded_close_probe") or {}
        if isinstance(probe, dict) and probe.get("allowed") is True:
            row["bounded_close_probe_authorized"] = True
        phases.append(row)
    if phases[-1]["phase"] != STOP_PHASE[status]:
        return None
    attempted = {row["phase"] for row in phases}
    result = {"schema": "grasp_phase_evidence_v1", "macro_status": status,
        "phases": phases, "close_attempted": "close" in attempted,
        "lift_attempted": "lift" in attempted,
        "semantics": "Historical primitive outcomes of this receipt only; not current contact, object identity, retention, task success or permission to act. Reinspect current evidence before acting."}
    last = phases[-1]
    orientation_error = orientation_receipt_error(trace[-1].get("error"))
    if (orientation_error and last.get("execution_state") == "not_started"
            and last.get("physics_executed") is False):
        result["interpretation"] = (orientation_error["summary"] +
            " This phase was not dispatched; earlier macro phases may have executed. "
            "Selecting another point alone does not repair invalid orientation parameters.")
        result["repair"] = orientation_error["repair"]
    elif (status in {"contact_failed", "approach_failed"}
            and last["phase"] == status.removesuffix("_failed")
            and last.get("execution_state") == "partial"
            and last.get("reason_code") == "servo_horizon"
            and last.get("physics_executed") is True
            and not last.get("error_present") and "close" not in attempted):
        result["interpretation"] = (
            "Motion was dispatched but did not reach the requested pose within its servo horizon. "
            "This is not a stale-point validation rejection or evidence of a completed close. "
            "Close and lift were not attempted; changing lift height cannot repair this failed phase. "
            "Use the current scene and measured residual to choose a different feasible contact site, "
            "approach or jaw orientation; merely renaming or remeasuring a nearby site does not "
            "establish a corrected motion strategy. The receipt alone does not identify the obstruction.")
    elif status == "grasp_contact_not_established" and "close" in attempted:
        result["interpretation"] = (
            "Closure was attempted but the macro did not establish bilateral pad contact. "
            "Inspect current contact and object geometry before choosing another grasp; this is "
            "different from a motion failure before closing. No lift or successful retention is implied.")
    elif status == "lift_failed" and "lift" in attempted:
        result["interpretation"] = (
            "The failure occurred during lift after closure. Inspect current holding and clearance "
            "evidence before deciding whether to continue, release or reacquire; do not assume the "
            "earlier contact evidence proves current retention.")
    return result
