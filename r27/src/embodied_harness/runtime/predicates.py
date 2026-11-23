"""Small public-sensor vocabulary. These facts never identify an object."""
import math

SENSOR_PREDICATES = ("bilateral_contact", "no_finger_contact", "gripper_open", "grasp_lift_receipt")


def evaluate_condition(condition, context):
    """Separate a live state from a witnessed occurrence in this exact stage."""
    predicate = condition["predicate"]
    when = condition.get("when", "current")
    if when == "current":
        return evaluate(predicate, context)
    if when != "during_stage" or predicate not in SENSOR_PREDICATES:
        raise ValueError("Unsupported temporal sensor condition")
    milestone_id = context.milestone.id if context.milestone else None
    fact = getattr(context, "stage_evidence", {}).get(predicate)
    if fact and fact.get("milestone_id") == milestone_id:
        return {**fact, "when": when}
    return {"predicate": predicate, "when": when, "status": "unknown", "milestone_id": milestone_id,
            "source": "no_recorded_stage_occurrence", "semantic_identity_verified": False,
            "semantics": "missing_history_is_not_proof_of_absence_or_current_state"}


def evaluate(predicate, context):
    obs = context.latest
    state = obs.state if obs else {}
    tactile = state.get("tactile") or {}
    left, right = tactile.get("left_fingerpad_contact"), tactile.get("right_fingerpad_contact")
    value = None
    if predicate in {"bilateral_contact", "no_finger_contact"}:
        if type(left) is bool and type(right) is bool:
            value = left and right if predicate == "bilateral_contact" else not left and not right
    elif predicate == "gripper_open":
        openness = ((state.get("robot") or {}).get("gripper") or {}).get("openness")
        if type(openness) in (int, float) and math.isfinite(openness):
            value = openness >= .95
    elif predicate == "grasp_lift_receipt":
        last = context.local_state.get("last_action") or {}
        if last:
            value = (last.get("tool") == "grasp_point"
                     and last.get("status") == "lift_completed_verify_retention_visually")
    else:
        raise ValueError("Unsupported public sensor predicate")
    return {"predicate": predicate, "status": "unknown" if value is None else "satisfied" if value else "contradicted",
            "observation_id": obs.id if obs else None, "action_seq": context.action_seq,
            "source": "public_sensor_or_receipt", "semantic_identity_verified": False}
