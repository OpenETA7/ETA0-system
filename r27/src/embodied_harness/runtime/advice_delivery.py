"""Timestamped episode message history, without expiry or motion authority."""
import math
import re

# Reject recognizable executable bindings. This does not certify free prose as
# true or safe: accepted prose remains an explicitly historical hypothesis.
BOUND_REFERENCE = re.compile(
    r"\bp\d+[ob]-c\d+\b|\bc\d+\b|\bpoint[-_ ]\d+\b|\bdirect-\d+\b|"
    r"\bcandidate[_ ]id\b|\bpixels?\b|\bcoordinates?\b|"
    r"\b[xyz]\s*[=:~]\s*-?\d|\[\s*-?\d+(?:[.,\d\s]+)\]|"
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm|meters?|metres?|radians?|rad|degrees?|m)\b|"
    r"\b(?:move_to|mark_point|locate_point|select_candidate|grasp_point|place_point|"
    r"pull_handle|gripper_control)\s*\(", re.I)

RECHECK_INSTRUCTION = (
    "This is a Slow hypothesis based on the recorded earlier review, not a current "
    "pose, held-object or completion fact and not an instruction to replay an old action. "
    "It remains in this task's history; use its time and original stage to decide relevance. "
    "Check CURRENT images, contact and receipts before using it; explain applicability "
    "or rejection in advice_response. It cannot change the milestone, authorize a tool, "
    "override a pause or prove task success. Ground every action in fresh measurements."
)


def unbound_strategy(text):
    return isinstance(text, str) and bool(text.strip()) and not BOUND_REFERENCE.search(text)


def strategy_invalid_reason(advice, context, now):
    m = context.milestone
    basis = advice.get("review_basis", {})
    if not isinstance(basis, dict):
        return "invalid_review_basis"
    if (advice.get("kind") != "advice" or advice.get("scope") != "milestone_strategy"
            or advice.get("authority") != "hypothesis_requiring_current_recheck"
            or advice.get("expires_at") is not None
            or not unbound_strategy(advice.get("text"))
            or any(advice.get(k) is not None for k in (
                "expected_action_seq", "expected_action_id", "expected_action_phase", "expected_visual_scope"))):
        return "invalid_strategy_envelope"
    if not m or basis.get("mission") != m.mission:
        return "different_mission"
    if (advice.get("generation") != m.generation or not context.latest
            or context.latest.generation != advice.get("generation")):
        return "generation_changed"
    if (not isinstance(advice.get("milestone_id"), str) or not advice["milestone_id"]
            or type(advice.get("revision")) is not int or advice["revision"] < 1):
        return "invalid_source_milestone"
    observed_at, published_at = basis.get("observed_at"), basis.get("published_at")
    if any(type(t) not in (int, float) or not math.isfinite(t) for t in (observed_at, published_at, now)):
        return "invalid_clock"
    # Validate provenance, not semantic freshness. Elapsed time never removes a message.
    if not observed_at <= published_at <= now:
        return "invalid_clock_order"
    seq = basis.get("action_seq")
    if type(seq) is not int or not 0 <= seq <= context.action_seq:
        return "invalid_review_action"
    if type(basis.get("ticket")) is not int or basis["ticket"] < 0:
        return "invalid_review_ticket"
    if type(basis.get("control_version")) is not int or not 0 <= basis["control_version"] <= context.control_version:
        return "invalid_review_control"
    if not isinstance(basis.get("observation_id"), str) or not basis["observation_id"]:
        return "missing_review_observation"
    return None


def strategy_projection(advice, context, now):
    return {**advice, "status": "historical_review_not_current_state",
            "retention": "episode_history_no_time_expiry",
            "age_actions": context.action_seq - advice["review_basis"]["action_seq"],
            "age_s": max(0, now - advice["review_basis"]["observed_at"]),
            "since_publication_s": max(0, now - advice["review_basis"]["published_at"]),
            "relation_to_current": {
                "same_milestone": (advice["milestone_id"], advice["revision"]) ==
                    (context.milestone.id, context.milestone.revision),
                "same_control_version": advice["review_basis"]["control_version"] == context.control_version},
            "instruction": RECHECK_INSTRUCTION}
