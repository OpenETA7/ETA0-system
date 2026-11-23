"""Conservative held-load bookkeeping from public ETA receipts."""
import math


def released(body):
    motion = (body.get("feedback") or {}).get("motion") or {}
    openness = ((body.get("robot") or {}).get("gripper") or {}).get("openness")
    return bool(not body.get("error") and isinstance(motion, dict)
                and motion.get("execution_state") == "completed"
                and motion.get("physics_executed") is True
                and type(openness) in (int, float) and math.isfinite(openness)
                and openness >= .95)


def possible_hold_after(previous, action, body):
    motion = (body.get("feedback") or {}).get("motion") or {}
    if action == "open":
        return False if released(body) else previous
    if not isinstance(motion, dict):
        return True
    if motion.get("physics_executed") is False or motion.get("execution_state") == "not_started":
        return previous
    # A close with missing, partial, or unknown feedback may have acquired a load.
    return True
