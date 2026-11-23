"""Typed, bounded diagnostics for rejected grasp orientation parameters."""


ORIENTATION_ERRORS = {
    "grasp_orientation_pair_required": {
        "summary": "Only one orientation axis was supplied; a complete orientation needs both axes.",
        "repair": "Supply both approach_world and jaw_world, or omit BOTH. With a current measured suggested_jaw_world, omission uses that closing direction and downward approach. Without it, omission preserves wrist orientation.",
    },
    "grasp_orientation_invalid": {
        "summary": "An orientation axis is malformed, non-finite, zero, or numerically too small/large.",
        "repair": "Each supplied axis must contain three finite numbers and have a nonzero usable magnitude. Derive the approach and closing axes from current geometry, or omit BOTH to use the documented default.",
    },
    "grasp_orientation_collinear": {
        "summary": "Approach and finger-closing axes are parallel or antiparallel; they cannot define a wrist orientation. This is a parameter error, not evidence of collision or an obstructed grasp point.",
        "repair": "Do not set approach_world to jaw_world or its negative. Approach is the direction toward the contact site; jaw is the direction across which the fingers close. Use independent axes from current geometry. With a current suggested_jaw_world, omit BOTH to use the measured jaw and downward approach; changing the point alone does not repair collinearity.",
    },
}


class GraspOrientationError(ValueError):
    def __init__(self, code):
        self.code = code
        facts = ORIENTATION_ERRORS[code]
        # Generic error history is also visible to the tool-free supervisor.
        # Executor-specific repair guidance travels separately in its projection.
        super().__init__(code + ": " + facts["summary"])

    def diagnostic(self):
        return {"code": self.code, **ORIENTATION_ERRORS[self.code]}


def orientation_receipt_error(error):
    """Map only known host diagnostics, never copy arbitrary free-form errors."""
    if (not isinstance(error, dict) or error.get("code") != "direct_tool_rejected"
            or not isinstance(error.get("message"), str)):
        return None
    code = {
        "Collinear orientation basis": "grasp_orientation_collinear",
        "Invalid orientation basis": "grasp_orientation_invalid",
        "Supply both approach_world and jaw_world, or omit both": "grasp_orientation_pair_required",
    }.get(error.get("message"))
    return {"code": code, **ORIENTATION_ERRORS[code]} if code else None
