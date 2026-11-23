"""Optional prompt-only clarification of existing measured rim geometry.

No task-name dispatch, geometric computation, action repair or permission change.
The executor remains responsible for interpreting the current visual evidence.
"""

POLICY_ID = "rim_selection_before_orientation_v1"

_SELECTION = """2. Prefer select_candidate with a visible ID; use part=top_surface for picking a can/box,
   part=placement_center for a destination container/plate. This estimates a drop point over its visible 3D footprint and upper edge; it is NOT a measured floor. A normal image-box center can hit the front wall, and part=top_rim,
   bottom_rim,left_rim,right_rim for a bowl-wall pinch. A rim candidate supplies an estimated jaw direction
   and object-center offset automatically; omit orientation overrides unless visual evidence requires one."""

_RIM_SELECTION = """2. Choose the contact geometry BEFORE grasping. For a wide open bowl, select_candidate with a CURRENT
   top_rim, bottom_rim, left_rim or right_rim part measures one wall and can supply an estimated closing direction.
   Do not substitute center or top_surface for a bowl rim pinch. Use top_surface for a can/box;
   use placement_center for a destination container/plate. A placement_center estimates the opening/upper edge,
   not a measured floor or a grasp point. Prefer a visible rim candidate over a manually guessed bowl pixel
   when its segmentation is suitable; inspect and use manual pixel tools when no suitable rim region exists."""

_ORIENTATION = """   A wide bowl requires pinching ONE RIM WALL (one finger inside, one outside), not grasping its center.
   Set jaw_world so closing spans the local rim wall, and approach_world=[0,0,-1]. Both vectors together."""

_MEASURED_ORIENTATION = """   A wide bowl requires pinching ONE RIM WALL (one finger inside, one outside), not grasping its center.
   Check the measured point's visual_candidate.suggested_jaw_world. When present, OMIT BOTH jaw_world and
   approach_world to let grasp_point use that measured estimate. Override both only when current geometric
   evidence justifies a different orientation; screen-horizontal does not imply world [0,1,0].
   If the point has no suggested_jaw_world, omission preserves the current wrist orientation; it does NOT
   infer a bowl-wall pinch. Obtain a suitable current rim measurement, or explicitly derive both vectors
   from current measured geometry. After a contact failure, keep the intended object's identity and measure
   another visible rim site instead of switching to a different object or repeating an unsupported axis.
   A suggested direction is fallible geometry, not proof of target identity, clearance, contact or success."""


def rim_guided_prompt(base: str) -> str:
    """Replace only the two contradictory geometry instructions in FAST."""
    if base.count(_SELECTION) != 1 or base.count(_ORIENTATION) != 1:
        raise ValueError("Rim guidance no longer matches the executor prompt; review the ablation")
    return base.replace(_SELECTION, _RIM_SELECTION).replace(_ORIENTATION, _MEASURED_ORIENTATION)
