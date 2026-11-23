"""Optional executor guidance for obtaining an unoccluded recovery view.

This text does not detect occlusion, choose a motion, or grant a permission.
"""

POLICY_ID = "failed_grasp_view_recovery_v1"

VIEW_RECOVERY = """
VIEW RECOVERY AFTER A FAILED GRASP: If CURRENT images show that the gripper itself hides the needed
object surface, repeatedly selecting regions in that occluded view can select robot geometry. First
resolve the load state using CURRENT robot, tactile, image and receipt evidence. Do not release a
possibly carried object merely to improve the view. A view-clearing retreat is an option only after
possible_hold is resolved, the gripper is measurably open and current evidence supports an empty hand.
If the short reverse-approach path is visibly clear and the current contract permits motion, you may
choose ONE small move_to with delta_m in world coordinates opposite the CURRENT robot.approach_world,
at most 0.08 m. Derive the direction from measured robot state, not screen axes or an old point ID;
omit both orientation vectors to preserve orientation. This is a proposal for ordinary bounded
motion, not a collision-free path certificate. If the load, direction or clearance is uncertain,
inspect another available view or escalate instead of moving blindly.
After any move, use its fresh returned scene to reidentify the intended object and measure a new
site. Do not reuse prior candidate/point IDs or infer identity, retention or success from a clearer
view. Do not keep repeating retreat when it fails to reveal useful evidence. A failed grasp with an
already clear view does not call for this extra motion.
"""


def view_recovery_prompt(base: str) -> str:
    return base + VIEW_RECOVERY
