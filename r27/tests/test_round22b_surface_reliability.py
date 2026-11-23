import numpy as np
import pytest

from embodied_harness.adapters.vision_geometry import candidate_surface, reliable_surface_mask


def scene():
    y, x = np.mgrid[:21, :21]
    cloud = np.stack([x * .002, y * .002, np.full_like(x, .1, dtype=float)], axis=-1)
    labels = np.zeros((21, 21), dtype=int)
    labels[4:17, 4:17] = 1
    # A diagonal background hole defeats a central-difference normal while
    # still making the grasp validator's 3x3 depth range discontinuous.
    labels[11, 11] = 0
    cloud[11, 11, 2] = .5
    candidate = {"component": 1, "bbox_xyxy": [4, 4, 17, 17],
                 "center_pixel_xy": [10, 10], "center_xyz_m": [.02, .02, .1]}
    return candidate, labels, cloud


@pytest.mark.parametrize("part", ["center", "placement_center", "top_surface", "top_rim",
                                  "bottom_rim", "left_rim", "right_rim"])
def test_selection_stays_on_requested_component_and_avoids_depth_hole(part):
    candidate, labels, cloud = scene()
    x, y, _ = candidate_surface(candidate, labels, cloud, part, reliable=True)
    assert labels[y, x] == candidate["component"]
    assert reliable_surface_mask(cloud)[y, x]
    assert np.ptp(cloud[y-1:y+2, x-1:x+2, 2]) < .04


def test_old_top_surface_can_choose_diagonal_discontinuity():
    candidate, labels, cloud = scene()
    x, y, _ = candidate_surface(candidate, labels, cloud, "top_surface")
    assert not reliable_surface_mask(cloud)[y, x]
    assert np.ptp(cloud[y-1:y+2, x-1:x+2, 2]) > .04


def test_no_reliable_patch_reports_failure_instead_of_inventing_geometry():
    candidate, labels, cloud = scene()
    cloud[..., 2] = np.indices(labels.shape).sum(axis=0) % 2
    with pytest.raises(ValueError, match="No reliable visible depth patch"):
        candidate_surface(candidate, labels, cloud, "top_surface", reliable=True)


def test_continuity_bound_is_rotation_invariant_and_rejects_nonfinite_neighbors():
    _, _, cloud = scene()
    mask = reliable_surface_mask(cloud)
    assert not mask[10, 10]
    rotated = cloud @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_equal(mask, reliable_surface_mask(rotated))
    cloud[5, 5] = np.nan
    assert not reliable_surface_mask(cloud)[4:7, 4:7].any()
