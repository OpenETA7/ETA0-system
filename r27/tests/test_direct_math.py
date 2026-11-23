import numpy as np
import pytest
from embodied_harness.adapters.libero_direct import unproject, bounded_control


def test_depth_unprojection_matches_known_pinhole_rays():
    depth = np.ones((100,100)) * 2
    assert np.allclose(unproject(50,50,depth,90,np.eye(3),[1,2,3]), [1,2,1])
    assert np.allclose(unproject(75,25,depth,90,np.eye(3),[1,2,3]), [2,3,1])


def test_bounded_servo_normalizes_using_actual_controller_scale():
    scale = [.05,.05,.05,.5,.5,.5]
    action = bounded_control([1,-1,.01], [.4,-.4,.05], scale)
    assert np.allclose(action, [.4,-.4,.2,.24,-.24,.1])
    with pytest.raises(ValueError):
        bounded_control([0,0,0], [0,0,0], [0]*6)
