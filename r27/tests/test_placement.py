import asyncio
import pytest
from test_grasp_skill import Adapter
from embodied_harness.adapters.placement import place_point,validate_place


def adapter(**kwargs):
    a=Adapter(**kwargs);a.possible_hold=True
    a.last["robot"]={"grip_xyz_m":[0,0,1.2]}
    return a


def run(a):return asyncio.run(place_point(a,{"point_id":"p0"},a.cancelled,lambda _:None))


def test_placement_uses_measured_destination_height_and_preserves_orientation():
    a=adapter();validate_place(a,{"point_id":"p0"})
    assert run(a)["macro_status"]=="released_verify_placement_visually"
    assert a.calls[0][1]["xyz_m"]==[.1,.2,1.2]
    assert a.calls[1][1]["xyz_m"]==[.1,.2,1.02]
    assert all("approach_world" not in args for _,args in a.calls)
    assert not a.possible_hold


def test_unknown_lowering_never_opens_gripper():
    a=adapter(fail=2)
    assert run(a)["state_unknown"]
    assert all(name!="gripper_control" for name,_ in a.calls)
    assert a.possible_hold


def test_cancelled_transport_never_releases():
    a=adapter(cancel_after=1)
    assert run(a)["stop_confirmed"]
    assert len(a.calls)==1


def test_empty_gripper_and_old_destination_are_rejected():
    a=adapter();a.possible_hold=False
    with pytest.raises(ValueError,match="held load"):validate_place(a,{"point_id":"p0"})
    a.possible_hold=True;a.last["views"][0]["source_packet_id"]="new"
    with pytest.raises(ValueError,match="Remeasure"):validate_place(a,{"point_id":"p0"})


@pytest.mark.parametrize(
    ("inferred_opening", "expected_offset"),
    [(True, .06), (False, .12)],
)
def test_placement_default_lowering_uses_inferred_opening_or_manual_surface_offset(
    inferred_opening, expected_offset,
):
    a=adapter()
    if inferred_opening:
        a.points["p0"]["placement_anchor"]="estimated_visible_region_opening"
    args={"point_id":"p0"}
    validate_place(a,args)

    result=asyncio.run(place_point(a,args,a.cancelled,lambda _:None))

    assert result["macro_status"]=="released_verify_placement_visually"
    lower=a.calls[1][1]
    assert lower["xyz_m"][2]==pytest.approx(.9+expected_offset)


def test_placement_explicit_release_height_override_controls_actual_lowering():
    a=adapter()
    args={"point_id":"p0","release_height_m":.17,"clearance_m":.2}
    validate_place(a,args)

    asyncio.run(place_point(a,args,a.cancelled,lambda _:None))

    assert a.calls[1][1]["xyz_m"][2]==pytest.approx(1.07)


def test_placement_rejects_release_height_above_transport_clearance():
    a=adapter()
    with pytest.raises(ValueError,match="Release height exceeds transport clearance"):
        validate_place(a,{"point_id":"p0","release_height_m":.19,"clearance_m":.18})
