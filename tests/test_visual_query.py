import json

import numpy as np
import pytest
from PIL import Image

from harness import Observation, run
from tcp_overlay import CameraCalibration, RobotGeometry
from visual_query import QUERY_STAGES, QueryView, query


@pytest.mark.parametrize("budget,expected_queries,expected_rejections", [(1, 1, 1), (2, 2, 0)])
def test_mixed_queries_share_budget_and_never_execute_physics(
    tmp_path, budget, expected_queries, expected_rejections
):
    class ReadOnlyRobot:
        def __init__(self):
            self.observations = 0

        def observe(self):
            self.observations += 1
            return observation()

        def success(self):
            return False

        def validate_action(self, *args):
            raise AssertionError("queries are not robot actions")

        def execute(self, *args):
            raise AssertionError("queries must not execute physics")

    robot = ReadOnlyRobot()
    sequence = iter(
        [
            request(stage="bbox"),
            request(stage="mark_point"),
            dict(stage="finish", observation_id="frame-0"),
        ]
    )
    seen = []

    def policy(messages):
        seen.append(messages)
        return response(next(sequence))

    result = run(
        robot,
        policy,
        instruction="query test",
        action_prompt="test",
        output_dir=tmp_path / "run",
        max_queries_per_observation=budget,
    )
    assert result["queries"] == expected_queries
    assert result["rejections"] == expected_rejections
    assert result["actions"] == 0 and robot.observations == 1
    assert '"stage":"bbox"' in seen[0][0]["content"]
    assert '"stage":"mark_point"' in seen[0][0]["content"]
    if budget == 2:
        assert json.loads(seen[1][-1]["content"][0]["text"])["stage"] == "bbox"
        assert json.loads(seen[2][-1]["content"][0]["text"])["stage"] == "mark_point"


def test_mixed_schema_and_unknown_stages_are_rejected():
    box = request(stage="bbox")
    box["point_xy_px"] = [18, 12]
    with pytest.raises(ValueError, match="exactly"):
        query(box, observation())
    with pytest.raises(ValueError, match="must be bbox or mark_point"):
        query(dict(stage="unknown"), observation())


def observation(depth=True, mirrored=False):
    cam = CameraCalibration(
        [[-100 if mirrored else 100, 0, 16], [0, 100, 12], [0, 0, 1]],
        np.eye(4),
        (32, 24),
        "base",
        "frame-0",
    )
    rgb = Image.new("RGB", (32, 24))
    obs = Observation(rgb, rgb, cam, RobotGeometry([0, 0, 1], "base", "frame-0", "tip"), {})
    if depth:
        obs.query_views["wrist"] = QueryView(rgb, cam, np.ones((24, 32)))
    return obs


def request(oid="frame-0", *, stage):
    out = dict(stage=stage, observation_id=oid, camera="wrist")
    out.update({"bbox_xyxy_px": [16, 12, 18, 14]} if stage == "bbox" else {"point_xy_px": [18, 12]})
    return out


def response(action):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(action)}}]}


@pytest.mark.parametrize("stage", QUERY_STAGES)
def test_missing_depth_is_annotation_only_and_raw_image_untouched(stage):
    obs = observation(depth=False)
    before = obs.wrist_rgb.tobytes()
    annotated, feedback = query(request(stage=stage), obs)
    assert annotated.size == obs.wrist_rgb.size and obs.wrist_rgb.tobytes() == before
    assert not feedback["geometry_valid"] and feedback["executed_ticks"] == 0
    assert feedback["invalid_reason"] == "metric_depth_not_supplied"
    json.dumps(feedback, allow_nan=False)


@pytest.mark.parametrize("stage", QUERY_STAGES)
def test_exact_geometry_and_signed_intrinsics(stage):
    for mirrored in [False, True]:
        obs = observation(mirrored=mirrored)
        _, feedback = query(request(stage=stage), obs)
        assert feedback["geometry_valid"]
        if stage == "bbox":
            assert feedback["valid_points"] == 4
            assert feedback["axis_percentiles_mm"]["z"]["p50"] == 1000
            assert feedback["axis_percentiles_mm"]["x"]["min"] == pytest.approx(
                -10 if mirrored else 0
            )
            assert feedback["axis_percentiles_mm"]["x"]["max"] == pytest.approx(
                0 if mirrored else 10
            )
        else:
            np.testing.assert_allclose(
                feedback["surface_point_reference_mm"],
                [-20 if mirrored else 20, 0, 1000],
                atol=1e-9,
            )


@pytest.mark.parametrize("stage", QUERY_STAGES)
def test_invalid_depth_never_snaps_to_neighbour(stage):
    obs = observation()
    if stage == "bbox":
        obs.query_views["wrist"].depth_optical_m[12:14, 16:18] = np.nan
    else:
        obs.query_views["wrist"].depth_optical_m[12, 18] = np.nan
    _, feedback = query(request(stage=stage), obs)
    assert not feedback["geometry_valid"]
    if stage == "mark_point":
        assert feedback["surface_point_reference_mm"] is None
        assert feedback["local_depth_p10_p50_p90_mm"] == [1000, 1000, 1000]
    json.dumps(feedback, allow_nan=False)


@pytest.mark.parametrize("stage", QUERY_STAGES)
def test_bad_coordinates_stale_ids_and_depth_registration_rejected(stage):
    obs = observation()
    with pytest.raises(ValueError, match="stale"):
        query(request("old", stage=stage), obs)
    bad = request(stage=stage)
    key = "bbox_xyxy_px" if stage == "bbox" else "point_xy_px"
    bad[key][0] = -1
    with pytest.raises(ValueError):
        query(bad, obs)
    obs.query_views["wrist"].depth_optical_m = np.ones((23, 32))
    with pytest.raises(ValueError, match="registered"):
        query(request(stage=stage), obs)


@pytest.mark.parametrize("stage", QUERY_STAGES)
def test_query_returns_image_without_physics_or_new_observation(tmp_path, stage):
    class Robot:
        def __init__(self):
            self.observations = 0

        def observe(self):
            self.observations += 1
            return observation()

        def success(self):
            return False

        def validate_action(self, *args):
            raise AssertionError("query must not be dispatched to action validation")

        def execute(self, *args):
            raise AssertionError("query must not execute physics")

    robot = Robot()
    seen = []
    replies = iter(
        [request(stage=stage), request(stage=stage), dict(stage="finish", observation_id="frame-0")]
    )

    def policy(messages):
        seen.append(messages)
        return response(next(replies))

    result = run(
        robot,
        policy,
        instruction="test",
        action_prompt="test",
        output_dir=tmp_path / "run",
        max_queries_per_observation=1,
    )
    assert result["queries"] == 1 and result["actions"] == 0
    assert result["rejections"] == 1 and robot.observations == 1
    assert seen[1][-1]["content"][-1]["type"] == "image_url"
    assert (tmp_path / "run/observation-000/query-01.png").exists()


@pytest.mark.parametrize("stage", QUERY_STAGES)
def test_query_images_evicted_with_own_observation_and_budget_resets(tmp_path, stage):
    from test_harness import Robot

    robot = Robot(target=3)
    sequence = iter(
        [
            request(stage=stage),
            dict(stage="move", observation_id="frame-0"),
            request("frame-1", stage=stage),
            dict(stage="move", observation_id="frame-1"),
            dict(stage="finish", observation_id="frame-2"),
        ]
    )
    seen = []

    def policy(messages):
        seen.append(messages)
        return response(next(sequence))

    result = run(
        robot,
        policy,
        instruction="test",
        action_prompt="test",
        output_dir=tmp_path / "run",
        keep_observations=1,
        max_queries_per_observation=1,
    )
    assert result["queries"] == 2 and result["actions"] == 2
    assert result["rejections"] == 0
    images = [
        p
        for m in seen[-1]
        if isinstance(m["content"], list)
        for p in m["content"]
        if p["type"] == "image_url"
    ]
    assert len(images) == 2
    assert any(
        "metric_depth_not_supplied" in p.get("text", "")
        for m in seen[-1]
        if isinstance(m["content"], list)
        for p in m["content"]
    )
