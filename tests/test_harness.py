import copy
import json

import numpy as np
import pytest
from PIL import Image

from harness import Observation, run
from tcp_overlay import CameraCalibration, RobotGeometry


class Robot:
    def __init__(self, target=1):
        self.steps = 0
        self.target = target
        self.fail = False

    def observe(self):
        oid = f"frame-{self.steps}"
        camera = CameraCalibration([[100, 0, 160], [0, 100, 120], [0, 0, 1]],
                                   np.eye(4), (320, 240), "base", oid)
        geometry = RobotGeometry([0, 0, 1], "base", oid, "tip")
        rgb = Image.new("RGB", (320, 240))
        return Observation(rgb, rgb, camera, geometry, {"steps": self.steps})

    def validate_action(self, action, observation):
        if action.get("stage") != "move" or set(action) != {"stage", "observation_id"}:
            raise ValueError("invalid move schema")

    def execute(self, action):
        self.steps += 1
        if self.fail:
            raise RuntimeError("physical error after a step")
        return {"measured_step": self.steps}

    def success(self):
        return self.steps >= self.target


def response(stage="move", oid="frame-0", finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {
        "content": json.dumps({"stage": stage, "observation_id": oid}),
        "reasoning_content": "recorded reasoning",
    }}]}


def invoke(tmp_path, robot, policy, **kwargs):
    return run(robot, policy, instruction="test", action_prompt="move then finish",
               output_dir=tmp_path / "run", **kwargs)


def test_one_action_native_success_and_two_images(tmp_path):
    robot = Robot()
    seen = []

    def policy(messages):
        seen.append(messages)
        return response()

    result = invoke(tmp_path, robot, policy)
    assert result["success"] and result["actions"] == 1 and result["calls"] == 1
    parts = seen[0][-1]["content"]
    assert sum(p["type"] == "image_url" for p in parts) == 2
    assert (tmp_path / "run/observation-000/wrist_raw.png").exists()
    assert (tmp_path / "run/RESULT.json").exists()


def test_model_finish_does_not_claim_success(tmp_path):
    result = invoke(tmp_path, Robot(), lambda messages: response("finish"))
    assert result["reason"] == "model_finish" and not result["success"]
    assert result["actions"] == 0


@pytest.mark.parametrize("bad", [response(oid="old"), response(finish="length"),
                                  response(stage="unknown")])
def test_rejection_executes_nothing_then_recovers(tmp_path, bad):
    robot = Robot()
    replies = iter([bad, response()])
    result = invoke(tmp_path, robot, lambda messages: next(replies))
    assert robot.steps == 1 and result["rejections"] == 1 and result["calls"] == 2


def test_physical_error_is_never_retried(tmp_path):
    robot = Robot()
    robot.fail = True
    with pytest.raises(RuntimeError, match="physical error"):
        invoke(tmp_path, robot, lambda messages: response())
    assert robot.steps == 1
    events = [json.loads(line) for line in (tmp_path / "run/events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "infrastructure_error"
    assert sum(e["event"] == "action_started" for e in events) == 1
    assert not any(e["event"] == "rejection" for e in events)


def test_history_evicts_only_old_images_and_keeps_reasoning(tmp_path):
    seen = []

    def policy(messages):
        seen.append(copy.deepcopy(messages))
        context = json.loads(messages[-1]["content"][-1]["text"])
        return response(oid=context["observation_id"])

    result = invoke(tmp_path, Robot(target=3), policy, keep_observations=1)
    assert result["actions"] == 3
    images = [p for m in seen[-1] if isinstance(m["content"], list)
              for p in m["content"] if p["type"] == "image_url"]
    assert len(images) == 2
    assert sum(m.get("reasoning_content") == "recorded reasoning" for m in seen[-1]) == 2


def test_budgets_and_existing_output_directory(tmp_path):
    robot = Robot(target=10)
    result = invoke(tmp_path, robot, lambda messages: response(), max_actions=1)
    assert result["reason"] == "action_budget" and not result["success"]
    with pytest.raises(FileExistsError):
        invoke(tmp_path, robot, lambda messages: response())


def test_bad_generations_are_call_bounded(tmp_path):
    robot = Robot()
    result = invoke(tmp_path, robot, lambda messages: response(finish="length"), max_calls=2)
    assert result["reason"] == "call_budget" and robot.steps == 0
