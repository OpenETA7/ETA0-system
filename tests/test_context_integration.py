import copy
import json

import pytest
from test_harness import Robot, response

from harness import run
from rollout_context import CompactionError, HistoryCompactor


def test_compact_recovers_one_observation_without_old_cot_and_keeps_audit(tmp_path):
    seen, summaries, events = [], [], []

    def summarize(messages, attempt, schema):
        summaries.append(messages)
        assert schema is None
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "An incomplete reply executed nothing; continue from frame-0."
                    },
                }
            ]
        }

    def count_fixture_tokens(messages):
        # Test-only counter: don't treat base64 image bytes as text tokens.
        wire = copy.deepcopy(messages)
        for message in wire:
            if isinstance(message.get("content"), list):
                for part in message["content"]:
                    if part.get("type") == "image_url":
                        part["image_url"]["url"] = "fixture-image"
        return len(json.dumps(wire))

    compact = HistoryCompactor(
        task="test",
        count_tokens=count_fixture_tokens,
        summarize=summarize,
        record=events.append,
        threshold=8000,
        max_model_len=50000,
        max_output_tokens=128,
    )

    def policy(messages):
        seen.append(messages)
        result = response(finish="length" if len(seen) == 1 else "stop")
        result["choices"][0]["message"]["reasoning_content"] = (
            "OLD_COT_" + "x" * 15000 if len(seen) == 1 else "fresh thought"
        )
        return result

    result = run(
        Robot(),
        policy,
        instruction="test",
        action_prompt="move",
        output_dir=tmp_path / "run",
        context_compactor=compact,
    )
    assert result["success"] and result["actions"] == 1
    assert compact.compactions == 1
    assert "OLD_COT_" in json.dumps(summaries[0])
    assert "OLD_COT_" not in json.dumps(seen[1])
    assert "OLD_COT_" in (tmp_path / "run/events.jsonl").read_text()
    assert events[-1]["retained_assistant_reasoning_chars"] == 0


def test_finish_feedback_does_not_stop_or_execute_physics(tmp_path):
    robot, seen = Robot(), []

    def policy(messages):
        seen.append(messages)
        return response(stage="finish" if len(seen) == 1 else "move")

    result = run(
        robot,
        policy,
        instruction="test",
        action_prompt="move",
        output_dir=tmp_path / "run",
        native_finish_feedback=True,
    )
    assert result["success"] and result["calls"] == 2 and robot.steps == 1
    feedback = json.loads(seen[1][-1]["content"][0]["text"])
    assert feedback["native_success"] is False and feedback["executed_ticks"] == 0


def test_observation_examples_bind_to_actual_id_without_rewriting_response(tmp_path):
    seen = []

    def policy(messages):
        seen.append(messages)
        return response(oid="CURRENT")

    result = run(
        Robot(),
        policy,
        instruction="test",
        action_prompt='Example: {"observation_id":"CURRENT"}',
        output_dir=tmp_path / "run",
        bind_observation_ids=True,
        max_calls=1,
    )
    assert '"observation_id":"frame-0"' in seen[0][0]["content"]
    assert result["actions"] == 0 and result["rejections"] == 1


def test_failed_compact_never_executes_a_physical_action(tmp_path):
    robot = Robot()
    compact = HistoryCompactor(
        task="test",
        count_tokens=lambda m: 9000,
        summarize=lambda *args: {
            "choices": [{"finish_reason": "stop", "message": {"content": ""}}]
        },
        threshold=8000,
        max_model_len=50000,
        max_output_tokens=128,
    )
    with pytest.raises(CompactionError):
        run(
            robot,
            lambda m: response(),
            instruction="test",
            action_prompt="move",
            output_dir=tmp_path / "run",
            context_compactor=compact,
        )
    assert robot.steps == 0 and compact.compactions == 0
    assert '"event": "infrastructure_error"' in (tmp_path / "run/events.jsonl").read_text()
