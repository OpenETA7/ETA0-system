import json

import pytest

from examples.hierarchy_demo import TwoStageRobot, lower, reply, upper
from harness import run
from hierarchy import Hierarchy


def episode(tmp_path, director, executor=lower):
    return run(
        TwoStageRobot(),
        executor,
        instruction="Place both items.",
        action_prompt="fake_step",
        output_dir=tmp_path / "episode",
        hierarchy=director,
    )


def events(tmp_path):
    path = tmp_path / "episode/events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_visual_supervision_and_teacher_use_one_environment(tmp_path):
    received = []

    def executor(messages):
        received.append(json.loads(messages[0]["content"].splitlines()[-1]))
        return lower(messages)

    result = episode(tmp_path, Hierarchy(upper, review_every=1), executor)
    assert result["success"] and result["actions"] == 2 and result["upper_calls"] == 4
    assert received[0]["active"] != received[1]["active"]
    assert received[1]["completed"] == [received[0]["active"]["instruction"]]
    review = [e for e in events(tmp_path) if e.get("role") == "supervisor"][0]
    assert review["frames"] == ["observation-000", "observation-001"]
    assert (tmp_path / "episode/terminal/external.png").is_file()


def test_ambiguous_goal_cannot_dispatch(tmp_path):
    result = episode(tmp_path, Hierarchy(lambda messages: reply({"question": "Which mug?"})))
    assert (result["reason"], result["actions"], result["detail"]) == (
        "needs_clarification",
        0,
        "Which mug?",
    )


def test_native_goal_overrides_last_subgoal_finish(tmp_path):
    def planner(messages):
        if messages[0]["content"].startswith("Interpret"):
            return reply({"steps": [{"instruction": "Place both.", "done_when": "Both placed."}]})
        return reply(
            dict(
                decision="stop",
                evidence=["observation-000"],
                reason="Target absent.",
                hint="Check the table.",
            )
        )

    def finish(messages):
        state = json.loads(messages[-1]["content"][-1]["text"])
        return reply(dict(stage="finish", observation_id=state["observation_id"]))

    result = episode(tmp_path, Hierarchy(planner, teacher=False), finish)
    assert (result["reason"], result["actions"], result["success"]) == ("supervisor_stop", 0, False)


def test_unknown_frame_rejected_before_next_action(tmp_path):
    def invalid(messages):
        if messages[0]["content"].startswith("Review"):
            return reply(dict(decision="advance", evidence=["imaginary"], reason="done", hint="go"))
        return upper(messages)

    with pytest.raises(ValueError, match="supplied frame"):
        episode(tmp_path, Hierarchy(invalid, review_every=1))
    assert sum(e["event"] == "action_started" for e in events(tmp_path)) == 1


def test_replan_replaces_suffix_and_regenerates_teacher(tmp_path):
    def revise(messages):
        if messages[0]["content"].startswith("Review"):
            return reply(
                dict(
                    decision="replan",
                    evidence=["observation-001"],
                    reason="Bowl moved.",
                    hint="Use visible handle.",
                    steps=[dict(instruction="Locate bowl again.", done_when="Bowl identified.")],
                )
            )
        return upper(messages)

    result = episode(tmp_path, Hierarchy(revise, review_every=1))
    steps = [e for e in events(tmp_path) if e["event"] == "subgoal"]
    assert result["actions"] == 2 and len(steps) == 2
    assert steps[1]["active"]["instruction"] == "Locate bowl again."
    assert len(steps[1]["remaining"]) == 1 and not steps[1]["completed"]
    assert (
        len(
            [
                e
                for e in events(tmp_path)
                if e.get("role") == "teacher" and e["event"] == "upper_request"
            ]
        )
        == 2
    )


def test_clarification_enters_both_planner_and_executor_context(tmp_path):
    def clarify(messages):
        if messages[0]["content"].startswith("Interpret"):
            goal = json.loads(messages[-1]["content"][0]["text"])
            if not goal["clarification"]:
                return reply({"question": "Which mug?"})
            assert goal["question"] == "Which mug?"
        return upper(messages)

    def executor(messages):
        guidance = json.loads(messages[0]["content"].splitlines()[-1])
        assert guidance["clarification"] == "Yellow mug."
        return lower(messages)

    director = Hierarchy(clarify, clarify=lambda question: "Yellow mug.", review_every=1)
    assert episode(tmp_path, director, executor)["success"]


def test_teacher_budget_stops_before_motion(tmp_path):
    result = episode(tmp_path, Hierarchy(upper, max_upper_calls=1))
    assert result["reason"] == "upper_call_budget"
    assert result["actions"] == result["calls"] == 0


def test_truncated_mode_retries_only_generation_then_restores_thinking():
    from eta0_run import ExecutorPolicy

    class Endpoint:
        def __init__(self):
            self.calls = []

        def chat(self, messages, **options):
            self.calls.append((messages, options))
            return {"choices": [{"finish_reason": "length" if len(self.calls) == 1 else "stop"}]}

    model = Endpoint()
    policy = ExecutorPolicy(model, 12, recover_length=True)
    original = [{"role": "user", "content": "real observation"}]
    for _ in range(3):
        policy(original)
    assert [c["chat_template_kwargs"]["enable_thinking"] for _, c in model.calls] == [
        True,
        False,
        True,
    ]
    assert [len(messages) for messages, _ in model.calls] == [1, 2, 1]
    assert [c["seed"] for _, c in model.calls] == [12, 13, 14]


def test_annotated_citation_repaired_at_same_observation(tmp_path):
    reviews = []

    def director(messages):
        if messages[0]["content"].startswith("Review"):
            state = json.loads(messages[-1]["content"][0]["text"])
            reviews.append(state)
            label = state["latest_frame_label"]
            return reply(
                dict(
                    decision="advance",
                    evidence=[label + " wrist view: aligned" if len(reviews) == 1 else label],
                    reason="Aligned",
                    hint="Continue",
                )
            )
        return upper(messages)

    result = episode(tmp_path, Hierarchy(director, review_every=1))
    assert result["success"] and result["actions"] == 2 and len(reviews) == 2
    assert reviews[0]["allowed_evidence_labels"] == reviews[1]["allowed_evidence_labels"]
    assert reviews[1]["correction"]
    assert sum(e["event"] == "invalid_citation" for e in events(tmp_path)) == 1
