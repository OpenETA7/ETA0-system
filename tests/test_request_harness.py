import hashlib
import itertools
import json
import sys
from types import SimpleNamespace

import pytest

import eta0_run
from deploy.build_demo_library import build
from examples.hierarchy_demo import TwoStageRobot, reply
from request_harness import DemoLibrary, run_request


@pytest.fixture
def library(tmp_path):
    ep = tmp_path / "archive"
    sim = ep / "simulation"
    folder = ep / "rollout/observation-000"
    sim.mkdir(parents=True)
    folder.mkdir(parents=True)
    TwoStageRobot().observe().external_rgb.save(folder / "external.png")
    (ep / "rollout/RESULT.json").write_text(
        json.dumps(dict(reason="native_success", success=True, actions=1))
    )
    (sim / "task.json").write_text(
        json.dumps(
            dict(
                suite="libero_goal",
                task_id=8,
                initial_index=0,
                instruction="Place the can on the rack",
                initial_state_sha256="archive",
                task_name="can_on_rack",
            )
        )
    )
    (sim / "latest.json").write_text('{"native_success": true}')
    pose = {"position_m": [0.2, 0, 0.3]}
    end = dict(
        before_eef_pose_base=pose,
        actual_eef_pose_base=pose,
        requested_final_pose={"gripper_command": 0},
        executed_ticks=5,
        termination="done",
    )
    (sim / "controller.jsonl").write_text(json.dumps(dict(action={"stage": "move"}, result=end)))
    folder = tmp_path / "demos"
    build([ep], folder)
    return folder / "library.json"


@pytest.mark.parametrize("rewrite,retrieve", list(itertools.product((False, True), repeat=2)))
def test_request_cli_preserves_round_boundary(tmp_path, monkeypatch, library, rewrite, retrieve):
    instances, calls = [], []
    goal = "Place both cans on the rack."
    rewritten = "Place both silver cans on the rack."

    class Robot(TwoStageRobot):
        def __init__(self, *args, **kwargs):
            super().__init__()
            instances.append(self)
            self.instruction = goal
            self.meta = dict(initial_state_sha256="test")
            self.completion = None

        def close(self):
            pass

    class Upper:
        model = "scripted-upper"
        chat_calls = 0

        def chat(self, messages, **options):
            self.chat_calls += 1
            n = instances[0].actions
            calls.append(("upper", n))
            data = json.loads(messages[1]["content"][0]["text"])
            assert sum(p["type"] == "image_url" for p in messages[1]["content"]) == 2
            assert data["original_goal"] == goal
            if n:
                assert data["previous_attempt"]["reason"] == "action_budget"
                assert "latest_measured_state" in data
            answer = {}
            if rewrite:
                answer["instruction"] = rewritten
            if retrieve:
                answer["demo_id"] = None if n else data["candidates"][0]["id"]
            return reply(answer)

    upper = Upper()

    def lower(self, url, payload):
        n = instances[0].actions
        calls.append(("lower", n))
        wire = json.dumps(payload["messages"])
        assert (rewritten in wire) == rewrite
        assert ("historical_success_example" in wire) == (retrieve and n == 0)
        assert goal in wire
        assert not any(m["role"] == "assistant" for m in payload["messages"])
        oid = instances[0].observe().robot.observation_id
        return reply(dict(stage="fake_step", observation_id=oid))

    monkeypatch.setitem(
        sys.modules,
        "rollout_libero_minimal",
        SimpleNamespace(ACTION_PROMPT="Use fake_step.", LiberoAdapter=Robot),
    )
    monkeypatch.setattr(
        eta0_run,
        "upper_endpoint",
        lambda *a: (
            upper
            if rewrite or retrieve
            else pytest.fail("baseline must not initialize an upper model")
        ),
    )
    monkeypatch.setattr(eta0_run.ChatEndpoint, "count", lambda *a: 100)
    monkeypatch.setattr(eta0_run.ChatEndpoint, "post", lower)
    monkeypatch.delenv("ETA0_LIBERO_ROOT", raising=False)
    client = tmp_path / "client.json"
    client.write_text('{"base_url":"unused","model":"27b","api_key":""}')
    out = tmp_path / "request"
    args = [
        "eta0-run",
        "--task",
        "0",
        "--initial-index",
        "5",
        "--output",
        str(out),
        "--executor-client",
        str(client),
        "--instruction-mode",
        "original",
        "--no-completion-query",
        "--max-actions",
        "1",
        "--no-clarify",
        "--rewrite" if rewrite else "--no-rewrite",
        "--retrieve" if retrieve else "--no-retrieve",
    ]
    if retrieve:
        args.extend(["--demo-library", str(library)])
    monkeypatch.setattr(sys, "argv", args)
    eta0_run.main()
    result = json.loads((out / "rollout/RESULT.json").read_text())
    assert result["success"] and not result["first_attempt_success"]
    assert [a["reason"] for a in result["attempts"]] == ["action_budget", "native_success"]
    assert result["upper_calls"] == (2 if rewrite or retrieve else 0)
    assert len(instances) == 1
    assert calls == [
        (role, n)
        for n in range(2)
        for role in (("upper", "lower") if rewrite or retrieve else ("lower",))
    ]


def test_clarification_answer_or_abstention(tmp_path):
    goal = "Pick up that can."
    for interactive in (False, True):
        robot = TwoStageRobot()
        replies = iter([{"question": "Which can?"}, {}])

        def chat(*args, script=replies, **kwargs):
            return reply(next(script))

        def attempt(path, budget, obs, context, number, robot=robot):
            assert "silver" in context[0]["text"]
            robot.actions = 2
            return dict(
                reason="native_success", success=True, calls=1, actions=1, completion_queries=0
            )

        r = run_request(
            robot,
            goal=goal,
            output=tmp_path / str(interactive),
            chat=chat,
            clarify=True,
            ask=(lambda q: "silver") if interactive else None,
            execute_attempt=attempt,
        )
        assert r["success"] == interactive
        assert r["upper_calls"] == (2 if interactive else 1)
        if not interactive:
            assert r["reason"] == "needs_clarification" and robot.actions == 0


def successful_attempt(robot, contexts):
    def execute(path, budget, obs, context, number):
        contexts.append(context)
        robot.actions = 2
        return dict(reason="native_success", success=True, actions=1, calls=1, completion_queries=0)

    return execute


def test_provenance_and_unknown_demo(tmp_path, library):
    lib = DemoLibrary(library)
    assert lib.candidates("can", "archive") == []
    key = lib.candidates("can", "test")[0]["id"]
    assert len(lib.example(key)) == 3
    robot = TwoStageRobot()
    robot.meta = dict(initial_state_sha256="test")
    contexts = []
    r = run_request(
        robot,
        goal="Place the can",
        output=tmp_path / "unknown",
        library=lib,
        retrieve=True,
        chat=lambda *a, **kw: reply(dict(demo_id="invented")),
        execute_attempt=successful_attempt(robot, contexts),
    )
    assert r["success"] and r["preparation_fallbacks"] == 1 and r["upper_calls"] == 5
    assert contexts == [[]]
    frame = library.parent / lib.entries[key]["frames"][0]["file"]
    frame.write_bytes(hashlib.sha256(b"corrupt").digest())
    with pytest.raises(ValueError, match="provenance"):
        lib.example(key)


def test_upper_waits_through_query_and_false_finish(tmp_path):
    from harness import run

    robot = TwoStageRobot()
    robot.completion = {"status": "pending"}
    order = []
    stages = iter(["check_task_completion", "finish", "fake_step", "fake_step"])
    oid = robot.observe().robot.observation_id

    def lower(messages):
        stage = next(stages)
        order.append(stage)
        return reply(dict(stage=stage, observation_id=oid))

    def upper(*args, **kwargs):
        order.append("prepare")
        return reply(dict(instruction="Put the two cans away."))

    def attempt(path, budget, obs, context, index):
        return run(
            robot,
            lower,
            instruction="Put the two cans away.",
            action_prompt="Use fake_step",
            output_dir=path,
            native_finish_feedback=True,
            progress_access="on_request",
            budget=budget,
            attempt_context=context,
            initial_observation=obs,
        )

    result = run_request(
        robot,
        output=tmp_path / "run",
        goal="Put the two cans away.",
        rewrite=True,
        chat=upper,
        execute_attempt=attempt,
    )
    assert result["first_attempt_success"] and result["completion_queries"] == 1
    assert order == ["prepare", "check_task_completion", "finish", "fake_step", "fake_step"]


def test_pilot_excludes_incomplete_and_unmatched_cases(tmp_path):
    from deploy.request_comparison import score

    row = dict(status="scored", identity=["a"], success=False)
    assert score(dict(baseline=row, dual=dict(row, success=True))) == "win"
    assert score(dict(dual=row, baseline=dict(row, success=True))) == "loss"
    assert score(dict(baseline=row, dual=dict(row, identity=["b"]))) == "mismatch"
    assert score(dict(baseline=row, dual=dict(status="unscored"))) == "incomplete"


def test_upper_call_budget_retains_confirmed_answer(tmp_path):
    robot, contexts = TwoStageRobot(), []
    r = run_request(
        robot,
        goal="Pick up that can.",
        output=tmp_path / "request",
        max_upper_calls=1,
        clarify=True,
        chat=lambda *a, **kw: reply({"question": "Which can?"}),
        ask=lambda _: "Silver.",
        execute_attempt=successful_attempt(robot, contexts),
    )
    assert r["success"] and r["upper_calls"] == 1 and r["preparation_fallbacks"] == 1
    handoff = json.loads(contexts[0][0]["text"])
    assert handoff["grounded_instruction"] == "Pick up that can."
    assert handoff["user_clarifications"] == [{"question": "Which can?", "answer": "Silver."}]


@pytest.mark.parametrize("named", [False, True])
def test_cloud_preview_packages_pilot_and_only_public_model_settings(tmp_path, monkeypatch, named):
    import ast
    import base64
    import io
    import zipfile

    from deploy import boyue_episode

    profile = tmp_path / "provider.json"
    profile.write_text('{"boyue":{"base_url":"https://example.invalid","api_key":"fake-secret"}}')
    if named:
        profile.write_text(
            json.dumps(
                {
                    "kimi": {
                        "base_url": "https://example.invalid",
                        "model": "kimi-k3",
                        "allowed_returned_models": ["kimi-k3"],
                        "api_key": "fake-secret",
                    },
                    "other": {"api_key": "other-secret"},
                }
            )
        )
    model = "kimi-k3" if named else "kimi-k3.1-pro"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit",
            *(["--config-profile", "kimi"] if named else []),
            "--request-comparison",
            "--config",
            str(profile),
            "--remote-client",
            "/shared/client.json",
        ],
    )

    def preview(cli, name, program, *, runner, submit, cpu_workers, priority):
        assert priority == 5
        assert name == "eta0-request-comparison" and runner and not submit and cpu_workers == 2
        assert "fake-secret" not in program
        opts = ast.literal_eval(program.splitlines()[0].removeprefix("options = "))
        assert opts["profile"]["allowed_returned_models"] == [model]
        assert opts["episode_args"][opts["episode_args"].index("--upper-model") + 1] == model
        assert "other-secret" not in program
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(opts["source"]))) as bundle:
            assert {
                opts["entrypoint"],
                "deploy/build_demo_library.py",
                "request_harness.py",
            } <= set(bundle.namelist())

    monkeypatch.setattr(boyue_episode, "create_job", preview)
    boyue_episode.main()


def test_40_task_report_keeps_missing_cases_out_of_denominator(tmp_path):
    from deploy.request_comparison import ARMS, cases, report

    assert len(set(cases())) == len(cases()) == 40
    suite, task = cases()[0]
    folder = tmp_path / suite / f"task-{task:02d}"
    folder.mkdir(parents=True)
    for arm in ARMS:
        row = dict(
            status="scored",
            identity=["same"],
            replica=1,
            success=arm == "dual",
            first_attempt_success=False,
            elapsed_s=10,
            lower_wall_s=7,
            upper_wall_s=3,
            calls=2,
            upper_calls=int(arm == "dual"),
            completion_queries=0,
            attempts=2,
            selected_demos=0,
        )
        (folder / f"{arm}.json").write_text(json.dumps(row))
    result = report(tmp_path)
    assert len(result["pairs"]) == 40 and result["matched"] == 1
    assert result["pairs"][0]["outcome"] == "win"
    assert result["means"]["dual"]["success"] == 1


def test_resume_retains_completed_and_replays_only_interrupted(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from deploy import request_comparison as runner

    original = tmp_path / "first/batch"
    resume = tmp_path / "retry/batch"
    args = SimpleNamespace(
        target=original,
        output=resume,
        resume_from=original,
        remote_client=["same-client"],
        boyue_config="private-profile",
        upper_model=runner.MODEL,
    )
    folder = original / "libero_goal/task-00"
    folder.mkdir(parents=True)
    final = {"status": "scored", "success": False, "reason": "wall_time_budget"}
    (folder / "baseline.json").write_text(json.dumps(final))
    (folder / "dual").mkdir()
    (folder / "dual/partial").write_text("interrupted")
    (folder / "dual.log").write_text("first execution")
    calls = []

    def outcome(out, lib, index, suite, task, arm, upper_model):
        if arm == "dual" and not calls:
            raise FileNotFoundError()
        return final

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner, "read_outcome", outcome)
    monkeypatch.setattr(runner.subprocess, "run", run)
    for arm in runner.ARMS:
        runner.launch(args, tmp_path / "library.json", 0, "libero_goal", 0, arm)
    assert (resume / "interrupted/libero_goal/task-00/dual/partial").read_text() == "interrupted"
    assert len(calls) == 1 and str(original.parent / "source/deploy/run_libero.py") in calls[0]
    assert json.loads((folder / "baseline.json").read_text()) == final


def test_preparation_failure_is_preserved_and_reported_as_unfulfilled(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from deploy import request_comparison as runner

    original = tmp_path / "first/batch"
    suite, task = runner.cases()[0]
    folder = original / suite / f"task-{task:02d}"
    folder.mkdir(parents=True)
    failure = dict(
        status="unscored",
        identity=["hash"],
        replica=1,
        success=False,
        first_attempt_success=False,
        elapsed_s=35,
        lower_wall_s=0,
        upper_wall_s=35,
        calls=0,
        upper_calls=1,
        reason="preparation_error",
        completion_queries=0,
        selected_demos=0,
    )
    success = dict(
        failure,
        status="scored",
        success=True,
        first_attempt_success=True,
        reason="native_success",
        lower_wall_s=35,
        upper_wall_s=0,
        calls=7,
        upper_calls=0,
        completion_queries=2,
    )
    (folder / "dual.json").write_text(json.dumps(failure))
    (folder / "baseline.json").write_text(json.dumps(success))
    args = SimpleNamespace(
        target=original,
        resume_from=original,
        output=tmp_path / "new/batch",
        upper_model=runner.MODEL,
    )
    monkeypatch.setattr(runner, "read_outcome", lambda *a: failure)
    runner.launch(args, tmp_path / "library.json", 0, suite, task, "dual")
    r = runner.report(original)
    assert r["matched"] == 0 and r["terminal_pairs"] == 1
    report = runner.render_report(r)
    assert "Kimi + 27B 完成 0/1" in report and "native_success / preparation_error" in report
    assert "| 27B | 1/1 | 0.6 | 0.6 | 0.0 | 7 | 0 | 2 |" in report
    assert "| Kimi + 27B | 0/1 | 0.6 | 0.0 | 0.6 | 0 | 1 | 0 |" in report


def test_truncated_preparation_records_provider_evidence(tmp_path):
    robot, contexts = TwoStageRobot(), []
    output = tmp_path / "request"
    result = run_request(
        robot,
        goal="Pick up the silver can.",
        output=output,
        rewrite=True,
        chat=lambda *args, **kw: {
            "choices": [{"finish_reason": "length", "message": {"content": None}}],
            "usage": {"completion_tokens": 1500},
        },
        execute_attempt=successful_attempt(robot, contexts),
    )
    assert result["success"] and result["upper_calls"] == 5 and contexts == [[]]
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    event = next(e for e in events if e["event"] == "preparation_response")
    assert event["finish_reason"] == "length"
    assert event["completion_tokens"] == 1500 and event["content_chars"] == 0

    assert [e["retry"] for e in events if e["event"] == "preparation_invalid"] == list(range(5))
    assert [e["cause"] for e in events if e["event"] == "preparation_fallback"] == ["truncated"]


def test_preparation_recovers_on_last_retry(tmp_path):
    goal = "Pick the can up."
    robot, contexts = TwoStageRobot(), []
    caps = []
    responses = iter(
        [
            {"choices": [{"finish_reason": "stop", "message": {"content": " "}}]},
            {"choices": [{"finish_reason": "length", "message": {"content": '{"instruction":'}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": "bad JSON"}}]},
            {"choices": []},
            reply({"instruction": "Pick up the silver can."}),
        ]
    )

    def chat(messages, *, deadline, max_tokens):
        caps.append(max_tokens)
        assert json.loads(messages[-1]["content"][0]["text"])["original_goal"] == goal
        return next(responses)

    result = run_request(
        robot,
        goal=goal,
        output=tmp_path / "request",
        rewrite=True,
        chat=chat,
        execute_attempt=successful_attempt(robot, contexts),
    )
    assert result["success"] and result["upper_calls"] == 5 and result["preparation_fallbacks"] == 0
    assert caps == [1500, 1500, 4096, 4096, 4096]
    assert json.loads(contexts[0][0]["text"])["grounded_instruction"] == "Pick up the silver can."


def test_second_round_fallback_discards_previous_rewrite_and_demo(tmp_path, library):
    robot = TwoStageRobot()
    robot.meta = {"initial_state_sha256": "test"}
    demo = DemoLibrary(library)
    chosen = next(iter(demo.entries))
    goal = "Place the can on the rack."
    plan = reply({"instruction": "Move the silver can to the rack.", "demo_id": chosen})
    answers = iter([plan, *[reply({"instruction": "partial"})] * 4])

    def attempt(path, budget, obs, context, number):
        assert budget.lower_s == 123 and budget.upper_elapsed == 0
        if number == 1:
            assert chosen in str(context) and "silver can" in str(context)
        else:
            assert robot.actions == 1 and context == []
        robot.actions = number
        return dict(
            success=robot.success(),
            reason="native_success" if number == 2 else "action_budget",
            actions=1,
            calls=1,
            completion_queries=0,
        )

    result = run_request(
        robot,
        goal=goal,
        output=tmp_path / "request",
        rewrite=True,
        retrieve=True,
        library=demo,
        chat=lambda *a, **kw: next(answers),
        execute_attempt=attempt,
        max_upper_calls=5,
        wall_seconds=123,
    )
    assert result["upper_calls"] == 5 and result["preparation_fallbacks"] == 1 and result["success"]
    events = [
        json.loads(line) for line in (tmp_path / "request/events.jsonl").read_text().splitlines()
    ]
    last = [e for e in events if e["event"] == "attempt_started"][-1]
    assert last["instruction"] == goal and last["demo_id"] is None


def test_exhausted_upper_wall_hands_off_without_model_call(tmp_path):
    robot, contexts = TwoStageRobot(), []
    r = run_request(
        robot,
        goal="Lift the can.",
        output=tmp_path / "request",
        rewrite=True,
        upper_seconds=0,
        chat=lambda *a, **kw: pytest.fail("no budget"),
        execute_attempt=successful_attempt(robot, contexts),
    )
    assert r["success"] and r["upper_calls"] == 0 and r["preparation_fallbacks"] == 1
    assert contexts == [[]]
