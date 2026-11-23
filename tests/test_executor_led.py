import itertools
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from examples.hierarchy_demo import TwoStageRobot, reply
from executor_led import EpisodeBudget, ExecutorLed, StallDetector
from harness import run
from libero_semantics import SemanticEpisode, task_semantics


def receipt(*flags):
    return {
        "status": "checked",
        "feedback": {
            "conditions": [
                {"requirement_id": str(i), "satisfied_in_last_evaluation": value}
                for i, value in enumerate(flags)
            ]
        },
    }


def test_native_plateau_requires_consecutive_known_action_receipts():
    detector = StallDetector(failures=3, plateau=2)
    state = {
        "eef_pose_base": {"position_mm": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
        "gripper_open_width_mm": 32,
    }

    def observe(*flags):
        return detector.observe("execution", {}, receipt(*flags), state)

    assert observe(False, False) is None
    assert observe(False, False) is None
    assert observe(None, False) is None
    for _ in range(4):
        assert observe(False, False) is None
    assert observe(False, False) == "unchanged_goal_conditions"
    assert observe(False, False) is None
    for _ in range(4):
        assert observe(True, False) is None
    assert observe(True, False) == "unchanged_goal_conditions"
    moving = StallDetector()
    for step in range(20):
        state["eef_pose_base"]["position_mm"][0] = step * 15
        assert moving.observe("execution", {}, receipt(False), state) is None


def test_tracking_error_needs_current_and_repeated_measured_misses():
    detector = StallDetector()

    def result(mm, *goals, termination="path_complete"):
        return detector.observe(
            "execution",
            {"termination": termination, "final_goal_error": {"position_error_mm": mm}},
            receipt(*goals),
        )

    for _ in range(13):
        assert result(0) is None
    for _ in range(2):
        assert result(30) is None
    assert result(30) == "repeated_tracking_error"
    for _ in range(10):
        assert result(55) is None
    for _ in range(8):
        assert result(10) is None
    for _ in range(2):
        assert result(31) is None
    assert result(31) == "repeated_tracking_error"

    # Actual newly satisfied predicates clear both windows before another hint.
    assert result(45, False) is None
    assert result(45, True) is None
    assert detector.actions_since_progress == 0
    for _ in range(15):
        assert result(45, True) is None
    assert result(45, True) == "repeated_tracking_error"


def test_unknown_or_interrupted_feedback_cannot_complete_tracking_window():
    detector = StallDetector()

    def action(error, termination="path_complete"):
        return detector.observe(
            "execution",
            {"termination": termination, "final_goal_error": {"position_error_mm": error}},
        )

    for _ in range(16):
        assert action(2) is None
    assert action(35) is None
    assert action(35) is None
    assert action(70, "tracking_lag_reobserve") is None
    assert action(35) is None
    for value in (None, "30", float("nan"), float("inf"), -3):
        assert action(value) is None
    for _ in range(7):
        assert action(35) is None
    assert action(2) is None
    assert action(35) == "repeated_tracking_error"


def test_zero_tick_queries_and_normal_paths_are_not_hard_failures():
    detector = StallDetector(failures=3)
    assert detector.observe("rejection") is None
    assert (
        detector.observe("execution", {"termination": "path_complete", "executed_ticks": 8}) is None
    )
    for _ in range(12):
        assert detector.observe("visual_query") is None
    assert detector.observe("execution", {"termination": "tracking_lag_reobserve"}) is None
    assert detector.observe("execution", {"executed_ticks": 0}) is None
    assert detector.observe("finish_feedback") == "consecutive_failures"
    assert detector.observe("rejection") is None


@pytest.mark.parametrize(
    "planner,supervisor,teacher", list(itertools.product((False, True), repeat=3))
)
def test_independent_roles_native_query_and_one_action_authority(
    tmp_path, planner, supervisor, teacher
):
    robot = TwoStageRobot()
    robot.completion = {"status": "pending", "reason": "no_native_evaluation"}
    oid = robot.observe().robot.observation_id
    stages = iter(["check_task_completion", "finish", "bad", "bad", "fake_step", "fake_step"])

    def lower(messages):
        assert "Optional guidance" not in messages[0]["content"]
        current = next(stages)
        if current == "finish":
            assert json.loads(messages[-1]["content"][-1]["text"]) == robot.completion
        return reply({"stage": current, "observation_id": oid})

    def upper(messages):
        prompt = messages[0]["content"]
        if '"plan"' in prompt:
            return reply({"plan": "Place both items in the basket."})
        if '"hint"' in prompt:
            return reply({"hint": "Use the measured action schema."})
        return reply(
            {"cue": "Repeated rejection", "skill": "Read the schema", "expected": "Valid action"}
        )

    budget = EpisodeBudget(100, 50)
    director = ExecutorLed(
        upper, budget, planner=planner, supervise=supervisor, teacher=teacher, long_task=True
    )
    result = run(
        robot,
        lower,
        instruction="Place both items",
        action_prompt="fake",
        output_dir=tmp_path / "out",
        hierarchy=director,
        budget=budget,
    )
    events = [json.loads(line) for line in (tmp_path / "out/events.jsonl").read_text().splitlines()]
    assert Counter(e["role"] for e in events if e["event"] == "upper_request") == Counter(
        {
            "planner": int(planner),
            "supervisor": int(supervisor),
            "teacher": int(teacher),
        }
    )
    assert (result["success"], result["actions"], result["queries"], result["calls"]) == (
        True,
        2,
        1,
        6,
    )
    if supervisor or teacher:
        stall_index = next(i for i, e in enumerate(events) if e["event"] == "stall_intervention")
        assert sum(e["event"] == "response" for e in events[:stall_index]) == 4


@pytest.mark.parametrize("bad_reply", ["timeout", "clarify", "stop"])
def test_upper_wait_and_failure_preserve_lower_wall_budget(tmp_path, bad_reply):
    now = [100.0]
    budget = EpisodeBudget(3, 10, clock=lambda: now[0])
    robot = TwoStageRobot()
    oid = robot.observe().robot.observation_id

    def upper(messages):
        assert budget.upper_deadline == 110
        now[0] += 10
        if bad_reply == "timeout":
            raise TimeoutError()
        return reply(
            {"question": "Which object?"} if bad_reply == "clarify" else {"decision": "stop"}
        )

    director = ExecutorLed(
        upper, budget, planner=True, supervise=True, teacher=True, long_task=True, failures=1
    )
    stages = iter(["finish", "fake_step", "fake_step"])

    def lower(messages):
        now[0] += 0.5
        return reply({"stage": next(stages), "observation_id": oid})

    result = run(
        robot,
        lower,
        instruction="Place both",
        action_prompt="fake",
        output_dir=tmp_path / "out",
        hierarchy=director,
        budget=budget,
    )
    assert result["success"] and result["calls"] == 3 and result["upper_calls"] == 1
    assert (result["episode_wall_s"], result["lower_wall_s"], result["upper_wall_s"]) == (
        11.5,
        1.5,
        10,
    )
    assert budget.lower_deadline() == 113


def test_short_task_proceeds_without_planner(tmp_path):
    robot = TwoStageRobot()
    oid = robot.observe().robot.observation_id

    def lower(messages):
        return reply({"stage": "fake_step", "observation_id": oid})

    budget = EpisodeBudget(10, 5)
    director = ExecutorLed(
        lambda _: pytest.fail("unnecessary call"),
        budget,
        planner=True,
        supervise=True,
        teacher=True,
        long_task=False,
    )
    result = run(
        robot,
        lower,
        instruction="Place item",
        action_prompt="fake",
        output_dir=tmp_path / "out",
        hierarchy=director,
        budget=budget,
    )
    assert result["success"] and result["upper_calls"] == 0


def test_bind_all_revised_instructions_and_task_lengths(monkeypatch):
    packs = pytest.importorskip("openeta_bench.packs")
    pack = packs.load_pack("libero")
    monkeypatch.setattr(packs, "load_pack", lambda name: pack)
    sizes = Counter()
    long = 0
    for row in pack.document["tasks"]:
        robot = SimpleNamespace(
            meta={"task_name": row["task_id"], "suite": row["variant"]},
            instruction=row["instruction"]["original"]["text"],
        )
        semantic = SemanticEpisode(robot, "revised", False)
        assert semantic.instruction == row["instruction"]["revised"]["text"]
        assert semantic.long_task == (len(row["native_goals"]) > 1)
        assert semantic.completion is None
        sizes[row["variant"]] += 1
        long += semantic.long_task
    assert sizes == dict(
        libero_spatial=10, libero_object=10, libero_goal=10, libero_10=10, libero_90=90
    )
    assert long == 20
    with pytest.raises(ValueError, match="suite"):
        task_semantics(pack, row["task_id"], "wrong")


def test_official_passive_query_reads_latest_transition(tmp_path, monkeypatch):
    libero = pytest.importorskip("openeta_bench.adapters.libero")
    pack = libero.TaskPack
    # Only source attestation is substituted for this simulator double.
    monkeypatch.setattr(pack, "verify_sources", lambda *a: None)
    monkeypatch.setattr(libero, "require_method_source", lambda *a: None)
    key = "put_the_wine_bottle_on_the_rack"
    from openeta_bench.packs import load_pack

    p = load_pack("libero")

    class Native:
        bddl_file_name = str(p.source_path(tmp_path, key))
        parsed_problem = {"goal_state": [libero.LiberoAdapter.SINGLE_GOALS[key]]}
        checks = 0

        def _eval_predicate(self, predicate):
            return False

        def _check_success(self):
            self.checks += 1
            return all(self._eval_predicate(g) for g in self.parsed_problem["goal_state"])

        def step(self):
            return self._check_success()

        def reward(self):
            return self._check_success()

    native = Native()

    def execute(action):
        native.step()
        return {"executed_ticks": 1}

    robot = SimpleNamespace(
        meta={"task_name": key, "suite": "libero_goal"},
        instruction="original",
        env=SimpleNamespace(env=native),
        execute=execute,
    )
    semantic = SemanticEpisode(robot, "original", True, tmp_path)
    assert semantic.instruction == "original" and semantic.completion["status"] == "pending"
    assert semantic.execute({}) == {"executed_ticks": 1}
    result = semantic.completion
    assert result["status"] == "checked" and result["source_action_index"] == 1
    assert result["feedback"]["conditions"][0]["satisfied_in_last_evaluation"] is False
    semantic.refresh()
    assert (
        native.checks == 1 and semantic.completion["feedback"]["native_evaluator_calls_added"] == 0
    )


@pytest.mark.parametrize("conditions", [(False,), (True, False, None)])
def test_executor_can_read_goal_progress_then_continue_unassisted(tmp_path, conditions):
    robot = TwoStageRobot()
    robot.completion = receipt(*conditions)
    oid = robot.observe().robot.observation_id
    stages = enumerate(["fake_step", "check_task_completion", "check_task_completion", "fake_step"])

    def lower(messages):
        decision, stage = next(stages)
        assert "check_task_completion" in messages[0]["content"]
        if decision < 2:
            # Ordinary physical observations don't include an unsolicited progress report.
            assert "conditions" not in messages[-1]["content"][-1]["text"]
        else:
            assert json.loads(messages[-1]["content"][0]["text"]) == robot.completion
            assert robot.actions == 1
        return reply({"stage": stage, "observation_id": oid})

    result = run(
        robot,
        lower,
        instruction="Complete the goal",
        action_prompt="fake",
        output_dir=tmp_path / "out",
    )
    assert result["completion_queries"] == result["queries"] == 2
    assert result["success"] and result["actions"] == 2 and result["calls"] == 4


def test_unsupported_progress_tool_cannot_reach_a_physical_controller(tmp_path):
    from examples.dry_run import FakeRobot

    robot = FakeRobot()
    stages = iter(["check_task_completion", "fake_step"])
    oid = robot.observe().robot.observation_id

    def lower(messages):
        stage = next(stages)
        assert "check_task_completion" not in messages[0]["content"]
        if stage == "fake_step":
            assert "task progress tool unavailable" in messages[-1]["content"][0]["text"]
        return reply({"stage": stage, "observation_id": oid})

    def validate(action, obs):
        assert action["stage"] == "fake_step"

    robot.validate_action = validate
    result = run(
        robot,
        lower,
        instruction="Complete the goal",
        action_prompt="fake",
        output_dir=tmp_path / "out",
    )
    assert result["success"] and result["actions"] == result["rejections"] == 1
    assert result["completion_queries"] == 0


@pytest.mark.parametrize("override", [False, True])
def test_episode_entrypoint_with_real_pack_and_scripted_executor(tmp_path, monkeypatch, override):
    import sys

    import eta0_run
    from examples.dry_run import FakeRobot, fake_policy

    packs = pytest.importorskip("openeta_bench.packs")
    key = "put_the_wine_bottle_on_the_rack"

    class TestAdapter(FakeRobot):
        def __init__(self, *args, suite_name, **kwargs):
            super().__init__()
            self.instruction = "put the wine bottle on the rack"
            self.meta = dict(task_name=key, suite=suite_name)

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "rollout_libero_minimal",
        SimpleNamespace(ACTION_PROMPT="Use fake_step.", LiberoAdapter=TestAdapter),
    )
    monkeypatch.setattr(eta0_run.ChatEndpoint, "count", lambda *a: 100)
    monkeypatch.setattr(
        eta0_run.ChatEndpoint, "post", lambda self, url, payload: fake_policy(payload["messages"])
    )
    monkeypatch.setattr(
        eta0_run, "upper_endpoint", lambda *a: pytest.fail("short planner used upper")
    )
    monkeypatch.delenv("ETA0_LIBERO_ROOT", raising=False)
    cfg = tmp_path / "client.json"
    cfg.write_text(json.dumps({"base_url": "unused", "model": "27b", "api_key": ""}))
    out = tmp_path / "episode"
    args = [
        "eta0-run",
        "--suite",
        "libero_goal",
        "--task",
        "9",
        "--initial-index",
        "0",
        "--mode",
        "base",
        "--planner",
        "--no-completion-query",
        "--executor-client",
        str(cfg),
        "--output",
        str(out),
    ]
    if override:
        args.extend(["--instruction", "My wording"])
    monkeypatch.setattr(sys, "argv", args)
    eta0_run.main()
    meta = json.loads((out / "run.json").read_text())
    result = json.loads((out / "rollout/RESULT.json").read_text())
    assert meta["pack_version"] == "0.3.2" and meta["upper"] is None
    assert meta["user_goal"] == (
        "My wording" if override else packs.load_pack("libero").instruction(key, "revised")
    )
    assert result["success"] and result["lower_chat_calls"] == result["calls"] == 1
    assert result["lower_wall_s"] == result["episode_wall_s"] and result["upper_wall_s"] == 0


def test_upper_absolute_timeout_stops_wait_while_lower_has_time():
    import time

    budget = EpisodeBudget(10, 0.03)
    with pytest.raises(TimeoutError, match="upper wall budget"):
        with budget.upper_interval():
            time.sleep(0.25)
    assert budget.upper_elapsed >= 0.02
    assert budget.lower_deadline() - time.monotonic() > 9


def test_batch_watchdog_reserves_both_budgets(tmp_path, monkeypatch):
    from deploy import libero40

    episode = tmp_path / "libero_goal/task-00/teacher/rollout"
    episode.mkdir(parents=True)
    (episode / "RESULT.json").write_text('{"success":true,"upper_wall_s":12,"lower_chat_calls":4}')

    class Process:
        returncode = 0

        def __init__(self, cmd, **kw):
            assert "--no-planner" in cmd and "--supervisor" in cmd and "--no-teacher" in cmd
            assert cmd[cmd.index("--upper-wall-seconds") + 1] == "123"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self, timeout):
            assert timeout == 2700 + 123 + 150

    monkeypatch.setattr(libero40.subprocess, "Popen", Process)
    args = SimpleNamespace(
        output=tmp_path,
        initial_index=0,
        boyue_config="unused",
        planner=False,
        supervisor=True,
        teacher=False,
        upper_wall_seconds=123,
    )
    row = libero40.run_episode(args, "unused", 0, ("libero_goal", 0), "teacher")
    assert row["lower_chat_calls"] == 4 and row["upper_wall_s"] == 12


@pytest.mark.parametrize(
    "mode,frozen,expected",
    [
        ("base", False, "--no-completion-query"),
        ("teacher", False, "--no-completion-query"),
        ("base", True, None),
        ("teacher", True, "--no-completion-query"),
    ],
)
def test_progress_ablation_reaches_the_paired_runner(tmp_path, monkeypatch, mode, frozen, expected):
    from deploy import libero40

    episode = tmp_path / f"libero_spatial/task-00/{mode}/rollout"
    episode.mkdir(parents=True)
    (episode / "RESULT.json").write_text('{"success":true,"completion_queries":0}')
    args = SimpleNamespace(
        output=tmp_path,
        initial_index=0,
        upstream_base=frozen,
        completion_query=False,
        boyue_config="unused",
    )

    class Process:
        returncode = 0

        def __init__(self, cmd, **kw):
            flags = [part for part in cmd if part.endswith("completion-query")]
            assert flags == ([expected] if expected else [])
            assert ("--upstream-base" in cmd) == (mode == "base" and frozen)

        def wait(self, timeout):
            pass

    monkeypatch.setattr(libero40.subprocess, "Popen", Process)
    row = libero40.run_episode(args, "unused", 0, ("libero_spatial", 0), mode)
    assert row["completion_queries"] == 0
