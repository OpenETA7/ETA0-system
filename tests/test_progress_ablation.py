import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from deploy import progress_ablation as study
from examples.hierarchy_demo import TwoStageRobot, reply
from harness import run


@pytest.mark.parametrize("access", ["on_request", "hidden"])
def test_finish_and_observation_do_not_disclose_receipts(access, tmp_path):
    robot = TwoStageRobot()
    robot.completion = {
        "status": "checked",
        "feedback": {"conditions": ["private-condition-marker"]},
    }
    oid = robot.observe().robot.observation_id
    stages = iter(["finish", "check_task_completion", "fake_step", "fake_step"])

    def policy(messages):
        stage = next(stages)
        if stage == "check_task_completion":
            assert "private-condition-marker" not in json.dumps(messages)
        if stage == "fake_step" and robot.actions == 0:
            assert ("private-condition-marker" in json.dumps(messages)) == (access == "on_request")
        assert ("Optional task progress tool" in messages[0]["content"]) == (access == "on_request")
        return reply({"stage": stage, "observation_id": oid})

    r = run(
        robot,
        policy,
        instruction="test",
        action_prompt="move",
        output_dir=tmp_path / "episode",
        native_finish_feedback=True,
        progress_access=access,
    )
    assert (
        r["success"] and r["completion_queries"] == (access == "on_request") and r["actions"] == 2
    )
    assert "completion_evidence" in (tmp_path / "episode/events.jsonl").read_text()


def test_entire_schedule_and_report_offline(tmp_path, monkeypatch):
    plan = study.manifest()
    assert len(plan["pairs"]) == 200
    assert Counter(p["first"] for p in plan["pairs"]) == {"off": 100, "on": 100}
    indexed = {(p["suite"], p["task"], p["initial"]): p for p in plan["pairs"]}
    assert len(indexed) == 200
    for case in study.batch.cases():
        assert len({indexed[case[0], case[1], i]["state"] for i in range(5)}) == 5
    client = tmp_path / "client.json"
    client.write_text(
        json.dumps(
            dict(model=study.MODEL, target=study.TARGET, draft=plan["draft"], base_url="unused")
        )
    )
    monkeypatch.setattr(study.batch, "ready", lambda _: None)
    seen = []

    def episode(args, client_path, slot, case, mode):
        assert mode == "base" and args.progress_query_ablation and client_path == client
        p = indexed[case[0], case[1], args.initial_index]
        on = args.completion_query
        arm = "on" if on else "off"
        seen.append((case, args.initial_index, arm))
        _, folder = study.locations(tmp_path, p, arm)
        assert Path(args.output) / case[0] / f"task-{case[1]:02d}" == folder
        native = folder / "base/simulation"
        native.mkdir(parents=True)

        def put(path, **fields):
            path.write_text(json.dumps(fields))

        put(
            native / "task.json",
            suite=p["suite"],
            task_id=p["task"],
            task_name=p["name"],
            initial_index=p["initial"],
            seed=study.batch.SEED,
        )
        put(
            folder / "base/run.json",
            mode="base",
            executor=study.MODEL,
            upper=None,
            instruction_mode="revised",
            completion_query=on,
            progress_access="on_request" if on else "hidden",
            roles=dict(planner=False, supervisor=False, teacher=False),
            pack_sha256=study.PACK,
            wall_seconds=2700,
            max_actions=120,
            max_calls=360,
            recover_length=True,
        )
        success = args.initial_index == 1 or (
            args.initial_index == 0 and (on if case[1] == 0 else not on)
        )
        put(
            folder / "base.result.json",
            returncode=0,
            success=success,
            bddl_sha256=p["bddl"],
            initial_state_sha256=p["state"],
            completion_queries=int(on),
            completion_query_events=int(on),
            upper_models=[],
            upper_wall_s=0,
            initial_view_sha256="a" * 64,
            replica=slot,
            elapsed_s=9 if on else 12,
        )

    monkeypatch.setattr(study.batch, "run_episode", episode)
    args = SimpleNamespace(output=tmp_path, client=[client], libero_source_root=tmp_path, slots=2)
    study.run_queue(args, plan)
    study.run_queue(args, plan)
    assert len(set(seen)) == len(seen) == 400
    assert study.report(tmp_path, plan) == 200
    summary = study.load(tmp_path / "report.json")
    assert summary["overall"]["on"]["queries"] == 200 and summary["overall"]["delta_s"] == -3
    assert Counter(t["direction"] for t in summary["tasks"]) == {"改善": 4, "退化": 36}
    p = plan["pairs"][0]
    _, off = study.locations(tmp_path, p, "off")
    path = off / "base.result.json"
    data = study.load(path)
    path.write_text(json.dumps(dict(data, initial_view_sha256="b" * 64)))
    assert study.report(tmp_path, plan) == 199
    assert (
        study.load(tmp_path / "report.json")["pair_details"][0]["status"]["pair"]
        == "view_or_replica_mismatch"
    )
    path.unlink()
    assert study.report(tmp_path, plan) == 199
    study.run_queue(args, plan)
    assert len(seen) == 401 and study.report(tmp_path, plan) == 200


@pytest.mark.parametrize("enabled", [False, True])
def test_real_batch_command_is_executor_only(tmp_path, monkeypatch, enabled):
    cfg = SimpleNamespace(
        output=tmp_path,
        initial_index=4,
        completion_query=enabled,
        progress_query_ablation=True,
        libero_source_root="/fixture/libero",
    )
    seen = []

    class Process:
        returncode = 0

        def __init__(self, cmd, **kwargs):
            seen.extend(cmd)
            rollout = Path(cmd[cmd.index("--output") + 1]) / "rollout"
            rollout.mkdir(parents=True)
            (rollout / "RESULT.json").write_text(
                json.dumps(
                    dict(success=False, reason="action_budget", completion_queries=int(enabled))
                )
            )
            if enabled:
                (rollout / "events.jsonl").write_text('{"event":"completion_query"}\n')

        def wait(self, timeout):
            assert timeout == study.batch.WALL + 150

    monkeypatch.setattr(study.batch.subprocess, "Popen", Process)
    r = study.batch.run_episode(cfg, "client", 0, ("libero_goal", 0), "base")
    assert seen[seen.index("--mode") + 1] == "base" and "--boyue-config" not in seen
    assert "--progress-query-ablation" in seen and "--upstream-base" not in seen
    assert seen[seen.index("--instruction-mode") + 1] == "revised"
    assert ("--completion-query" in seen) == enabled and (
        "--no-completion-query" in seen
    ) != enabled
    assert r["completion_queries"] == r["completion_query_events"] == int(enabled)
