import json
import sys

from deploy import compare_upper


def test_five_state_schedule_has_one_matching_block_per_task():
    lanes = compare_upper.schedule(0, 5, 4)
    assert [len(lane) for lane in lanes] == [50] * 4
    assert len({item for lane in lanes for item in lane}) == 200


def test_four_arms_preflight_and_report(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(dict(base_url="https://example.invalid/v1", api_key="test-secret")))
    out = tmp_path / "trial"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test",
            "--models",
            "a",
            "b",
            "c",
            "--initial-index",
            "0",
            "--initial-count",
            "2",
            "--output",
            str(out),
            "--replicas",
            "1",
            "--remote-client",
            "client",
            "--boyue-config",
            str(cfg),
        ],
    )
    monkeypatch.setattr(compare_upper.batch, "ready", lambda client: None)
    monkeypatch.setattr(compare_upper.batch, "cases", lambda: [("libero_goal", 0)])
    checks = []

    def check(cmd, **kwargs):
        from pathlib import Path

        checks.append(json.loads(Path(cmd[-2]).read_text())["model"])

    def run(args, client, slot, case, mode):
        arm = args.output.name
        return dict(
            suite=case[0],
            task=case[1],
            initial_index=args.initial_index,
            replica=slot,
            seed=1234,
            success=arm != "upper-2",
            elapsed_s=10,
            reason="done",
            initial_view_sha256="v",
            initial_state_sha256="s",
            bddl_sha256="b",
        )

    monkeypatch.setattr(compare_upper.subprocess, "run", check)
    monkeypatch.setattr(compare_upper.batch, "run_episode", run)
    compare_upper.main()
    r = json.loads((out / "report.json").read_text())
    assert checks == ["a", "b", "c"]
    assert len(r) == 6 and all(x["paired"] == 2 for x in r.values())
    assert r["upper-1_vs_upper-2"]["right_minus_left_pp"] == -100
    assert len(json.loads((out / "episodes.json").read_text())) == 8


def test_resume_keeps_results_and_archives_only_unfinished_arm(tmp_path, monkeypatch):
    prior = tmp_path / "original/batch"
    out = tmp_path / "recovery/batch"
    prior.mkdir(parents=True)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(dict(api_key="test", base_url="https://example.invalid/v1")))
    models = {"base": None, "upper-1": "a", "upper-2": "b", "upper-3": "c"}
    task = ("libero_goal", 0)
    (prior / "manifest.json").write_text(
        json.dumps(dict(arms=models, clients=["one"], lanes=[[(0, task)]]))
    )

    def result(mode):
        return dict(
            suite=task[0],
            task=0,
            initial_index=0,
            seed=1234,
            replica=0,
            mode=mode,
            success=False,
            elapsed_s=10,
            reason="done",
            initial_view_sha256="v",
            initial_state_sha256="s",
            bddl_sha256="b",
        )

    for arm in models:
        mode = "base" if arm == "base" else "teacher"
        folder = prior / "initial-00" / arm / task[0] / "task-00"
        folder.mkdir(parents=True)
        if arm == "upper-2":
            (folder / mode).mkdir()
            (folder / (mode + ".log")).write_text("interrupted")
        else:
            (folder / (mode + ".result.json")).write_text(json.dumps(result(mode)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--models",
            "a",
            "b",
            "c",
            "--initial-index",
            "0",
            "--initial-count",
            "1",
            "--remote-client",
            "one",
            "--replicas",
            "1",
            "--boyue-config",
            str(profile),
            "--output",
            str(out),
            "--resume-from",
            str(prior),
        ],
    )
    monkeypatch.setattr(compare_upper.batch, "ready", lambda client: None)
    monkeypatch.setattr(compare_upper.batch, "cases", lambda: [task])
    monkeypatch.setattr(compare_upper.subprocess, "run", lambda *a, **kw: None)
    calls = []

    def resume(args, client, slot, case, mode):
        assert args.episode_runner == prior.parent / "source/deploy/run_libero.py"
        calls.append(args.output.name)
        return result(mode)

    monkeypatch.setattr(compare_upper.batch, "run_episode", resume)
    compare_upper.main()
    assert calls == ["upper-2"]
    assert len(json.loads((prior / "episodes.json").read_text())) == 4
    assert (
        out / "interrupted/initial-00/upper-2/libero_goal/task-00/teacher.log"
    ).read_text() == "interrupted"
