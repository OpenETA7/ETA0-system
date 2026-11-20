import json
import sys

import pytest

from deploy import libero40


@pytest.mark.parametrize("count", [3, 4])
def test_paired_workers_cover_all_cases_once_on_matching_replicas(tmp_path, monkeypatch, count):
    output = tmp_path / "batch"

    def fake_replicas(*args):
        return list(range(count))

    def episode(args, client, replica, case, mode):
        assert client == replica
        suite, task = case
        return dict(
            mode=mode,
            suite=suite,
            task=task,
            success=task % 2 == (mode == "base"),
            reason="done",
            initial_index=args.initial_index,
            seed=1234,
            replica=replica,
            initial_view_sha256=str(case),
            initial_state_sha256=str(case),
            bddl_sha256=str(case),
            elapsed_s=5 if mode == "base" else 7,
        )

    monkeypatch.setattr(libero40, "start_replicas", fake_replicas)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "libero40",
            "--replicas",
            str(count),
            "--initial-index",
            "0",
            "--executor-client",
            "test",
            "--boyue-config",
            "test",
            "--output",
            str(output),
        ],
    )
    seen = []

    def recorded(*args):
        row = episode(*args)
        seen.append(row)
        return row

    monkeypatch.setattr(libero40, "run_episode", recorded)
    libero40.main()
    report = json.loads((output / "report.json").read_text())
    assert report["completed_episodes"] == 80 and report["paired_scored"] == 40
    assert report["paired_outcomes"] == {"base_only": 20, "teacher_only": 20}
    assert report["paired_teacher_minus_base_wall"]["mean_s"] == 2
    assert len({(r["suite"], r["task"], r["replica"]) for r in seen}) == 40
    assert len({(r["suite"], r["task"], r["mode"]) for r in seen}) == 80


def test_incomplete_and_mismatched_pairs_are_visible():
    row = dict(
        mode="base",
        suite="libero_goal",
        task=0,
        success=True,
        reason="native_success",
        elapsed_s=10,
        initial_index=0,
        seed=1234,
        initial_view_sha256="view",
        initial_state_sha256="state",
        bddl_sha256="bddl",
    )
    teacher = dict(row, mode="teacher", success=None, reason="incomplete")
    stats = libero40.summarize([row, teacher])
    assert stats["paired_scored"] == 0
    assert stats["arms"]["teacher"]["success_per_attempt"] == 0
    assert stats["arms"]["teacher"]["success_per_scored"] is None
    teacher.update(success=True, initial_view_sha256="different")
    stats = libero40.summarize([row, teacher])
    assert stats["initial_view_mismatches"] == [("libero_goal", 0, 0, 1234)]
    assert stats["paired_scored"] == 0
