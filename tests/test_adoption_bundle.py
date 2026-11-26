import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from deploy import first_run
from deploy.build_demo_library import build
from examples.dry_run import FakeRobot, fake_policy
from examples.hierarchy_demo import TwoStageRobot, reply
from harness import run, text
from request_harness import DemoLibrary, run_request
from rollout_context import strip_historical_examples
from skill_cards import SKILL_SCOPE, SkillLibrary, load_profile
from tcp_overlay import CameraCalibration, RobotGeometry, try_render_tcp_overlay

REPO = Path(__file__).resolve().parents[1]


def test_example_config_templates_and_catalog_cli(monkeypatch, capsys):
    client_tpl = json.loads((REPO / "examples/client.example.json").read_text())
    upper_tpl = json.loads((REPO / "examples/upper.example.json").read_text())
    assert {"base_url", "model", "api_key"} <= set(client_tpl)
    assert {"base_url", "model", "api_key", "allowed_returned_models"} <= set(upper_tpl)

    monkeypatch.setattr(sys, "argv", ["first_run", "--list-tasks"])
    first_run.main()
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 40

    monkeypatch.setattr(sys, "argv", ["first_run", "--list-tasks", "--suite", "libero_goal"])
    first_run.main()
    goal_lines = capsys.readouterr().out.strip().splitlines()
    assert len(goal_lines) == 10
    assert "open the middle drawer of the cabinet" in goal_lines[0]

    with pytest.raises(ValueError, match="out of range"):
        first_run.resolve_task("libero_goal", 99, 0)
    with pytest.raises(ValueError, match="initial-index"):
        first_run.resolve_task("libero_goal", 0, 999)


def test_first_run_profile_and_dry_run_with_cli_override(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ETA0_EXECUTOR_CLIENT", raising=False)
    for k in ("CONFIG", "MODEL", "BASE_URL", "API_KEY"):
        monkeypatch.delenv("ETA0_UPPER_" + k, raising=False)
    monkeypatch.delenv("ETA0_DEMO_LIBRARY", raising=False)

    out = tmp_path / "preview-out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "first_run",
            "--profile",
            str(REPO / "configs/profiles/libero_goal_drawer.json"),
            "--initial-index",
            "2",
            "--output",
            str(out),
            "--dry-run",
        ],
    )
    first_run.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["profile"] == "libero_goal_drawer"
    assert plan["suite"] == "libero_goal"
    assert plan["task"] == 0
    assert plan["initial_index"] == 2  # CLI overrides profile default 5
    assert plan["native_goal"] == "open the middle drawer of the cabinet"
    assert plan["skill_ids"] == ["articulated-interaction"]
    assert plan["executor_client_ready"] is False
    assert "--skills-dir" in plan["command"]
    assert "--skill" in plan["command"]
    assert not out.exists()


def test_demo_library_example_includes_anti_hallucination_limits(tmp_path):
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
                task_id=0,
                initial_index=0,
                instruction="open the middle drawer of the cabinet",
                initial_state_sha256="archive",
                task_name="open_middle_drawer",
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
    out = tmp_path / "lib"
    build([ep], out)
    lib = DemoLibrary(out / "library.json")
    parts = lib.example("libero_goal-00-00")
    header = json.loads(parts[0]["text"])
    assert "no current coordinates" in header["limits"]
    assert header["outcome"] == "native_success"


def test_skill_library_pinned_dynamic_and_compaction_stripping(tmp_path):
    lib = SkillLibrary(REPO / "skills")
    assert {c["id"] for c in lib.catalog()} >= {"pick-place", "articulated-interaction"}
    with pytest.raises(ValueError, match="digest mismatch"):
        SkillLibrary(REPO / "skills", expected_sha256={"pick-place": "0" * 64})

    profile = load_profile(REPO / "configs/profiles/libero_object_default.json", base_dir=REPO)
    assert profile["skill_ids"] == ["pick-place"]
    assert Path(profile["skills_dir"]).is_dir()

    # 1. Pinned read-only skill card works without any upper model.
    seen_contexts = []
    robot = TwoStageRobot()
    robot.meta = {"initial_state_sha256": "s1"}

    def attempt(path, budget, obs, context, number):
        seen_contexts.append(context)
        return run(
            robot,
            fake_policy,
            instruction="Pick up soup",
            action_prompt="Use fake_step.",
            output_dir=path,
            max_actions=2,
            budget=budget,
            initial_observation=obs,
            attempt_context=context,
        )

    res = run_request(
        robot,
        goal="Pick up soup",
        output=tmp_path / "pinned",
        execute_attempt=attempt,
        skills=lib,
        skill_ids=["pick-place"],
    )
    assert res["success"] and res["skills_sha256"] == lib.digest
    card_payload = json.loads(seen_contexts[0][0]["text"])
    assert card_payload["kind"] == "read_only_skill_guidance"
    assert card_payload["skill_id"] == "pick-place"
    assert card_payload["scope"] == SKILL_SCOPE

    # Static read-only skill guidance is stripped from compaction summary prefixes.
    stripped = strip_historical_examples(
        [{"role": "user", "content": [seen_contexts[0][0], text("External RGB")]}]
    )
    assert stripped[0]["content"] == [text("External RGB")]

    # 2. Dynamic skill selection by upper model when skill_ids is empty.
    robot2 = TwoStageRobot()
    robot2.meta = {"initial_state_sha256": "s2"}
    dynamic_contexts = []

    def upper(messages, **kwargs):
        payload = json.loads(messages[1]["content"][0]["text"])
        assert any(s["id"] == "articulated-interaction" for s in payload["skill_cards"])
        return reply(
            {
                "instruction": "Open the middle drawer smoothly.",
                "skill_id": "articulated-interaction",
            }
        )

    def attempt2(path, budget, obs, context, number):
        dynamic_contexts.append(context)
        return run(
            robot2,
            fake_policy,
            instruction="Open drawer",
            action_prompt="Use fake_step.",
            output_dir=path,
            max_actions=2,
            budget=budget,
            initial_observation=obs,
            attempt_context=context,
        )

    res2 = run_request(
        robot2,
        goal="Open drawer",
        output=tmp_path / "dynamic",
        execute_attempt=attempt2,
        chat=upper,
        rewrite=True,
        skills=lib,
    )
    assert res2["success"]
    assert any(
        json.loads(p["text"]).get("skill_id") == "articulated-interaction"
        for p in dynamic_contexts[0]
        if p["type"] == "text"
    )


def test_harness_pluggable_observation_writer_and_query_backend(tmp_path):
    robot = FakeRobot()
    writer_called = []
    query_called = []

    def custom_writer(observation, folder, instruction):
        writer_called.append(observation.robot.observation_id)
        ctx = dict(task=instruction, observation_id=observation.robot.observation_id, custom=True)
        return ctx, {"role": "user", "content": [text(ctx)]}

    def custom_query(action, observation):
        query_called.append(action["stage"])
        return Image.new("RGB", (16, 16), "blue"), {"stage": "triangulate", "ok": True}

    backend = SimpleNamespace(
        QUERY_NOTICE="Custom triangulate notice.",
        QUERY_STAGES={"triangulate"},
        query=custom_query,
    )

    step = {"n": 0}

    def policy(messages):
        assert "Custom triangulate notice." in messages[0]["content"]
        step["n"] += 1
        oid = robot.observe().robot.observation_id
        if step["n"] == 1:
            return reply({"stage": "triangulate", "observation_id": oid})
        return reply({"stage": "fake_step", "observation_id": oid})

    result = run(
        robot,
        policy,
        instruction="Custom backend task",
        action_prompt="Use fake_step.",
        output_dir=tmp_path / "pluggable",
        observation_writer=custom_writer,
        query_backend=backend,
    )
    assert result["success"] and result["queries"] == 1
    assert writer_called and query_called == ["triangulate"]


def test_try_render_tcp_overlay_fallback_and_franka_eef_label():
    img = Image.new("RGB", (64, 64), "gray")
    raw, meta = try_render_tcp_overlay(img, None, None)
    assert raw.size == (64, 64)
    assert meta["status"] == "unavailable"

    cam = CameraCalibration(
        [[60, 0, 32], [0, 60, 32], [0, 0, 1]],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        (64, 64),
        "robot0_base",
        "obs-1",
        "wrist",
    )
    bad_robot = RobotGeometry([0, 0, 0.2], "world", "obs-1", "measured TCP")
    _, bad_meta = try_render_tcp_overlay(img, cam, bad_robot)
    assert bad_meta["status"] == "unavailable" and "reference frames differ" in bad_meta["reason"]

    franka_robot = RobotGeometry(
        [0, 0, 0.2], "robot0_base", "obs-1", "configured Franka EEF pose"
    )
    ok_img, ok_meta = try_render_tcp_overlay(img, cam, franka_robot)
    assert ok_img.size == (64, 64)
    assert ok_meta["status"] == "available"
    assert "configured Franka EEF reference" in ok_meta["notice"]
