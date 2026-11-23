"""Offline checks for the public launcher; never runs physics or model requests."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from embodied_harness import app
from embodied_harness.runtime.topology import validate_two_role_options

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_reference", ROOT / "scripts/run_reference.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize("mode", launcher.MODES)
@pytest.mark.parametrize("suite,budget", [("libero_object", 24), ("libero_10", 36)])
def test_reference_cli_preserves_topology_and_budget(tmp_path, monkeypatch, mode, suite, budget):
    args = SimpleNamespace(suite=suite, task=3, initial_state=0, mode=mode,
                           output=tmp_path / "new-run", model_env=tmp_path / "private-env")
    command = launcher.command_for(args)
    captured = []
    async def capture(parsed):
        captured.append(parsed)
    monkeypatch.setattr(app, "episode", capture)
    monkeypatch.setattr(sys, "argv", ["embodied-eval", *command[3:]])
    app.main()
    parsed = captured[0]
    validate_two_role_options(parsed)
    assert parsed.max_decisions == budget
    assert parsed.backend == "direct_libero"
    assert parsed.two_role_runtime and not parsed.identity_check
    assert parsed.milestone_advice_delivery == (mode == "dual_async")
    assert parsed.scout_prompt_parity == (mode == "dual_async")
    assert not parsed.strong_target_judge and not parsed.strong_recovery_executor
    assert not Path(args.output).exists()


def test_launcher_refuses_existing_output(tmp_path):
    args = SimpleNamespace(suite="libero_object", task=3, initial_state=0,
                           mode="dual_async", output=tmp_path, model_env=tmp_path / "env")
    with pytest.raises(ValueError, match="already exists"):
        launcher.command_for(args)


@pytest.mark.parametrize("task,state", [(10, 0), (3, -1), (3, 50)])
def test_launcher_refuses_unknown_task_or_state(tmp_path, task, state):
    args = SimpleNamespace(suite="libero_object", task=task, initial_state=state,
                           mode="dual_async", output=tmp_path / "run", model_env=tmp_path / "env")
    with pytest.raises(ValueError):
        launcher.command_for(args)
