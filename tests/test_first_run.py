import json
import subprocess
import sys
from pathlib import Path

import pytest

from deploy import first_run


@pytest.mark.parametrize("upper", [False, True])
def test_first_run_from_config_paths(tmp_path, monkeypatch, upper, capsys):
    config = tmp_path / "private.json"
    config.write_text("{}")
    output = tmp_path / "receipt dir"
    monkeypatch.setenv("ETA0_EXECUTOR_CLIENT", str(config))
    monkeypatch.setenv("ETA0_SIM_PYTHON", sys.executable)
    monkeypatch.setenv("ETA0_CACHE", str(tmp_path / "cache"))
    for k in ("CONFIG", "MODEL", "BASE_URL", "API_KEY"):
        monkeypatch.delenv("ETA0_UPPER_" + k, raising=False)
    monkeypatch.delenv("ETA0_DEMO_LIBRARY", raising=False)
    if upper:
        monkeypatch.setenv("ETA0_UPPER_CONFIG", str(config))
        monkeypatch.setenv("ETA0_DEMO_LIBRARY", str(config))
    monkeypatch.setattr(first_run, "checkout", lambda _: tmp_path)
    monkeypatch.setattr(sys, "argv", ["first_run", "--output", str(output), "--gate", "8765"])

    def run(argv, *, check, env):
        assert check and argv[0] == sys.executable
        assert Path(argv[1]).name == "eta0_run.py"
        assert argv[argv.index("--output") + 1] == str(output)
        assert argv[argv.index("--gate") + 1] == "8765"
        assert ("--no-rewrite" in argv) != upper
        assert ("--demo-library" in argv) == upper
        (output / "rollout").mkdir(parents=True)
        (output / "run.json").write_text('{"native_goal":"open drawer","request_features":{}}')
        result = dict(
            success=False, reason="call_budget", lower_chat_calls=2, upper_calls=0, episode_wall_s=1
        )
        (output / "rollout/RESULT.json").write_text(json.dumps(result))

    monkeypatch.setattr(subprocess, "run", run)
    first_run.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["success"] is False and summary["reason"] == "call_budget"
    assert Path(summary["receipt"]).is_file()
