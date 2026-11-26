import json
import sys
import time
import urllib.request
from pathlib import Path

import pytest

import eta0_run
from deploy import first_run
from eta0_gate import RequestGate
from examples.dry_run import fake_policy
from examples.hierarchy_demo import TwoStageRobot
from harness import run
from request_harness import PROTOCOL, DemoLibrary, run_request
from rollout_context import strip_historical_examples


def test_adaptive_context_budget_and_tokenize_fallback():
    assert eta0_run.adaptive_context_budget(262144) == (262144, 16384, 200000)
    ctx_len, max_out, threshold = eta0_run.adaptive_context_budget(65536)
    assert ctx_len == 65536
    assert max_out == 16384
    assert threshold + max_out + 2048 <= ctx_len

    seen = []
    endpoint = eta0_run.ChatEndpoint(
        "http://127.0.0.1:8000/v1",
        "eta0-27b",
        "",
        time.monotonic() + 60,
        strict_context=False,
        on_context_len=seen.append,
    )
    endpoint.post = lambda url, payload: {"count": 1200, "max_model_len": 65536}
    assert endpoint.count([{"role": "user", "content": "hello"}]) == 1200
    assert seen == [65536]

    strict = eta0_run.ChatEndpoint(
        "http://127.0.0.1:8000/v1",
        "eta0-27b",
        "",
        time.monotonic() + 60,
        strict_context=True,
    )
    strict.post = lambda url, payload: {"count": 1200, "max_model_len": 65536}
    with pytest.raises(ValueError, match="pinned serving profile"):
        strict.count([{"role": "user", "content": "hello"}])

    # Non-vLLM endpoint without /tokenize falls back cleanly to local estimate.
    no_tok = eta0_run.ChatEndpoint(
        "http://127.0.0.1:11434/v1",
        "eta0-27b",
        "",
        time.monotonic() + 60,
        strict_context=False,
    )
    calls = []

    def fail_404(url, payload):
        calls.append(url)
        raise RuntimeError("Model eta0-27b returned HTTP 404 (not_found)")

    no_tok.post = fail_404
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello world"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
            ],
        }
    ]
    est1 = no_tok.count(msgs)
    est2 = no_tok.count(msgs)
    assert est1 == est2 and est1 > 1024
    assert len(calls) == 1  # cached fallback without repeated 404 probes


def test_executor_policy_openai_api_style_omits_vllm_only_fields():
    captured = {}

    class DummyEndpoint:
        def chat(self, messages, **options):
            captured.update(options)
            return {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}

    policy = eta0_run.ExecutorPolicy(DummyEndpoint(), seed=42, max_tokens=4096, api_style="openai")
    policy([{"role": "user", "content": "hi"}])
    assert captured["max_tokens"] == 4096
    assert "top_k" not in captured
    assert "repetition_penalty" not in captured
    assert "chat_template_kwargs" not in captured


def test_gate_live_observation_and_running_stop_halts_without_retry(tmp_path):
    gate = RequestGate(0, "Put both items away")
    base, token = gate.url.split("/#")
    auth = {"Authorization": "Bearer " + token}

    def get_state():
        req = urllib.request.Request(base + "/state", headers=auth)
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read())

    def post_stop():
        req = urllib.request.Request(
            base + "/decision",
            json.dumps({"action": "stop"}).encode(),
            {**auth, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status

    try:
        robot = TwoStageRobot()
        robot.meta = {"initial_state_sha256": "live-stop"}
        gate.scene(robot.observe(), "Put both items away")
        gate.running(1, "Put both items away", None)

        seen_live = []

        def on_obs(obs, actions, calls, overlay):
            gate.observe_live(obs, actions, calls, overlay)
            snap = get_state()
            seen_live.append((snap["live_actions"], snap["live_frame"]))
            if actions == 1:
                assert post_stop() == 200
                assert get_state()["stopping"] is True

        def attempt(path, budget, obs, context, number):
            return run(
                robot,
                fake_policy,
                instruction="Put both items away",
                action_prompt="Use fake_step.",
                output_dir=path,
                max_actions=5,
                budget=budget,
                initial_observation=obs,
                attempt_context=context,
                on_observation=on_obs,
                should_stop=gate.stop_requested,
            )

        result = run_request(
            robot,
            goal="Put both items away",
            output=tmp_path / "req",
            execute_attempt=attempt,
            max_attempts=2,
        )
        assert result["reason"] == "user_stopped"
        assert result["success"] is False
        assert len(result["attempts"]) == 1  # never retries after user_stopped
        assert robot.actions == 1
        assert seen_live == [(0, True), (1, True)]
    finally:
        gate.close()


def test_demo_library_chinese_recall_and_stratified_fallback(tmp_path):
    examples = []
    specs = [
        ("libero_10", 0, "turn on the stove and put the moka pot on it"),
        ("libero_10", 1, "put the black bowl in the bottom drawer of the cabinet"),
        ("libero_10", 2, "put the yellow and white mug in the microwave and close it"),
        ("libero_10", 3, "put the white mug on the left plate and put the yellow mug on the right"),
        ("libero_10", 4, "pick up the book and place it in the back compartment of the caddy"),
        ("libero_10", 5, "put the cream cheese in the bowl"),
        ("libero_10", 6, "put the wine bottle on the rack"),
        ("libero_10", 7, "put the soup in the basket"),
        ("libero_goal", 0, "open the middle drawer of the cabinet"),
        ("libero_object", 0, "pick up the alphabet soup and place it in the basket"),
        ("libero_spatial", 0, "pick up the black bowl next to the plate and place it on the plate"),
    ]
    for suite, tid, goal in specs:
        examples.append(
            dict(
                id=f"{suite}-{tid:02d}-00",
                suite=suite,
                task_id=tid,
                goal=goal,
                initial_state_sha256=f"hash-{suite}-{tid}",
                task_name=f"{suite}_{tid}",
                success=True,
                reason="native_success",
                segments=[],
                frames=[],
            )
        )
    lib_path = tmp_path / "library.json"
    lib_path.write_text(json.dumps(dict(version=1, protocol=PROTOCOL, examples=examples)))
    lib = DemoLibrary(lib_path)

    # Chinese instruction recalls the matching English LIBERO-Goal drawer demo first.
    zh_hits = lib.candidates("请打开柜子的中间抽屉", "other-state")
    assert zh_hits[0]["id"] == "libero_goal-00-00"

    # Zero-overlap query stratifies across all suites instead of only alphabetical libero_10.
    unseen_hits = lib.candidates("xyzzy frobnicate", "other-state")
    suites = {h["id"].split("-", 1)[0] for h in unseen_hits}
    assert suites == {"libero_10", "libero_goal", "libero_object", "libero_spatial"}


def test_strip_historical_examples_preserves_real_observations():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps({"historical_success_example": "other goal"})},
                {"type": "text", "text": "Historical external frame before action 1"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,DEMO"}},
                {"type": "text", "text": "External RGB"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,REAL"}},
            ],
        }
    ]
    cleaned = strip_historical_examples(messages)
    assert len(cleaned) == 1
    assert cleaned[0]["content"] == [
        {"type": "text", "text": "External RGB"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,REAL"}},
    ]


def test_first_run_skips_reinstall_when_stamp_matches(tmp_path, monkeypatch):
    config = tmp_path / "client.json"
    config.write_text("{}")
    cache = tmp_path / "cache"
    venv_bin = cache / "sim-py311" / "bin"
    venv_bin.mkdir(parents=True)
    fake_py = venv_bin / "python"
    fake_py.write_text("")
    monkeypatch.setenv("ETA0_EXECUTOR_CLIENT", str(config))
    monkeypatch.delenv("ETA0_SIM_PYTHON", raising=False)
    monkeypatch.setenv("ETA0_CACHE", str(cache))
    for k in ("CONFIG", "MODEL", "BASE_URL", "API_KEY"):
        monkeypatch.delenv("ETA0_UPPER_" + k, raising=False)
    monkeypatch.delenv("ETA0_DEMO_LIBRARY", raising=False)
    monkeypatch.setattr(first_run, "checkout", lambda _: tmp_path)
    monkeypatch.setattr(first_run.shutil, "which", lambda name: "/usr/bin/uv")

    invocations = []

    def fake_run(argv, *, check, env=None):
        invocations.append(argv)
        if Path(argv[1]).name == "eta0_run.py":
            out = Path(argv[argv.index("--output") + 1])
            (out / "rollout").mkdir(parents=True)
            (out / "run.json").write_text('{"native_goal":"open","request_features":{}}')
            (out / "rollout/RESULT.json").write_text(
                json.dumps(
                    dict(
                        success=True,
                        reason="native_success",
                        lower_chat_calls=1,
                        upper_calls=0,
                        episode_wall_s=1.0,
                    )
                )
            )

    monkeypatch.setattr(first_run.subprocess, "run", fake_run)

    # First run installs and writes stamp.
    monkeypatch.setattr(sys, "argv", ["first_run", "--output", str(tmp_path / "r1")])
    first_run.main()
    assert sum(cmd[0] == "/usr/bin/uv" and cmd[1] == "pip" for cmd in invocations) == 1

    # Second run reuses stamp and skips uv pip install.
    monkeypatch.setattr(sys, "argv", ["first_run", "--output", str(tmp_path / "r2")])
    first_run.main()
    assert sum(cmd[0] == "/usr/bin/uv" and cmd[1] == "pip" for cmd in invocations) == 1

    # Explicit --reinstall refreshes packages.
    monkeypatch.setattr(
        sys, "argv", ["first_run", "--output", str(tmp_path / "r3"), "--reinstall"]
    )
    first_run.main()
    assert sum(cmd[0] == "/usr/bin/uv" and cmd[1] == "pip" for cmd in invocations) == 2
