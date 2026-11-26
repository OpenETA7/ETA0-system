import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

from eta0_gate import RequestGate
from examples.request_demo import demo
from request_harness import GateCancelled


def test_gate_http_binds_decision_to_current_question():
    gate = RequestGate(0, "native goal")
    base, token = gate.url.split("/#")

    def request(path, *, data=None, **headers):
        req = urllib.request.Request(
            base + path, json.dumps(data).encode() if data is not None else None, headers
        )
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.read()

    try:
        auth = {"Authorization": "Bearer " + token}
        assert "任务交接" in request("/").decode()
        with pytest.raises(urllib.error.HTTPError, match="403"):
            request("/state")
        with pytest.raises(urllib.error.HTTPError, match="403"):
            request("/state", Origin="https://unrelated.invalid", **auth)
        with ThreadPoolExecutor(1) as pool:
            started = threading.Event()

            def ask():
                with gate.condition:
                    started.set()
                    return gate.ask("Which object?")

            answer = pool.submit(ask)
            assert started.wait(2)
            version = json.loads(request("/state", **auth))["version"]
            decision = dict(action="answer", version=version, answer="The blue can")
            request("/decision", data=decision, **auth)
            assert answer.result(2) == "The blue can"
            with pytest.raises(urllib.error.HTTPError, match="409"):
                request("/decision", data=decision, **auth)
    finally:
        gate.close()


def test_user_correction_survives_retry(tmp_path):
    class Review:
        def scene(self, obs, goal):
            assert obs.external_rgb is not None

        def ask(self, question):
            return "The two cans"

        def review(self, instruction, demo_id, library, dialogue):
            assert dialogue[-1]["answer"] in instruction
            return "Move the two blue cans to the basket", False

        def running(self, attempt, instruction, demo_id):
            assert instruction == "Move the two blue cans to the basket"
            assert demo_id is None

    result = demo(tmp_path / "run", gate=Review())
    assert result["success"] and len(result["attempts"]) == 2
    assert result["upper_calls"] == 2  # question + handoff; confirmed edit carries over
    events = list(map(json.loads, (tmp_path / "run/events.jsonl").read_text().splitlines()))
    assert sum(e["event"] == "handoff_reviewed" for e in events) == 1
    assert all(
        e["dialogue"][-1]["answer"] == e["instruction"]
        for e in events
        if e["event"] == "attempt_started"
    )


@pytest.mark.parametrize("stage", ["question", "review"])
def test_cancel_before_lower_call(tmp_path, stage):
    class Cancel:
        def scene(self, obs, goal):
            pass

        def ask(self, question):
            if stage == "question":
                raise GateCancelled()
            return "The soup and butter"

        def review(self, *args):
            raise GateCancelled()

    result = demo(tmp_path / "run", gate=Cancel())
    assert result["reason"] == "user_cancelled"
    assert (result["attempts"], result["calls"], result["actions"]) == ([], 0, 0)


def test_gate_state_includes_telemetry_dialogue_and_finish_metrics(tmp_path):
    from examples.hierarchy_demo import TwoStageRobot

    gate = RequestGate(0, "Put the bowl in the drawer")
    base, token = gate.url.split("/#")
    auth = {"Authorization": "Bearer " + token}

    def get_state():
        req = urllib.request.Request(base + "/state", headers=auth)
        with urllib.request.urlopen(req, timeout=3) as response:
            return json.loads(response.read())

    def post_decision(payload):
        req = urllib.request.Request(
            base + "/decision",
            json.dumps(payload).encode(),
            {**auth, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status

    try:
        obs = TwoStageRobot().observe()
        obs.state["gripper_open_width_mm"] = 42.5
        gate.scene(obs, "Put the two items away")
        snap = get_state()
        assert snap["observation_id"] == obs.robot.observation_id
        assert snap["reference_frame"] == obs.robot.reference_frame
        assert snap["gripper_open_width_mm"] == 42.5

        with ThreadPoolExecutor(1) as pool:
            q_future = pool.submit(gate.ask, "Which two items?")
            for _ in range(40):
                snap = get_state()
                if snap["phase"] == "question":
                    break
            assert snap["phase"] == "question"
            payload = dict(action="answer", version=snap["version"], answer="Soup & butter")
            assert post_decision(payload) == 200
            assert q_future.result(2) == "Soup & butter"
            snap = get_state()
            assert snap["dialogue"] == [
                dict(question="Which two items?", answer="Soup & butter")
            ]

            r_future = pool.submit(
                gate.review,
                "Put the soup and butter into the basket.",
                None,
                None,
                snap["dialogue"],
            )
            for _ in range(40):
                snap = get_state()
                if snap["phase"] == "review":
                    break
            assert snap["phase"] == "review"
            assert snap["proposed_instruction"] == "Put the soup and butter into the basket."
            assert (
                post_decision(
                    dict(
                        action="start",
                        version=snap["version"],
                        instruction="Put the soup and butter into the basket.",
                        keep_demo=False,
                    )
                )
                == 200
            )
            assert r_future.result(2) == ("Put the soup and butter into the basket.", False)

        gate.running(1, "Put the soup and butter into the basket.", None)
        snap = get_state()
        assert snap["phase"] == "running" and isinstance(snap["running_since"], float)

        receipt = tmp_path / "RESULT.json"
        with ThreadPoolExecutor(1) as pool:
            done = pool.submit(
                gate.finish,
                dict(
                    success=True,
                    reason="native_success",
                    calls=5,
                    actions=2,
                    upper_calls=2,
                    attempts=[{"success": True}],
                    first_attempt_success=True,
                    elapsed_s=12.3,
                ),
                receipt,
            )
            for _ in range(40):
                snap = get_state()
                if snap["phase"] == "finished":
                    break
            done.result(3)
        assert snap["phase"] == "finished"
        assert (snap["attempts"], snap["actions"], snap["calls"], snap["upper_calls"]) == (
            1,
            2,
            5,
            2,
        )
        assert snap["receipt"] == str(receipt)
    finally:
        gate.close()
