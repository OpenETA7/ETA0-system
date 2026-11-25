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
