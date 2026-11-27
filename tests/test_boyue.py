import io
import json
import time
import urllib.error
import urllib.request
from unittest.mock import Mock

import pytest

from eta0_run import ChatEndpoint, upper_endpoint


def http_failure(status, code="rate_limit", headers=None):
    body = json.dumps({"error": {"type": "api_error", "code": code}}).encode()
    return urllib.error.HTTPError(
        "https://example.invalid", status, "", headers or {}, io.BytesIO(body)
    )


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.delenv("ETA0_UPPER_EXPECTED_MODEL", raising=False)
    for suffix, value in [
        ("BASE_URL", "https://example.invalid"),
        ("MODEL", "flash"),
        ("API_KEY", "test"),
    ]:
        monkeypatch.setenv("ETA0_UPPER_" + suffix, value)
    endpoint = upper_endpoint(None, time.monotonic() + 1000)
    opener, sleep = Mock(), Mock()
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: opener)
    monkeypatch.setattr(time, "sleep", sleep)
    return endpoint, opener.open, sleep


def test_transient_failures_stop_after_four_retries(transport):
    endpoint, send, sleep = transport
    send.side_effect = urllib.error.URLError(TimeoutError("read timed out"))
    with pytest.raises(urllib.error.URLError):
        endpoint.chat([])
    assert send.call_count == 5
    assert all(c.kwargs["timeout"] == 120 for c in send.call_args_list)
    assert [int(c.args[0]) for c in sleep.call_args_list] == [1, 2, 4, 8]


@pytest.mark.parametrize(
    "status,code", [(401, "invalid_key"), (403, "forbidden"), (429, "insufficient_quota")]
)
def test_permanent_failure_stops_immediately(transport, status, code):
    endpoint, send, sleep = transport
    send.side_effect = http_failure(status, code)
    with pytest.raises(RuntimeError, match=code):
        endpoint.chat([])
    assert send.call_count == 1
    sleep.assert_not_called()


def test_retry_after_must_fit_in_remaining_budget(transport):
    endpoint, send, sleep = transport
    endpoint.deadline = time.monotonic() + 10
    send.side_effect = http_failure(429, headers={"Retry-After": "11"})
    with pytest.raises(RuntimeError, match="HTTP 429"):
        endpoint.chat([])
    assert send.call_count == 1 and 0 < send.call_args.kwargs["timeout"] <= 10
    sleep.assert_not_called()


def test_executor_timeout_is_not_replayed(transport):
    upper, send, sleep = transport
    executor = ChatEndpoint(upper.url, "27b", "", upper.deadline, direct=True)
    send.side_effect = TimeoutError()
    with pytest.raises(TimeoutError):
        executor.chat([])
    assert send.call_count == 1 and send.call_args.kwargs["timeout"] == 600
    sleep.assert_not_called()


def test_teacher_recovers_at_subgoal_transition_without_replaying_action(transport, tmp_path):
    from examples.hierarchy_demo import TwoStageRobot, lower, upper
    from harness import run
    from hierarchy import Hierarchy

    endpoint, send, sleep = transport
    robot = TwoStageRobot()
    retries = []

    class ReadTimeout(io.BytesIO):
        def read(self, *args):
            raise TimeoutError("read timed out")

    def respond(request, timeout):
        messages = json.loads(request.data)["messages"]
        if messages[0]["content"].startswith("Generate") and robot.actions == 1:
            retries.append(request.data)
            if len(retries) == 1:
                return ReadTimeout()
            if len(retries) == 2:
                raise http_failure(503, headers={"Retry-After": "7"})
        return io.BytesIO(json.dumps(upper(messages)).encode())

    send.side_effect = respond
    result = run(
        robot,
        lower,
        instruction="Place both items.",
        action_prompt="fake_step",
        output_dir=tmp_path / "episode",
        hierarchy=Hierarchy(endpoint.chat, review_every=1),
    )
    assert result["success"] and result["actions"] == robot.actions == 2
    assert result["upper_calls"] == 4 and send.call_count == 6
    assert len(retries) == 3 and len(set(retries)) == 1
    assert sleep.call_args_list[-1].args == (7,)


def test_boyue_uses_selected_profile_and_checks_returned_model(tmp_path, monkeypatch):
    profile = tmp_path / "boyue.json"
    profile.write_text(
        json.dumps(
            {
                "boyue": {
                    "base_url": "https://example.invalid/v1",
                    "model": "flash",
                    "api_key": "test-only",
                    "allowed_returned_models": ["flash-v1"],
                }
            }
        )
    )
    monkeypatch.setenv("ETA0_UPPER_MODEL", "unrelated")
    endpoint = upper_endpoint(profile, deadline=0)
    assert endpoint.model == "flash"
    monkeypatch.setattr(endpoint, "post", lambda *args: {"model": "flash-v1"})
    assert endpoint.chat([])["model"] == "flash-v1"
    monkeypatch.setattr(endpoint, "post", lambda *args: {"model": "other"})
    with pytest.raises(ValueError, match="allowlist"):
        endpoint.chat([])


def test_generic_upper_config_cannot_mix_in_other_credentials(monkeypatch):
    monkeypatch.setenv("ETA0_UPPER_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("ETA0_UPPER_MODEL", "model")
    monkeypatch.delenv("ETA0_UPPER_API_KEY", raising=False)
    monkeypatch.setenv("KIMI_CODE_API_KEY", "other-origin")
    with pytest.raises(KeyError, match="ETA0_UPPER_API_KEY"):
        upper_endpoint(None, deadline=0)


def test_official_deepseek_profile_preserves_credential_origin(tmp_path):
    profile = tmp_path / "upper.json"
    profile.write_text(
        json.dumps(
            dict(base_url="https://api.deepseek.com", model="deepseek-flash", api_key="test")
        )
    )
    client = upper_endpoint(profile, deadline=0)
    assert client.url == "https://api.deepseek.com"
    assert client.model == "deepseek-flash"
    assert client.allowed_models == ["deepseek-flash"]
