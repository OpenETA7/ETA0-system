import json
import shutil
import threading
import time
from pathlib import Path

import pytest

import preparation_relay
from eta0_run import upper_endpoint


def test_relay_delivers_validated_response_without_publishing_key(tmp_path, monkeypatch):
    mailbox = tmp_path / "preparation-mailbox"
    mailbox.mkdir()
    src = tmp_path / "source"
    src.mkdir()
    shutil.copy(preparation_relay.__file__, src)
    shell = tmp_path / "remote"
    shell.write_text('#!/bin/sh\nexec sh -c "$1"\n')
    shell.chmod(0o700)
    profile = dict(model="test-kimi", allowed_returned_models=["test-kimi"])
    config = tmp_path / "public.json"
    config.write_text(json.dumps(dict(profile, preparation_mailbox=str(mailbox))))
    calls = []

    def forward(private, item):
        assert "api_key" not in item["payload"]
        calls.append(item["payload"]["model"])
        return {"model": "test-kimi", "choices": [{"finish_reason": "stop"}]}

    monkeypatch.setattr(preparation_relay, "forward", forward)
    thread = threading.Thread(
        target=preparation_relay.serve,
        args=(
            str(shell),
            tmp_path,
            dict(profile, api_key="private-test-value"),
            tmp_path / "cache",
        ),
    )
    thread.start()
    try:
        ep = upper_endpoint(config, time.monotonic() + 8)
        assert ep.chat([], max_tokens=1500)["model"] == "test-kimi"
        assert ep.chat_calls == 1 and calls == ["test-kimi"]
    finally:
        (mailbox / "closed").touch()
        thread.join(8)
    assert not thread.is_alive() and list(mailbox.iterdir()) == [mailbox / "closed"]


def test_relay_timeout_removes_unclaimed_request(tmp_path):
    end = time.monotonic() + 0.05
    with pytest.raises(TimeoutError):
        preparation_relay.request_preparation(Path(tmp_path), {}, lambda: end - time.monotonic())
    assert not list(tmp_path.iterdir())
