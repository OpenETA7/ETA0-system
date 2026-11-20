import json

import pytest

from eta0_run import upper_endpoint


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


def test_cloud_handoff_delivers_profile_without_secret_in_job_specs(tmp_path, monkeypatch):
    import base64
    import shutil
    import subprocess
    import sys

    from deploy import boyue_episode

    if not shutil.which("openssl"):
        pytest.skip("the YiCloud key handoff requires openssl")
    profile = tmp_path / "boyue.json"
    secret = "dummy-unit-test-key"
    profile.write_text(
        json.dumps(
            {
                "boyue": {
                    "api_key": secret,
                    "model": "flash",
                    "base_url": "https://example.invalid/v1",
                    "allowed_returned_models": ["flash"],
                }
            }
        )
    )
    checkout = tmp_path / "checkout"
    (checkout / "deploy").mkdir(parents=True)
    (checkout / "deploy/run_libero.py").write_text(
        "import sys,json,pathlib,hashlib\n"
        "p=pathlib.Path(sys.argv[sys.argv.index('--boyue-config')+1])\n"
        "assert p.stat().st_mode & 0o077 == 0\n"
        "print(hashlib.sha256(json.loads(p.read_text())['boyue']['api_key'].encode()).hexdigest())"
    )
    monkeypatch.setattr(boyue_episode, "CHECKOUT", checkout)
    monkeypatch.setattr(boyue_episode, "ROOT", str(tmp_path / "gpfs"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "boyue_episode",
            "--config",
            str(profile),
            "--task",
            "3",
            "--initial-index",
            "0",
            "--submit",
        ],
    )
    workers = []

    def create(cli, name, program, *, runner, submit):
        assert secret not in program
        if runner:
            workers.append(
                subprocess.Popen(
                    [sys.executable, "-u", "-c", program], stdout=subprocess.PIPE, text=True
                )
            )
        else:
            subprocess.run([sys.executable, "-c", program], check=True)
        return name

    def recipient(cli, job):
        tag, public_key = workers[0].stdout.readline().split()
        assert tag == "RECIPIENT_PUBLIC_KEY"
        return base64.b64decode(public_key)

    monkeypatch.setattr(boyue_episode, "create_job", create)
    monkeypatch.setattr(boyue_episode, "await_recipient", recipient)
    try:
        import hashlib

        boyue_episode.main()
        output, _ = workers[0].communicate(timeout=10)
        assert workers[0].returncode == 0
        assert output.strip() == hashlib.sha256(secret.encode()).hexdigest()
    finally:
        for worker in workers:
            worker.kill()
            worker.wait()
