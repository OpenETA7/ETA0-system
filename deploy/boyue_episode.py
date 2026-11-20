"""Submit a private Boyue + ETA0 teacher episode from the Mac to YiCloud."""

import argparse
import base64
import io
import json
import secrets
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

PROJECT = "fdj-infra"
CHECKOUT = Path(__file__).resolve().parents[1]
ROOT = "/root/tests/inference/eta0-system"
DEFAULT_CLIENT = ROOT + "/rjob-8d131ad6dcf558ed-88c9a4c797eb2ec1-0/client.json"


def create_job(cli, name, program, *, runner, submit):
    command = [
        cli,
        "--json",
        "job",
        "create",
        "--name",
        name,
        "--project",
        PROJECT,
        "--quota-group",
        "fdj_cpu_task",
        "--priority",
        "3",
        "--gpu",
        "0",
        "--cpu",
        "8" if runner else "1",
        "--memory",
        "32Gi" if runner else "4Gi",
        "--image",
        "registry.gate.yicloud.com.cn/fdj-infra/eta0-train:v01",
        "--mount",
        "gpfs://gpfs5/yfzhang:/root",
    ]
    if runner:
        command.extend(
            [
                "--internet",
                "--mount",
                "gpfs://gpfs2/avalanche-share2:/mnt/share",
                "--mount",
                "gpfs://gpfs2/avalanche-shared-model2:/mnt/model",
            ]
        )
    for arg in ("python3", "-u", "-c", program):
        command.extend(["--command-argv", arg])
    preview = json.loads(subprocess.check_output([*command, "--dry-run"]))
    print("Preview", preview["Name"], preview["QuotaGroup"], flush=True)
    if submit:
        result = json.loads(subprocess.check_output([*command, "--submit"]))
        print(name, result["JobId"], flush=True)
        return result["JobId"]
    return None


def await_recipient(cli, job):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        logs = subprocess.run(
            [
                cli,
                "job",
                "logs",
                job,
                "--project",
                PROJECT,
                "--keyword",
                "RECIPIENT_PUBLIC_KEY",
                "--limit",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=45,
        )
        for line in logs.stdout.splitlines():
            tag = "RECIPIENT_PUBLIC_KEY "
            if tag in line:
                return base64.b64decode(line.split(tag, 1)[1].strip(), validate=True)
        time.sleep(4)
    raise TimeoutError(f"Job {job} did not publish its recipient key within ten minutes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path.home() / ".config/robodojo/boyue-api.json"
    )
    parser.add_argument("--cli", default="yicloud")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--initial-index", type=int, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--executor-client", default=DEFAULT_CLIENT)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    profile = json.loads(args.config.read_text())["boyue"]
    # The job spec carries only source, routing and model identifiers.
    public = {k: profile[k] for k in ("base_url", "model", "allowed_returned_models")}
    if not public["base_url"].startswith("https://"):
        raise ValueError("Boyue requires an HTTPS endpoint")
    work = ROOT + "/boyue-teacher-" + secrets.token_hex(8)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in [*sorted(CHECKOUT.glob("*.py")), CHECKOUT / "deploy/run_libero.py"]:
            bundle.write(path, path.relative_to(CHECKOUT))
    episode_args = [
        "--mode",
        "teacher",
        "--recover-length",
        "--suite",
        args.suite,
        "--task",
        str(args.task),
        "--initial-index",
        str(args.initial_index),
        "--executor-client",
        args.executor_client,
        "--output",
        work + "/episode",
    ]
    values = dict(
        work=work,
        profile=public,
        episode_args=episode_args,
        source=base64.b64encode(buf.getvalue()).decode(),
    )
    program = (
        "options = " + repr(values) + "\n" + Path(__file__).with_name("boyue_worker.py").read_text()
    )
    run = create_job(args.cli, "eta0-boyue-teacher", program, runner=True, submit=args.submit)
    if run is None:
        return
    print("Episode directory", work, flush=True)
    pub = await_recipient(args.cli, run)
    with tempfile.NamedTemporaryFile() as f:
        f.write(pub)
        f.flush()
        envelope = subprocess.check_output(
            [
                "openssl",
                "pkeyutl",
                "-encrypt",
                "-pubin",
                "-inkey",
                f.name,
                "-pkeyopt",
                "rsa_padding_mode:oaep",
                "-pkeyopt",
                "rsa_oaep_md:sha256",
            ],
            input=profile["api_key"].encode(),
        )
    delivery = (
        "import os,base64,pathlib\nos.umask(0o077)\n"
        f"path=pathlib.Path({work!r})/'envelope'\n"
        f"temp=path.with_suffix('.part')\ntemp.write_bytes(base64.b64decode({base64.b64encode(envelope)!r}))\n"
        "temp.replace(path)"
    )
    create_job(args.cli, "eta0-boyue-envelope", delivery, runner=False, submit=True)


if __name__ == "__main__":
    main()
