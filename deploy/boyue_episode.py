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


def create_job(cli, name, program, *, runner, submit, gpus=0, tp2_node=None):
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
        "fdj_gpu_taskpool" if gpus else "fdj_cpu_task",
        "--priority",
        "3",
        "--gpu",
        str(gpus),
        "--cpu",
        str(10 * gpus) if gpus else ("8" if runner else "1"),
        "--memory",
        f"{60 * gpus}Gi" if gpus else ("32Gi" if runner else "512Mi"),
        "--image",
        "registry.gate.yicloud.com.cn/fdj-infra/eta0-train:v01",
        "--mount",
        "gpfs://gpfs5/yfzhang:/root",
    ]
    if gpus:
        command.extend(["--positive-tag", "H200"])
    if tp2_node:
        command.extend(["--positive-tag", "node/" + tp2_node])
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
    next_phase_check = 0
    while True:
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
        if time.monotonic() >= next_phase_check:
            status = subprocess.run(
                [cli, "--json", "job", "get", job, "--project", PROJECT],
                capture_output=True,
                text=True,
                timeout=45,
            )
            if status.returncode == 0:
                phase = json.loads(status.stdout)["Phase"]
                if phase in {"Failed", "Stopped", "Closed", "Succeeded"}:
                    raise RuntimeError(f"Job {job} reached {phase} before key handoff")
            next_phase_check = time.monotonic() + 60
        time.sleep(4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path.home() / ".config/robodojo/boyue-api.json"
    )
    parser.add_argument("--cli", default="yicloud")
    jobs = parser.add_mutually_exclusive_group(required=True)
    jobs.add_argument("--task", type=int)
    jobs.add_argument("--libero40", action="store_true", help="40 paired base/teacher tasks")
    parser.add_argument("--initial-index", type=int, default=0)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--executor-client", default=DEFAULT_CLIENT)
    parser.add_argument(
        "--tp2-node",
        help="colocate with an existing TP2, otherwise allocate one entire eight-GPU node",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        choices=[2, 3, 4],
        default=4,
        help="total TP2 copies on the selected node",
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.tp2_node and not args.libero40:
        parser.error("--tp2-node requires --libero40")
    profile = json.loads(args.config.read_text())["boyue"]
    # The job spec carries only source, routing and model identifiers.
    public = {k: profile[k] for k in ("base_url", "model", "allowed_returned_models")}
    if not public["base_url"].startswith("https://"):
        raise ValueError("Boyue requires an HTTPS endpoint")
    name = "libero40-paired" if args.libero40 else "boyue-teacher"
    work = ROOT + "/" + name + "-" + secrets.token_hex(8)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in [*sorted(CHECKOUT.glob("*.py")), CHECKOUT / "deploy/run_libero.py"]:
            bundle.write(path, path.relative_to(CHECKOUT))
        if args.libero40:
            bundle.write(CHECKOUT / "deploy/libero40.py", "deploy/libero40.py")
            bundle.write(CHECKOUT / "deploy/serve.py", "deploy/serve.py")
            baseline = subprocess.check_output(
                ["git", "show", "cd6fcd91d19472ce24ea4dbd226930bafbf907c5:harness.py"], cwd=CHECKOUT
            )
            bundle.writestr("upstream_harness.py", baseline)
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
    if args.libero40:
        values.update(
            entrypoint="deploy/libero40.py",
            episode_args=[
                *(["--executor-client", args.executor_client] if args.tp2_node else []),
                "--initial-index",
                str(args.initial_index),
                "--replicas",
                str(args.replicas),
                "--output",
                work + "/batch",
            ],
        )
    program = (
        "options = " + repr(values) + "\n" + Path(__file__).with_name("boyue_worker.py").read_text()
    )
    resources = (
        {"tp2_node": args.tp2_node, "gpus": 2 * (args.replicas - bool(args.tp2_node))}
        if args.libero40
        else {}
    )
    run = create_job(
        args.cli, "eta0-" + name, program, runner=True, submit=args.submit, **resources
    )
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
