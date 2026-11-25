"""Submit a private Boyue + ETA0 teacher episode from the Mac to YiCloud."""

import argparse
import base64
import io
import json
import secrets
import shlex
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

if __package__:
    from .libero40 import harness_options
else:
    from libero40 import harness_options

PROJECT = "fdj-infra"
CHECKOUT = Path(__file__).resolve().parents[1]
ROOT = "/root/tests/inference/eta0-system"
DEFAULT_CLIENT = ROOT + "/cpu-remote-cc57436d67/replica-1/client.json"


def create_job(
    cli,
    name,
    program,
    *,
    runner,
    submit,
    gpus=0,
    tp2_node=None,
    cpu_workers=0,
    serving_only=False,
    priority=3,
):
    cpu, memory = (
        (max(8, 2 * cpu_workers), f"{max(32, 4 * cpu_workers)}Gi") if runner else (1, "512Mi")
    )
    if gpus:
        cpu = gpus * (4 if serving_only else 10)
        memory = f"{gpus * (40 if serving_only else 60)}Gi"
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
        "fdj_gpu_taskpool" if gpus or tp2_node else "fdj_cpu_task",
        "--priority",
        str(priority),
        "--gpu",
        str(gpus),
        "--cpu",
        str(cpu),
        "--memory",
        memory,
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
        try:
            result = json.loads(subprocess.check_output([*command, "--submit"]))
        except subprocess.CalledProcessError:
            # Create can commit before its HTTP response times out. A recovery worker
            # has a random, unique name, so look it up before issuing another create.
            if not name.startswith("eta0-request-comparison-"):
                raise
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                reply = subprocess.run(
                    [cli, "--json", "job", "list", "--project", PROJECT, "--name", name],
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
                if reply.returncode == 0:
                    matches = [j for j in json.loads(reply.stdout)["Items"] if j["Name"] == name]
                    if len(matches) == 1:
                        result = matches[0]
                        break
                time.sleep(10)
            else:
                raise
        print(name, result["JobId"], flush=True)
        return result["JobId"]
    return None


def await_recipient(cli, job, shared=None):
    if shared:
        ssh, directory = shared
        read_key = [ssh, "cat " + shlex.quote(directory + "/recipient-public.pem")]
    else:
        read_key = [
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
        ]
    next_phase_check = 0
    while True:
        reply = subprocess.run(read_key, capture_output=True, text=True, timeout=45)
        if reply.returncode == 0:
            if shared and reply.stdout.startswith("-----BEGIN PUBLIC KEY-----"):
                return reply.stdout.encode()
            for line in reply.stdout.splitlines():
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
        time.sleep(15 if shared else 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path.home() / ".config/robodojo/boyue-api.json"
    )
    parser.add_argument("--config-profile", help="named upper API profile for --request-comparison")
    parser.add_argument("--cli", default="yicloud")
    parser.add_argument("--shared-ssh", help="SSH wrapper for encrypted delivery via the job GPFS")
    parser.add_argument(
        "--mac-preparation",
        action="store_true",
        help="dispatch this comparison's upper calls on the submitting Mac",
    )
    jobs = parser.add_mutually_exclusive_group(required=True)
    jobs.add_argument("--task", type=int)
    jobs.add_argument("--check-model", metavar="MODEL", help="test a visual teacher from a CPU job")
    jobs.add_argument("--request-comparison", action="store_true", help="40 held-out request pairs")
    jobs.add_argument("--libero40", action="store_true", help="40 paired base/teacher tasks")
    parser.add_argument("--initial-index", type=int, default=0)
    parser.add_argument("--compare-models", nargs=3, metavar=("MODEL1", "MODEL2", "MODEL3"))
    parser.add_argument("--initial-count", type=int, default=1)
    parser.add_argument(
        "--repair-from", help="retry transport-interrupted trials from a finished comparison"
    )
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--executor-client", default=DEFAULT_CLIENT)
    parser.add_argument(
        "--remote-client",
        action="append",
        default=[],
        help="run all simulations on CPU using one client.json per remote TP2",
    )
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
    parser.add_argument(
        "--resume-from", help="continue a closed comparison in its original result tree"
    )
    parser.add_argument("--resume-job", help="closed job ID owning the result tree")
    harness_options(parser)
    args = parser.parse_args()
    if args.mac_preparation and not (
        args.request_comparison and args.shared_ssh and args.config_profile
    ):
        parser.error("--mac-preparation requires a named comparison profile and --shared-ssh")
    if args.upstream_base and not args.libero40:
        parser.error("--upstream-base selects the baseline for --libero40")
    if args.remote_client and (
        not (args.libero40 or args.request_comparison)
        or args.tp2_node
        or len(args.remote_client) > 4
        or len(set(args.remote_client)) != len(args.remote_client)
    ):
        parser.error("use 1-4 distinct --remote-client paths with --libero40 and no --tp2-node")
    if args.request_comparison and (
        not args.remote_client or args.compare_models or args.repair_from or args.initial_index != 0
    ):
        parser.error(
            "--request-comparison uses existing --remote-client endpoints and frozen cases"
        )
    if args.remote_client:
        args.replicas = len(args.remote_client)
    if args.compare_models and (
        not args.remote_client or len(set(args.compare_models)) != 3 or args.initial_index < 0
    ):
        parser.error(
            "--compare-models requires distinct IDs, CPU simulation and a nonnegative initial index"
        )
    if args.initial_count < 1 or (args.initial_count != 1 and not args.compare_models):
        parser.error(
            "multiple initial states require --compare-models and a positive --initial-count"
        )
    if bool(args.resume_from) != bool(args.resume_job) or (
        args.resume_from and not (args.compare_models or args.request_comparison)
    ):
        parser.error("resuming requires an original closed job, result tree and comparison mode")
    if args.resume_job and args.submit:
        job = json.loads(
            subprocess.check_output(
                [args.cli, "--json", "job", "get", args.resume_job, "--project", PROJECT]
            )
        )
        if job["Phase"] not in {"Failed", "Stopped", "Closed", "Succeeded"}:
            parser.error("original CPU job must have terminated before resuming its output tree")
    if args.repair_from and not args.compare_models:
        parser.error("--repair-from requires --compare-models")
    if args.tp2_node and not args.libero40:
        parser.error("--tp2-node requires --libero40")
    config = json.loads(args.config.read_text())
    if args.config_profile:
        if not args.request_comparison or not isinstance(config.get(args.config_profile), dict):
            parser.error(
                "--config-profile requires an existing named profile and --request-comparison"
            )
        profile = config[args.config_profile]
    else:
        profile = config.get("boyue", config)
    # The job spec carries only source, routing and model identifiers.
    if args.request_comparison:
        if __package__:
            from .request_comparison import MODEL
        else:
            from request_comparison import MODEL
        model = profile["model"] if args.config_profile else MODEL
    else:
        model = (
            args.compare_models[0] if args.compare_models else args.check_model or profile["model"]
        )
    public = dict(base_url=profile["base_url"], model=model)
    public["allowed_returned_models"] = profile.get("allowed_returned_models", [model])
    if args.check_model:
        public.update(model=args.check_model, allowed_returned_models=[args.check_model])
    if args.request_comparison:
        if args.config_profile and model not in public["allowed_returned_models"]:
            parser.error("requested comparison model is missing from the profile allowlist")
        public["allowed_returned_models"] = [model]
    if not args.mac_preparation and not public["base_url"].startswith("https://"):
        raise ValueError("Boyue requires an HTTPS endpoint")
    name = "libero40-paired" if args.libero40 else "boyue-teacher"
    if args.request_comparison:
        name = "request-comparison"
    work = ROOT + "/" + name + "-" + secrets.token_hex(8)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in [*sorted(CHECKOUT.glob("*.py")), CHECKOUT / "deploy/run_libero.py"]:
            bundle.write(path, path.relative_to(CHECKOUT))
        if args.request_comparison:
            for tool in ("request_comparison", "build_demo_library"):
                bundle.write(CHECKOUT / f"deploy/{tool}.py", f"deploy/{tool}.py")
        if args.libero40 or args.check_model:
            bundle.write(CHECKOUT / "deploy/check_upper.py", "deploy/check_upper.py")
        if args.libero40:
            bundle.write(CHECKOUT / "deploy/libero40.py", "deploy/libero40.py")
            if args.compare_models:
                bundle.write(CHECKOUT / "deploy/compare_upper.py", "deploy/compare_upper.py")
                bundle.write(CHECKOUT / "deploy/audit_comparison.py", "deploy/audit_comparison.py")
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
        upper_wall_seconds=900 if args.upper_wall_seconds is None else args.upper_wall_seconds,
        profile=public,
        shared_recipient=bool(args.shared_ssh),
        mac_preparation=args.mac_preparation,
        episode_args=episode_args,
        source=base64.b64encode(buf.getvalue()).decode(),
    )
    if args.request_comparison:
        values.update(
            entrypoint="deploy/request_comparison.py",
            episode_args=[
                "--upper-model",
                model,
                *[arg for client in args.remote_client for arg in ("--remote-client", client)],
                "--output",
                work + "/batch",
            ],
        )
    if args.check_model:
        values.update(check_model=True, episode_args=[str(args.initial_index), "--discover"])
    if args.libero40:
        values.update(
            entrypoint="deploy/libero40.py",
            episode_args=[
                *(["--executor-client", args.executor_client] if args.tp2_node else []),
                *[arg for client in args.remote_client for arg in ("--remote-client", client)],
                "--initial-index",
                str(args.initial_index),
                "--replicas",
                str(args.replicas),
                "--output",
                work + "/batch",
            ],
        )
    if args.compare_models:
        values.update(entrypoint="deploy/compare_upper.py")
        values["episode_args"].extend(
            ["--models", *args.compare_models, "--initial-count", str(args.initial_count)]
        )
    if args.resume_from:
        values["episode_args"].extend(["--resume-from", args.resume_from])
    if args.repair_from:
        values["episode_args"].extend(["--repair-from", args.repair_from])
    if not (args.check_model or args.request_comparison):
        for option in ("planner", "supervisor", "teacher", "upstream_base", "completion_query"):
            value = getattr(args, option)
            if value is not None and (args.libero40 or option != "upstream_base"):
                values["episode_args"].append(
                    ("--" if value else "--no-") + option.replace("_", "-")
                )
        for option in ("libero_source_root", "upper_wall_seconds"):
            value = getattr(args, option)
            if value is not None:
                values["episode_args"].extend(["--" + option.replace("_", "-"), str(value)])
    program = (
        "options = " + repr(values) + "\n" + Path(__file__).with_name("boyue_worker.py").read_text()
    )
    resources = (
        {"tp2_node": args.tp2_node, "gpus": 2 * (args.replicas - bool(args.tp2_node))}
        if args.libero40
        else {}
    )
    if args.remote_client:
        resources = {
            "cpu_workers": (2 if args.repair_from or args.request_comparison else 4) * args.replicas
        }
    if args.request_comparison:
        resources["priority"] = 5
    job_name = "eta0-" + name
    if (args.resume_from or args.mac_preparation) and args.request_comparison:
        job_name += "-" + work.rsplit("-", 1)[-1][:8]
        print("Recovery directory", work, flush=True)
    run = create_job(args.cli, job_name, program, runner=True, submit=args.submit, **resources)
    if run is None:
        return
    print("Episode directory", work, flush=True)
    if args.mac_preparation:
        sys.path.insert(0, str(CHECKOUT))
        from preparation_relay import serve

        cache = CHECKOUT / "outputs" / work.rsplit("/", 1)[-1] / "relay"
        serve(args.shared_ssh, Path(work), profile, cache)
        return
    if args.shared_ssh:
        pub = await_recipient(args.cli, run, shared=(args.shared_ssh, work))
    else:
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
    if args.shared_ssh:
        command = (
            "umask 077 && cd "
            + shlex.quote(work)
            + " && cat > envelope.ssh.part && mv envelope.ssh.part envelope"
        )
        subprocess.run([args.shared_ssh, command], input=envelope, check=True, timeout=45)
        print("Encrypted credential delivered to the worker", flush=True)
        return
    delivery = (
        "import os,base64,pathlib\nos.umask(0o077)\n"
        f"path=pathlib.Path({work!r})/'envelope'\n"
        f"temp=path.with_suffix('.part')\ntemp.write_bytes(base64.b64decode({base64.b64encode(envelope)!r}))\n"
        "temp.replace(path)"
    )
    placement = {}
    if args.libero40 and not args.remote_client:
        replicas = json.loads(
            subprocess.check_output(
                [
                    args.cli,
                    "--json",
                    "job",
                    "replicas",
                    run,
                    "--project",
                    PROJECT,
                ]
            )
        )["Items"]
        if len(replicas) != 1 or not replicas[0].get("NodeName"):
            raise RuntimeError("could not identify the evaluation host for credential delivery")
        placement["tp2_node"] = replicas[0]["NodeName"]
    create_job(args.cli, "eta0-boyue-envelope", delivery, runner=False, submit=True, **placement)


if __name__ == "__main__":
    main()
