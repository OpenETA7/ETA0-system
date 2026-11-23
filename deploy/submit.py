"""Preview or submit the independent two-GPU YiCloud service via an installed yicloud CLI."""

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--name", default="eta0-system-fullmix-tp2")
    parser.add_argument("--node", help="require this physical host for the persistent TP2")
    parser.add_argument("--cli", default="yicloud", help="CLI executable or wrapper")
    args = parser.parse_args()
    source = Path(__file__).with_name("serve.py").read_bytes()
    encoded = base64.b64encode(source).decode()
    start = f"import base64;exec(compile(base64.b64decode('{encoded}'),'serve.py','exec'))"
    command = [
        args.cli,
        "--json",
        "job",
        "create",
        "--name",
        args.name,
        "--project",
        "fdj-infra",
        "--zone-id",
        "cn-shanghai-a",
        "--quota-group",
        "fdj_gpu_taskpool",
        "--priority",
        "3",
        "--replicas",
        "1",
        "--cpu",
        "16",
        "--memory",
        "160Gi",
        "--gpu",
        "2",
        "--positive-tag",
        "H200",
        "--image",
        "registry.gate.yicloud.com.cn/fdj-infra/eta0-train:v01",
        "--mount",
        "gpfs://gpfs5/yfzhang:/root",
        "--mount",
        "gpfs://gpfs2/avalanche-share2:/mnt/share",
        "--mount",
        "gpfs://gpfs2/avalanche-shared-model2:/mnt/model",
        "--command-argv",
        "python3",
        "--command-argv",
        "-u",
        "--command-argv",
        "-c",
        "--command-argv",
        start,
    ]
    if args.node:
        command.extend(["--positive-tag", "node/" + args.node])
    command.append("--submit" if args.submit else "--dry-run")
    data = json.loads(subprocess.check_output(command, text=True))
    for spec in data.get("TaskSpecs", {}).values():
        spec["Command"] = ["python3", "serve.py", "sha256=" + hashlib.sha256(source).hexdigest()]
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
