"""Submit one GPU inference node for CPU-based LIBERO trials."""

import argparse
import base64
import io
import zipfile
from pathlib import Path

from boyue_episode import create_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-root", required=True, help="fresh GPFS directory for TP2 clients")
    parser.add_argument("--batch", required=True, help="CPU job's GPFS batch result directory")
    parser.add_argument("--replicas", type=int, choices=[2, 3, 4], default=4)
    parser.add_argument("--tp2-node", help="node hosting an existing TP2 service")
    parser.add_argument("--existing", help="client.json of the existing service")
    parser.add_argument("--cli", default="yicloud")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if bool(args.existing) != bool(args.tp2_node):
        parser.error("--existing and --tp2-node must be specified together")
    folder = Path(__file__).resolve().parent
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        for name in ["libero40.py", "serve.py", "serve_pool.py"]:
            out.write(folder / name, name)
    command = [
        "serve_pool.py",
        "--root",
        args.client_root,
        "--batch",
        args.batch,
        "--total-replicas",
        str(args.replicas),
    ]
    if args.existing:
        command += ["--existing", args.existing]
    bootstrap = f"""import base64,io,subprocess,sys,tempfile,zipfile
payload = {base64.b64encode(archive.getvalue())!r}
with tempfile.TemporaryDirectory() as tmp:
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as bundle:
        bundle.extractall(tmp)
    subprocess.run([sys.executable,'-u',*{command!r}],cwd=tmp,check=True)
"""
    create_job(
        args.cli,
        "eta0-libero40-executor-pool",
        bootstrap,
        runner=True,
        serving_only=True,
        gpus=2 * (args.replicas - bool(args.existing)),
        tp2_node=args.tp2_node,
        submit=args.submit,
    )


if __name__ == "__main__":
    main()
