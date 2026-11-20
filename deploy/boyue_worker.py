"""YiCloud job entrypoint. The submitter supplies nonsecret options."""

import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path


def run(options):
    os.umask(0o077)
    work = Path(options["work"])
    work.mkdir(parents=True, exist_ok=False)
    src = work / "source"
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(options["source"]))) as bundle:
        bundle.extractall(src)
    with tempfile.TemporaryDirectory() as tmp:
        private = Path(tmp) / "recipient.pem"
        subprocess.run(
            [
                "openssl",
                "genpkey",
                "-algorithm",
                "RSA",
                "-pkeyopt",
                "rsa_keygen_bits:3072",
                "-out",
                str(private),
            ],
            check=True,
            stderr=subprocess.DEVNULL,
        )
        pub = subprocess.check_output(["openssl", "pkey", "-in", str(private), "-pubout"])
        print("RECIPIENT_PUBLIC_KEY", base64.b64encode(pub).decode(), flush=True)
        envelope = work / "envelope"
        deadline = time.monotonic() + 300
        while not envelope.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("no encrypted credential received")
            time.sleep(2)
        key = subprocess.check_output(
            [
                "openssl",
                "pkeyutl",
                "-decrypt",
                "-inkey",
                str(private),
                "-pkeyopt",
                "rsa_padding_mode:oaep",
                "-pkeyopt",
                "rsa_oaep_md:sha256",
                "-in",
                str(envelope),
            ]
        )
        private.unlink()
        config = Path(tmp) / "boyue.json"
        config.write_text(json.dumps({"boyue": dict(options["profile"], api_key=key.decode())}))
        subprocess.run(
            [
                sys.executable,
                str(src / "deploy/run_libero.py"),
                "--boyue-config",
                str(config),
                *options["episode_args"],
            ],
            check=True,
            timeout=2850,
        )


if __name__ == "__main__":
    run(globals()["options"])
