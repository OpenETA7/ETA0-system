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
    if options.get("mac_preparation"):
        mailbox = work / "preparation-mailbox"
        mailbox.mkdir()
        profile = work / "upper-routing.json"
        profile.write_text(json.dumps(dict(options["profile"], preparation_mailbox=str(mailbox))))
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(src / options["entrypoint"]),
                    "--boyue-config",
                    str(profile),
                    *options["episode_args"],
                ],
                check=True,
            )
        finally:
            (mailbox / "closed").touch()
        return
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
        if options.get("shared_recipient"):
            tmp_pub = work / "recipient-public.part"
            tmp_pub.write_bytes(pub)
            tmp_pub.replace(work / "recipient-public.pem")
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
                str(src / options.get("entrypoint", "deploy/run_libero.py")),
                "--check-upper" if options.get("check_model") else "--boyue-config",
                str(config),
                *options["episode_args"],
            ],
            check=True,
            timeout=None
            if options.get("entrypoint")
            else 2850 + options.get("upper_wall_seconds", 0),
        )


if __name__ == "__main__":
    run(globals()["options"])
