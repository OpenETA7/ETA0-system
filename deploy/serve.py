"""YiCloud single-pod TP2 service. Uses the runtime pinned by the LIBERO evaluation."""

import argparse
import base64
import json
import os
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

TARGET = "/mnt/share/yfzhang/training-mixture/exports/fullmix-qwen38-27b-step2500-20261117/hf"
DRAFT = "/root/tests/inference/dflash2-qwen38-27b-fullmix-step2500-20261117/finetuned-dflash2"
PLAN = Path("/root/tests/inference/libero10-text-compact-20261119/plan.json")
MODEL = "eta0-system-fullmix-2500"
PORT = 18680


def serve(session=None, port=PORT):
    os.umask(0o077)
    session = session or Path("/root/tests/inference/eta0-system") / socket.gethostname()
    session.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(32)
    draft = os.environ.get("ETA0_DFLASH_DRAFT", DRAFT)
    for path in (TARGET, draft):
        if not (Path(path) / "config.json").is_file():
            raise FileNotFoundError(path)
    runtime = Path(json.loads(PLAN.read_text())["runtime_mirror"])
    if not (runtime / "READY.json").is_file():
        raise ValueError("incomplete runtime mirror")
    target = Path("/opt/eta0-dflash2")
    try:
        target.symlink_to(runtime, target_is_directory=True)
    except FileExistsError:
        if not target.is_dir():
            raise
    python = str(target / "serve/bin/python")
    env = {
        k: v
        for k, v in os.environ.items()
        if "proxy" not in k.lower()
        and k
        not in {
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "NODE_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
        }
    }
    env.update(
        VLLM_API_KEY=key,
        LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib",
        PATH=str(target / "serve/bin") + ":" + env["PATH"],
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        OMP_NUM_THREADS="8",
        TOKENIZERS_PARALLELISM="false",
        NCCL_SOCKET_IFNAME="=eth0",
        VLLM_BATCH_INVARIANT="0",
        VLLM_CACHE_ROOT=f"/tmp/eta0-system-vllm-{port}",
        VLLM_PORT=str(port + 100),
        MASTER_PORT=str(port + 200),
    )
    subprocess.run(
        [
            python,
            "-c",
            "import torch,vllm; print(torch.__version__,vllm.__version__); "
            "assert torch.cuda.device_count()==2 and vllm.__version__=='0.33.0'",
        ],
        env=env,
        check=True,
    )
    args = [
        python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        TARGET,
        "--served-model-name",
        MODEL,
        "--tensor-parallel-size",
        "2",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "262144",
        "--max-num-seqs",
        "2",
        "--max-num-batched-tokens",
        "8192",
        "--block-size",
        "128",
        "--gpu-memory-utilization",
        "0.8",
        "--no-enable-prefix-caching",
        "--limit-mm-per-prompt",
        '{"image":96,"video":0}',
        "--mm-processor-kwargs",
        '{"max_pixels":262144}',
        "--reasoning-parser",
        "qwen3",
        "--generation-config",
        "vllm",
        "--chat-template-content-format",
        "string",
        "--mm-processor-cache-gb",
        "0",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--speculative-config",
        json.dumps(dict(method="dflash", model=draft, num_speculative_tokens=7)),
    ]
    with (session / "server.log").open("w") as log:
        proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        url = f"http://127.0.0.1:{port}/v1"
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 1800
        while proc.poll() is None and time.monotonic() < deadline:
            try:
                http.open(url + "/models", timeout=4).close()
            except urllib.error.HTTPError as err:
                if err.code == 401:
                    break
                raise
            except OSError:
                time.sleep(4)
                continue
            raise RuntimeError("server accepted an unauthenticated model request")
        else:
            tail = (session / "server.log").read_text(errors="replace")[-4000:].replace(key, "***")
            raise RuntimeError("model readiness failed: " + tail)
        headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
        with http.open(urllib.request.Request(url + "/models", headers=headers)) as reply:
            assert MODEL in {model["id"] for model in json.load(reply)["data"]}
        # Encode the test PNG with the serving runtime (Pillow is a vLLM dependency).
        png = subprocess.check_output(
            [
                python,
                "-c",
                "from PIL import Image; import sys; "
                "Image.new('RGB',(512,512),'red').save(sys.stdout.buffer,'PNG')",
            ],
            env=env,
        )
        image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()},
        }
        payload = dict(
            model=MODEL,
            temperature=0,
            max_tokens=32,
            chat_template_kwargs={"enable_thinking": False},
            messages=[
                {
                    "role": "user",
                    "content": [
                        image,
                        {
                            "type": "text",
                            "text": "What color fills the image? Reply with one English word.",
                        },
                    ],
                }
            ],
        )
        with http.open(
            urllib.request.Request(
                url + "/chat/completions", json.dumps(payload).encode(), headers
            ),
            timeout=240,
        ) as reply:
            choice = json.load(reply)["choices"][0]
        answer = choice["message"]["content"].strip()
        if choice["finish_reason"] != "stop" or "red" not in answer.lower().split(".")[0].split():
            raise RuntimeError("visual canary failed: " + answer[:80])
        info = dict(
            target=TARGET,
            draft=draft,
            base_url=f"http://{socket.gethostbyname(socket.gethostname())}:{port}/v1",
            model=MODEL,
            canary=answer,
            client_file=str(session / "client.json"),
        )
        client = session / "client.json"
        temp = client.with_suffix(".tmp")
        temp.write_text(json.dumps(dict(**info, api_key=key)))
        temp.replace(client)
        print("ETA0_READY " + json.dumps(info), flush=True)
        raise SystemExit(proc.wait())
    finally:
        (session / "client.json").unlink(missing_ok=True)
        if proc.poll() is None:
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    serve(args.session, args.port)
