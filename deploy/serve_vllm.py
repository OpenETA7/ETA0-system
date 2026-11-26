"""Portable vLLM launcher for ETA0-27B (standard vLLM or DFlash2 speculative decoding).

Detects local NVIDIA GPUs, chooses sensible tensor-parallel and context defaults,
runs an authenticated multimodal canary check, and writes a private client.json
ready for `deploy/first_run.py` or `eta0_run.py`.
"""

import argparse
import base64
import io
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Placeholders to be updated with the published HuggingFace model repositories before release.
DEFAULT_MODEL_REPO = "https://huggingface.co/OpenETA7/ETA0-27B-PLACEHOLDER"
DEFAULT_DRAFT_REPO = "https://huggingface.co/OpenETA7/ETA0-27B-DFlash2-PLACEHOLDER"
DEFAULT_SERVED_NAME = "eta0-27b"
DEFAULT_PORT = 18680


def detect_gpus():
    """Return a list of dicts: [{"index": int, "name": str, "vram_gb": float}, ...]."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpus.append(
                dict(
                    index=int(parts[0]),
                    name=parts[1],
                    vram_gb=round(float(parts[2]) / 1024.0, 1),
                )
            )
        except ValueError:
            continue
    return gpus


def choose_profile(
    gpus,
    *,
    tensor_parallel=None,
    max_model_len=None,
    max_num_seqs=None,
    gpu_memory_utilization=None,
    dflash=False,
):
    """Select hardware-appropriate vLLM defaults without assuming a fixed cluster."""
    count = len(gpus)
    min_vram = min((g["vram_gb"] for g in gpus), default=0.0)

    if tensor_parallel is not None:
        tp = tensor_parallel
    elif count <= 1:
        tp = 1
    elif min_vram >= 30.0:
        tp = 2
    elif count >= 4:
        tp = 4
    else:
        tp = 2

    if tp < 1:
        raise ValueError("tensor_parallel must be positive")
    if count and tp > count:
        raise ValueError(f"requested tensor_parallel={tp} exceeds detected GPU count ({count})")

    effective_vram = min_vram * tp if count else 0.0
    if max_model_len is not None:
        ctx_len = max_model_len
    elif effective_vram >= 120.0:
        ctx_len = 262144
    elif effective_vram >= 60.0:
        ctx_len = 131072
    else:
        ctx_len = 65536

    if max_num_seqs is not None:
        seqs = max_num_seqs
    elif effective_vram and effective_vram < 48.0:
        seqs = 1
    else:
        seqs = 2

    if gpu_memory_utilization is not None:
        mem_util = gpu_memory_utilization
    else:
        mem_util = 0.85 if tp == 1 else 0.8

    return dict(
        tensor_parallel_size=tp,
        max_model_len=ctx_len,
        max_num_seqs=seqs,
        gpu_memory_utilization=mem_util,
        block_size=128 if dflash else None,
        effective_vram_gb=round(effective_vram, 1) if count else None,
    )


def build_vllm_command(
    *,
    python,
    model,
    served_model_name,
    host,
    port,
    profile,
    dtype="bfloat16",
    quantization=None,
    draft=None,
    num_speculative_tokens=7,
    block_size=None,
    max_num_batched_tokens=8192,
    trust_remote_code=False,
    extra_args=(),
):
    cmd = [
        python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--served-model-name",
        served_model_name,
        "--tensor-parallel-size",
        str(profile["tensor_parallel_size"]),
        "--dtype",
        dtype,
        "--max-model-len",
        str(profile["max_model_len"]),
        "--max-num-seqs",
        str(profile["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(profile["gpu_memory_utilization"]),
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
        host,
        "--port",
        str(port),
    ]
    resolved_block_size = block_size if block_size is not None else profile.get("block_size")
    if resolved_block_size is not None:
        cmd.extend(["--block-size", str(resolved_block_size)])
    if quantization:
        cmd.extend(["--quantization", quantization])
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if draft:
        spec = dict(
            method="dflash",
            model=draft,
            num_speculative_tokens=num_speculative_tokens,
        )
        cmd.extend(["--speculative-config", json.dumps(spec)])
    cmd.extend(extra_args)
    return cmd


def _red_canary_png():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (512, 512), "red").save(buf, format="PNG")
    return buf.getvalue()


def verify_ready(proc, url, model, api_key, log_path, timeout_s=1800):
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_s
    while proc.poll() is None and time.monotonic() < deadline:
        try:
            http.open(url + "/models", timeout=4).close()
        except urllib.error.HTTPError as err:
            if err.code == 401:
                break
            raise
        except OSError:
            time.sleep(3)
            continue
        raise RuntimeError("server accepted an unauthenticated /models request")
    else:
        tail = log_path.read_text(errors="replace")[-4000:].replace(api_key, "***")
        raise RuntimeError("vLLM readiness failed:\n" + tail)

    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    with http.open(urllib.request.Request(url + "/models", headers=headers), timeout=30) as reply:
        models = {entry["id"] for entry in json.load(reply)["data"]}
    if model not in models:
        raise RuntimeError(f"served model {model!r} not listed in /v1/models: {sorted(models)}")

    image = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(_red_canary_png()).decode("ascii")
        },
    }
    payload = dict(
        model=model,
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
        urllib.request.Request(url + "/chat/completions", json.dumps(payload).encode(), headers),
        timeout=240,
    ) as reply:
        choice = json.load(reply)["choices"][0]
    answer = (choice["message"].get("content") or "").strip()
    if choice["finish_reason"] != "stop" or "red" not in answer.lower().split(".")[0].split():
        raise RuntimeError("visual canary failed: " + answer[:80])
    return answer


def advertise_host(bind_host):
    if bind_host in {"0.0.0.0", "::"}:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    return bind_host


def write_client_file(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    temp.replace(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        help="HuggingFace repo ID or local path to the ETA0-27B weights",
    )
    p.add_argument(
        "--draft",
        help="optional HuggingFace repo ID or local path to DFlash2 draft weights",
    )
    p.add_argument("--served-model-name", default=DEFAULT_SERVED_NAME)
    p.add_argument("--host", default="127.0.0.1", help="bind address (use 0.0.0.0 for remote)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--tensor-parallel-size", type=int)
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--max-num-seqs", type=int)
    p.add_argument("--gpu-memory-utilization", type=float)
    p.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16"])
    p.add_argument("--quantization", help="optional vLLM quantization method (e.g. fp8, awq)")
    p.add_argument("--block-size", type=int)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--num-speculative-tokens", type=int, default=7)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--python", default=sys.executable, help="Python executable with vLLM installed")
    p.add_argument("--session-dir", type=Path, help="directory for server.log and client.json")
    p.add_argument("--client-file", type=Path, help="override path for generated client.json")
    p.add_argument("--api-key", help="optional explicit Bearer key (auto-generated by default)")
    p.add_argument("--ready-timeout", type=int, default=1800)
    p.add_argument("--skip-canary", action="store_true", help="skip the 512x512 visual canary")
    p.add_argument("--dry-run", action="store_true", help="print detected GPUs and vLLM command")
    p.add_argument(
        "--extra-vllm-arg",
        action="append",
        default=[],
        help="repeatable extra flag forwarded verbatim to vLLM",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    gpus = detect_gpus()
    profile = choose_profile(
        gpus,
        tensor_parallel=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dflash=bool(args.draft),
    )
    cache_root = Path(os.getenv("ETA0_CACHE", Path.home() / ".cache" / "eta0-system")).expanduser()
    session = (args.session_dir or cache_root / f"serve-{args.port}").expanduser().resolve()
    client_path = (args.client_file or session / "client.json").expanduser().resolve()

    cmd = build_vllm_command(
        python=args.python,
        model=args.model,
        served_model_name=args.served_model_name,
        host=args.host,
        port=args.port,
        profile=profile,
        dtype=args.dtype,
        quantization=args.quantization,
        draft=args.draft,
        num_speculative_tokens=args.num_speculative_tokens,
        block_size=args.block_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=args.trust_remote_code,
        extra_args=args.extra_vllm_arg,
    )
    plan = dict(
        serving_mode="dflash2" if args.draft else "vllm",
        model=args.model,
        draft=args.draft,
        served_model_name=args.served_model_name,
        detected_gpus=gpus,
        profile=profile,
        client_file=str(client_path),
        command=cmd,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    os.umask(0o077)
    session.mkdir(parents=True, exist_ok=True)
    api_key = args.api_key or secrets.token_urlsafe(32)
    env = os.environ.copy()
    env["VLLM_API_KEY"] = api_key
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    log_path = session / "server.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)

    def handle_signal(signum, _frame):
        if proc.poll() is None:
            proc.terminate()

    prev_sigterm = signal.signal(signal.SIGTERM, handle_signal)
    try:
        local_url = f"http://127.0.0.1:{args.port}/v1"
        if args.skip_canary:
            canary = "skipped"
            http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + args.ready_timeout
            while proc.poll() is None and time.monotonic() < deadline:
                try:
                    http.open(local_url + "/models", timeout=4).close()
                except urllib.error.HTTPError as err:
                    if err.code == 401:
                        break
                    raise
                except OSError:
                    time.sleep(3)
                    continue
                raise RuntimeError("server accepted an unauthenticated /models request")
            else:
                tail = log_path.read_text(errors="replace")[-4000:].replace(api_key, "***")
                raise RuntimeError("vLLM readiness failed:\n" + tail)
        else:
            canary = verify_ready(
                proc, local_url, args.served_model_name, api_key, log_path, args.ready_timeout
            )
        public_base = f"http://{advertise_host(args.host)}:{args.port}/v1"
        info = dict(
            serving_mode="dflash2" if args.draft else "vllm",
            target=args.model,
            draft=args.draft,
            base_url=public_base,
            model=args.served_model_name,
            max_model_len=profile["max_model_len"],
            canary=canary,
            client_file=str(client_path),
            log_file=str(log_path),
        )
        write_client_file(client_path, dict(**info, api_key=api_key))
        print("ETA0_READY " + json.dumps(info, ensure_ascii=False), flush=True)
        print(f"export ETA0_EXECUTOR_CLIENT={client_path}", flush=True)
        raise SystemExit(proc.wait())
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        client_path.unlink(missing_ok=True)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
