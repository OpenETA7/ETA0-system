"""Exchange this experiment's preparation messages through its private shared directory."""

import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def write_message(path, data):
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    try:
        with temp.open("x") as stream:
            json.dump(data, stream, separators=(",", ":"))
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def request_preparation(mailbox, payload, remaining):
    request_id = uuid.uuid4().hex
    request = mailbox / (request_id + ".request")
    answer = request.with_suffix(".answer")
    seconds = remaining()
    if seconds <= 0:
        raise TimeoutError("upper deadline expired before handoff")
    try:
        write_message(request, dict(payload=payload, expires=time.time() + seconds))
        while remaining() > 0:
            if answer.is_file():
                data = json.loads(answer.read_bytes())
                if "response" in data:
                    return data["response"]
                if data["error"] == "invalid_json":
                    raise json.JSONDecodeError("invalid upstream JSON", "", 0)
                if data["error"] == "timeout":
                    raise TimeoutError("upper preparation timed out")
                raise RuntimeError("upper preparation relay: " + data["error"])
            time.sleep(min(0.25, max(0, remaining())))
        raise TimeoutError("upper deadline expired waiting for the relay")
    finally:
        request.unlink(missing_ok=True)
        answer.unlink(missing_ok=True)


def exchange(mailbox):
    """Run on the shared-volume host. This side only exchanges files for one job."""

    def receive():
        try:
            for line in sys.stdin:
                row = json.loads(line)
                ident = row["id"]
                if (
                    re.fullmatch("[0-9a-f]{32}", ident)
                    and (mailbox / (ident + ".request")).is_file()
                ):
                    retry_at = row["answer"].get("retry_at")
                    if retry_at:
                        write_message(mailbox / "next-admission.json", dict(epoch=retry_at))
                    write_message(mailbox / (ident + ".answer"), row["answer"])
        finally:
            stopped.set()

    stopped = threading.Event()
    threading.Thread(target=receive, daemon=True).start()
    sent = set()
    while not stopped.is_set() and not (mailbox / "closed").exists():
        for file in mailbox.glob("*.request"):
            if file.stem in sent or not re.fullmatch("[0-9a-f]{32}", file.stem):
                continue
            try:
                request = json.loads(file.read_bytes())
            except FileNotFoundError:
                continue
            print(json.dumps(dict(id=file.stem, **request), separators=(",", ":")), flush=True)
            sent.add(file.stem)
        stopped.wait(0.25)


def forward(profile, item):
    from eta0_run import ChatEndpoint
    from request_harness import BRIEF

    payload = item["payload"]
    if payload["model"] != profile["model"] or set(payload) != {"model", "messages", "max_tokens"}:
        raise ValueError("unexpected model or request options")
    system, user = payload["messages"]
    if system["role"] != "system" or not system["content"].startswith(BRIEF):
        raise ValueError("expected a task preparation")
    if user["role"] != "user" or payload["max_tokens"] not in (1500, 4096):
        raise ValueError("unexpected preparation request")
    views = [p["image_url"]["url"] for p in user["content"] if p["type"] == "image_url"]
    if len(views) != 2 or not all(v.startswith("data:image/jpeg;base64,") for v in views):
        raise ValueError("expected two inline observation images")
    left = min(900, item["expires"] - time.time())
    if left <= 0:
        raise TimeoutError("expired")
    client = ChatEndpoint(
        profile["base_url"],
        profile["model"],
        profile["api_key"],
        time.monotonic() + left,
        direct=True,
        allowed_models=profile["allowed_returned_models"],
        retries=4,
    )
    return client.chat(payload["messages"], max_tokens=payload["max_tokens"])


def serve(ssh, work, profile, cache):
    """Run on the Mac. Cache replies before delivery so an SSH reconnect can resend them."""
    from eta0_run import UpperRateLimit

    os.umask(0o077)
    cache.mkdir(parents=True, exist_ok=True)
    hold = cache.parent / "upstream-retry-at.json"
    mailbox = work / "preparation-mailbox"
    script = work / "source/preparation_relay.py"
    command = "python3 -u " + shlex.quote(str(script)) + " --exchange " + shlex.quote(str(mailbox))

    def handle(item, proc, lock):
        ident = item["id"]
        if not re.fullmatch("[0-9a-f]{32}", ident):
            raise ValueError("invalid request identifier")
        receipt = cache / (ident + ".json")
        if receipt.exists():
            answer = json.loads(receipt.read_bytes())["answer"]
        else:
            start = time.monotonic()
            try:
                retry_at = json.loads(hold.read_bytes())["epoch"] if hold.is_file() else 0
                if retry_at > time.time():
                    if retry_at >= item["expires"]:
                        raise UpperRateLimit("shared upstream rate window", retry_at - time.time())
                    time.sleep(max(0, retry_at - time.time()))
                answer = {"response": forward(profile, item)}
            except UpperRateLimit as exc:
                write_message(hold, dict(epoch=exc.retry_at))
                answer = {"error": "rate_limit_error", "retry_at": exc.retry_at}
                print(ident, "rate_limit_error", int(exc.retry_at), flush=True)
            except (json.JSONDecodeError, UnicodeDecodeError):
                answer = {"error": "invalid_json"}
            except TimeoutError:
                answer = {"error": "timeout"}
            except Exception as exc:
                answer = {"error": type(exc).__name__}
                print(ident, "failed", type(exc).__name__, flush=True)
            write_message(receipt, dict(answer=answer, elapsed_s=time.monotonic() - start))
        try:
            with lock:
                proc.stdin.write(json.dumps(dict(id=ident, answer=answer)) + "\n")
                proc.stdin.flush()
        except (OSError, ValueError):
            pass  # The completed response will be available on reconnect.

    while True:
        with subprocess.Popen(
            [ssh, command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        ) as proc:
            lock = threading.Lock()
            with ThreadPoolExecutor(max_workers=8) as workers:
                pending = []
                for line in proc.stdout:
                    pending.append(workers.submit(handle, json.loads(line), proc, lock))
                for future in pending:
                    future.result()
            if proc.wait() == 0:
                return
        time.sleep(5)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--exchange":
        exchange(Path(sys.argv[2]))
