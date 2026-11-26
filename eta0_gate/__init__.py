"""Local browser review of a single robot request."""

import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files

from hierarchy import phrase
from request_harness import GateCancelled, image_url


class _ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def _is_loopback_host(host: str) -> bool:
    if not host:
        return False
    if host.startswith("["):
        return host.split("]", 1)[0] == "[::1"
    return host.split(":", 1)[0] in {"127.0.0.1", "localhost"}


class RequestGate:
    def __init__(self, port, benchmark_goal):
        self.condition = threading.Condition()
        self.state = dict(
            phase="preparing",
            version=0,
            benchmark_goal=benchmark_goal,
            dialogue=[],
            started_at=time.time(),
        )
        self.answer = None
        self.client_seen = False
        self.final_seen = threading.Event()
        self._stop = threading.Event()
        self.token = secrets.token_urlsafe(32)
        gate = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass  # Keep request content and access tokens out of logs.

            def respond(self, status, body=b"", kind="application/json", etag=None):
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                if etag:
                    self.send_header("ETag", etag)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src data:; "
                    "frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
                )
                self.end_headers()
                self.wfile.write(body)

            def authorized(self):
                host = self.headers.get("Host", "")
                origin = self.headers.get("Origin")
                return (
                    _is_loopback_host(host)
                    and origin in {None, "http://" + host}
                    and secrets.compare_digest(
                        self.headers.get("Authorization", "").encode(),
                        ("Bearer " + gate.token).encode(),
                    )
                )

            def do_GET(self):
                assets = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/gate.js": ("gate.js", "text/javascript; charset=utf-8"),
                    "/gate.css": ("gate.css", "text/css; charset=utf-8"),
                }
                if self.path in assets:
                    name, kind = assets[self.path]
                    self.respond(200, files("eta0_gate").joinpath(name).read_bytes(), kind)
                elif self.path == "/state" and self.authorized():
                    with gate.condition:
                        gate.client_seen = True
                        snapshot = dict(gate.state)
                    etag = f'"{snapshot["version"]}"'
                    if self.headers.get("If-None-Match") == etag:
                        self.send_response(304)
                        self.end_headers()
                        if snapshot["phase"] == "finished":
                            gate.final_seen.set()
                        return
                    self.respond(
                        200, json.dumps(snapshot, ensure_ascii=False).encode(), etag=etag
                    )
                    if snapshot["phase"] == "finished":
                        gate.final_seen.set()
                else:
                    self.respond(403)

            def do_POST(self):
                if self.path != "/decision" or not self.authorized():
                    return self.respond(403)
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 32_768:
                        return self.respond(400)
                    gate.submit(json.loads(self.rfile.read(size)))
                except (ValueError, TypeError, KeyError):
                    return self.respond(400)
                except RuntimeError:
                    return self.respond(409)
                self.respond(200, b"{}")

        self.server = _ReusableHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/#{self.token}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"ETA0 Gate  {self.url}", flush=True)

    def update(self, phase, **values):
        with self.condition:
            self.state.update(phase=phase, version=self.state["version"] + 1, **values)

    def scene(self, obs, goal):
        state = getattr(obs, "state", None) or {}
        robot = getattr(obs, "robot", None)
        self.update(
            "preparing",
            goal=goal,
            dialogue=[],
            external=image_url(obs.external_rgb)["image_url"]["url"],
            wrist=image_url(obs.wrist_rgb)["image_url"]["url"],
            observation_id=getattr(robot, "observation_id", None),
            reference_frame=getattr(robot, "reference_frame", None),
            gripper_open_width_mm=state.get("gripper_open_width_mm"),
            captured_at=time.time(),
        )

    def stop_requested(self):
        return self._stop.is_set()

    def observe_live(self, obs, actions, calls, wrist_overlay=None):
        state = getattr(obs, "state", None) or {}
        robot = getattr(obs, "robot", None)
        wrist_img = wrist_overlay if wrist_overlay is not None else obs.wrist_rgb
        with self.condition:
            if self.state["phase"] != "running":
                return
            self.state.update(
                version=self.state["version"] + 1,
                external=image_url(obs.external_rgb)["image_url"]["url"],
                wrist=image_url(wrist_img)["image_url"]["url"],
                observation_id=getattr(robot, "observation_id", None),
                reference_frame=getattr(robot, "reference_frame", None),
                gripper_open_width_mm=state.get("gripper_open_width_mm"),
                live_actions=actions,
                live_calls=calls,
                live_frame=True,
            )

    def submit(self, data):
        if not isinstance(data, dict):
            raise ValueError("expected an object")
        with self.condition:
            phase = self.state["phase"]
            if phase == "running" and data.get("action") == "stop":
                self._stop.set()
                self.update("running", stopping=True)
                self.condition.notify_all()
                return
            if phase not in {"question", "review"} or data.get("version") != self.state["version"]:
                raise RuntimeError("this handoff has changed")
            extra = {}
            if data.get("action") == "cancel":
                self.answer = GateCancelled()
            elif phase == "question" and data.get("action") == "answer":
                reply_text = phrase(data["answer"])
                self.answer = reply_text
                dialogue = list(self.state.get("dialogue") or [])
                dialogue.append(dict(question=self.state.get("question", ""), answer=reply_text))
                extra["dialogue"] = dialogue
            elif phase == "review" and data.get("action") == "start":
                if type(data.get("keep_demo")) is not bool:
                    raise ValueError("invalid example decision")
                self.answer = (phrase(data["instruction"]), data["keep_demo"])
            else:
                raise ValueError("unexpected decision")
            self.update("preparing", **extra)
            self.condition.notify_all()

    def wait(self, phase, **values):
        with self.condition:
            self.answer = None
            self.update(phase, **values)
            self.condition.wait_for(lambda: self.answer is not None)
            if isinstance(self.answer, GateCancelled):
                raise self.answer
            return self.answer

    def ask(self, question):
        return self.wait("question", question=question)

    def review(self, instruction, demo_id, library, dialogue):
        demo = None
        if demo_id:
            entry = library.entries[demo_id]
            parts = library.example(demo_id)
            preview = next((p["image_url"]["url"] for p in parts if p["type"] == "image_url"), None)
            demo = dict(
                id=demo_id,
                goal=entry["goal"],
                preview=preview,
                suite=entry.get("suite"),
                segments=entry.get("segments", []),
                frames_count=len(entry.get("frames", [])),
            )
        return self.wait(
            "review",
            instruction=instruction,
            proposed_instruction=instruction,
            demo=demo,
            dialogue=list(dialogue),
        )

    def running(self, attempt, instruction, demo_id):
        self.update(
            "running",
            attempt=attempt,
            instruction=instruction,
            demo_id=demo_id,
            running_since=time.time(),
            live_actions=0,
            live_calls=0,
            stopping=self._stop.is_set(),
        )

    def finish(self, result, receipt):
        self.update(
            "finished",
            success=result["success"],
            reason=result["reason"],
            calls=result.get("calls", 0),
            actions=result.get("actions", 0),
            upper_calls=result.get("upper_calls", 0),
            attempts=len(result.get("attempts", [])),
            first_attempt_success=result.get("first_attempt_success"),
            elapsed_s=result.get("elapsed_s"),
            receipt=str(receipt),
        )
        self.final_seen.wait(timeout=10 if self.client_seen else 0.5)

    def failed(self, error_type):
        self.update("finished", success=False, reason="runtime_error", error_type=error_type)
        self.final_seen.wait(timeout=10 if self.client_seen else 0.5)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
