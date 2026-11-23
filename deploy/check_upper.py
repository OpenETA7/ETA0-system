"""Exercise the visual teacher contract in the deployed simulation interpreter."""

import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from eta0_run import upper_endpoint
from executor_led import TEACH, short_field
from harness import image_part, parse_action, text
from tcp_overlay import render_tcp_overlay


def check(profile, frame, *, discover=False):
    endpoint = upper_endpoint(profile, time.monotonic() + 100)
    if discover:
        request = urllib.request.Request(endpoint.url + "/models")
        request.add_unredirected_header("Authorization", "Bearer " + endpoint.key)
        with urllib.request.urlopen(request, timeout=20) as response:
            models = [item["id"] for item in json.load(response)["data"]]
        print("LIBERO40_UPPER_CATALOG", json.dumps(models), flush=True)
        if endpoint.model == "kimi-k3.1" and endpoint.model not in models:
            candidates = [
                model for model in models if "kimi" in model.lower() and "k3.1" in model.lower()
            ]
            if len(candidates) == 1:
                endpoint.model = candidates[0]
                endpoint.allowed_models = candidates
    messages = [
        {"role": "system", "content": TEACH + " Return one JSON object."},
        {
            "role": "user",
            "content": [
                text(
                    {
                        "original_goal": "Place the object in the target region.",
                        "trigger": "consecutive_failures",
                    }
                ),
                text("External RGB"),
                image_part(frame / "external.png"),
                text("Wrist RGB / TCP"),
                image_part(frame / "wrist_overlay.png"),
            ],
        },
    ]
    reply = endpoint.chat(messages)
    choice = reply["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError("incomplete teacher preflight")
    data = parse_action(choice["message"]["content"])
    for key in ("cue", "skill", "expected"):
        short_field(data, key, 120)
    print(
        "LIBERO40_UPPER_READY",
        json.dumps(dict(model=reply["model"], role="teacher")),
        flush=True,
    )


if __name__ == "__main__":
    from rollout_libero_minimal import LiberoAdapter

    with tempfile.TemporaryDirectory() as tmp:
        frame = Path(tmp)
        adapter = LiberoAdapter(
            0,
            int(sys.argv[2]),
            1234,
            frame / "simulation",
            visual_queries=True,
            suite_name="libero_goal",
        )
        try:
            obs = adapter.observe()
            overlay, _ = render_tcp_overlay(obs.wrist_rgb, obs.wrist_camera, obs.robot)
            obs.external_rgb.save(frame / "external.png")
            overlay.save(frame / "wrist_overlay.png")
        finally:
            adapter.close()
        check(Path(sys.argv[1]), frame, discover="--discover" in sys.argv[3:])
