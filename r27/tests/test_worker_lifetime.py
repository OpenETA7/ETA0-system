"""Linux integration regression: simulator survives a transient request thread."""
import subprocess
import sys
import threading
import time
import pytest


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux PDEATHSIG semantics")


def worker_module():
    return pytest.importorskip("sim.mcp_server.worker_mgr")


def test_original_parent_thread_retirement_reproduces_sigterm():
    module = worker_module()
    children = []
    def start():
        proc = subprocess.Popen([sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(15)"],
                                stdout=subprocess.PIPE, text=True, preexec_fn=module._die_with_parent)
        assert proc.stdout.readline().strip() == "ready"
        children.append(proc)
    thread = threading.Thread(target=start)
    thread.start(); thread.join(timeout=10)
    assert children
    try:
        assert children[0].wait(timeout=5) == -15
    finally:
        if children[0].poll() is None:
            children[0].kill(); children[0].wait()


def test_persistent_spawner_survives_caller_thread_exit():
    module = worker_module()
    children = []
    def start():
        proc = module._spawn_with_stable_parent(
            [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(15)"],
            stdout=subprocess.PIPE, text=True, preexec_fn=module._die_with_parent)
        assert proc.stdout.readline().strip() == "ready"
        children.append(proc)
    thread = threading.Thread(target=start)
    thread.start(); thread.join(timeout=10)
    assert children
    try:
        time.sleep(0.1)
        assert children[0].poll() is None
    finally:
        children[0].terminate(); children[0].wait(timeout=5)
