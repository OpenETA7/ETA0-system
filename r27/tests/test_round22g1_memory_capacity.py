"""Offline tests for Linux and cgroup memory-capacity preflight."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import run_round22_targeted as runner  # noqa: E402


def mock_files(monkeypatch, files):
    reads = []

    def is_file(path):
        return str(path) in files

    def read_text(path, *args, **kwargs):
        key = str(path)
        reads.append(key)
        if key not in files:
            raise FileNotFoundError(key)
        return files[key]

    monkeypatch.setattr(Path, "is_file", is_file)
    monkeypatch.setattr(Path, "read_text", read_text)
    return reads


def meminfo(available_kib):
    return f"MemTotal: 2000000 kB\nMemFree: 10000 kB\nMemAvailable: {available_kib} kB\n"


def test_linux_memavailable_is_not_clamped_to_free_page_count(monkeypatch):
    mock_files(monkeypatch, {
        "/proc/meminfo": meminfo(900_000),
        "/proc/self/cgroup": "0::/\n",
        "/proc/self/mountinfo": mountinfo("v2", "/sys/fs/cgroup"),
    })
    monkeypatch.setattr(runner.sys, "platform", "linux")

    def unexpected_sysconf(_name):
        pytest.fail("MemAvailable should be preferred over the free-page fallback")

    monkeypatch.setattr(runner.os, "sysconf", unexpected_sysconf)
    available = runner.available_memory_bytes()
    assert available == 900_000 * 1024


def test_free_page_count_is_used_when_meminfo_is_unavailable(monkeypatch):
    mock_files(monkeypatch, {
        "/proc/self/cgroup": "0::/\n",
        "/proc/self/mountinfo": mountinfo("v2", "/sys/fs/cgroup"),
    })
    monkeypatch.setattr(runner.sys, "platform", "linux")
    values = {"SC_PAGE_SIZE": 4096, "SC_AVPHYS_PAGES": 512}
    monkeypatch.setattr(runner.os, "sysconf", lambda name: values[name])

    assert runner.available_memory_bytes() == 4096 * 512


def mountinfo(version, mountpoint, *, root="/"):
    if version == "v2":
        fs_type, source, super_options = "cgroup2", "cgroup", "rw"
    else:
        fs_type, source, super_options = "cgroup", "cgroup", "rw,memory"
    return f"36 25 0:31 {root} {mountpoint} rw,nosuid,nodev - {fs_type} {source} {super_options}\n"


@pytest.mark.parametrize("membership", [None, "\n"])
def test_linux_missing_or_empty_membership_fails_closed(monkeypatch, membership):
    files = {"/proc/meminfo": meminfo(900_000)}
    if membership is not None:
        files["/proc/self/cgroup"] = membership
    mock_files(monkeypatch, files)
    monkeypatch.setattr(runner.sys, "platform", "linux")

    with pytest.raises(ValueError, match="membership"):
        runner.available_memory_bytes()


@pytest.mark.parametrize("version,cgroup_record", [
    ("v2", "0::/team/job\n"),
    ("v1", "5:memory:/team/job\n"),
])
def test_leaf_and_ancestor_finite_headroom_are_both_applied(monkeypatch, version, cgroup_record):
    gib = 1024**3
    if version == "v2":
        root = Path("/run/container/cgroup2")
        names = ("memory.max", "memory.current")
        root_limit = {str(root / names[0]): "max\n"}
    else:
        root = Path("/run/container/memory")
        names = ("memory.limit_in_bytes", "memory.usage_in_bytes")
        root_limit = {str(root / names[0]): "max\n"}
    files = {"/proc/meminfo": meminfo(100 * 1024 * 1024),
             "/proc/self/cgroup": cgroup_record,
             "/proc/self/mountinfo": mountinfo(version, str(root)), **root_limit}
    files.update({
        str(root / "team" / "job" / names[0]): f"{20 * gib}\n",
        str(root / "team" / "job" / names[1]): f"{10 * gib}\n",
        str(root / "team" / names[0]): f"{15 * gib}\n",
        str(root / "team" / names[1]): f"{12 * gib}\n",
    })
    mock_files(monkeypatch, files)
    monkeypatch.setattr(runner.os, "sysconf", lambda name: 4096 if name == "SC_PAGE_SIZE" else 1)

    assert runner.available_memory_bytes() == 3 * gib


def test_zero_cgroup_headroom_is_preserved_and_preflight_rejects_dispatch(monkeypatch):
    root = Path("/var/lib/container/cgroup")
    files = {
        "/proc/meminfo": meminfo(100 * 1024 * 1024),
        "/proc/self/cgroup": "0::/single-job\n",
        "/proc/self/mountinfo": mountinfo("v2", str(root)),
        str(root / "memory.max"): "max\n",
        str(root / "single-job" / "memory.max"): "0\n",
        str(root / "single-job" / "memory.current"): "0\n",
    }
    mock_files(monkeypatch, files)
    monkeypatch.setattr(runner.os, "sysconf", lambda name: 4096 if name == "SC_PAGE_SIZE" else 1)

    assert runner.available_memory_bytes() == 0
    with pytest.raises(ValueError, match="fewer than one episode"):
        runner.concurrency_for_memory({}, 0)


@pytest.mark.parametrize("membership,mount_record", [
    ("0::/../../etc\n", mountinfo("v2", "/sys/fs/cgroup")),
    ("0::/team/job\n", ""),  # Membership exists, but no matching cgroup2 mount is known.
    ("0::/tenant/job\n", mountinfo("v2", "/sys/fs/cgroup", root="/tenant")),
])
def test_invalid_unknown_or_delegated_cgroup_membership_fails_closed(monkeypatch, membership, mount_record):
    files = {"/proc/meminfo": meminfo(128 * 1024),
             "/proc/self/cgroup": membership,
             "/proc/self/mountinfo": mount_record,
             "/sys/fs/cgroup/memory.max": "max\n"}
    reads = mock_files(monkeypatch, files)
    monkeypatch.setattr(runner.os, "sysconf", lambda name: 4096 if name == "SC_PAGE_SIZE" else 1)

    with pytest.raises(ValueError, match="cgroup|container|membership|mount"):
        runner.available_memory_bytes()
    assert not any("/etc/" in path or "/../" in path for path in reads)


@pytest.mark.parametrize("missing_path", ["/worker/job", "/worker"])
def test_unreadable_nonroot_cgroup_ancestor_limit_fails_closed(monkeypatch, missing_path):
    root = Path("/srv/container/memory")
    names = ("memory.max", "memory.current")
    files = {
        "/proc/meminfo": meminfo(100 * 1024 * 1024),
        "/proc/self/cgroup": "0::/worker/job\n",
        "/proc/self/mountinfo": mountinfo("v2", str(root)),
        str(root / names[0]): "max\n",
    }
    present_path = "/worker" if missing_path == "/worker/job" else "/worker/job"
    files[str(root / present_path.lstrip("/") / names[0])] = "8\n"
    files[str(root / present_path.lstrip("/") / names[1])] = "0\n"
    mock_files(monkeypatch, files)
    monkeypatch.setattr(runner.os, "sysconf", lambda name: 4096 if name == "SC_PAGE_SIZE" else 1)

    with pytest.raises(ValueError, match="cgroup|limit|memory"):
        runner.available_memory_bytes()


def test_root_cgroup2_may_omit_memory_max(monkeypatch):
    root = Path("/run/host/cgroup2")
    files = {
        "/proc/meminfo": meminfo(128 * 1024),
        "/proc/self/cgroup": "0::/\n",
        "/proc/self/mountinfo": mountinfo("v2", str(root)),
    }
    mock_files(monkeypatch, files)
    monkeypatch.setattr(runner.os, "sysconf", lambda name: 4096 if name == "SC_PAGE_SIZE" else 1)

    assert runner.available_memory_bytes() == 128 * 1024 * 1024


def test_requested_cap_reserve_and_per_episode_limits_are_unchanged():
    gib = 1024**3
    scheduler = {"max_concurrent_episodes": 24, "memory_reserve_gib": 8,
                 "memory_per_episode_gib": 2}
    capacity = runner.concurrency_for_memory(scheduler, 128 * gib)
    assert capacity["requested_cap"] == 24
    assert capacity["memory_derived_cap"] == 60
    assert capacity["effective_concurrency"] == 24
    assert capacity["reserve_gib"] == 8
    assert capacity["estimated_gib_per_episode"] == 2

    scheduler["max_concurrent_episodes"] = 80
    capped = runner.concurrency_for_memory(scheduler, 128 * gib)
    assert capped["requested_cap"] == capped["effective_concurrency"] == 60
    assert capped["authorized_hard_cap"] == 60

    limited = runner.concurrency_for_memory(scheduler, 26 * gib)
    assert limited["memory_derived_cap"] == limited["effective_concurrency"] == 9
    with pytest.raises(ValueError, match="fewer than one episode"):
        runner.concurrency_for_memory(scheduler, 9 * gib)
