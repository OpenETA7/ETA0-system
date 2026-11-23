#!/usr/bin/env python3
"""Verify the published core, original tests, and reference pixels against R27."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / "docs/source-provenance.json").read_text())
    errors = []
    for name, expected in manifest["unchanged_files_sha256"].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(name)
    expected_core = {name for name in manifest["unchanged_files_sha256"]
                     if name.startswith("src/")}
    actual_core = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")}
    errors.extend(sorted(actual_core - expected_core))
    memory = ROOT / "memory/verified-experience.json"
    if (not memory.is_file()
            or hashlib.sha256(memory.read_bytes()).hexdigest()
            != manifest["public_memory_manifest_sha256"]):
        errors.append("memory/verified-experience.json")
    if errors:
        raise SystemExit("R27 source differs: " + ", ".join(errors))
    print(f"Verified {len(manifest['unchanged_files_sha256'])} unchanged R27 files")


if __name__ == "__main__":
    verify()
