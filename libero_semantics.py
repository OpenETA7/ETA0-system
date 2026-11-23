"""Bind the pinned OpenETA semantic pack to the simulator already owned by the runner."""

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

PACK_SHA256 = "1325eeaca2d9eb257b788bf3732108426798a4fadb819cbb0f1a639fda7bd94f"


def use_libero_checkout(root, output):
    """Select both code and assets before the first simulator import, scoped to this process."""
    root = root.resolve(strict=True)
    benchmark = root / "libero/libero"
    config = output / "libero-config"
    config.mkdir()
    paths = dict(
        benchmark_root=benchmark,
        bddl_files=benchmark / "bddl_files",
        init_states=benchmark / "init_files",
        assets=benchmark / "assets",
        datasets=root / "datasets",
    )
    (config / "config.yaml").write_text(json.dumps({k: str(v) for k, v in paths.items()}))
    os.environ["LIBERO_CONFIG_PATH"] = str(config.resolve())
    sys.path.insert(0, str(root))


def task_semantics(pack, task_id, suite):
    if pack.version != "0.3.2" or pack.sha256 != PACK_SHA256:
        raise ValueError("expected the pinned LIBERO full-v1 0.3.2 pack")
    task = pack.task(task_id)
    if task["variant"] != suite:
        raise ValueError("semantic task and simulator suite differ")
    return task


class SemanticEpisode:
    def __init__(self, adapter, mode, completion_enabled, source_root=None):
        from openeta_bench.adapters.libero import LiberoAdapter
        from openeta_bench.instructions import materialize_instruction
        from openeta_bench.packs import load_pack
        from openeta_bench.tools import CompletionTool

        self.adapter = adapter
        self.pack = pack = load_pack("libero")
        task_id = adapter.meta["task_name"]
        task = task_semantics(pack, task_id, adapter.meta["suite"])
        binding = pack.source(task["binding"]["task_source"])
        if adapter.meta.get("bddl_sha256", binding["sha256"]) != binding["sha256"]:
            raise ValueError("semantic pack and simulator BDDL differ")
        self.instruction = (
            materialize_instruction(pack, task_id, "revised")
            if mode == "revised"
            else adapter.instruction
        )
        self.long_task = len(task["native_goals"]) > 1
        self.episode_id = uuid4().hex
        self.index = 0
        self.provider = None
        self.completion = None
        if completion_enabled:
            native = adapter.env.env
            bddl = Path(native.bddl_file_name).resolve()
            relative = Path(binding["path"])
            root = source_root or bddl.parents[len(relative.parts) - 1]
            self.provider = LiberoAdapter(pack, task_id, native, root)
            self.provider.begin_episode(self.episode_id)
            self.tool = CompletionTool(self.provider, profile="native_goal_conditions_v1")
            self.refresh()

    def refresh(self):
        self.completion = self.tool({"episode_id": self.episode_id, "action_index": self.index})

    def execute(self, action):
        if self.provider is None:
            return self.adapter.execute(action)
        self.index += 1
        with self.provider.observe_transition(self.index):
            feedback = self.adapter.execute(action)
        self.refresh()
        return feedback

    def __getattr__(self, key):
        return getattr(self.adapter, key)
