#!/usr/bin/env bash
# Store the Boyue credential for the multi-model evaluation on this Mac.
set -eu
umask 077
exec python3 - <<'PY'
import getpass
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from urllib.parse import urlsplit

folder = Path.home() / ".config" / "robodojo"
source = folder / "boyue-api.json"
target = folder / "boyue-multimodel.json"
try:
    current = json.loads(source.read_text())
    base_url = current.get("boyue", current)["base_url"]
    if urlsplit(base_url).scheme != "https" or not urlsplit(base_url).hostname:
        raise ValueError("Boyue 地址需要是 HTTPS URL")
except (OSError, ValueError, KeyError, TypeError) as error:
    sys.exit(f"无法从 {source} 读取现有 Boyue 地址：{error}")

try:
    with open("/dev/tty", "w", encoding="utf-8") as terminal:
        if target.exists():
            terminal.write(f"{target} 已存在。要替换这份多模型配置吗？[y/N] ")
            terminal.flush()
            with open("/dev/tty", encoding="utf-8") as reply:
                if reply.readline().strip().lower() != "y":
                    sys.exit("已取消。")
        terminal.write(f"Boyue 服务地址为 {base_url}\n")
        terminal.flush()
        warnings.simplefilter("error", getpass.GetPassWarning)
        key = getpass.getpass("粘贴新 Boyue key，然后按回车（输入隐藏）：", stream=terminal).strip()
except (OSError, EOFError, getpass.GetPassWarning):
    sys.exit("请在 Mac 的交互式终端中运行此脚本。")
except KeyboardInterrupt:
    sys.exit("已取消。")
if not key or any(c.isspace() for c in key):
    sys.exit("key 为空或含空白字符，请重新运行。")

pending = None
try:
    with tempfile.NamedTemporaryFile("w", dir=folder, delete=False, encoding="utf-8") as temp:
        pending = Path(temp.name)
        os.fchmod(temp.fileno(), 0o600)
        json.dump({"boyue": {"base_url": base_url, "api_key": key}}, temp)
        temp.write("\n")
    os.replace(pending, target)
finally:
    if pending is not None:
        pending.unlink(missing_ok=True)
print(f"已保存 {target}（权限 600）。")
PY
