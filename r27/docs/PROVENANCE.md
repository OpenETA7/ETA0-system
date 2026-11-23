# 来源与发布范围

本目录导入冻结 R27 提交 `f0c490f1e914027205044003b4e0c50eb444bed4`。原开发分支为 `codex/round27-grasp-contract`，配套 LIBERO 提交为 `8f1084e3132a39270c3a13ebe37270a43ece2a01`。

`src/` 的 34 个 Python 文件、原始 62 个测试文件和 4 张参考 PNG 保持字节不变。逐文件 SHA-256 见 [source-provenance.json](source-provenance.json)，运行 `python scripts/verify_source.py` 可核对。导入不携带旧仓库 Git 历史。

发布整理包括：

- 独立 `r27/` 安装包、说明文档、参考配置和固定题单。
- 为离线测试补充 SciPy 安装依赖；新增预览优先的单题入口、入口测试和源码核验脚本。
- 保留原 runner 和任务目录以支持旧测试导入；不发布私有 profile、历史服务器启动脚本、密钥或原始运行日志。
- 将记忆来源记录中的机器绝对路径改为 `archive/` 相对标识。原图像、caption、证据摘要和原始证据哈希不变；manifest 文件哈希因此改变，前后哈希均在来源 JSON 中记录。这些来源标识不表示原始完整轨迹已随包公开。

参考图来自以往仿真任务，包含 alphabet soup、BBQ sauce、ketchup、tomato sauce。它们仅提供外观经验，不证明当前物体身份、位置或成功。运行时发送给模型的参考内容是 caption 和图像，机器来源路径不作为参考提示发送。

仓库根目录的既有系统保留，仅增加 R27 入口和独立测试配置。R27 核心不纳入根目录的 Ruff 风格重写，以保持冻结代码。软件授权沿用仓库根目录的 [LICENSE](../../LICENSE)；LIBERO 和其他依赖保留各自授权，不随本目录重新授权。
