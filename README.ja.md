# ETA0 System

[English](README.md) · [简体中文](README.zh-CN.md) · **日本語**

ETA0-27B（`fullmix step2500`）はユーザーの最終目標を受け取り、1 ラウンド内で自律的に観察・計画・ネイティブ進捗確認・アクション実行を行います。デフォルトの `request` モードでは、実行前に上位マルチモーダルモデルが 2 視点カメラ画像を確認し、必要時のみユーザーへ確認質問を行い、指示を明確化したうえで成功軌跡ライブラリから最大 1 件の例を選択できます。予算を使い切った場合は終了時の環境状態から引き継ぎ内容を更新し、次のラウンドで 27B が同一環境から続行して、ネイティブ成功判定を満たした時点で終了します。上位モデルの応答が空・途中で途切れた・形式不正だった場合は上位予算内で最大 4 回再試行し、それでも失敗した場合は元の指示と確認済み回答のみ（例なし）で 27B が続行します。

[27B モデル配備ガイド（vLLM / DFlash2）](docs/serving.md) · [リクエスト単位実行と対照評価](docs/request-harness.md) · [旧版ラウンド内監督方式](docs/executor-led.md) · [ベースハーネス仕様](README.base.md)

27B は自律的に `check_task_completion` を呼び出し、直近のネイティブ評価で達成済みの条件を確認してから次の行動を決定できます。全タスクで同一ツールプロトコルを使用します。この機能のみを検証する場合は `--mode base --completion-query` と `--mode base --no-completion-query` で 27B 単体比較を行えます。40 タスク × 5 初期状態の正式評価は[進捗クエリ対照アブレーション](docs/progress-query-ablation.md)を参照してください。

## モデル重みと推論配備（HuggingFace）

| 重み | 用途 | HuggingFace リポジトリ（公開前に正式 URL へ置換） |
|---|---|---|
| **ETA0-27B** | メイン実行モデル（標準 vLLM / DFlash2 対応） | `https://huggingface.co/OpenETA7/ETA0-27B-PLACEHOLDER` |
| **ETA0-27B-DFlash2** | DFlash2 投機的デコーディング用ドラフト重み（**RTX 5090 単GPU での高速ローカル起動に対応**） | `https://huggingface.co/OpenETA7/ETA0-27B-DFlash2-PLACEHOLDER` |

本リポジトリには社内パス依存を排除した汎用起動スクリプト [deploy/serve_vllm.py](deploy/serve_vllm.py) が含まれており、ローカル NVIDIA GPU の枚数と VRAM を自動検出して TP・コンテキスト長を決定し、512×512 画像カナリア検証後に `client.json` を自動出力します：

```bash
# 方式 1：標準 vLLM（GPU 枚数・VRAM に応じて TP とコンテキスト長を自動設定）
python3 deploy/serve_vllm.py --model OpenETA7/ETA0-27B-PLACEHOLDER

# 方式 2：vLLM + DFlash2 投機的デコーディング（例：RTX 5090 単GPU ローカル高速配備）
python3 deploy/serve_vllm.py \
  --model OpenETA7/ETA0-27B-PLACEHOLDER \
  --draft OpenETA7/ETA0-27B-DFlash2-PLACEHOLDER
```

RTX 5090 単GPU／マルチ GPU 推奨設定、量子化オプション、`--dry-run` 確認、リモート接続手順は[27B モデル配備ガイド](docs/serving.md)を参照してください。

## コーディングエージェント向けワンプロンプト配備

ターミナル実行権限を持つコーディングエージェント（Claude Code、Codex、Cursor、Antigravity など）へ以下をそのまま送信してください：

> `OpenETA7/ETA0-system` をクローンし、README に従って ETA0 System をセットアップして 1 件のリクエストを完走させてください。特定のハードウェアや推論基盤を前提にせず、まず現在の OS・GPU/VRAM・Python・オフスクリーン描画環境を確認してください。そのうえで私の実際の 27B 環境（既存の OpenAI 互換エンドポイントを再利用、`deploy/serve_vllm.py` でローカルまたはリモート GPU に標準 vLLM または DFlash2 で 27B を起動、あるいは 27B 未準備ならオフライン対話デモで先に検証）に応じて不足している必須設定だけを質問し、準備完了後に可能なら `--gate` ブラウザ引き継ぎ UI 付きで実行して、検出環境・採用した配備ルート・有効化した機能・環境検証目標・成功フラグ・レシートのパスを報告してください。

### コーディングエージェント配備ルート判断表（ハードウェア非依存）

エージェントはまず環境確認（OS、`nvidia-smi` / Apple Silicon、Python / `uv`、ネットワーク、既存エンドポイント）を行い、状況に応じて下記ルートを選択します：

| シナリオ | 対象環境と前提 | 推奨ルート |
|---|---|---|
| **A. 利用可能な 27B サービスが既にある** | ローカル・LAN・クラウドの 27B へ接続できる任意の Linux / macOS | 接続情報を `client.json` に保存し、`python3 deploy/first_run.py --gate 8765` を実行 |
| **B. 手元の GPU に 27B を配備する（vLLM / DFlash2）** | RTX 5090 単GPU、80 GB 単GPU、またはマルチ GPU NVIDIA 環境 | [deploy/serve_vllm.py](deploy/serve_vllm.py)（[配備ガイド](docs/serving.md)参照）で GPU 自動検出・起動して `client.json` を生成。実行器はサーバー側のコンテキスト長に自動追従（`/tokenize` 非対応時はローカル推定へフォールバック。厳密な `262144` 検証には `--strict-serving-profile` を指定）し、ルート A へ進む |
| **C. 27B 未準備／ノート PC で手軽に体験** | macOS / Linux / CPU のみ、またはインストールと GUI 確認を先行 | 下の「ローカル・オフライン体験」の `examples/request_demo.py --gate 8765` を実行 |

### 最初の実 LIBERO リクエスト

アクセス可能な 27B OpenAI Chat Completions エンドポイント（vLLM `/tokenize` があれば精密カウント、無ければ保守的ローカル推定に自動切替）、Git、`uv` が必要です。オフスクリーン描画要件は OS により異なります：
- **ヘッドレス Linux**：既定は Mesa（`MUJOCO_GL=osmesa`、Ubuntu では `libosmesa6`・`libgl1` が必要）。GPU 搭載環境では `MUJOCO_GL=egl` も利用可能。
- **macOS**：OS 標準のグラフィックスを使用するため Mesa 不要。

リポジトリルートから実行します（`--list-tasks` で内蔵 40 タスクを一覧表示、`--dry-run` でシミュレーションを起動せずにコマンドを事前確認可能。設定テンプレートは [examples/client.example.json](examples/client.example.json) と [examples/upper.example.json](examples/upper.example.json) を参照し、秘密ファイルは `chmod 600` を推奨）：

```bash
# 内蔵タスク一覧の確認・宣言的プロファイルのドライラン
python3 deploy/first_run.py --list-tasks --suite libero_goal
python3 deploy/first_run.py --profile configs/profiles/libero_goal_drawer.json --dry-run

# 実 LIBERO リクエストの実行
export ETA0_EXECUTOR_CLIENT=/path/to/private/client.json
python3 deploy/first_run.py --profile configs/profiles/libero_object_default.json --gate 8765
```

ランチャーはユーザーキャッシュ内に Python 3.11 環境を作成し、依存フィンガープリントでインストール済み状態をキャッシュ（2 回目以降は再インストール不要、更新時は `--reinstall`）、固定リビジョンの LIBERO アセットを取得します。既存環境を再利用する場合は `ETA0_SIM_PYTHON`・`ETA0_LIBERO_ROOT`、キャッシュ先は `ETA0_CACHE` で指定できます。既定タスクは `libero_object` task 0・初期状態 5 で、毎回新しい `outputs/first-request-*` に結果を保存します。

| プライベート設定 | 有効になる機能 |
|---|---|
| `ETA0_EXECUTOR_CLIENT` | 27B 自律実行。`base_url`、`model`、`api_key`、任意の `max_model_len`・`api_style`（`"vllm"` / `"openai"`）を含む JSON |
| `ETA0_UPPER_CONFIG`（または `ETA0_UPPER_BASE_URL`/`MODEL`/`API_KEY`）を追加 | 上位視覚モデルによる確認質問と指示グラウンディング。`allowed_returned_models` 指定可 |
| `ETA0_DEMO_LIBRARY` を追加 | 成功軌跡ライブラリ（`library.json`）からのデモ選択 |
| 任意の `--profile` / `--skills-dir` / `--skill` | 宣言的タスクプロファイルおよび SHA-256 検証済み読み取り専用 Markdown スキルカード（[skills/](skills/)、厳格な Tool/Skill 分離） |

**ビジュアル GUI Gate の起動**：`python3 deploy/first_run.py --gate 8765` を実行し、端末に表示された `#<token>` 付き完全 URL をブラウザで開きます。リモート環境の場合は先にローカル PC から `ssh -N -L 8765:127.0.0.1:8765 <SSH エイリアス>` を実行してください。5 段階ステッパー、拡大ライトボックス付き 2 視点カメラモニタ、シーンテレメトリ、確認対話履歴、指示差分バッジとワンクリック復元、デモ採用トグル、**実行中ライブ映像更新と安全停止ボタン**、コピー可能な完了レシートを備えています。`--gate` を省略するとヘッドレスで直接実行されます。

**マルチロボット・実機エコシステム**：本リポジトリは ETA0-27B 向けリクエストコンソールと LIBERO 評価に特化しており、コア制御ループ [harness.py](harness.py) は差し替え可能な `observation_writer`・`query_backend`・`try_render_tcp_overlay` を提供します。双腕 PiPER・Galaxea R1 Pro・実機 Franka/DROID アダプタ（`readback → preview → execute` 3 段階オンボーディング）は姉妹リポジトリ [OpenETA7/ETA-Harness](https://github.com/OpenETA7/ETA-Harness) をご参照ください。設計背景と過去実験の詳細は[11 月 24 日設計提案](docs/attempt-design-2026-11-24.md)にまとめています。

## ローカル・オフライン体験

Python 3.10 以上が必要です。スクリプトモデルと 2 ステップ模擬ロボットを使い、GPU や API キーなしで 2 ラウンド継続と GUI Gate の全フローを体験できます。

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
.venv/bin/python examples/request_demo.py --output outputs/first-request
# オプション：ブラウザ上で確認質問への回答と引き継ぎレビューを体験
.venv/bin/python examples/request_demo.py --gate 8765 --output outputs/gui-demo
.venv/bin/python -m pytest -q
```

## 詳細 CLI とマルチモード評価

27B 単体の `--mode base` や旧来の `--mode teacher` / `--mode judge` を直接実行する場合は、[deploy/serve_vllm.py](deploy/serve_vllm.py) が生成した（または手動作成した）`client.json` を指定します：

```bash
export ETA0_EXECUTOR_CLIENT=/path/to/private/client.json
python3 deploy/run_libero.py --executor-client "$ETA0_EXECUTOR_CLIENT" \
  --mode base --suite libero_10 --task 3 --initial-index 0 --output /tmp/eta0-bowl-base
```

上位マルチモーダルモデルを環境変数で指定する場合は、`ETA0_UPPER_BASE_URL`（末尾 `/v1`）、`ETA0_UPPER_MODEL`、`ETA0_UPPER_API_KEY` を設定します（または [examples/upper.example.json](examples/upper.example.json) 形式の JSON ファイルを使用）。上位 API の一時障害（タイムアウト・切断・HTTP 408/429/5xx）は同一フレーム上で最大 4 回リトライし、その間ロボットは静止します：

```bash
python3 deploy/run_libero.py --executor-client "$ETA0_EXECUTOR_CLIENT" \
  --mode teacher --suite libero_10 --task 3 --initial-index 0 --output /tmp/eta0-bowl-teacher
```

`--planner` / `--supervisor` / `--teacher`（および `--no-*`）の組み合わせや `--instruction`、`--recover-length`、LIBERO-40 対照評価の詳細は[接続ガイド](docs/executor-led.md)と[LIBERO-40 評価ガイド](docs/libero40.md)を参照してください。他のベンチマークも `harness.run(benchmark, policy, ...)` から直接統合できます。
