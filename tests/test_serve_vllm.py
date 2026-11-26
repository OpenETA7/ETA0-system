import json
import stat

import pytest

from deploy import serve_vllm


def test_choose_profile_scales_across_single_5090_single_80g_and_multi_gpu():
    rtx5090 = [dict(index=0, name="NVIDIA GeForce RTX 5090", vram_gb=32.0)]
    p5090 = serve_vllm.choose_profile(rtx5090, dflash=True)
    assert p5090["tensor_parallel_size"] == 1
    assert p5090["max_model_len"] == 65536
    assert p5090["max_num_seqs"] == 1
    assert p5090["gpu_memory_utilization"] == 0.85
    assert p5090["block_size"] == 128

    single_h100 = [dict(index=0, name="NVIDIA H100 80GB", vram_gb=80.0)]
    p80 = serve_vllm.choose_profile(single_h100, dflash=False)
    assert p80["tensor_parallel_size"] == 1
    assert p80["max_model_len"] == 131072
    assert p80["max_num_seqs"] == 2
    assert p80["block_size"] is None

    dual_h100 = [
        dict(index=0, name="NVIDIA H100 80GB", vram_gb=80.0),
        dict(index=1, name="NVIDIA H100 80GB", vram_gb=80.0),
    ]
    p2x80 = serve_vllm.choose_profile(dual_h100, dflash=True)
    assert p2x80["tensor_parallel_size"] == 2
    assert p2x80["max_model_len"] == 262144
    assert p2x80["max_num_seqs"] == 2
    assert p2x80["gpu_memory_utilization"] == 0.8

    quad_4090 = [dict(index=i, name="RTX 4090", vram_gb=24.0) for i in range(4)]
    p4x24 = serve_vllm.choose_profile(quad_4090)
    assert p4x24["tensor_parallel_size"] == 4
    assert p4x24["max_model_len"] == 131072

    with pytest.raises(ValueError, match="exceeds detected GPU count"):
        serve_vllm.choose_profile(rtx5090, tensor_parallel=2)


def test_build_vllm_command_standard_and_dflash2():
    rtx5090 = [dict(index=0, name="NVIDIA GeForce RTX 5090", vram_gb=32.0)]
    std_profile = serve_vllm.choose_profile(rtx5090, dflash=False)
    std_cmd = serve_vllm.build_vllm_command(
        python="/usr/bin/python3",
        model="OpenETA7/ETA0-27B-PLACEHOLDER",
        served_model_name="eta0-27b",
        host="127.0.0.1",
        port=18680,
        profile=std_profile,
        quantization="fp8",
    )
    assert "--speculative-config" not in std_cmd
    assert "--block-size" not in std_cmd
    assert std_cmd[std_cmd.index("--quantization") + 1] == "fp8"
    assert std_cmd[std_cmd.index("--reasoning-parser") + 1] == "qwen3"
    assert std_cmd[std_cmd.index("--tensor-parallel-size") + 1] == "1"

    df_profile = serve_vllm.choose_profile(rtx5090, dflash=True)
    df_cmd = serve_vllm.build_vllm_command(
        python="/usr/bin/python3",
        model="OpenETA7/ETA0-27B-PLACEHOLDER",
        served_model_name="eta0-27b",
        host="127.0.0.1",
        port=18680,
        profile=df_profile,
        draft="OpenETA7/ETA0-27B-DFlash2-PLACEHOLDER",
        num_speculative_tokens=7,
        extra_args=("--enforce-eager",),
    )
    spec = json.loads(df_cmd[df_cmd.index("--speculative-config") + 1])
    assert spec == {
        "method": "dflash",
        "model": "OpenETA7/ETA0-27B-DFlash2-PLACEHOLDER",
        "num_speculative_tokens": 7,
    }
    assert df_cmd[df_cmd.index("--block-size") + 1] == "128"
    assert df_cmd[-1] == "--enforce-eager"


def test_dry_run_and_private_client_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        serve_vllm,
        "detect_gpus",
        lambda: [dict(index=0, name="NVIDIA GeForce RTX 5090", vram_gb=32.0)],
    )
    client = tmp_path / "client.json"
    serve_vllm.main(
        [
            "--model",
            "OpenETA7/ETA0-27B-PLACEHOLDER",
            "--draft",
            "OpenETA7/ETA0-27B-DFlash2-PLACEHOLDER",
            "--client-file",
            str(client),
            "--dry-run",
        ]
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["serving_mode"] == "dflash2"
    assert plan["profile"]["tensor_parallel_size"] == 1
    assert plan["profile"]["max_model_len"] == 65536
    assert plan["client_file"] == str(client.resolve())
    assert not client.exists()

    serve_vllm.write_client_file(client, dict(base_url="http://127.0.0.1:18680/v1", api_key="k"))
    assert stat.S_IMODE(client.stat().st_mode) == 0o600
    assert json.loads(client.read_text())["api_key"] == "k"
