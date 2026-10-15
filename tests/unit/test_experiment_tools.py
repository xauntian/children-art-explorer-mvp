"""实验工具应记录真实输入与缓存证据，并复用生产检查方式；不调用模型。"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import load_app_config
from app.generator import ComfyGenerator, load_workflow
from tools import bench_generate, prompt_grid


@pytest.mark.parametrize(
    ("cached", "samplers", "expected"),
    [
        (["1", "2", "11"], ["11"], "hit"),
        (["1", "2"], ["11"], "miss"),
        ([], ["11"], "miss"),
        (["11"], ["11", "21"], "partial"),
    ],
)
def test_cache_status_tracks_sampler_not_model_cache(cached, samplers, expected):
    history = {
        "p1": {"status": {"messages": [["execution_cached", {"prompt_id": "p1", "nodes": cached}]]}}
    }
    evidence = bench_generate.cache_evidence(history, "p1", samplers)
    assert evidence["status"] == expected
    assert evidence["cached_nodes"] == sorted(cached)


@pytest.mark.parametrize(
    "history",
    [
        {},
        {"p1": {"status": {"messages": []}}},
        {"p1": {"status": {"messages": [["execution_cached", {"prompt_id": "other", "nodes": []}]]}}},
        {"p1": {"status": {"messages": [["execution_cached", {"prompt_id": "p1", "nodes": "11"}]]}}},
    ],
)
def test_missing_or_unrelated_cache_evidence_stays_unknown(history):
    assert bench_generate.cache_evidence(history, "p1", ["11"])["status"] == "unknown"


@pytest.mark.parametrize("explicit_seed", [None, 0])
@pytest.mark.parametrize("uses_unet", [False, True])
@pytest.mark.parametrize("experiment_config", [False, True])
def test_bench_prompt_seed_and_metadata_match_submission(
    monkeypatch, tmp_path, capsys, explicit_seed, uses_unet, experiment_config,
):
    submitted = {}
    config = load_app_config(bench_generate.ROOT)
    if not uses_unet:
        config.comfy_workflow_path = bench_generate.ROOT / "workflows" / "image_edit.json"
        config.comfy_controlnet = "scribble.safetensors"
    config_path = bench_generate.ROOT / "configs" / "app.json"
    if experiment_config:
        workflow = json.loads(config.comfy_workflow_path.read_text(encoding="utf-8"))
        if uses_unet:
            workflow["5"] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["1", 0], "upscale_method": "lanczos",
                    "width": 1184, "height": 888, "crop": "disabled",
                },
            }
        workflow_path = tmp_path / "experiment-workflow.json"
        workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
        raw = config.raw
        raw["comfy"]["workflow_path"] = str(workflow_path)
        raw["comfy"]["controlnet"] = config.comfy_controlnet
        raw["preprocess"]["max_input_side"] = 640
        config_path = tmp_path / "experiment-app.json"
        config_path.write_text(json.dumps(raw), encoding="utf-8")
        config = load_app_config(bench_generate.ROOT, config_path=config_path)
        # 使用相对 CLI 路径，仍以项目根目录解析配置内的资源路径。
        monkeypatch.chdir(tmp_path)

        def config_loader(root, *, config_path):
            assert root == bench_generate.ROOT
            assert config_path == tmp_path / "experiment-app.json"
            return load_app_config(root, config_path=config_path)

        monkeypatch.setattr(bench_generate, "load_app_config", config_loader)
    else:
        # 保留原来没有 --config 时的单位置参数调用。
        monkeypatch.setattr(bench_generate, "load_app_config", lambda _: config)
    monkeypatch.setattr(bench_generate, "import_photo", lambda _: None)
    monkeypatch.setattr(
        bench_generate,
        "prepare_model_input",
        lambda *args, **kwargs: (SimpleNamespace(output_size=(64, 64)), b"input-image"),
    )
    monkeypatch.setattr(bench_generate, "decode_image_bytes", lambda *args: np.zeros((64, 64, 3)))
    monkeypatch.setattr(bench_generate.secrets, "randbelow", lambda _: 999)

    class Client:
        def __init__(self, **kwargs):
            pass

        def history(self, prompt_id):
            return {prompt_id: {"status": {"messages": [
                ["execution_cached", {"prompt_id": prompt_id, "nodes": ["1", *submitted["samplers"]]}]
            ]}}}

    class Generator(ComfyGenerator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            submitted["generator"] = kwargs
            submitted["samplers"] = self.binding["ksampler"]

        def verify_checkpoint(self):
            pass

        def generate(self, **kwargs):
            submitted["request"] = kwargs
            return b"candidate-image", "prompt-unit-test"

    monkeypatch.setattr(bench_generate, "ComfyClient", Client)
    monkeypatch.setattr(bench_generate, "ComfyGenerator", Generator)
    argv = [
        "--photo", str(tmp_path / "photo.jpg"), "--out", str(tmp_path / "outputs"),
        "--prompt", "Custom {style_name}: {preservation_rules}",
        "--steps", "8", "--lora-strength", "0.7",
    ]
    if explicit_seed is not None:
        argv.extend(["--seed", str(explicit_seed)])
    if experiment_config:
        argv.extend(["--config", "experiment-app.json", "--artwork"])
    config_bytes = config_path.read_bytes()
    assert bench_generate.main(argv) == 0
    request = submitted["request"]
    assert request["positive_prompt"].startswith("Custom ")
    assert "{style_name}" not in request["positive_prompt"]
    assert "{preservation_rules}" not in request["positive_prompt"]
    assert request["seed"] == (999 if explicit_seed is None else explicit_seed)
    metadata = json.loads(next((tmp_path / "outputs").glob("*.json")).read_text(encoding="utf-8"))
    assert metadata["positive_prompt"] == request["positive_prompt"]
    assert metadata["seed"] == request["seed"]
    assert metadata["sampler_params"]["steps"] == 8
    assert metadata["lora_strength"] == (0.7 if uses_unet else None)
    expected_model = config.comfy_unet_name if uses_unet else config.comfy_checkpoint
    assert metadata["model"] == expected_model
    printed = capsys.readouterr().out
    assert expected_model in printed
    if not uses_unet:
        assert config.comfy_unet_name not in printed
        assert config.comfy_lora_name not in printed
    assert metadata["prompt_id"] == "prompt-unit-test"
    assert metadata["cache"]["status"] == "hit"
    assert metadata["config_path"] == str(config_path.resolve())
    assert metadata["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert config_path.read_bytes() == config_bytes
    assert metadata["workflow_sha256"] == hashlib.sha256(config.comfy_workflow_path.read_bytes()).hexdigest()
    assert metadata["preprocess"]["max_input_side"] == config.pre_max_input_side
    resolved = metadata["resolved_workflow_before_upload"]
    assert resolved[submitted["samplers"][0]]["inputs"]["steps"] == 8
    if experiment_config:
        assert metadata["preprocess"]["frame_detect"] is False
        assert metadata["preprocess"]["border_trim_ratio"] == 0.0
        if uses_unet:
            assert resolved["5"]["inputs"]["crop"] == "disabled"
            assert resolved["5"]["inputs"]["width"] == 1184
            assert resolved["5"]["inputs"]["height"] == 888


@pytest.mark.parametrize("uses_unet", [False, True])
@pytest.mark.parametrize("use_lora", [False, True])
def test_bench_metadata_uses_only_active_workflow_dependencies(uses_unet, use_lora):
    root = bench_generate.ROOT
    path = root / "workflows" / ("image_edit_qwen.json" if uses_unet else "image_edit.json")
    nodes = load_workflow(path)
    # An unused loader must not change the recorded model route or dependencies.
    nodes["unused"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "unused.gguf"}}
    generator = ComfyGenerator(
        client=object(), workflow_nodes=nodes, checkpoint="sd.safetensors", negative_prompt="negative",
        controlnet="scribble.safetensors", unet_name="qwen.gguf", text_encoder="qwen-vl.gguf",
        vae_name="qwen-vae.safetensors", lora_name="lightning.safetensors" if use_lora else None,
        lora_strength=0.7, shift=3.0,
    )
    workflow = generator.build_workflow(
        input_image_name="input.png", positive_prompt="prompt", width=64, height=64, seed=42,
        params={"steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"},
    )
    actual = bench_generate.workflow_model_metadata(workflow, generator.binding["save_image"])
    assert actual["model_route"] == ("unet" if uses_unet else "checkpoint")
    assert actual["model"] == ("qwen.gguf" if uses_unet else "sd.safetensors")
    assert actual["text_encoder"] == ("qwen-vl.gguf" if uses_unet else None)
    assert actual["vae"] == ("qwen-vae.safetensors" if uses_unet else None)
    assert actual["shift"] == (3.0 if uses_unet else None)
    assert actual["lora_name"] == ("lightning.safetensors" if uses_unet and use_lora else None)
    assert actual["lora_strength"] == (0.7 if uses_unet and use_lora else None)


@pytest.mark.parametrize("mode", ["pairwise", "two_stage"])
def test_grid_checks_through_configured_mode(monkeypatch, tmp_path, mode):
    config = load_app_config(prompt_grid.ROOT)
    config.checker_mode = mode
    config.checker_control = True
    observed = {"checks": []}
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(prompt_grid, "load_app_config", lambda _: config)
    monkeypatch.setattr(
        prompt_grid,
        "prepare_model_input",
        lambda *args, **kwargs: (SimpleNamespace(output_size=(64, 64), image=image), b"image"),
    )
    monkeypatch.setattr(prompt_grid, "image_stats", lambda _: {})

    class Checker:
        def __init__(self, **kwargs):
            observed["mode"] = kwargs["mode"]
            observed["control"] = kwargs["control"]

        def check_pair(self, **kwargs):
            observed["checks"].append(kwargs)
            return SimpleNamespace(style_match="match", content_preserved=True, passed=True, reason="test")

        def check(self, *args, **kwargs):
            raise AssertionError("must not bypass configured two-stage dispatch")

    class Generator:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            return b"candidate", "prompt-test"

    monkeypatch.setattr(prompt_grid, "VisionChecker", Checker)
    monkeypatch.setattr(prompt_grid, "ComfyClient", lambda **kwargs: object())
    monkeypatch.setattr(prompt_grid, "ComfyGenerator", Generator)
    assert prompt_grid.main(["--photo", "input.jpg", "--out", str(tmp_path)]) == 0
    assert observed["mode"] == mode
    assert observed["control"] is True
    assert len(observed["checks"]) == 4
    style = config.load_styles().resolve("fauvism")
    assert all(call["style_name"] == style.name for call in observed["checks"])
    assert all(call["preservation_rules"] == style.preservation_rules for call in observed["checks"])
