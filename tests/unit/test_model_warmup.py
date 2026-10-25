"""启动预热只接收合成输入，不创建作品任务或消耗随机画派。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from app import model_warmup
from app.errors import ConfigError, GenerationError
from app.preprocess import decode_image_bytes
from tests.fixtures import make_app_config, make_registry, solid_png


def _config(tmp_path, *, lcm=True, lora=True):
    config = make_app_config(
        tmp_path,
        comfy={"lora_name": "lcm.safetensors" if lora else None},
        preprocess={"max_input_side": 768},
        checker={"enabled": False},
    )
    workflow = {
        "1": {
            "class_type": "ModelSamplingDiscrete",
            "inputs": {"sampling": "lcm" if lcm else "eps"},
        }
    }
    config.comfy_workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    return config


@pytest.mark.parametrize("lcm,lora", [(False, True), (True, False), (False, False)])
def test_non_lcm_or_missing_lora_does_not_construct_app(tmp_path, monkeypatch, lcm, lora):
    _config(tmp_path, lcm=lcm, lora=lora)

    def forbidden(**kwargs):
        raise AssertionError("other routes must not generate")

    monkeypatch.setattr(model_warmup, "App", forbidden)
    assert model_warmup.warmup_generation(tmp_path) is None
    assert not (tmp_path / "runs/runtime/generation-warmup.json").exists()


def _fake_app(monkeypatch, *, output=None, error=None):
    calls = []
    verified = []
    registry = make_registry()

    def generate(**kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        return (
            output if output is not None else solid_png(kwargs["width"], kwargs["height"])
        ), "warmup-prompt"

    app = SimpleNamespace(
        registry=registry,
        generator=SimpleNamespace(
            generate=generate, verify_checkpoint=lambda: verified.append(True)
        ),
        pipeline=SimpleNamespace(build_generate_prompt=lambda style: "paint in " + style.name),
    )
    monkeypatch.setattr(model_warmup, "App", lambda **kwargs: app)
    return calls, verified


def test_warmup_uses_synthetic_input_and_first_style_without_job_or_random_choice(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    calls, verified = _fake_app(monkeypatch)
    report = model_warmup.warmup_generation(tmp_path)
    assert verified == [True]
    assert len(calls) == 1
    call = calls[0]
    decoded = decode_image_bytes(call["input_bytes"])
    assert decoded.shape == (512, 768, 3)
    assert np.unique(decoded).tolist() == [192]
    assert call["positive_prompt"] == "paint in Fauvism"
    assert 0 <= call["seed"] < 2**31
    assert call["params"] == config.sampler_for("fauvism")
    assert call["filename_prefix"] == "runtime/generation-warmup"
    assert report["status"] == "completed"
    assert report["checkpoint"] == config.comfy_checkpoint
    assert report["input_source"] == "synthetic_solid_colour"
    assert report["creates_user_job"] is False
    assert report["prompt_id"] == "warmup-prompt"
    assert report["seed"] == call["seed"]
    assert report["output_size"] == [768, 512]
    assert report["total_sec"] >= 0
    path = tmp_path / "runs/runtime/generation-warmup.json"
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert list((tmp_path / "runs").iterdir()) == [path.parent]


@pytest.mark.parametrize(
    "error,output",
    [(GenerationError("model offline"), None), (None, b"invalid PNG")],
)
def test_failed_warmup_records_failure_and_prevents_ready(tmp_path, monkeypatch, error, output):
    _config(tmp_path)
    calls, _ = _fake_app(monkeypatch, error=error, output=output)
    with pytest.raises(ConfigError, match="模型预热失败"):
        model_warmup.warmup_generation(tmp_path)
    report = json.loads(
        (tmp_path / "runs/runtime/generation-warmup.json").read_text(encoding="utf-8")
    )
    assert len(calls) == 1
    assert report["status"] == "failed"
    assert report["error"]["message"]
    assert report["total_sec"] >= 0
    assert "prompt_id" not in report


def _flux2_config(tmp_path, *, scheduler=True, reference=True, unet=True, size=1024):
    config = make_app_config(
        tmp_path,
        comfy={"unet_name": "flux2-klein-4b.safetensors" if unet else None},
        preprocess={"max_input_side": size},
        checker={"enabled": False},
    )
    workflow = {
        "_meta": {"description": "Official FLUX.2 image-edit route"},
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "2": {"class_type": "CLIPLoader", "inputs": {"type": "flux2"}},
        "3": {"class_type": "VAELoader", "inputs": {}},
    }
    if scheduler:
        workflow["4"] = {"class_type": "Flux2Scheduler", "inputs": {}}
    if reference:
        workflow["5"] = {"class_type": "ReferenceLatent", "inputs": {}}
    config.comfy_workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    return config


@pytest.mark.parametrize(
    "scheduler,reference,unet", [(False, True, True), (True, False, True), (True, True, False)]
)
def test_flux2_detection_requires_scheduler_reference_and_configured_unet(
    tmp_path, monkeypatch, scheduler, reference, unet
):
    _flux2_config(tmp_path, scheduler=scheduler, reference=reference, unet=unet)
    monkeypatch.setattr(model_warmup, "App", lambda **kwargs: pytest.fail("not a supported route"))
    assert model_warmup.warmup_generation(tmp_path) is None


def test_flux2_prewarms_every_enabled_static_style_at_configured_size(tmp_path, monkeypatch):
    config = _flux2_config(tmp_path)
    calls, verified = _fake_app(monkeypatch)
    report = model_warmup.warmup_generation(tmp_path)
    expected_ids = [style.id for style in make_registry().require_enabled()]
    assert verified == [True]
    assert len(calls) == 3
    assert report["route"] == "flux2"
    assert report["checkpoint"] is None
    assert report["planned_styles"] == expected_ids
    assert [item["style_id"] for item in report["style_prewarms"]] == expected_ids
    assert len({call["seed"] for call in calls}) == 3
    assert report["width"] == 1024 and report["height"] == 688
    for call, item, style in zip(
        calls, report["style_prewarms"], make_registry().require_enabled(), strict=True
    ):
        assert call["positive_prompt"] == "paint in " + style.name
        assert call["params"] == config.sampler_for(style.id)
        assert (call["width"], call["height"]) == (1024, 688)
        decoded = decode_image_bytes(call["input_bytes"])
        assert decoded.shape == (688, 1024, 3) and np.unique(decoded).tolist() == [192]
        assert item["status"] == "completed"
        assert item["total_sec"] >= 0 and item["prompt_id"] == "warmup-prompt"
        assert item["output_size"] == [1024, 688]
    assert report["style_id"] == expected_ids[0]
    assert report["prompt_id"] == report["style_prewarms"][0]["prompt_id"]
    assert report["seed"] == calls[0]["seed"]
    assert report["creates_user_job"] is False
    path = tmp_path / "runs/runtime/generation-warmup.json"
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert list((tmp_path / "runs").iterdir()) == [path.parent]


def test_lcm_retains_one_style_and_768_limit_even_if_configured_larger(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.raw["preprocess"]["max_input_side"] = 1024
    (tmp_path / "configs/app.json").write_text(json.dumps(config.raw), encoding="utf-8")
    calls, _ = _fake_app(monkeypatch)
    report = model_warmup.warmup_generation(tmp_path)
    assert len(calls) == 1
    assert (report["width"], report["height"]) == (768, 512)
    assert report["route"] == "lcm" and len(report["style_prewarms"]) == 1


def test_flux2_failure_on_second_style_keeps_first_evidence_and_stops(tmp_path, monkeypatch):
    _flux2_config(tmp_path)
    calls, _ = _fake_app(monkeypatch)
    original_app = model_warmup.App(config=None)
    generate = original_app.generator.generate

    def fails_second(**kwargs):
        if calls:
            calls.append(kwargs)
            raise GenerationError("second style failed")
        return generate(**kwargs)

    original_app.generator.generate = fails_second
    with pytest.raises(ConfigError, match="模型预热失败"):
        model_warmup.warmup_generation(tmp_path)
    report = json.loads(
        (tmp_path / "runs/runtime/generation-warmup.json").read_text(encoding="utf-8")
    )
    assert len(calls) == 2
    assert report["status"] == "failed"
    assert [item["status"] for item in report["style_prewarms"]] == ["completed", "failed"]
    assert report["style_prewarms"][0]["prompt_id"] == "warmup-prompt"
    assert report["style_prewarms"][1]["error"]["message"] == "second style failed"


def test_flux2_warmup_does_not_use_style_assigner(tmp_path, monkeypatch):
    _flux2_config(tmp_path)
    calls, _ = _fake_app(monkeypatch)
    app = model_warmup.App(config=None)

    class ForbiddenAssigner:
        def assign(self):
            pytest.fail("prewarming cannot consume a user style assignment")

    app.pipeline.style_assigner = ForbiddenAssigner()
    report = model_warmup.warmup_generation(tmp_path)
    assert report["status"] == "completed" and len(calls) == 3


def test_prewarms_use_distinct_new_sampling_seeds_across_startups(tmp_path, monkeypatch):
    _flux2_config(tmp_path)
    calls, _ = _fake_app(monkeypatch)
    seeds = iter([100, 1000])
    monkeypatch.setattr(model_warmup.time, "time_ns", lambda: next(seeds))
    model_warmup.warmup_generation(tmp_path)
    model_warmup.warmup_generation(tmp_path)
    assert [call["seed"] for call in calls] == [100, 101, 102, 1000, 1001, 1002]


def test_wrong_sized_warmup_output_cannot_report_ready(tmp_path, monkeypatch):
    _flux2_config(tmp_path)
    _fake_app(monkeypatch, output=solid_png(768, 512))
    with pytest.raises(ConfigError, match="尺寸"):
        model_warmup.warmup_generation(tmp_path)


def test_flux2_only_prewarms_enabled_registry_styles(tmp_path, monkeypatch):
    _flux2_config(tmp_path)
    calls, _ = _fake_app(monkeypatch)
    app = model_warmup.App(config=None)
    app.registry = make_registry(enabled={"cubism": False})
    report = model_warmup.warmup_generation(tmp_path)
    assert report["planned_styles"] == ["fauvism", "impressionism"]
    assert len(calls) == 2


def _qwen21_config(tmp_path, *, encoder=True, cache=True, unet=True, size=1024):
    config = make_app_config(
        tmp_path,
        comfy={
            "unet_name": "qwen-image-2.1-turbo.safetensors" if unet else None,
            "text_encoder": "qwen_2.5_vl_3b.safetensors",
            "vae_name": "qwen_image_vae.safetensors",
        },
        preprocess={"max_input_side": size},
        checker={"enabled": False},
        generation={
            "samplers": {
                style.id: {
                    "steps": 8,
                    "cfg": 1.0,
                    "denoise": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                }
                for style in make_registry().require_enabled()
            }
        },
    )
    workflow = {
        "_meta": {"description": "Official Qwen-Image 2.1 Turbo image-edit route"},
        "1": {"class_type": "UNETLoader", "inputs": {}},
    }
    if encoder:
        workflow["2"] = {"class_type": "TextEncodeQwenImage21", "inputs": {}}
    if cache:
        workflow["3"] = {"class_type": "QwenImage21Cache", "inputs": {}}
    config.comfy_workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    return config


@pytest.mark.parametrize(
    "encoder,cache,unet", [(False, True, True), (True, False, True), (True, True, False)]
)
def test_qwen21_requires_encoder_cache_and_configured_unet(
    tmp_path, monkeypatch, encoder, cache, unet
):
    _qwen21_config(tmp_path, encoder=encoder, cache=cache, unet=unet)
    monkeypatch.setattr(model_warmup, "App", lambda **kwargs: pytest.fail("unsupported route"))
    assert model_warmup.warmup_generation(tmp_path) is None
    assert not (tmp_path / "runs/runtime/generation-warmup.json").exists()


@pytest.mark.parametrize(
    "size,expected", [(1024, (1024, 672)), (896, (896, 608)), (1001, (992, 672))]
)
def test_qwen21_prewarms_all_static_styles_at_32_aligned_size(
    tmp_path, monkeypatch, size, expected
):
    config = _qwen21_config(tmp_path, size=size)
    calls, verified = _fake_app(monkeypatch)
    app = model_warmup.App(config=None)

    class ForbiddenAssigner:
        def assign(self):
            pytest.fail("prewarming must not consume user style assignments")

    app.pipeline.style_assigner = ForbiddenAssigner()
    app.pipeline.run = lambda **kwargs: pytest.fail("prewarming must not create user jobs")
    report = model_warmup.warmup_generation(tmp_path)
    assert verified == [True]
    assert report["status"] == "completed"
    assert report["route"] == "qwen_image21"
    assert report["checkpoint"] is None
    assert report["unet_name"] == config.comfy_unet_name
    assert report["text_encoder"] == config.comfy_text_encoder
    assert report["vae_name"] == config.comfy_vae_name
    assert (report["width"], report["height"]) == expected
    styles = make_registry().require_enabled()
    assert report["planned_styles"] == [style.id for style in styles]
    assert len(calls) == len(styles) == 3
    assert len({call["seed"] for call in calls}) == 3
    for call, item, style in zip(calls, report["style_prewarms"], styles, strict=True):
        assert (call["width"], call["height"]) == expected
        assert call["positive_prompt"] == "paint in " + style.name
        assert call["params"] == config.sampler_for(style.id)
        assert call["params"]["steps"] == 8 and call["params"]["cfg"] == 1.0
        assert call["filename_prefix"] == f"runtime/generation-warmup/{style.id}"
        decoded = decode_image_bytes(call["input_bytes"])
        assert decoded.shape == (expected[1], expected[0], 3)
        assert np.unique(decoded).tolist() == [192]
        assert item["status"] == "completed" and item["style_id"] == style.id
        assert item["prompt_id"] == "warmup-prompt"
        assert item["output_size"] == list(expected) and item["total_sec"] >= 0
    assert report["creates_user_job"] is False
    assert report["style_id"] == styles[0].id
    assert report["prompt_id"] == report["style_prewarms"][0]["prompt_id"]
    path = tmp_path / "runs/runtime/generation-warmup.json"
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert list((tmp_path / "runs").iterdir()) == [path.parent]


def test_qwen21_invalid_output_blocks_ready_and_records_actual_models(tmp_path, monkeypatch):
    config = _qwen21_config(tmp_path)
    calls, _ = _fake_app(monkeypatch, output=solid_png(1024, 688))
    with pytest.raises(ConfigError, match="尺寸"):
        model_warmup.warmup_generation(tmp_path)
    report = json.loads(
        (tmp_path / "runs/runtime/generation-warmup.json").read_text(encoding="utf-8")
    )
    assert len(calls) == 1
    assert report["status"] == "failed"
    assert report["route"] == "qwen_image21" and report["checkpoint"] is None
    assert report["unet_name"] == config.comfy_unet_name
    assert report["text_encoder"] == config.comfy_text_encoder
    assert report["vae_name"] == config.comfy_vae_name
    assert report["style_prewarms"][0]["status"] == "failed"
    assert "prompt_id" not in report


def test_qwen21_a4_warmup_matches_native_a4_landscape_bucket(tmp_path, monkeypatch):
    config = _qwen21_config(tmp_path, size=1088)
    config.raw["preprocess"].update(paper_format="a4", output_short_side=720)
    (tmp_path / "configs/app.json").write_text(json.dumps(config.raw), encoding="utf-8")
    calls, _ = _fake_app(monkeypatch)
    report = model_warmup.warmup_generation(tmp_path)
    assert report["status"] == "completed"
    assert (report["width"], report["height"]) == (1088, 768)
    assert len(calls) == 3
    assert all((call["width"], call["height"]) == (1088, 768) for call in calls)
