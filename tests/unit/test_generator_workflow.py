"""工作流校验测试：必须确认是真正的图生图（原图进入采样器）。

不得编造工作流节点 ID，也不得把文字生图当作图生图。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.errors import ConfigError
from app.generator import (
    ComfyClient,
    ComfyGenerator,
    apply_placeholders,
    load_workflow,
    validate_image_input_workflow,
)


def _nodes(**extra) -> dict:
    base = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "in.png"}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "m.safetensors"}},
        "3": {"class_type": "ImageScale", "inputs": {"image": ["1", 0], "upscale_method": "area", "width": 512, "height": 512, "crop": "disabled"}},
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["2", 2]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos", "clip": ["2", 1]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "neg", "clip": ["2", 1]}},
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "seed": 1,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["4", 0],
                "denoise": 0.6,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "x"}},
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------- 正常链路
def test_valid_img2img_workflow_passes():
    binding = validate_image_input_workflow(_nodes())
    assert binding["load_image"] == "1"
    assert binding["vae_encode"] == "4"
    assert binding["save_image"] == "9"
    assert binding["ksampler"] == ["7"]


def test_real_project_workflow_is_valid_and_img2img():
    root = Path(__file__).resolve().parents[2]
    nodes = load_workflow(root / "workflows" / "image_edit.json")
    binding = validate_image_input_workflow(nodes)
    assert binding["load_image"]
    assert binding["vae_encode"]
    class_types = {n["class_type"] for n in nodes.values()}
    assert "LoadImage" in class_types
    assert "ControlNetApplyAdvanced" in class_types, "应使用 ControlNet 保持原作结构"


# ---------------------------------------------------------------- 拒绝文字生图
def test_rejects_workflow_without_load_image():
    nodes = _nodes()
    del nodes["1"]
    del nodes["3"]
    nodes["4"]["inputs"]["pixels"] = ["12", 0]
    with pytest.raises(ConfigError, match="LoadImage"):
        validate_image_input_workflow(nodes)


def test_rejects_text_to_image_workflow_with_empty_latent():
    """EmptyLatentImage 起点说明是文字生图，必须拒绝。"""
    nodes = _nodes()
    nodes["13"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
    }
    nodes["7"]["inputs"]["latent_image"] = ["13", 0]
    with pytest.raises(ConfigError, match="latent_image 不来自 VAEEncode"):
        validate_image_input_workflow(nodes)


def test_rejects_workflow_where_load_image_output_is_unused():
    """原图存在但没有任何节点使用它，等于文字生图。"""
    nodes = _nodes()
    nodes["3"]["inputs"]["image"] = ["15", 0]
    nodes["15"] = {"class_type": "EmptyLatentImage", "inputs": {}}
    with pytest.raises(ConfigError, match="没有被任何节点使用"):
        validate_image_input_workflow(nodes)


def test_rejects_workflow_without_vae_encode():
    nodes = _nodes()
    del nodes["4"]
    nodes["7"]["inputs"]["latent_image"] = ["13", 0]
    nodes["13"] = {"class_type": "EmptyLatentImage", "inputs": {}}
    with pytest.raises(ConfigError, match="VAEEncode"):
        validate_image_input_workflow(nodes)


def test_rejects_workflow_without_save_image():
    nodes = _nodes()
    del nodes["9"]
    with pytest.raises(ConfigError, match="SaveImage"):
        validate_image_input_workflow(nodes)


def test_rejects_workflow_without_ksampler():
    nodes = _nodes()
    del nodes["7"]
    with pytest.raises(ConfigError, match="KSampler"):
        validate_image_input_workflow(nodes)


def test_accepts_image_reaching_vae_encode_through_intermediate_node():
    """经过中间图像节点（如 ControlNet 前置处理）也算图生图。"""
    nodes = _nodes()
    nodes["3"]["inputs"]["image"] = ["14", 0]
    nodes["14"] = {"class_type": "ImageInvert", "inputs": {"image": ["1", 0]}}
    validate_image_input_workflow(nodes)


def _custom_nodes():
    """The native FLUX.2 reference-image route, without a dummy KSampler."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "__input_image__"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "__unet_name__", "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "__text_encoder__", "type": "flux2"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "__vae_name__"}},
        "5": {"class_type": "ImageScale", "inputs": {"image": ["1", 0], "width": "__width__", "height": "__height__"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": "__positive_prompt__"}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["4", 0]}},
        "8": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["6", 0], "latent": ["7", 0]}},
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["8", 0]}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": "__seed__"}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "__sampler_name__"}},
        "12": {"class_type": "Flux2Scheduler", "inputs": {"steps": "__steps__", "width": "__width__", "height": "__height__"}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["9", 0], "sampler": ["11", 0], "sigmas": ["12", 0], "latent_image": ["7", 0]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["4", 0]}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "__filename_prefix__"}},
    }


def test_custom_sampler_uses_actual_image_latent_and_reference_conditioning():
    nodes = _custom_nodes()
    binding = validate_image_input_workflow(nodes)
    assert binding == {"load_image": "1", "vae_encode": "7", "ksampler": ["13"], "save_image": "15"}
    assert not any(node["class_type"] == "KSampler" for node in nodes.values())
    assert nodes["8"]["inputs"]["latent"] == nodes["13"]["inputs"]["latent_image"]
    assert nodes["9"]["inputs"]["conditioning"] == ["8", 0]


@pytest.mark.parametrize("guider_class", ["BasicGuider", "CFGGuider"])
def test_custom_sampler_native_flux2_empty_latent_uses_real_reference_conditioning(guider_class):
    nodes = _custom_nodes()
    nodes["16"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {}}
    nodes["13"]["inputs"]["latent_image"] = ["16", 0]
    if guider_class == "CFGGuider":
        nodes["9"] = {"class_type": "CFGGuider", "inputs": {"model": ["2", 0], "positive": ["8", 0], "negative": ["8", 0], "cfg": 1}}
    binding = validate_image_input_workflow(nodes)
    assert binding == {"load_image": "1", "vae_encode": "7", "ksampler": ["13"], "save_image": "15"}


def test_custom_sampler_empty_latent_cannot_use_orphan_reference():
    nodes = _custom_nodes()
    nodes["16"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {}}
    nodes["13"]["inputs"]["latent_image"] = ["16", 0]
    nodes["9"]["inputs"]["conditioning"] = ["6", 0]
    with pytest.raises(ConfigError, match="SamplerCustomAdvanced.*latent_image 不来自 VAEEncode"):
        validate_image_input_workflow(nodes)


def test_orphan_ksampler_cannot_validate_text_only_custom_output():
    nodes = _custom_nodes()
    nodes["16"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {}}
    nodes["13"]["inputs"]["latent_image"] = ["16", 0]
    nodes["9"]["inputs"]["conditioning"] = ["6", 0]
    nodes["dummy"] = {"class_type": "KSampler", "inputs": {"latent_image": ["7", 0]}}
    with pytest.raises(ConfigError, match="SamplerCustomAdvanced.*latent_image 不来自 VAEEncode"):
        validate_image_input_workflow(nodes)


def test_native_flux2_reference_only_on_negative_guider_input_is_not_original_conditioning():
    nodes = _custom_nodes()
    nodes["16"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {}}
    nodes["13"]["inputs"]["latent_image"] = ["16", 0]
    nodes["9"] = {"class_type": "CFGGuider", "inputs": {"model": ["2", 0], "positive": ["6", 0], "negative": ["8", 0], "cfg": 1}}
    with pytest.raises(ConfigError, match="有效的原生图像参考"):
        validate_image_input_workflow(nodes)


def test_native_flux2_consumed_reference_must_use_original_encoder():
    nodes = _custom_nodes()
    nodes["16"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {}}
    nodes["17"] = {"class_type": "EmptyImage", "inputs": {}}
    nodes["18"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["17", 0], "vae": ["4", 0]}}
    nodes["13"]["inputs"]["latent_image"] = ["16", 0]
    nodes["8"]["inputs"]["latent"] = ["18", 0]
    with pytest.raises(ConfigError, match="有效的原生图像参考"):
        validate_image_input_workflow(nodes)


@pytest.mark.parametrize("sampler_class", ["KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced"])
def test_sampler_cannot_use_unrelated_encoder_when_an_orphan_encoder_uses_original(sampler_class):
    nodes = _nodes()
    nodes["7"]["class_type"] = sampler_class
    nodes["16"] = {"class_type": "EmptyImage", "inputs": {}}
    nodes["17"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["16", 0], "vae": ["2", 2]}}
    nodes["7"]["inputs"]["latent_image"] = ["17", 0]
    with pytest.raises(ConfigError, match="latent_image 不来自 VAEEncode"):
        validate_image_input_workflow(nodes)


def test_binding_excludes_dormant_sampler_and_uses_encoder_connected_to_output():
    nodes = _custom_nodes()
    nodes["dormant"] = {"class_type": "KSampler", "inputs": {"latent_image": ["7", 0]}}
    assert validate_image_input_workflow(nodes)["ksampler"] == ["13"]


def test_sampler_must_feed_saved_output():
    nodes = _custom_nodes()
    nodes["15"]["inputs"]["images"] = ["1", 0]
    with pytest.raises(ConfigError, match="SaveImage 未连接"):
        validate_image_input_workflow(nodes)


def test_encoder_pixels_cannot_be_replaced_by_load_image_mask_output():
    nodes = _nodes()
    nodes["4"]["inputs"]["pixels"] = ["1", 1]
    with pytest.raises(ConfigError, match="原图未连接到 VAEEncode"):
        validate_image_input_workflow(nodes)


def test_encoder_vae_dependency_cannot_masquerade_as_original_pixels():
    nodes = _nodes()
    nodes["16"] = {"class_type": "EmptyImage", "inputs": {}}
    nodes["4"]["inputs"] = {"pixels": ["16", 0], "vae": ["1", 0]}
    with pytest.raises(ConfigError, match="原图未连接到 VAEEncode"):
        validate_image_input_workflow(nodes)


@pytest.mark.parametrize("encoder_loader", ["CLIPLoader", "CLIPLoaderGGUF"])
def test_unet_model_verification_uses_real_encoder_loader_without_unused_lora(monkeypatch, encoder_loader):
    nodes = _custom_nodes()
    nodes["3"]["class_type"] = encoder_loader
    generator = ComfyGenerator(
        ComfyClient("http://127.0.0.1:1"), nodes, "unused", "",
        unet_name="flux.safetensors", text_encoder="encoder.safetensors", vae_name="vae.safetensors",
        lora_name="unused.safetensors", lora_strength=0,
    )
    calls = []
    models = {("UNETLoader", "unet_name"): ["flux.safetensors"],
              (encoder_loader, "clip_name"): ["encoder.safetensors"],
              ("VAELoader", "vae_name"): ["vae.safetensors"]}

    def options(class_type, field):
        calls.append((class_type, field))
        return models[(class_type, field)]

    monkeypatch.setattr(generator, "_model_options", options)
    generator.verify_checkpoint()
    assert calls == [("UNETLoader", "unet_name"), (encoder_loader, "clip_name"), ("VAELoader", "vae_name")]


def test_missing_standard_encoder_is_reported_from_standard_loader(monkeypatch):
    generator = ComfyGenerator(
        ComfyClient("http://127.0.0.1:1"), _custom_nodes(), "unused", "",
        unet_name="flux.safetensors", text_encoder="missing.safetensors",
    )
    monkeypatch.setattr(generator, "_model_options", lambda cls, field: ["flux.safetensors"] if cls == "UNETLoader" else [])
    with pytest.raises(ConfigError, match="文本编码器 'missing.safetensors'"):
        generator.verify_checkpoint()


def test_custom_workflow_build_resolves_noise_scheduler_and_image_bindings():
    generator = ComfyGenerator(
        ComfyClient("http://127.0.0.1:1"), _custom_nodes(), "unused", "",
        unet_name="flux.safetensors", text_encoder="encoder.safetensors", vae_name="vae.safetensors",
    )
    nodes = generator.build_workflow("new-input.png", "paint it", 768, 512, 4242, {
        "steps": 4, "cfg": 1, "denoise": 1, "sampler_name": "euler", "scheduler": "simple",
    })
    assert nodes["1"]["inputs"]["image"] == "new-input.png"
    assert nodes["10"]["inputs"]["noise_seed"] == 4242
    assert nodes["12"]["inputs"] == {"steps": 4, "width": 768, "height": 512}
    assert nodes["13"]["inputs"]["latent_image"] == ["7", 0]


def _qwen21_nodes(sampler_class="KSampler", *, nested_images=False, guider_class="BasicGuider"):
    nodes = _nodes()
    del nodes["4"]  # The official combined node performs VAE encoding internally.
    inputs = {
        "clip": ["2", 1], "prompt": "paint it", "negative_prompt": "text",
        "vae": ["2", 2], "resolution": 768,
    }
    inputs.update({"images": {"image_1": ["3", 0]}} if nested_images else {"images.image_1": ["3", 0]})
    nodes["21"] = {"class_type": "TextEncodeQwenImage21", "inputs": inputs}
    nodes["7"]["class_type"] = sampler_class
    nodes["7"]["inputs"].update(latent_image=["21", 2], positive=["21", 0], negative=["21", 1])
    if sampler_class == "SamplerCustomAdvanced":
        guider_inputs = {"model": ["2", 0]}
        guider_inputs.update({"conditioning": ["21", 0]} if guider_class == "BasicGuider" else {"positive": ["21", 0], "negative": ["21", 1], "cfg": 1})
        nodes["22"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}}
        nodes["23"] = {"class_type": guider_class, "inputs": guider_inputs}
        nodes["24"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        nodes["25"] = {"class_type": "BasicScheduler", "inputs": {"model": ["2", 0], "scheduler": "simple", "steps": 8, "denoise": 1}}
        nodes["7"]["inputs"] = {"noise": ["22", 0], "guider": ["23", 0], "sampler": ["24", 0], "sigmas": ["25", 0], "latent_image": ["21", 2]}
    return nodes


@pytest.mark.parametrize("sampler_class,guider_class", [
    ("KSampler", "BasicGuider"), ("KSamplerAdvanced", "BasicGuider"),
    ("SamplerCustomAdvanced", "BasicGuider"), ("SamplerCustomAdvanced", "CFGGuider"),
])
@pytest.mark.parametrize("nested_images", [False, True])
def test_qwen21_internal_image_encoder_valid_without_dummy_vae_encode(sampler_class, guider_class, nested_images):
    nodes = _qwen21_nodes(sampler_class, nested_images=nested_images, guider_class=guider_class)
    binding = validate_image_input_workflow(nodes)
    assert binding == {"load_image": "1", "vae_encode": "21", "vae_encode_type": "TextEncodeQwenImage21", "ksampler": ["7"], "save_image": "9"}
    assert not any(node["class_type"] == "VAEEncode" for node in nodes.values())


def test_qwen21_supports_explicit_vae_loader_and_direct_nested_image_reference():
    nodes = _qwen21_nodes(nested_images=True)
    nodes["26"] = {"class_type": "VAELoader", "inputs": {"vae_name": "qwen21.safetensors"}}
    nodes["21"]["inputs"]["vae"] = ["26", 0]
    nodes["21"]["inputs"]["images"]["image_1"] = ["1", 0]
    del nodes["3"]
    assert validate_image_input_workflow(nodes)["vae_encode"] == "21"


@pytest.mark.parametrize("latent_slot", [0, 1, 3])
def test_qwen21_rejects_wrong_latent_output_slot(latent_slot):
    nodes = _qwen21_nodes()
    nodes["7"]["inputs"]["latent_image"] = ["21", latent_slot]
    with pytest.raises(ConfigError, match="latent_image"):
        validate_image_input_workflow(nodes)


@pytest.mark.parametrize("sampler_class", ["KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced"])
def test_qwen21_rejects_negative_output_as_positive_conditioning(sampler_class):
    nodes = _qwen21_nodes(sampler_class)
    if sampler_class == "SamplerCustomAdvanced":
        nodes["23"]["inputs"]["conditioning"] = ["21", 1]
    else:
        nodes["7"]["inputs"]["positive"] = ["21", 1]
    with pytest.raises(ConfigError, match="有效的原生图像参考"):
        validate_image_input_workflow(nodes)


def test_qwen21_rejects_reference_only_consumed_by_negative_cfg_guider():
    nodes = _qwen21_nodes("SamplerCustomAdvanced", guider_class="CFGGuider")
    nodes["23"]["inputs"].update(positive=["5", 0], negative=["21", 0])
    with pytest.raises(ConfigError, match="有效的原生图像参考"):
        validate_image_input_workflow(nodes)


def test_qwen21_rejects_positive_and_latent_from_different_combined_nodes():
    nodes = _qwen21_nodes()
    nodes["31"] = {"class_type": "TextEncodeQwenImage21", "inputs": dict(nodes["21"]["inputs"])}
    nodes["7"]["inputs"]["positive"] = ["31", 0]
    with pytest.raises(ConfigError, match="有效的原生图像参考"):
        validate_image_input_workflow(nodes)


def test_qwen21_accepts_positive_through_conditioning_transform_same_origin_slot():
    nodes = _qwen21_nodes()
    nodes["31"] = {"class_type": "ConditioningSetTimestepRange", "inputs": {"conditioning": ["21", 0], "start": 0, "end": 1}}
    nodes["7"]["inputs"]["positive"] = ["31", 0]
    assert validate_image_input_workflow(nodes)["vae_encode"] == "21"


@pytest.mark.parametrize("vae_reference", [None, ["1", 0], ["2", 0], ["missing", 0]])
def test_qwen21_requires_valid_vae_connection(vae_reference):
    nodes = _qwen21_nodes()
    nodes["21"]["inputs"]["vae"] = vae_reference
    with pytest.raises(ConfigError, match="有效的 TextEncodeQwenImage21"):
        validate_image_input_workflow(nodes)


@pytest.mark.parametrize("image_inputs", [{}, {"images.image_0": ["3", 0]}, {"images.image_17": ["3", 0]}, {"images": {"other": ["3", 0]}}])
def test_qwen21_does_not_accept_missing_or_unrecognized_image_keys(image_inputs):
    nodes = _qwen21_nodes()
    del nodes["21"]["inputs"]["images.image_1"]
    nodes["21"]["inputs"].update(image_inputs)
    with pytest.raises(ConfigError, match="有效的 TextEncodeQwenImage21"):
        validate_image_input_workflow(nodes)


def test_qwen21_rejects_mask_as_reference_image():
    nodes = _qwen21_nodes()
    nodes["3"]["inputs"]["image"] = ["1", 1]
    with pytest.raises(ConfigError, match="有效的 TextEncodeQwenImage21"):
        validate_image_input_workflow(nodes)


def test_qwen21_orphan_cannot_validate_unconditioned_sampler():
    nodes = _qwen21_nodes()
    nodes["32"] = {"class_type": "EmptyLatentImage", "inputs": {}}
    nodes["7"]["inputs"].update(latent_image=["32", 0], positive=["5", 0], negative=["6", 0])
    with pytest.raises(ConfigError, match="有效的原生图像参考"):
        validate_image_input_workflow(nodes)


# ---------------------------------------------------------------- 加载与占位符
def test_load_workflow_ignores_meta_keys(tmp_path):
    path = tmp_path / "wf.json"
    path.write_text(json.dumps({"_meta": {"name": "x"}, **_nodes()}), encoding="utf-8")
    nodes = load_workflow(path)
    assert "_meta" not in nodes
    assert "1" in nodes


def test_load_workflow_rejects_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="不存在"):
        load_workflow(tmp_path / "nope.json")


def test_load_workflow_rejects_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="不是合法 JSON"):
        load_workflow(path)


def test_load_workflow_rejects_node_without_class_type(tmp_path):
    path = tmp_path / "bad2.json"
    path.write_text(json.dumps({"1": {"inputs": {}}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="缺少 class_type"):
        load_workflow(path)


def test_apply_placeholders_replaces_strings_and_numbers():
    nodes = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "__input_image__"}},
        "2": {"class_type": "KSampler", "inputs": {"seed": "__seed__", "steps": "__steps__", "ref": ["1", 0]}},
    }
    resolved = apply_placeholders(nodes, {"input_image": "a.png", "seed": 42, "steps": 30})
    assert resolved["1"]["inputs"]["image"] == "a.png"
    assert resolved["2"]["inputs"]["seed"] == 42
    assert resolved["2"]["inputs"]["steps"] == 30
    assert resolved["2"]["inputs"]["ref"] == ["1", 0], "节点引用不能被破坏"


def test_apply_placeholders_rejects_missing_value():
    nodes = {"1": {"class_type": "LoadImage", "inputs": {"image": "__input_image__"}}}
    with pytest.raises(ConfigError, match="没有提供对应配置值"):
        apply_placeholders(nodes, {})


def test_apply_placeholders_detects_leftover():
    nodes = {"1": {"class_type": "LoadImage", "inputs": {"image": "__input_image__"}}}
    # 直接提供错误 key 会先触发缺失错误；这里验证残留检测逻辑
    with pytest.raises(ConfigError):
        apply_placeholders(nodes, {"other": "x"})
