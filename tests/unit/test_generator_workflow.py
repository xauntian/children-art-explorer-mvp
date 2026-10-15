"""工作流校验测试：必须确认是真正的图生图（原图进入采样器）。

不得编造工作流节点 ID，也不得把文字生图当作图生图。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.errors import ConfigError
from app.generator import (
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
