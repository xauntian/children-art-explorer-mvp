"""真实本地服务集成测试。

这些测试需要真实的 ComfyUI 与本地视觉模型服务。
仅运行规则测试时请显式指定 tests/unit。
缺失服务时标记为 skip，不计为通过。

运行：
    python -m pytest tests/integration -v -m integration
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.checker import VisionChecker
from app.config import load_app_config
from app.errors import PipelineError, ServiceUnavailableError
from app.generator import (
    ComfyClient,
    ComfyGenerator,
    load_workflow,
    validate_image_input_workflow,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def config():
    return load_app_config(ROOT)


# ---------------------------------------------------------------- 视觉模型
def test_vision_service_is_reachable_and_model_installed(config):
    checker = VisionChecker(
        base_url=config.checker_base_url,
        model=config.checker_model,
        connect_timeout=config.checker_connect_timeout,
        request_timeout=config.checker_request_timeout,
    )
    try:
        models = checker.list_models()
    except PipelineError as exc:
        pytest.skip(f"本地视觉模型服务不可用: {exc.message}")
    assert models, "视觉模型服务没有返回任何已安装模型"
    checker.verify_model()


def test_vision_checker_uses_configured_mode(config):
    """真实服务协议检查；返回合法结构不代表画派或内容质量验收通过。"""
    import cv2
    import numpy as np

    checker = VisionChecker(
        base_url=config.checker_base_url,
        model=config.checker_model,
        api=config.checker_api,
        connect_timeout=config.checker_connect_timeout,
        request_timeout=config.checker_request_timeout,
        keep_alive=config.checker_keep_alive,
        options=config.checker_options,
        mode=config.checker_mode,
        control=config.checker_control,
    )
    try:
        checker.verify_model()
    except ServiceUnavailableError as exc:
        pytest.skip(f"本地视觉模型不可用: {exc.message}")

    # 原图：白纸 + 粗马克笔方块与圆
    original = np.full((256, 320, 3), 245, dtype=np.uint8)
    cv2.rectangle(original, (40, 40), (150, 150), (0, 0, 200), 10)
    cv2.circle(original, (230, 120), 45, (200, 0, 0), 10)
    ok, buf = cv2.imencode(".png", original)
    assert ok
    original_bytes = buf.tobytes()

    # 确定性合成候选：保留构图，改变颜色并添加固定短笔触。
    candidate = np.full((256, 320, 3), 250, dtype=np.uint8)
    cv2.rectangle(candidate, (40, 40), (150, 150), (60, 180, 40), -1)
    cv2.circle(candidate, (230, 120), 45, (40, 40, 220), -1)
    for y in range(55, 140, 15):
        for x in range(55, 140, 15):
            cv2.line(candidate, (x, y), (x + 7, y + 3), (40, 210, 100), 3)
    for y in range(90, 151, 15):
        for x in range(200, 261, 15):
            if (x - 230) ** 2 + (y - 120) ** 2 < 30**2:
                cv2.line(candidate, (x, y), (x + 7, y + 3), (100, 70, 250), 3)
    ok, buf = cv2.imencode(".png", candidate)
    assert ok
    candidate_bytes = buf.tobytes()

    style = config.load_styles().resolve("fauvism")
    _, check_template = config.load_prompts()
    prompt = check_template
    for key, value in {
        "style_name": style.name,
        "style_features": style.style_features,
        "preservation_rules": style.preservation_rules,
    }.items():
        prompt = prompt.replace("{" + key + "}", value)

    result = checker.check_pair(
        original_bytes=original_bytes,
        candidate_bytes=candidate_bytes,
        prompt=prompt,
        style_name=style.name,
        style_features=style.style_features,
        preservation_rules=style.preservation_rules,
    )
    assert result.style_match in ("match", "mismatch", "uncertain")
    assert result.content_preserved in (True, False, None)
    assert isinstance(result.reason, str)
    assert result.raw_text, "应保留模型原始响应文本"
    assert result.extra["mode"] == config.checker_mode


# ---------------------------------------------------------------- ComfyUI
def test_comfyui_is_reachable(config):
    client = ComfyClient(
        base_url=config.comfy_base_url,
        connect_timeout=config.comfy_connect_timeout,
        request_timeout=config.comfy_request_timeout,
    )
    try:
        stats = client.system_stats()
    except PipelineError as exc:
        pytest.skip(f"ComfyUI 不可用: {exc.message}")
    assert stats.get("devices"), "ComfyUI 未报告任何计算设备"


def test_configured_workflow_models_exist_in_comfyui(config):
    client = ComfyClient(
        base_url=config.comfy_base_url,
        connect_timeout=config.comfy_connect_timeout,
        request_timeout=config.comfy_request_timeout,
    )
    try:
        client.system_stats()
    except PipelineError as exc:
        pytest.skip(f"ComfyUI 不可用: {exc.message}")

    generator = ComfyGenerator(
        client=client,
        workflow_nodes=load_workflow(config.comfy_workflow_path),
        checkpoint=config.comfy_checkpoint,
        negative_prompt=config.negative_prompt,
        controlnet=config.comfy_controlnet,
        controlnet_strength=config.controlnet_strength,
        unet_name=config.comfy_unet_name,
        text_encoder=config.comfy_text_encoder,
        vae_name=config.comfy_vae_name,
        shift=config.comfy_shift,
        lora_name=config.comfy_lora_name,
        lora_strength=config.comfy_lora_strength,
    )
    # verify_checkpoint 根据实际工作流选择 checkpoint 或 UNet 路径；
    # UNet 路径还检查配置的编码器、VAE 和 LoRA，缺失模型应失败而非 skip。
    generator.verify_checkpoint()


def test_real_workflow_is_img2img(config):
    nodes = load_workflow(config.comfy_workflow_path)
    binding = validate_image_input_workflow(nodes)
    assert binding["load_image"] and binding["vae_encode"] and binding["save_image"]


def test_controlnet_file_present(config):
    if not config.comfy_controlnet:
        pytest.skip("未配置 ControlNet")
    client = ComfyClient(
        base_url=config.comfy_base_url,
        connect_timeout=config.comfy_connect_timeout,
        request_timeout=config.comfy_request_timeout,
    )
    try:
        response = client.session.get(
            f"{client.base_url}/object_info/ControlNetLoader",
            timeout=client.request_timeout,
        )
        options = response.json()["ControlNetLoader"]["input"]["required"]["control_net_name"][0]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"无法读取 ControlNet 列表: {exc}")
    assert config.comfy_controlnet in options, (
        f"缺少 ControlNet {config.comfy_controlnet}，当前可用: {options}"
    )
