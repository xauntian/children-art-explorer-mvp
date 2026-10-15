"""共享测试夹具：临时项目、假 ComfyUI 生成器与假视觉检查器。

这些夹具只用于验证流程规则；真实画质与模型误判必须通过
tests/integration 的真实模型测试验证。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app import states
from app.checker import CheckResult
from app.config import AppConfig, StyleConfig, StyleRegistry


# ---------------------------------------------------------------- 配置
def make_app_config(tmp_path: Path, **overrides) -> AppConfig:
    """构造一份指向临时目录的最小可用配置。"""
    root = tmp_path
    for sub in ("configs", "runs", "prompts", "workflows"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    (root / "prompts" / "generate.txt").write_text(
        "style={style_name}\nfeatures={style_features}\nrules={preservation_rules}\n",
        encoding="utf-8",
    )
    (root / "prompts" / "check.txt").write_text(
        "style={style_name}\nfeatures={style_features}\nrules={preservation_rules}\n",
        encoding="utf-8",
    )
    (root / "workflows" / "image_edit.json").write_text(
        json.dumps(_img2img_workflow()), encoding="utf-8"
    )

    raw = {
        "config_version": "test",
        "comfy": {
            "base_url": "http://127.0.0.1:1",
            "workflow_path": "workflows/image_edit.json",
            "checkpoint": "test.safetensors",
            "controlnet": None,
            "connect_timeout_sec": 1,
            "request_timeout_sec": 1,
            "job_timeout_sec": 5,
            "poll_interval_sec": 0.01,
        },
        "checker": {
            "base_url": "http://127.0.0.1:2",
            "api": "chat",
            "model": "test-vlm",
            "connect_timeout_sec": 1,
            "request_timeout_sec": 1,
            "keep_alive": "1m",
            "options": {},
        },
        "capture": {
            "device_index": 0,
            "backend": "dshow",
            "width": 640,
            "height": 480,
            "warmup_frames": 1,
            "settle_ms": 0,
        },
        "preprocess": {
            "max_input_side": 128,
            "frame_detect": False,
            "border_trim_ratio": 0.0,
            "crop_ratios": [0, 0, 0, 0],
            "min_area_ratio": 0.15,
            "max_area_ratio": 0.98,
            "approx_epsilon_ratio": 0.02,
            "output_format": "png",
        },
        "generation": {
            "max_attempts": 2,
            "base_seed": 1000,
            "negative_prompt": "text",
            "samplers": {
                sid: {
                    "steps": 4,
                    "cfg": 7.0,
                    "denoise": 0.6,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                }
                for sid in ("fauvism", "cubism", "impressionism")
            },
            "controlnet_strength": 0.5,
            "seed_stride": 1,
        },
        "paths": {
            "runs_dir": "runs",
            "workflow_dir": "workflows",
            "prompt_dir": "prompts",
            "style_config": "configs/styles.json",
            "temp_suffix": ".tmp",
        },
        "print": {"print_enabled": False},
        "logging": {"level": "INFO"},
    }
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)

    (root / "configs" / "app.json").write_text(json.dumps(raw), encoding="utf-8")
    return AppConfig(raw, root)


def _img2img_workflow() -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "__input_image__"}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "__checkpoint__"}},
        "3": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["1", 0],
                "upscale_method": "area",
                "width": "__width__",
                "height": "__height__",
                "crop": "disabled",
            },
        },
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["2", 2]}},
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "__positive_prompt__", "clip": ["2", 1]},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "__negative_prompt__", "clip": ["2", 1]},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "seed": "__seed__",
                "steps": "__steps__",
                "cfg": "__cfg__",
                "sampler_name": "__sampler_name__",
                "scheduler": "__scheduler__",
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["4", 0],
                "denoise": "__denoise__",
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 2]}},
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "__filename_prefix__"},
        },
    }


def make_registry(*, enabled: dict[str, bool] | None = None) -> StyleRegistry:
    enabled = enabled or {}
    specs = [
        ("fauvism", "Fauvism", "bold non-natural colour, flat blocks"),
        ("cubism", "Cubism", "overlapping planes, geometric fragmentation"),
        ("impressionism", "Impressionism", "short broken brush strokes, light"),
    ]
    styles = [
        StyleConfig(
            {
                "id": sid,
                "name": name,
                "enabled": enabled.get(sid, True),
                "style_features": features,
                "preservation_rules": "keep main subjects and layout",
            },
            index,
        )
        for index, (sid, name, features) in enumerate(specs)
    ]
    return StyleRegistry(styles, "test-1.0.0")


# ---------------------------------------------------------------- 图像
def solid_png(width: int = 96, height: int = 96, value: int = 200) -> bytes:
    import cv2

    image = np.full((height, width, 3), value, dtype=np.uint8)
    image[10:40, 10:40] = (0, 0, 255)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def marker_drawing_png(width: int = 200, height: int = 160) -> bytes:
    """合成一张「白色画纸 + 粗马克笔线条」的图。"""
    import cv2

    image = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (110, 110), (0, 0, 200), 6)
    cv2.circle(image, (150, 90), 30, (200, 0, 0), 6)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


# ---------------------------------------------------------------- 假适配器
class FakeGenerator:
    """假的 ComfyUI 生成器：产生可解码的候选图，并记录调用参数。"""

    def __init__(self, fail_on: set[int] | None = None) -> None:
        self.calls: list[dict] = []
        self.fail_on = fail_on or set()
        self.workflow_nodes = _img2img_workflow()
        self.binding = {
            "load_image": "1",
            "vae_encode": "4",
            "ksampler": ["7"],
            "save_image": "9",
        }

    def generate(self, **kwargs):
        attempt = len(self.calls) + 1
        self.calls.append(kwargs)
        if attempt in self.fail_on:
            from app.errors import GenerationError

            raise GenerationError("模拟生成失败", stage=states.GENERATING)
        return solid_png(value=120 + attempt * 10), f"prompt-{attempt}"


class FakeChecker:
    """假的视觉检查器：按预设序列返回结果。"""

    def __init__(
        self, payloads: list | None = None, technical_error: Exception | None = None
    ) -> None:
        self.payloads = list(payloads or [])
        self.calls: list[dict] = []
        self.technical_error = technical_error

    def _verdict(self, kwargs) -> CheckResult:
        self.calls.append(kwargs)
        if self.technical_error is not None:
            raise self.technical_error
        payload = (
            self.payloads.pop(0)
            if self.payloads
            else {"style_match": "match", "content_preserved": True, "reason": "默认"}
        )
        return CheckResult(
            style_match=payload["style_match"],
            content_preserved=payload["content_preserved"],
            reason=payload.get("reason", ""),
        )

    def check(self, **kwargs) -> CheckResult:
        return self._verdict(kwargs)

    def check_pair(self, **kwargs) -> CheckResult:
        """与真实 VisionChecker 接口一致的入口，供 Pipeline 调用。"""
        return self._verdict(kwargs)
