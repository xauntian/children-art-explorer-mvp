"""配置加载与启动前校验。

只使用标准库。加载 configs/app.json 与 configs/styles.json，
并在进入流程前校验关键字段，避免把非法配置带入运行阶段。
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .errors import ConfigError

VALID_STYLE_IDS_FIELD = "id"


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"配置文件无法读取: {path} ({exc})") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"配置文件不是合法 JSON: {path} ({exc})") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是 JSON 对象: {path}")
    return data


def _require(obj: dict, key: str, path: str) -> Any:
    if key not in obj:
        raise ConfigError(f"配置缺少字段 {path}.{key}")
    return obj[key]


def _require_str(obj: dict, key: str, path: str, allow_none: bool = False) -> str | None:
    value = _require(obj, key, path)
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"配置字段 {path}.{key} 必须是非空字符串")
    return value


def _require_number(obj: dict, key: str, path: str, allow_none: bool = False) -> float | None:
    value = _require(obj, key, path)
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"配置字段 {path}.{key} 必须是数字")
    return value


def _require_int(obj: dict, key: str, path: str, allow_none: bool = False) -> int | None:
    value = _require(obj, key, path)
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"配置字段 {path}.{key} 必须是整数")
    return value


def _require_bool(obj: dict, key: str, path: str) -> bool:
    value = _require(obj, key, path)
    if not isinstance(value, bool):
        raise ConfigError(f"配置字段 {path}.{key} 必须是布尔值")
    return value


def _require_dict(obj: dict, key: str, path: str) -> dict:
    value = _require(obj, key, path)
    if not isinstance(value, dict):
        raise ConfigError(f"配置字段 {path}.{key} 必须是 JSON 对象")
    return value


class StyleConfig:
    """单条画派配置。"""

    def __init__(self, raw: dict, index: int) -> None:
        path = f"styles[{index}]"
        if not isinstance(raw, dict):
            raise ConfigError(f"配置 {path} 必须是 JSON 对象")

        style_id = _require_str(raw, "id", path)
        if not style_id.replace("_", "").replace("-", "").isalnum():
            raise ConfigError(f"配置字段 {path}.id 只能包含字母、数字、下划线和连字符")

        self.id: str = style_id
        self.name: str = _require_str(raw, "name", path)
        self.name_zh: str | None = raw.get("name_zh")

        enabled = _require(raw, "enabled", path)
        if not isinstance(enabled, bool):
            raise ConfigError(f"配置字段 {path}.enabled 必须是布尔值")
        self.enabled: bool = enabled

        self.style_features: str = _require_str(raw, "style_features", path)
        self.preservation_rules: str = _require_str(raw, "preservation_rules", path)
        self.raw = raw

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "name_zh": self.name_zh,
            "enabled": self.enabled,
            "style_features": self.style_features,
            "preservation_rules": self.preservation_rules,
        }


class StyleRegistry:
    """画派注册表：校验列表、字段与启用状态。"""

    def __init__(self, styles: list[StyleConfig], prompt_version: str | None) -> None:
        self.styles = styles
        self.prompt_version = prompt_version
        seen: set[str] = set()
        for style in styles:
            if style.id in seen:
                raise ConfigError(f"画派 id 重复: {style.id}")
            seen.add(style.id)

    @property
    def enabled(self) -> list[StyleConfig]:
        return [s for s in self.styles if s.enabled]

    def get(self, style_id: str) -> StyleConfig:
        for style in self.styles:
            if style.id == style_id:
                return style
        raise ConfigError(f"未知画派 id: {style_id}")

    def require_enabled(self) -> list[StyleConfig]:
        """返回启用画派；没有启用画派时抛出配置错误。"""
        enabled = self.enabled
        if not enabled:
            raise ConfigError(
                "没有启用任何画派，请检查 configs/styles.json 中各画派的 enabled 字段"
            )
        return enabled

    def resolve(self, style_id: str) -> StyleConfig:
        """显式指定画派（测试入口覆盖用），必须是已启用的画派。"""
        style = self.get(style_id)
        if not style.enabled:
            raise ConfigError(f"画派 {style_id} 未启用，不能用于生成")
        return style


class AppConfig:
    """应用配置访问器。"""

    def __init__(self, raw: dict, project_root: Path) -> None:
        self.raw = raw
        self.project_root = project_root
        self.config_version = raw.get("config_version", "unknown")

        comfy = _require_dict(raw, "comfy", "app")
        self.comfy_base_url: str = str(_require_str(comfy, "base_url", "app.comfy")).rstrip("/")
        self.comfy_workflow_path: Path = self._resolve(
            _require_str(comfy, "workflow_path", "app.comfy")
        )
        self.comfy_checkpoint: str = str(_require_str(comfy, "checkpoint", "app.comfy"))
        controlnet = comfy.get("controlnet")
        if controlnet is not None and not isinstance(controlnet, str):
            raise ConfigError("配置字段 app.comfy.controlnet 必须是字符串或 null")
        self.comfy_controlnet: str | None = controlnet

        # 编辑模型（Qwen-Image-Edit 等）的可选模型字段
        for key in ("unet_name", "text_encoder", "vae_name", "lora_name"):
            value = comfy.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"配置字段 app.comfy.{key} 必须是非空字符串或 null")
        self.comfy_unet_name: str | None = comfy.get("unet_name")
        self.comfy_text_encoder: str | None = comfy.get("text_encoder")
        self.comfy_vae_name: str | None = comfy.get("vae_name")
        self.comfy_lora_name: str | None = comfy.get("lora_name")
        lora_strength = comfy.get("lora_strength", 1.0)
        if isinstance(lora_strength, bool) or not isinstance(lora_strength, (int, float)):
            raise ConfigError("配置字段 app.comfy.lora_strength 必须是数字")
        self.comfy_lora_strength: float = float(lora_strength)
        shift = comfy.get("shift", 3.0)
        if isinstance(shift, bool) or not isinstance(shift, (int, float)):
            raise ConfigError("配置字段 app.comfy.shift 必须是数字")
        self.comfy_shift: float = float(shift)

        self.comfy_connect_timeout: float = float(
            _require_number(comfy, "connect_timeout_sec", "app.comfy")
        )
        self.comfy_request_timeout: float = float(
            _require_number(comfy, "request_timeout_sec", "app.comfy")
        )
        self.comfy_job_timeout: float = float(
            _require_number(comfy, "job_timeout_sec", "app.comfy")
        )
        self.comfy_poll_interval: float = float(
            _require_number(comfy, "poll_interval_sec", "app.comfy")
        )

        checker = _require_dict(raw, "checker", "app")
        self.checker_base_url: str = str(_require_str(checker, "base_url", "app.checker")).rstrip("/")
        self.checker_api: str = str(_require_str(checker, "api", "app.checker"))
        if self.checker_api not in ("chat", "generate"):
            raise ConfigError("配置字段 app.checker.api 只能是 'chat' 或 'generate'")
        self.checker_model: str = str(_require_str(checker, "model", "app.checker"))
        # 检查方式：
        #   comparison —— 单张 A/B 图检查内容与渲染变化，再粗判画派
        #   two_stage —— 先分别描述原图与候选图，再依据描述判定
        #   pairwise  —— 一次提交两张图让模型直接比较（要求模型真正支持多图）
        self.checker_mode: str = str(checker.get("mode", "two_stage"))
        if self.checker_mode not in ("two_stage", "pairwise", "comparison"):
            raise ConfigError("配置字段 app.checker.mode 只能是 'two_stage'、'pairwise' 或 'comparison'")
        control = checker.get("two_stage_control", False)
        if not isinstance(control, bool):
            raise ConfigError("配置字段 app.checker.two_stage_control 必须是布尔值")
        self.checker_control: bool = control
        self.checker_connect_timeout: float = float(
            _require_number(checker, "connect_timeout_sec", "app.checker")
        )
        self.checker_request_timeout: float = float(
            _require_number(checker, "request_timeout_sec", "app.checker")
        )
        self.checker_keep_alive: str | None = checker.get("keep_alive")
        options = checker.get("options", {})
        if not isinstance(options, dict):
            raise ConfigError("配置字段 app.checker.options 必须是 JSON 对象")
        self.checker_options: dict = options

        capture = _require_dict(raw, "capture", "app")
        self.capture_device_index: int = int(_require_int(capture, "device_index", "app.capture"))
        self.capture_backend: str = str(_require_str(capture, "backend", "app.capture"))
        self.capture_width: int = int(_require_int(capture, "width", "app.capture"))
        self.capture_height: int = int(_require_int(capture, "height", "app.capture"))
        self.capture_warmup_frames: int = int(
            _require_int(capture, "warmup_frames", "app.capture")
        )
        self.capture_settle_ms: int = int(_require_int(capture, "settle_ms", "app.capture"))

        pre = _require_dict(raw, "preprocess", "app")
        self.pre_max_input_side: int = int(_require_int(pre, "max_input_side", "app.preprocess"))
        if self.pre_max_input_side < 64:
            raise ConfigError("配置字段 app.preprocess.max_input_side 必须 >= 64")
        self.pre_frame_detect: bool = _require_bool(pre, "frame_detect", "app.preprocess")
        self.pre_border_trim_ratio: float = float(
            _require_number(pre, "border_trim_ratio", "app.preprocess")
        )
        ratios = _require(pre, "crop_ratios", "app.preprocess")
        if not isinstance(ratios, list) or len(ratios) != 4:
            raise ConfigError("配置字段 app.preprocess.crop_ratios 必须是 4 个数字的数组")
        for r in ratios:
            if isinstance(r, bool) or not isinstance(r, (int, float)):
                raise ConfigError("配置字段 app.preprocess.crop_ratios 的元素必须是数字")
        self.pre_crop_ratios: tuple[float, float, float, float] = tuple(float(r) for r in ratios)  # type: ignore[assignment]
        self.pre_min_area_ratio: float = float(
            _require_number(pre, "min_area_ratio", "app.preprocess")
        )
        self.pre_max_area_ratio: float = float(
            _require_number(pre, "max_area_ratio", "app.preprocess")
        )
        self.pre_approx_epsilon_ratio: float = float(
            _require_number(pre, "approx_epsilon_ratio", "app.preprocess")
        )
        self.pre_output_format: str = str(_require_str(pre, "output_format", "app.preprocess"))
        if self.pre_output_format != "png":
            raise ConfigError("当前仅支持 app.preprocess.output_format = 'png'")

        gen = _require_dict(raw, "generation", "app")
        self.max_attempts: int = int(_require_int(gen, "max_attempts", "app.generation"))
        if self.max_attempts != 2:
            raise ConfigError("本阶段要求 app.generation.max_attempts = 2")
        base_seed = gen.get("base_seed")
        if base_seed is not None and (isinstance(base_seed, bool) or not isinstance(base_seed, int)):
            raise ConfigError("配置字段 app.generation.base_seed 必须是整数或 null")
        self.base_seed: int | None = base_seed
        self.negative_prompt: str = str(_require_str(gen, "negative_prompt", "app.generation"))
        samplers = _require_dict(gen, "samplers", "app.generation")
        self.samplers: dict[str, dict] = {}
        for style_id, params in samplers.items():
            if not isinstance(params, dict):
                raise ConfigError(f"配置字段 app.generation.samplers.{style_id} 必须是 JSON 对象")
            for key in ("steps", "cfg", "denoise", "sampler_name", "scheduler"):
                if key not in params:
                    raise ConfigError(f"配置字段 app.generation.samplers.{style_id} 缺少 {key}")
            if not (0.0 < float(params["denoise"]) <= 1.0):
                raise ConfigError(
                    f"配置字段 app.generation.samplers.{style_id}.denoise 必须在 (0, 1] 区间"
                )
            self.samplers[style_id] = dict(params)
        self.controlnet_strength: float = float(
            _require_number(gen, "controlnet_strength", "app.generation")
        )
        self.seed_stride: int = int(_require_int(gen, "seed_stride", "app.generation"))

        paths = _require_dict(raw, "paths", "app")
        self.runs_dir: Path = self._resolve(_require_str(paths, "runs_dir", "app.paths"), ensure=True)
        self.workflow_dir: Path = self._resolve(_require_str(paths, "workflow_dir", "app.paths"))
        self.prompt_dir: Path = self._resolve(_require_str(paths, "prompt_dir", "app.paths"))
        self.style_config_path: Path = self._resolve(
            _require_str(paths, "style_config", "app.paths")
        )
        self.temp_suffix: str = str(_require_str(paths, "temp_suffix", "app.paths"))
        self.generate_prompt_path = self.prompt_dir / "generate.txt"
        self.check_prompt_path = self.prompt_dir / "check.txt"

        print_cfg = _require_dict(raw, "print", "app")
        self.print_enabled: bool = _require_bool(print_cfg, "print_enabled", "app.print")

        logging_cfg = raw.get("logging", {})
        if not isinstance(logging_cfg, dict):
            raise ConfigError("配置字段 app.logging 必须是 JSON 对象")
        self.log_level: str = str(logging_cfg.get("level", "INFO")).upper()

    def _resolve(self, rel: str, ensure: bool = False) -> Path:
        p = Path(rel)
        if not p.is_absolute():
            p = self.project_root / p
        p = p.resolve()
        if ensure:
            p.mkdir(parents=True, exist_ok=True)
        return p

    def sampler_for(self, style_id: str) -> dict:
        if style_id not in self.samplers:
            raise ConfigError(f"缺少画派 {style_id} 的生成参数，请检查 app.generation.samplers")
        return self.samplers[style_id]

    def for_artwork(self) -> AppConfig:
        """完整画作导入：保留四周内容，并隔离本次覆盖与原始配置。"""
        raw = deepcopy(self.raw)
        raw["preprocess"].update(
            frame_detect=False, border_trim_ratio=0.0, crop_ratios=[0.0, 0.0, 0.0, 0.0]
        )
        return AppConfig(raw, self.project_root)

    def require_prompt_files(self) -> None:
        for path in (self.generate_prompt_path, self.check_prompt_path):
            if not path.is_file():
                raise ConfigError(f"Prompt 文件不存在: {path}")
        if not self.comfy_workflow_path.is_file():
            raise ConfigError(f"ComfyUI 工作流文件不存在: {self.comfy_workflow_path}")

    def load_styles(self) -> StyleRegistry:
        raw = _read_json(self.style_config_path)
        styles_raw = _require(raw, "styles", "styles")
        if not isinstance(styles_raw, list) or not styles_raw:
            raise ConfigError("配置 styles.styles 必须是非空数组")
        styles = [StyleConfig(item, i) for i, item in enumerate(styles_raw)]
        return StyleRegistry(styles, raw.get("prompt_version"))

    def load_prompts(self) -> tuple[str, str]:
        self.require_prompt_files()
        return (
            self.generate_prompt_path.read_text(encoding="utf-8"),
            self.check_prompt_path.read_text(encoding="utf-8"),
        )


def project_root_from(start: Path | None = None) -> Path:
    """从给定位置向上查找包含 configs/app.json 的目录。"""
    here = (start or Path(__file__).resolve()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "configs" / "app.json").is_file():
            return candidate
    raise ConfigError("找不到项目根目录（未发现 configs/app.json）")


def load_app_config(project_root: Path | None = None, config_path: Path | None = None) -> AppConfig:
    root = project_root or project_root_from()
    path = config_path or (root / "configs" / "app.json")
    raw = _read_json(path)
    return AppConfig(raw, root)
