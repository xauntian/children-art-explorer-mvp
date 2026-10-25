"""手机入口模型预热；合成输入不创建用户任务，也不消耗随机画派。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import load_app_config
from .errors import ConfigError
from .generator import load_workflow
from .main import App
from .preprocess import a4_dimensions, decode_image_bytes, encode_png
from .storage import atomic_write_json, utc_now_iso


def warmup_generation(project_root: Path) -> dict | None:
    """LCM 预热首画派，FLUX.2 / Qwen 2.1 预热每个启用画派。

    调用者须在 PhoneRunLock 内、采集与任务计时开始前调用。合成图与各画派
    的静态生成提示用于完整生成，避免首次随机选择某画派时才加载其文本条件。
    不支持的工作流返回 None，不调用服务；报告中的时间不计入 P 任务耗时。
    """
    started = time.perf_counter()
    config = load_app_config(project_root)
    workflow = load_workflow(config.comfy_workflow_path)
    classes = {node["class_type"] for node in workflow.values()}
    is_qwen21 = {"TextEncodeQwenImage21", "QwenImage21Cache"}.issubset(classes) and bool(
        config.comfy_unet_name
    )
    is_flux2 = {"Flux2Scheduler", "ReferenceLatent"}.issubset(classes) and bool(
        config.comfy_unet_name
    )
    is_lcm = bool(config.comfy_lora_name) and any(
        node["class_type"] == "ModelSamplingDiscrete"
        and node.get("inputs", {}).get("sampling") == "lcm"
        for node in workflow.values()
    )
    if not (is_qwen21 or is_flux2 or is_lcm):
        return None

    route = "qwen_image21" if is_qwen21 else "flux2" if is_flux2 else "lcm"
    uses_unet = is_qwen21 or is_flux2
    report_path = config.runs_dir / "runtime" / "generation-warmup.json"
    if config.pre_paper_format == "a4":
        (width, height), _ = a4_dimensions(config.pre_output_short_side or 720, landscape=True)
    elif uses_unet:
        alignment = 32 if is_qwen21 else 16
        width = max(alignment, config.pre_max_input_side // alignment * alignment)
        height = max(alignment, round(width * 2 / 3 / alignment) * alignment)
    else:
        width = min(768, config.pre_max_input_side) // 8 * 8
        height = max(8, round(width * 2 / 3) // 8 * 8)
    # 每次启动及每个画派均使用不同采样种子，避免只命中历史输出缓存。
    seed = time.time_ns() % (2**31)
    report = {
        "status": "running",
        "started_at": utc_now_iso(),
        "purpose": "startup_model_warmup",
        "input_source": "synthetic_solid_colour",
        "creates_user_job": False,
        "route": route,
        "checkpoint": None if uses_unet else config.comfy_checkpoint,
        "unet_name": config.comfy_unet_name,
        "text_encoder": config.comfy_text_encoder,
        "vae_name": config.comfy_vae_name,
        "lora_name": config.comfy_lora_name,
        "workflow_file": str(config.comfy_workflow_path),
        "width": width,
        "height": height,
        "seed": seed,
        "style_prewarms": [],
    }
    atomic_write_json(report_path, report)
    try:
        app = App(config=config)
        enabled = app.registry.require_enabled()
        styles = enabled if uses_unet else enabled[:1]
        report["planned_styles"] = [style.id for style in styles]
        app.generator.verify_checkpoint()
        image = np.full((height, width, 3), 192, dtype=np.uint8)
        input_bytes = encode_png(image)
        for index, style in enumerate(styles):
            style_started = time.perf_counter()
            params = dict(config.sampler_for(style.id))
            style_seed = (seed + index) % (2**31)
            item = {
                "style_id": style.id,
                "seed": style_seed,
                "params": params,
                "status": "running",
                "started_at": utc_now_iso(),
            }
            report["style_prewarms"].append(item)
            if index == 0:
                # Preserve the original single-style report fields for callers.
                report.update(style_id=style.id, params=params)
            atomic_write_json(report_path, report)
            try:
                output, prompt_id = app.generator.generate(
                    input_bytes=input_bytes,
                    input_filename="generation-warmup-synthetic.png",
                    positive_prompt=app.pipeline.build_generate_prompt(style),
                    width=width,
                    height=height,
                    seed=style_seed,
                    params=params,
                    filename_prefix=(
                        f"runtime/generation-warmup/{style.id}"
                        if uses_unet
                        else "runtime/generation-warmup"
                    ),
                )
                item["prompt_id"] = prompt_id
                decoded = decode_image_bytes(output, source_name="generation-warmup")
                if decoded.shape[:2] != (height, width):
                    raise ConfigError("模型预热输出尺寸与请求不一致")
                item.update(status="completed", output_size=[decoded.shape[1], decoded.shape[0]])
                if index == 0:
                    report.update(prompt_id=prompt_id, output_size=list(item["output_size"]))
            except Exception as exc:
                item.update(
                    status="failed", error={"type": type(exc).__name__, "message": str(exc)}
                )
                raise
            finally:
                item.update(
                    finished_at=utc_now_iso(),
                    total_sec=round(time.perf_counter() - style_started, 3),
                )
                atomic_write_json(report_path, report)
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise ConfigError(f"模型预热失败，尚未进入可拍摄状态: {exc}") from exc
    finally:
        report.update(
            finished_at=utc_now_iso(),
            total_sec=round(time.perf_counter() - started, 3),
        )
        atomic_write_json(report_path, report)
    return report
