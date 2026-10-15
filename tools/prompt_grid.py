"""生成配置网格实验：对同一张输入，比较不同 Prompt 与采样参数的实际输出。

输出每组的图像统计（颜色饱和度、色彩方差、边缘密度）与检查模型判断，
用于区分「输出确实没有风格特征」与「检查模型过严」。

用法：
    python -m tools.prompt_grid --photo samples/debug/debug_01_sun_house.jpg --style fauvism
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.checker import VisionChecker  # noqa: E402
from app.config import load_app_config  # noqa: E402
from app.generator import ComfyClient, ComfyGenerator, load_workflow  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.preprocess import decode_image_bytes, encode_png, prepare_model_input  # noqa: E402


def short_prompt(style) -> str:
    """指令式短 Prompt：命名风格 + 描述特征 + 明确必须保留的内容。"""
    return (
        f"Repaint this child's drawing in {style.name} style: {style.style_features}. "
        f"Keep {style.preservation_rules}. "
        "Keep the exact same number of subjects, their positions, relative sizes and "
        "the overall composition unchanged. Do not add new objects or text."
    )


def image_stats(image_bytes: bytes) -> dict:
    image = decode_image_bytes(image_bytes, "<candidate>")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = float(hsv[:, :, 1].mean())
    value = float(hsv[:, :, 2].mean())
    # 色彩丰富度：色相直方图的熵
    hue_hist = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
    hue_hist = hue_hist / max(1.0, hue_hist.sum())
    hue_entropy = float(-(hue_hist[hue_hist > 0] * np.log2(hue_hist[hue_hist > 0])).sum())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float((edges > 0).mean())
    # 局部纹理强度：拉普拉斯方差
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # 非自然色指标：高饱和像素占比
    vivid_ratio = float((hsv[:, :, 1] > 120).mean())
    return {
        "sat_mean": round(saturation, 1),
        "val_mean": round(value, 1),
        "hue_entropy": round(hue_entropy, 2),
        "edge_density": round(edge_density, 4),
        "texture_var": round(texture, 1),
        "vivid_ratio": round(vivid_ratio, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prompt 与参数网格实验")
    parser.add_argument("--photo", required=True)
    parser.add_argument("--style", default="fauvism")
    parser.add_argument("--input-side", type=int, default=None)
    parser.add_argument("--out", default="samples/_grid")
    args = parser.parse_args(argv)

    config = load_app_config(ROOT)
    registry = config.load_styles()
    style = registry.resolve(args.style)
    generate_template, _ = config.load_prompts()

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.generate_prompt_template = generate_template
    pipeline.check_prompt_template = config.load_prompts()[1]
    long_prompt = pipeline.build_generate_prompt(style)
    short = short_prompt(style)

    photo = Path(args.photo)
    input_side = args.input_side or config.pre_max_input_side
    result, input_bytes = prepare_model_input(
        photo,
        max_input_side=input_side,
        frame_detect=config.pre_frame_detect,
        border_trim_ratio=config.pre_border_trim_ratio,
        crop_ratios=config.pre_crop_ratios,
        min_area_ratio=config.pre_min_area_ratio,
        max_area_ratio=config.pre_max_area_ratio,
        approx_epsilon_ratio=config.pre_approx_epsilon_ratio,
    )
    width, height = result.output_size
    original_bytes = encode_png(result.image)

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
    check_prompt = pipeline.build_check_prompt(style)

    client = ComfyClient(
        base_url=config.comfy_base_url,
        connect_timeout=config.comfy_connect_timeout,
        request_timeout=config.comfy_request_timeout,
        job_timeout=config.comfy_job_timeout,
        poll_interval=config.comfy_poll_interval,
    )

    variants = [
        ("long_s4_lora", long_prompt, 4, True),
        ("short_s4_lora", short, 4, True),
        ("short_s8_lora", short, 8, True),
        ("short_s20_nolora", short, 20, False),
    ]

    print(f"输入: {photo.name} -> {width}x{height}")
    print(f"画派: {style.id}")
    print(f"原图统计: {json.dumps(image_stats(original_bytes), ensure_ascii=False)}")
    print("")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for label, prompt, steps, use_lora in variants:
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
            lora_name=config.comfy_lora_name if use_lora else None,
            lora_strength=config.comfy_lora_strength,
        )
        params = dict(config.sampler_for(style.id))
        params["steps"] = steps

        print(f"=== {label} (steps={steps}, lora={use_lora}) ===")
        started = time.monotonic()
        try:
            image_bytes, _ = generator.generate(
                input_bytes=input_bytes,
                input_filename=f"grid-{photo.stem}.png",
                positive_prompt=prompt,
                width=width,
                height=height,
                seed=12345,
                params=params,
                filename_prefix=f"grid/{photo.stem}_{label}",
                job_timeout=config.comfy_job_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [生成失败] {type(exc).__name__}: {str(exc)[:200]}")
            rows.append({"label": label, "error": str(exc)[:200]})
            continue
        gen_sec = time.monotonic() - started

        path = out_dir / f"{photo.stem}_{style.id}_{label}.png"
        path.write_bytes(image_bytes)
        stats = image_stats(image_bytes)

        check_started = time.monotonic()
        try:
            verdict = checker.check_pair(
                original_bytes=original_bytes,
                candidate_bytes=image_bytes,
                prompt=check_prompt,
                style_name=style.name,
                style_features=style.style_features,
                preservation_rules=style.preservation_rules,
            )
            verdict_dict = {
                "style_match": verdict.style_match,
                "content_preserved": verdict.content_preserved,
                "passed": verdict.passed,
                "reason": verdict.reason[:300],
            }
        except Exception as exc:  # noqa: BLE001
            verdict_dict = {"technical_error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        check_sec = time.monotonic() - check_started

        row = {
            "label": label,
            "steps": steps,
            "lora": use_lora,
            "gen_sec": round(gen_sec, 1),
            "check_sec": round(check_sec, 1),
            "checker_mode": config.checker_mode,
            "stats": stats,
            "verdict": verdict_dict,
            "file": str(path),
        }
        rows.append(row)
        print(f"  生成 {gen_sec:.1f}s  检查 {check_sec:.1f}s")
        print(f"  统计 {json.dumps(stats, ensure_ascii=False)}")
        print(f"  判断 {json.dumps(verdict_dict, ensure_ascii=False)}")
        print("")

    report = out_dir / f"{photo.stem}_{style.id}_grid.json"
    report.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果写入 {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
