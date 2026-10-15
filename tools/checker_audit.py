"""检查模型对照测试：验证误通过与误拒绝。

构造已知答案的样例，检验检查模型是否真的在区分风格与内容：
    A. 原图本身                       -> 应当不通过（没有目标画派特征）
    B. 原图仅做色调偏移（内容保留、风格不符） -> 应当不通过
    C. 内容被替换成无关图案（风格伪造）      -> 应当不通过（主体丢失）
    D. 人为上色的高饱和版本（内容保留 + 有一定风格） -> 允许 match 或 mismatch（灰区）

用法：
    python -m tools.checker_audit --photo samples/debug/debug_01_sun_house.jpg --style fauvism
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.checker import VisionChecker  # noqa: E402
from app.config import load_app_config  # noqa: E402
from app.preprocess import encode_png, load_image, preprocess  # noqa: E402


def tone_shift(image: np.ndarray) -> np.ndarray:
    """只改色调与亮度：内容（形状与布局）完全保留。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 40) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.6, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def unrelated_pattern(image: np.ndarray) -> np.ndarray:
    """形状与布局全部替换：风格可以任意，但主体已丢失。"""
    h, w = image.shape[:2]
    out = np.full((h, w, 3), 250, np.uint8)
    cv2.circle(out, (int(w * 0.3), int(h * 0.4)), int(min(h, w) * 0.18), (60, 20, 200), -1)
    cv2.rectangle(out, (int(w * 0.55), int(h * 0.5)), (int(w * 0.85), int(h * 0.8)), (200, 180, 20), -1)
    return out


def posterised_vivid(image: np.ndarray) -> np.ndarray:
    """高饱和 + 色块化，形状布局保留：内容保留、风格接近野兽派方向。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * 2.4 + 90, 0, 255).astype(np.uint8)
    vivid = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # 色块化：中值滤波后强量化，模拟平涂色块
    smoothed = cv2.medianBlur(vivid, 9)
    quant = (smoothed // 64) * 64 + 32
    return quant.astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查模型对照测试")
    parser.add_argument("--photo", required=True)
    parser.add_argument("--style", default="fauvism")
    parser.add_argument("--out", default="samples/_audit")
    args = parser.parse_args(argv)

    config = load_app_config(ROOT)
    style = config.load_styles().resolve(args.style)
    check_prompt = config.load_prompts()[1].replace("{style_name}", style.name).replace(
        "{style_features}", style.style_features).replace("{preservation_rules}", style.preservation_rules)

    image = load_image(Path(args.photo))
    result = preprocess(
        image,
        max_input_side=config.pre_max_input_side,
        frame_detect=config.pre_frame_detect,
        border_trim_ratio=config.pre_border_trim_ratio,
        crop_ratios=config.pre_crop_ratios,
        min_area_ratio=config.pre_min_area_ratio,
        max_area_ratio=config.pre_max_area_ratio,
        approx_epsilon_ratio=config.pre_approx_epsilon_ratio,
    )
    base = result.image
    original_bytes = encode_png(base)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("A_original_itself", base, "不通过", "原图没有目标画派特征"),
        ("B_tone_shift_only", tone_shift(base), "不通过", "内容保留但风格不符"),
        ("C_unrelated_pattern", unrelated_pattern(base), "不通过", "主体丢失"),
        ("D_vivid_posterised", posterised_vivid(base), "灰区", "内容保留 + 高饱和色块"),
    ]

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

    print(f"输入: {args.photo}   画派: {style.id}")
    print(f"检查方式: {config.checker_mode}")
    print("")
    rows = []
    for label, image_array, expectation, note in cases:
        path = out_dir / f"{Path(args.photo).stem}_{style.id}_{label}.png"
        candidate_bytes = encode_png(image_array)
        path.write_bytes(candidate_bytes)
        verdict = checker.check_pair(
            original_bytes=original_bytes,
            candidate_bytes=candidate_bytes,
            prompt=check_prompt,
            style_name=style.name,
            style_features=style.style_features,
            preservation_rules=style.preservation_rules,
        )
        hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
        row = {
            "case": label,
            "mode": config.checker_mode,
            "expectation": expectation,
            "note": note,
            "sat_mean": round(float(hsv[:, :, 1].mean()), 1),
            "style_match": verdict.style_match,
            "content_preserved": verdict.content_preserved,
            "passed": verdict.passed,
            "reason": verdict.reason[:250],
        }
        rows.append(row)
        flag = "OK " if (expectation == "不通过" and not verdict.passed) or expectation == "灰区" else "检查"
        print(f"[{flag}] {label}  期望={expectation}  sat={row['sat_mean']}")
        print(f"       match={verdict.style_match} preserved={verdict.content_preserved!r} passed={verdict.passed}")
        print(f"       {verdict.reason[:220]}")
        print("")

    report = out_dir / f"{Path(args.photo).stem}_{style.id}_audit.json"
    report.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果写入 {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
