"""比较不同视觉模型作为自动检查器的判定结果。

对同一批候选图，用指定的一个或多个视觉模型按配置方式执行检查，
输出对照表，用于评估检查模型的误通过与误拒绝。

用法：
    python -m tools.compare_checkers --candidates samples/_gen8 --style fauvism `
        --models qwen2.5vl:3b,qwen2.5vl:7b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.checker import VisionChecker  # noqa: E402
from app.config import load_app_config  # noqa: E402
from app.preprocess import encode_png, load_image, preprocess  # noqa: E402

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查模型对照")
    parser.add_argument("--original", required=True, help="原图照片路径")
    parser.add_argument("--candidates", required=True, help="候选图目录或单张图")
    parser.add_argument("--style", required=True, help="目标画派 id")
    parser.add_argument("--models", required=True, help="逗号分隔的视觉模型名")
    parser.add_argument("--control", action="store_true", help="同时执行对照判定")
    parser.add_argument("--out", default="docs/checker-compare.json")
    parser.add_argument("--input-side", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_app_config(ROOT)
    style = config.load_styles().resolve(args.style)
    check_prompt = config.load_prompts()[1].replace("{style_name}", style.name).replace(
        "{style_features}", style.style_features).replace("{preservation_rules}", style.preservation_rules)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    image = load_image(Path(args.original))
    result = preprocess(
        image,
        max_input_side=args.input_side or config.pre_max_input_side,
        frame_detect=config.pre_frame_detect,
        border_trim_ratio=config.pre_border_trim_ratio,
        crop_ratios=config.pre_crop_ratios,
        min_area_ratio=config.pre_min_area_ratio,
        max_area_ratio=config.pre_max_area_ratio,
        approx_epsilon_ratio=config.pre_approx_epsilon_ratio,
    )
    original_bytes = encode_png(result.image)

    candidates_dir = Path(args.candidates)
    if candidates_dir.is_file():
        candidate_paths = [candidates_dir]
    else:
        candidate_paths = [
            p
            for p in sorted(candidates_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("_")
        ]
    if not candidate_paths:
        print(f"[错误] 目录中没有候选图: {args.candidates}")
        return 2

    print(f"原图      : {args.original}")
    print(f"目标画派  : {style.id} ({style.name})")
    print(f"候选图    : {len(candidate_paths)} 张")
    print(f"检查模型  : {', '.join(models)}")
    print(f"检查方式  : {config.checker_mode}")
    print("")

    rows: list[dict] = []
    for model in models:
        checker = VisionChecker(
            base_url=config.checker_base_url,
            model=model,
            api=config.checker_api,
            connect_timeout=config.checker_connect_timeout,
            request_timeout=config.checker_request_timeout,
            keep_alive=config.checker_keep_alive,
            options=config.checker_options,
            mode=config.checker_mode,
            control=args.control,
        )
        try:
            checker.verify_model()
        except Exception as exc:  # noqa: BLE001
            print(f"[跳过] 模型 {model} 不可用: {exc}")
            continue

        print(f"===== {model} =====")
        for path in candidate_paths:
            started = time.monotonic()
            try:
                verdict = checker.check_pair(
                    original_bytes=original_bytes,
                    candidate_bytes=path.read_bytes(),
                    prompt=check_prompt,
                    style_name=style.name,
                    style_features=style.style_features,
                    preservation_rules=style.preservation_rules,
                )
                row = {
                    "model": model,
                    "mode": config.checker_mode,
                    "candidate": path.name,
                    "style_match": verdict.style_match,
                    "content_preserved": verdict.content_preserved,
                    "passed": verdict.passed,
                    "reason": verdict.reason[:300],
                    "original_description": verdict.extra.get("original_description", "")[:300],
                    "candidate_description": verdict.extra.get("candidate_description", "")[:300],
                    "control": verdict.extra.get("control"),
                    "seconds": round(time.monotonic() - started, 2),
                }
            except Exception as exc:  # noqa: BLE001
                row = {
                    "model": model,
                    "mode": config.checker_mode,
                    "candidate": path.name,
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                }
            rows.append(row)
            mark = "PASS" if row.get("passed") else ("ERR " if "error" in row else "fail")
            print(
                f"  [{mark}] {path.name:<52} match={row.get('style_match')} "
                f"preserved={row.get('content_preserved')} ({row.get('seconds')}s)"
            )
            if row.get("candidate_description"):
                print(f"        候选描述: {row['candidate_description'][:170]}")
            print(f"        理由    : {row.get('reason', row.get('error', ''))[:170]}")
        print("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
