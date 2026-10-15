"""按文件名中的画派标记，用真实检查模型评估一批候选图。

文件名需包含画派 id（例如 ..._fauvism_s8.png）。每个画派分别统计，
避免用一种画派的高通过率掩盖另一种的失败。

用法：
    python -m tools.grade_candidates --original X.jpg --candidates samples/_gen8 `
        --models qwen2.5vl:7b --report docs/gen-quality.md
"""

from __future__ import annotations

import argparse
import json
import statistics
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


def style_of(path: Path, style_ids: list[str]) -> str | None:
    name = path.stem.lower()
    for style_id in style_ids:
        if f"_{style_id}_" in f"_{name}_" or name.endswith(f"_{style_id}"):
            return style_id
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="候选图质量评估（真实检查模型）")
    parser.add_argument("--original", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--models", default=None, help="默认取配置中的检查模型")
    parser.add_argument("--report", default="docs/gen-quality.md")
    parser.add_argument("--json", default="docs/gen-quality.json")
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args(argv)

    config = load_app_config(ROOT)
    registry = config.load_styles()
    style_ids = [s.id for s in registry.enabled]
    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else [config.checker_model]
    )

    image = load_image(Path(args.original))
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
    original_bytes = encode_png(result.image)

    folder = Path(args.candidates)
    paths = [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("_")
    ]
    if not paths:
        print(f"[错误] 没有候选图: {folder}")
        return 2

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
            mode="two_stage",
            control=args.control,
        )
        try:
            checker.verify_model()
        except Exception as exc:  # noqa: BLE001
            print(f"[跳过] 检查模型不可用 {model}: {exc}")
            continue

        print(f"===== 检查模型 {model} =====")
        for path in paths:
            style_id = style_of(path, style_ids)
            if style_id is None:
                print(f"  [跳过] 文件名无法识别画派: {path.name}")
                continue
            style = registry.get(style_id)
            started = time.monotonic()
            try:
                verdict = checker.check_two_stage(
                    original_bytes=original_bytes,
                    candidate_bytes=path.read_bytes(),
                    style_name=style.name,
                    style_features=style.style_features,
                    preservation_rules=style.preservation_rules,
                    control=args.control,
                )
                row = {
                    "model": model,
                    "candidate": path.name,
                    "target_style": style_id,
                    "style_match": verdict.style_match,
                    "content_preserved": verdict.content_preserved,
                    "passed": verdict.passed,
                    "reason": verdict.reason,
                    "original_description": verdict.extra.get("original_description", ""),
                    "candidate_description": verdict.extra.get("candidate_description", ""),
                    "control": verdict.extra.get("control"),
                    "seconds": round(time.monotonic() - started, 2),
                }
            except Exception as exc:  # noqa: BLE001
                row = {
                    "model": model,
                    "candidate": path.name,
                    "target_style": style_id,
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
            rows.append(row)
            mark = "PASS" if row.get("passed") else ("ERR " if "error" in row else "fail")
            print(
                f"  [{mark}] {path.name:<50} 目标={style_id:<14} "
                f"match={row.get('style_match')} preserved={row.get('content_preserved')} "
                f"({row.get('seconds')}s)"
            )
        print("")

    # ---- 统计 ----
    summary: dict = {"models": {}, "rows": rows}
    for model in models:
        model_rows = [r for r in rows if r.get("model") == model and "error" not in r]
        if not model_rows:
            continue
        summary["models"][model] = {"total": len(model_rows), "passed": sum(1 for r in model_rows if r["passed"])}
        for style_id in style_ids:
            style_rows = [r for r in model_rows if r["target_style"] == style_id]
            if not style_rows:
                continue
            summary["models"][model][style_id] = {
                "total": len(style_rows),
                "passed": sum(1 for r in style_rows if r["passed"]),
                "rate": round(sum(1 for r in style_rows if r["passed"]) / len(style_rows), 3),
                "seconds_mean": round(statistics.mean([r["seconds"] for r in style_rows]), 2),
            }

    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 生成质量与自动检查评估",
        "",
        f"- 原图：`{args.original}`",
        f"- 候选图目录：`{args.candidates}`",
        f"- 检查模型：{', '.join(models)}",
        "- 检查方式：two_stage（先分别描述原图与候选图，再依据描述判定）",
        "",
        "## 通过统计（按画派分别统计）",
        "",
        "| 检查模型 | 画派 | 张数 | 通过 | 通过率 | 平均耗时(s) |",
        "|---|---|---|---|---|---|",
    ]
    for model, stats in summary["models"].items():
        for style_id in style_ids:
            if style_id in stats:
                s = stats[style_id]
                lines.append(
                    f"| {model} | {style_id} | {s['total']} | {s['passed']} | "
                    f"{s['rate'] * 100:.0f}% | {s['seconds_mean']} |"
                )
        lines.append(
            f"| {model} | **合计** | {stats['total']} | {stats['passed']} | "
            f"{stats['passed'] / stats['total'] * 100:.0f}% | — |"
        )

    lines += [
        "",
        "## 逐条明细",
        "",
        "| 检查模型 | 候选图 | 目标画派 | 判定 | 主体保留 | 通过 | 耗时(s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if "error" in row:
            lines.append(f"| {row['model']} | {row['candidate']} | {row['target_style']} | 技术错误 | — | — | — |")
            continue
        lines.append(
            f"| {row['model']} | {row['candidate']} | {row['target_style']} | "
            f"{row['style_match']} | {row['content_preserved']} | "
            f"{'是' if row['passed'] else '否'} | {row['seconds']} |"
        )

    lines += ["", "## 判定理由（供人工抽查误判）", ""]
    for row in rows:
        if "error" in row:
            lines.append(f"- **{row['candidate']}**（{row['model']}）：技术错误 {row['error']}")
            continue
        lines.append(
            f"- **{row['candidate']}**（{row['model']}，目标 {row['target_style']}）"
            f"→ {row['style_match']} / content_preserved={row['content_preserved']}"
        )
        lines.append(f"  - 理由：{row['reason']}")
        if row.get("candidate_description"):
            lines.append(f"  - 候选描述：{row['candidate_description'][:300]}")

    lines += [
        "",
        "> 自动检查属于粗略判断，必须结合人工抽查误通过与误拒绝，",
        "> 不能只以本表的通过率作为质量结论。",
        "",
    ]

    out_md = Path(args.report)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告写入 {out_md}")
    print(f"原始数据写入 {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
