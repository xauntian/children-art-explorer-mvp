"""各画派质量评估脚本（验收用）。

对一批输入照片，按固定配置分别以每种画派生成并自动确认，
统计每种画派的通过情况，并输出可人工抽查的对照表。

用法：
    python -m tools.evaluate_styles --inputs samples/heldout --per-style 5 --report docs/eval-report.md
    python -m tools.evaluate_styles --inputs samples/heldout --styles fauvism,cubism
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import states  # noqa: E402
from app.capture import PHOTO_SUFFIXES, import_photo  # noqa: E402
from app.config import load_app_config  # noqa: E402
from app.errors import PipelineError  # noqa: E402
from app.main import App  # noqa: E402


def list_inputs(folder: Path) -> list[Path]:
    if folder.is_file():
        return [folder]
    files = [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in PHOTO_SUFFIXES and not p.name.startswith("_")
    ]
    return files


def validate_report_paths(paths: list[Path]) -> None:
    """在运行模型前拒绝覆盖历史记录，且两个格式不能指向同一文件。"""
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Markdown 与 JSON 报告必须使用不同路径")
    for path in paths:
        if path.exists():
            raise ValueError(f"报告已存在，请指定新的报告文件名以保留历史记录: {path}")


def write_new_report(path: Path, content: str) -> None:
    """独占创建也防止模型运行期间出现同名文件后被覆盖。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def evaluation_metadata(app: App, inputs: list[Path], role: str, note: str) -> dict:
    """按实际工作流选择生成模型；目录名不是独立保留集的证明。"""
    config = app.config
    uses_unet = app.generator.uses_unet_loader()
    return {
        "generation_model": config.comfy_unet_name if uses_unet else config.comfy_checkpoint,
        "checkpoint": None if uses_unet else config.comfy_checkpoint,
        "unet_name": config.comfy_unet_name if uses_unet else None,
        "controlnet": config.comfy_controlnet,
        "workflow_file": str(config.comfy_workflow_path),
        "workflow_sha256": hashlib.sha256(config.comfy_workflow_path.read_bytes()).hexdigest(),
        "text_encoder": config.comfy_text_encoder if uses_unet else None,
        "vae_name": config.comfy_vae_name if uses_unet else None,
        "lora_name": config.comfy_lora_name if uses_unet else None,
        "lora_strength": config.comfy_lora_strength if uses_unet else None,
        "shift": config.comfy_shift if uses_unet else None,
        "generation_config": config.raw.get("generation", {}),
        "preprocess": {
            "frame_detect": config.pre_frame_detect,
            "border_trim_ratio": config.pre_border_trim_ratio,
            "crop_ratios": list(config.pre_crop_ratios),
        },
        "checker_model": config.checker_model,
        "checker_mode": config.checker_mode,
        "checker_options": dict(config.checker_options),
        "dataset": {
            "role": role,
            "note": note,
            "independence": "not_verified",
            "inputs": [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in inputs
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="各画派质量评估")
    parser.add_argument("--inputs", required=True, help="输入照片目录（或单张照片）")
    parser.add_argument(
        "--artwork", action="store_true", help="完整画作输入：关闭画框检测和四周裁切"
    )
    parser.add_argument("--styles", default=None, help="逗号分隔的画派 id；默认全部启用画派")
    parser.add_argument("--per-style", type=int, default=0, help="每个画派最多使用多少张输入；0 表示全部")
    parser.add_argument("--report", default=None, help="输出 Markdown 报告路径")
    parser.add_argument("--json", default=None, help="输出 JSON 结果路径")
    parser.add_argument(
        "--dataset-role", choices=("development", "heldout"), default="development",
        help="输入用途；默认 development。仅未参与调参的输入才可声明 heldout",
    )
    parser.add_argument("--dataset-note", default="", help="输入来源与是否参与过调参的说明")
    args = parser.parse_args(argv)

    if args.per_style < 0:
        parser.error("--per-style 必须大于等于 0")
    try:
        validate_report_paths([Path(p) for p in (args.report, args.json) if p])
    except ValueError as exc:
        parser.error(str(exc))

    app = App(config=load_app_config(ROOT).for_artwork()) if args.artwork else App()
    inputs = list_inputs(Path(args.inputs))
    if not inputs:
        print(f"[错误] 输入目录中没有可用的照片: {args.inputs}")
        return 2

    style_ids = (
        [s.strip() for s in args.styles.split(",") if s.strip()]
        if args.styles
        else [s.id for s in app.registry.enabled]
    )
    for style_id in style_ids:
        app.registry.resolve(style_id)

    per_style = args.per_style or len(inputs)
    selected = inputs[:per_style]
    metadata = evaluation_metadata(app, selected, args.dataset_role, args.dataset_note)

    print(f"画派: {style_ids}")
    print(f"输入: {len(selected)} 张（来自 {args.inputs}）")
    print("")

    records: list[dict] = []
    started_all = time.monotonic()

    for style_id in style_ids:
        for photo in selected:
            started = time.monotonic()
            try:
                data = import_photo(photo)
                summary = app.pipeline.run(
                    source_bytes=data, source_name=str(photo), style_id=style_id
                )
            except PipelineError as exc:
                summary = {
                    "state": states.FAILED,
                    "style_id": style_id,
                    "error": exc.to_dict(),
                    "job_id": None,
                }
            duration = round(time.monotonic() - started, 2)

            job = None
            check_result = None
            if summary.get("job_id"):
                job = app.store.load(summary["job_id"])
                for attempt in (1, 2):
                    payload = job.read_check_result(attempt)
                    if payload and "style_match" in payload:
                        check_result = payload
            record = {
                "style_id": style_id,
                "input": photo.name,
                "job_id": summary.get("job_id"),
                "state": summary.get("state"),
                "attempts": summary.get("attempt_count"),
                "duration_sec": duration,
                "passed": summary.get("state") == states.COMPLETED,
                "check": check_result,
                "error": summary.get("error"),
                "output": summary.get("output_path"),
            }
            records.append(record)
            mark = "PASS" if record["passed"] else "FAIL"
            detail = ""
            if check_result:
                detail = f" style_match={check_result.get('style_match')} content_preserved={check_result.get('content_preserved')}"
            print(
                f"[{mark}] {style_id:<14} {photo.name:<32} attempts={record['attempts']} "
                f"{duration:>6.1f}s{detail}"
            )

    total_duration = round(time.monotonic() - started_all, 2)

    # ---- 统计 ----
    from app import print_adapter

    print_summary: dict = {
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs_dir": str(args.inputs),
        "per_style": per_style,
        "total_duration_sec": total_duration,
        **metadata,
        "max_input_side": app.config.pre_max_input_side,
        "styles": {},
        "print_placeholder": print_adapter.export_to_print("eval", "n/a"),
    }

    print("")
    print("=== 每种画派统计（按输入张数计） ===")
    for style_id in style_ids:
        rows = [r for r in records if r["style_id"] == style_id]
        passed = [r for r in rows if r["passed"]]
        durations = [r["duration_sec"] for r in rows]
        retried = [r for r in rows if (r["attempts"] or 0) > 1]
        stats = {
            "inputs": len(rows),
            "passed": len(passed),
            "pass_rate": round(len(passed) / len(rows), 3) if rows else 0.0,
            "retried": len(retried),
            "failed": len(rows) - len(passed),
            "duration_sec_mean": round(statistics.mean(durations), 2) if durations else None,
            "duration_sec_max": max(durations) if durations else None,
        }
        print_summary["styles"][style_id] = stats
        print(
            f"{style_id:<14} 通过 {stats['passed']}/{stats['inputs']} "
            f"({stats['pass_rate'] * 100:.0f}%)  重试 {stats['retried']}  "
            f"失败 {stats['failed']}  平均 {stats['duration_sec_mean']}s"
        )

    print_summary["records"] = records

    if args.json:
        path = Path(args.json)
        write_new_report(path, json.dumps(print_summary, ensure_ascii=False, indent=2))
        print(f"JSON 结果已写入 {path}")

    if args.report:
        path = Path(args.report)
        write_new_report(path, render_report(print_summary))
        print(f"Markdown 报告已写入 {path}")

    return 0


def render_report(summary: dict) -> str:
    lines = [
        "# 各画派质量评估报告",
        "",
        f"- 评估时间（UTC）：{summary['evaluated_at']}",
        f"- 输入目录：`{summary['inputs_dir']}`",
        f"- 每个画派输入张数：{summary['per_style']}",
        f"- 生成模型：`{summary.get('generation_model', summary.get('checkpoint'))}`",
        f"- 工作流：`{summary.get('workflow_file', '历史报告未记录')}`",
        f"- LoRA：`{summary.get('lora_name')}`；强度：`{summary.get('lora_strength')}`",
        f"- ControlNet：`{summary['controlnet']}`",
        f"- 检查模型：`{summary['checker_model']}`",
        f"- 检查方式：`{summary.get('checker_mode', '历史报告未记录')}`",
        f"- 模型输入最长边：{summary['max_input_side']} px",
        f"- 总耗时：{summary['total_duration_sec']} s",
        f"- 输入用途：`{summary.get('dataset', {}).get('role', '未记录')}`（独立性未经脚本验证）",
        f"- 输入说明：{summary.get('dataset', {}).get('note') or '未提供'}",
        "- development 结果仅用于调试；heldout 标签不代表已证明未参与调参。",
        "",
        "## 每种画派统计",
        "",
        "| 画派 | 输入 | 通过 | 通过率 | 发生重试 | 失败 | 平均耗时(s) | 最长耗时(s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for style_id, stats in summary["styles"].items():
        lines.append(
            f"| {style_id} | {stats['inputs']} | {stats['passed']} | "
            f"{stats['pass_rate'] * 100:.0f}% | {stats['retried']} | {stats['failed']} | "
            f"{stats['duration_sec_mean']} | {stats['duration_sec_max']} |"
        )

    lines += [
        "",
        "## 逐条记录",
        "",
        "| 画派 | 输入 | job_id | 状态 | 尝试 | 耗时(s) | style_match | content_preserved | 输出 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for record in summary["records"]:
        check = record.get("check") or {}
        lines.append(
            f"| {record['style_id']} | {record['input']} | `{record['job_id']}` | "
            f"{record['state']} | {record['attempts']} | {record['duration_sec']} | "
            f"{check.get('style_match', '—')} | {check.get('content_preserved', '—')} | "
            f"{record.get('output') or '（无）'} |"
        )

    lines += [
        "",
        "## 打印占位接口",
        "",
        "```json",
        json.dumps(summary["print_placeholder"], ensure_ascii=False, indent=2),
        "```",
        "",
        "> 说明：本报告由自动检查结果生成。自动检查属于粗略判断，",
        "> 必须结合人工抽查误通过与误拒绝，不能只以本表作为质量结论。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
