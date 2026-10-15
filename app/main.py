"""后台启动入口。

用法示例（在项目根目录执行）：

    python -m app.main doctor
    python -m app.main import --photo path\\to\\photo.jpg
    python -m app.main import --photo photo.jpg --style cubism
    python -m app.main camera
    python -m app.main resume
    python -m app.main list

正常入口随机分配画派；--style 仅用于测试覆盖，不改变正常入口的随机行为。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import states
from .capture import CameraCapture, import_photo, list_camera_candidates
from .checker import VisionChecker, model_is_available
from .config import AppConfig, StyleRegistry, load_app_config
from .errors import ConfigError, PipelineError
from .generator import ComfyClient, ComfyGenerator, load_workflow
from .pipeline import Pipeline
from .storage import JobStore
from .styles import StyleAssigner, validate_registry


def build_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("amcc")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def log(message: str) -> None:
    print(message, flush=True)


class App:
    """把配置、模型适配器与流程装配在一起。"""

    def __init__(self, config: AppConfig | None = None, project_root: Path | None = None) -> None:
        self.config = config or load_app_config(project_root)
        self.logger = build_logger(self.config.log_level)
        self.registry: StyleRegistry = self.config.load_styles()
        validate_registry(self.registry)
        self.config.require_prompt_files()
        generate_prompt, check_prompt = self.config.load_prompts()

        workflow_nodes = load_workflow(self.config.comfy_workflow_path)
        self.comfy_client = ComfyClient(
            base_url=self.config.comfy_base_url,
            connect_timeout=self.config.comfy_connect_timeout,
            request_timeout=self.config.comfy_request_timeout,
            job_timeout=self.config.comfy_job_timeout,
            poll_interval=self.config.comfy_poll_interval,
        )
        self.generator = ComfyGenerator(
            client=self.comfy_client,
            workflow_nodes=workflow_nodes,
            checkpoint=self.config.comfy_checkpoint,
            negative_prompt=self.config.negative_prompt,
            controlnet=self.config.comfy_controlnet,
            controlnet_strength=self.config.controlnet_strength,
            unet_name=self.config.comfy_unet_name,
            text_encoder=self.config.comfy_text_encoder,
            vae_name=self.config.comfy_vae_name,
            shift=self.config.comfy_shift,
            lora_name=self.config.comfy_lora_name,
            lora_strength=self.config.comfy_lora_strength,
        )
        self.checker = VisionChecker(
            base_url=self.config.checker_base_url,
            model=self.config.checker_model,
            api=self.config.checker_api,
            connect_timeout=self.config.checker_connect_timeout,
            request_timeout=self.config.checker_request_timeout,
            keep_alive=self.config.checker_keep_alive,
            options=self.config.checker_options,
            mode=self.config.checker_mode,
            control=self.config.checker_control,
        )
        self.store = JobStore(self.config.runs_dir, self.config.temp_suffix)
        self.pipeline = Pipeline(
            config=self.config,
            registry=self.registry,
            store=self.store,
            generator=self.generator,
            checker=self.checker,
            generate_prompt_template=generate_prompt,
            check_prompt_template=check_prompt,
            style_assigner=StyleAssigner(self.registry),
        )

    # ---- 环境检查 ----
    def doctor(self, probe_camera: bool = False, probe_models: bool = True) -> dict:
        report: dict = {
            "python": sys.version.split()[0],
            "project_root": str(self.config.project_root),
            "enabled_styles": [s.id for s in self.registry.enabled],
            "max_attempts": self.config.max_attempts,
            "print_enabled": self.config.print_enabled,
            "runs_dir": str(self.config.runs_dir),
        }
        report["workflow"] = {
            "file": str(self.config.comfy_workflow_path),
            "binding": self.generator.binding,
        }
        uses_unet = self.generator.uses_unet_loader()
        report["config"] = {
            "generation_model": (
                self.config.comfy_unet_name if uses_unet else self.config.comfy_checkpoint
            ),
            "model_route": "unet" if uses_unet else "checkpoint",
            "checkpoint": None if uses_unet else self.config.comfy_checkpoint,
            "unet_name": self.config.comfy_unet_name if uses_unet else None,
            "text_encoder": self.config.comfy_text_encoder if uses_unet else None,
            "vae_name": self.config.comfy_vae_name if uses_unet else None,
            "lora_name": self.config.comfy_lora_name if uses_unet else None,
            "controlnet": self.config.comfy_controlnet,
            "checker_model": self.config.checker_model,
            "max_input_side": self.config.pre_max_input_side,
            "comfy_base_url": self.config.comfy_base_url,
            "checker_base_url": self.config.checker_base_url,
        }

        comfy: dict = {
            "probe_status": "not_run",
            "reachable": None,
            "generation_model_ok": None,
        }
        if probe_models:
            comfy.update(probe_status="completed", reachable=False, generation_model_ok=False)
            try:
                stats = self.comfy_client.system_stats()
                comfy["reachable"] = True
                devices = stats.get("devices") or []
                comfy["devices"] = [
                    {
                        "name": d.get("name"),
                        "vram_total_mb": round((d.get("vram_total") or 0) / 1024 / 1024, 1),
                        "vram_free_mb": round((d.get("vram_free") or 0) / 1024 / 1024, 1),
                    }
                    for d in devices
                ]
                comfy["comfyui_version"] = (stats.get("system") or {}).get("comfyui_version")
                self.generator.verify_checkpoint()
                comfy["generation_model_ok"] = True
            except PipelineError as exc:
                comfy["error"] = exc.message
        report["comfyui"] = comfy

        vision: dict = {"probe_status": "not_run", "reachable": None, "model_ok": None}
        if probe_models:
            vision.update(probe_status="completed", reachable=False, model_ok=False)
            try:
                models = self.checker.list_models()
                vision["reachable"] = True
                vision["models"] = models
                vision["model_ok"] = model_is_available(self.config.checker_model, models)
            except PipelineError as exc:
                vision["error"] = exc.message
        report["vision"] = vision

        if probe_camera:
            report["camera"] = list_camera_candidates(
                max_index=4, backend=self.config.capture_backend
            )
        active = self.store.find_active()
        report["active_job"] = None if active is None else active.to_summary()
        report["incomplete_jobs"] = [j.to_summary() for j in self.store.find_incomplete()]
        return report

    # ---- 运行 ----
    def run_photo(
        self, photo: str | Path, style_id: str | None = None, job_id: str | None = None,
        force: bool = False,
    ) -> dict:
        data = import_photo(photo)
        return self.pipeline.run(
            source_bytes=data,
            source_name=str(photo),
            style_id=style_id,
            job_id=job_id,
            force=force,
        )

    def run_camera(
        self, style_id: str | None = None, job_id: str | None = None, force: bool = False
    ) -> dict:
        camera = CameraCapture(
            device_index=self.config.capture_device_index,
            backend=self.config.capture_backend,
            width=self.config.capture_width,
            height=self.config.capture_height,
            warmup_frames=self.config.capture_warmup_frames,
            settle_ms=self.config.capture_settle_ms,
        )
        data = camera.capture_jpeg()
        return self.pipeline.run(
            source_bytes=data,
            source_name="camera",
            style_id=style_id,
            job_id=job_id,
            force=force,
        )


def _print_summary(summary: dict) -> None:
    log("")
    log("=== 任务结果 ===")
    log(f"job_id      : {summary.get('job_id')}")
    log(f"state       : {summary.get('state')}")
    log(f"style       : {summary.get('style_id')}")
    log(f"attempts    : {summary.get('attempt_count')}")
    log(f"output      : {summary.get('output_path') or '（无 output.png）'}")
    if summary.get("error"):
        log(f"error       : {json.dumps(summary['error'], ensure_ascii=False)}")
    if summary.get("timings"):
        log(f"timings     : {json.dumps(summary['timings'], ensure_ascii=False)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="儿童艺术风格探索装置 MVP 后台入口（粗马克笔 → 随机画派 → 本地生成 → 双图检查 → 自动确认）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="检查配置、本地服务、模型与摄像头")
    p_doctor.add_argument("--camera", action="store_true", help="同时探测摄像头索引")
    p_doctor.add_argument("--offline", action="store_true", help="不访问本地服务，只检查本地配置")

    p_import = sub.add_parser("import", help="导入已有照片并跑完整流程")
    p_import.add_argument("--photo", required=True, help="照片路径")
    p_import.add_argument(
        "--artwork", action="store_true", help="完整画作输入：关闭画框检测和四周裁切"
    )
    p_import.add_argument("--style", default=None, help="测试用：显式指定画派 id，覆盖随机分配")
    p_import.add_argument("--job-id", default=None, help="复用指定 job_id（需配合 --force）")
    p_import.add_argument("--force", action="store_true", help="重跑已失败任务，沿用其已保存画派")

    p_camera = sub.add_parser("camera", help="触发摄像头采集并跑完整流程")
    p_camera.add_argument("--style", default=None, help="测试用：显式指定画派 id，覆盖随机分配")
    p_camera.add_argument("--job-id", default=None, help="复用指定 job_id（需配合 --force）")
    p_camera.add_argument("--force", action="store_true", help="重跑已失败任务，沿用其已保存画派")

    sub.add_parser("resume", help="读取既有记录，处理中断任务（不重复提交已完成任务）")

    p_list = sub.add_parser("list", help="列出 runs/ 中的任务记录")
    p_list.add_argument("--all", action="store_true", help="包含终态任务")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        app = (
            App(config=load_app_config().for_artwork())
            if args.command == "import" and args.artwork else App()
        )
    except ConfigError as exc:
        log(f"[配置错误] {exc.message}")
        return 2

    if args.command == "doctor":
        report = app.doctor(probe_camera=args.camera, probe_models=not args.offline)
        log(json.dumps(report, ensure_ascii=False, indent=2))
        ok = True
        if not report.get("config", {}).get("generation_model"):
            log("[提示] 当前工作流未配置生成模型")
            ok = False
        if not args.offline:
            comfy = report.get("comfyui", {})
            if not comfy.get("reachable"):
                log(f"[提示] ComfyUI 未就绪: {comfy.get('error')}")
                ok = False
            elif not comfy.get("generation_model_ok"):
                log(f"[提示] 当前工作流的生成模型或依赖未就绪: {comfy.get('error')}")
                ok = False
            vision = report.get("vision", {})
            if not vision.get("reachable"):
                log(f"[提示] 视觉模型服务未就绪: {vision.get('error')}")
                ok = False
            elif not vision.get("model_ok"):
                log(f"[提示] 配置的视觉模型未安装: {vision.get('models')}")
                ok = False
        return 0 if ok else 1

    if args.command == "resume":
        reports = app.pipeline.resume_scan()
        log(json.dumps(reports, ensure_ascii=False, indent=2))
        if not reports:
            log("没有中断任务需要处理。")
        return 0

    if args.command == "list":
        jobs = app.store.list_jobs()
        if not args.all:
            jobs = [j for j in jobs if states.is_active(j.state)]
        if not jobs:
            log("没有符合条件的任务记录。")
            return 0
        for job in jobs:
            summary = job.to_summary()
            log(
                f"{summary['job_id']}  {summary['state']:<14} style={summary['style_id']}  "
                f"attempts={summary['attempt_count']}  output={summary['has_output']}"
            )
        return 0

    # 先扫描中断任务（不重复提交已完成任务），再判断是否有活动任务
    resume_reports = app.pipeline.resume_scan()
    for item in resume_reports:
        log(f"[恢复] {json.dumps(item, ensure_ascii=False)}")

    ignore_id = args.job_id if getattr(args, "force", False) else None
    try:
        app.pipeline.ensure_idle(ignore_job_id=ignore_id)
    except ConfigError as exc:
        log(f"[拒绝新任务] {exc.message}")
        log("请先运行 python -m app.main resume 处理中断任务，或检查 runs/ 中的活动任务。")
        return 2

    try:
        if args.command == "import":
            summary = app.run_photo(
                args.photo, style_id=args.style, job_id=args.job_id, force=args.force
            )
        elif args.command == "camera":
            summary = app.run_camera(
                style_id=args.style, job_id=args.job_id, force=args.force
            )
        else:  # pragma: no cover - argparse 已限制
            parser.error(f"未知命令: {args.command}")
            return 2
    except PipelineError as exc:
        log(f"[任务失败] {exc.category}: {exc.message}")
        return 1

    _print_summary(summary)
    return 0 if summary.get("state") == states.COMPLETED else 1


if __name__ == "__main__":
    sys.exit(main())
