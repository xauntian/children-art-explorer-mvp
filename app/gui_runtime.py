"""桌面入口的单件工作流；P 键锁覆盖启动、生成、打印和出纸确认。"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .config import load_app_config
from .errors import ConfigError
from .phone import PhoneController, load_phone_config
from .phone_runtime import PhoneRunLock, reconcile_phone_sessions
from .storage import atomic_write_json, utc_now_iso


def load_gui_config(root: Path, path: Path | None = None) -> dict:
    selected = path or root / "configs/gui.json"
    try:
        config = json.loads(selected.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict):
            raise ValueError("GUI 配置须为对象")
        for name in ("comfyui", "printer", "preview"):
            if not isinstance(config.get(name, {}), dict):
                raise ValueError(f"{name} 须为对象")
        if not isinstance(config.get("printer", {}).get("enabled", True), bool):
            raise ValueError("printer.enabled 须为布尔值")
        for name, value in (
            ("printer.timeout_sec", config.get("printer", {}).get("timeout_sec", 180)),
            ("printer.poll_interval_sec", config.get("printer", {}).get("poll_interval_sec", 0.5)),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 86400:
                raise ValueError(f"{name} 须为有效正数")
        phone_path = config.get("phone_config", "configs/phone_capture_adb.json")
        if not isinstance(phone_path, str) or not phone_path.strip():
            raise ValueError("phone_config 须为配置文件路径")
        printer_name = config.get("printer", {}).get("name")
        if printer_name is not None and (not isinstance(printer_name, str) or not printer_name.strip()):
            raise ValueError("printer.name 须为打印机名称或 null")
    except (OSError, ValueError) as exc:
        raise ConfigError(f"GUI 配置无法读取: {selected}: {exc}") from exc
    return config


def record(value) -> dict:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return vars(value).copy()


class GuiRuntime:
    def __init__(self, root: Path, config: dict, *, camera=None, printer=None,
                 service=None, warmup=None, controller_factory=None, preview_factory=None):
        self.root = root.resolve()
        self.config = config
        self.print_enabled = config.get("printer", {}).get("enabled", True)
        if not isinstance(self.print_enabled, bool):
            raise ConfigError("printer.enabled 须为布尔值")
        self.events: queue.Queue = queue.Queue()
        self.state = "NEW"
        self._gate = threading.RLock()
        self.preview_factory = preview_factory
        self.preview = None
        self.preview_roi = None
        self._preview_started = False
        self._worker: threading.Thread | None = None
        self._phone_lock = None
        self.camera = camera
        self.printer = printer
        self.printer_name = None
        self.service = service
        self.warmup = warmup
        self.controller_factory = controller_factory or PhoneController
        self.controller = None
        self.pending: dict | None = None
        self.pending_path = self.root / "runs/runtime/gui-pending-print.json"
        self.round_started = None

    def emit(self, kind: str, **data) -> None:
        self.events.put((kind, data))

    def _set_state(self, state: str, message: str) -> None:
        with self._gate:
            self.state = state
        self.emit("status", state=state, message=message)

    def _launch(self, allowed: tuple[str, ...], state: str, message: str, work) -> bool:
        with self._gate:
            if self.state not in allowed:
                return False
            self._set_state(state, message)
            self._worker = threading.Thread(target=work, daemon=True, name="amcc-gui-workflow")
            self._worker.start()
        return True

    def start(self) -> bool:
        message = "正在准备手机、AI 服务和打印机…" if self.print_enabled else "正在准备手机和 AI 服务…"
        return self._launch(("NEW", "ERROR"), "STARTING", message, self._prepare)

    def trigger(self) -> bool:
        return self._launch(("READY",), "RUNNING", "P 已锁定，正在开始新一轮…", self._run_round)

    def _dependency(self, name: str, status: str, message: str) -> None:
        self.emit("dependency", name=name, status=status, message=message)

    def _prepare(self) -> None:
        try:
            phone_path = self.root / self.config.get("phone_config", "configs/phone_capture_adb.json")
            phone_config = load_phone_config(phone_path, self.root)
            phone_config["capture_backend"] = "adb"
            phone_config["phone"]["fallback_photo"] = False
            phone_config["extraction"]["max_output_side"] = 1536
            self.preview_roi = phone_config["phone"].get("preview_roi")
            if self._phone_lock is None:
                phone_lock = PhoneRunLock(phone_config["capture_runs_dir"])
                phone_lock.__enter__()
                self._phone_lock = phone_lock
            if self.camera is None:
                from .gui_camera import GuiCamera
                self.camera = GuiCamera(self.root, phone_config["phone"])
            if self.print_enabled and self.printer is None:
                from .gui_print import PrinterAdapter
                self.printer = PrinterAdapter(printer_name=self.config.get("printer", {}).get("name"))
            self.controller = self.controller_factory(phone_config, camera=self.camera, observer=self._phone_event)
            reports = reconcile_phone_sessions(phone_config["capture_runs_dir"], self.controller.generation_store)
            if any(item["blocking"] for item in reports):
                raise ConfigError("存在尚未解决的旧生成任务，请查看 runs 中的任务记录后处理；不会重复提交。")
            active = self.controller.generation_store.find_active()
            if active is not None:
                raise ConfigError(f"已有生成任务 {active.job_id} 仍在处理中，请先解决该任务。")
            errors = []
            devices = [("phone", self.camera.prepare)]
            if self.print_enabled:
                devices.append(("printer", self.printer.prepare))
            else:
                self._dependency("printer", "skipped", "已跳过 · 仅保存数字图片")
                self.emit("stage", step="print", status="skipped", message="实体打印已关闭")
            for name, prepare in devices:
                self._dependency(name, "working", "正在检查…")
                try:
                    result = record(prepare())
                    if name == "phone":
                        if result.get("available") is not True:
                            raise ConfigError(result.get("error") or result.get("message") or "手机相机尚未就绪")
                        if not self._preview_started:
                            self._dependency("phone", "working", "正在连接手机屏幕视频…")
                            if self.preview_factory is None:
                                from .gui_video import ScreenVideoPreview
                                self.preview_factory = ScreenVideoPreview
                            self.preview = self.preview_factory(self.root, phone_config["phone"], self.config.get("preview", {}))
                            try:
                                self.preview.start(serial=result.get("serial"))
                            except Exception:
                                self.preview.close()
                                self.preview = None
                                raise
                            self._preview_started = True
                        label = "手机实时视频已连接"
                    else:
                        if result.get("available") is False:
                            raise ConfigError(result.get("error") or result.get("message") or "打印机尚未就绪")
                        label = result.get("name") or result.get("printer")
                        if not isinstance(label, str) or not label.strip():
                            raise ConfigError("打印机未返回有效的设备名称")
                        self.printer_name = label
                    self._dependency(name, "ready", label)
                except Exception as exc:
                    errors.append(str(exc))
                    self._dependency(name, "error", str(exc))
            self._dependency("ai", "working", "正在检查本地服务与模型…")
            try:
                if self.service is None:
                    from .gui_services import ensure_generation_service
                    self.service = ensure_generation_service
                self.service(self.root, load_app_config(self.root), self.config,
                             progress=lambda message: self._dependency("ai", "working", message))
                model = load_app_config(self.root).comfy_unet_name or "本地生成模型"
                self._dependency("ai", "ready", "ComfyUI · " + model)
            except Exception as exc:
                errors.append(str(exc))
                self._dependency("ai", "error", str(exc))
            if errors:
                raise ConfigError("；".join(errors))
            if self.print_enabled:
                self._read_pending()
            else:
                self.pending = None
            if self.pending:
                self._restore_images()
                self._set_state("PRINT_ERROR", "发现上一轮待核实的打印。请检查原作业；P 保持锁定。")
                self.emit("print_controls", confirm=False, retry=True)
                return
            if phone_config.get("generation_warmup", True):
                self._dependency("ai", "working", "正在预热本地模型，首次准备需要一些时间…")
                if self.warmup is None:
                    from .model_warmup import warmup_generation
                    self.warmup = warmup_generation
                report = self.warmup(self.root)
                suffix = ""
                if report:
                    completed = sum(
                        item.get("status") == "completed"
                        for item in report.get("style_prewarms", [])
                    )
                    styles = f"{completed} 种画派 · " if completed else ""
                    suffix = f" · {styles}预热 {report['total_sec']:.1f} 秒"
                self._dependency("ai", "ready", "AI 模型已就绪" + suffix)
            self._set_state("READY", "准备完成。放好画作后，在本窗口按 P 开始。")
        except Exception as exc:
            logging.exception("GUI 启动准备失败")
            self._set_state("ERROR", str(exc))
            self.emit("error", message=str(exc))

    def _phone_event(self, event: str, session) -> None:
        self.emit("session", directory=str(session.directory), session_id=session.directory.name)
        if event == "capturing":
            self.emit("stage", step="capture", status="working", message="读取手机当前取景框…")
        elif event == "captured":
            self.emit("image", key="original", path=str(session.directory / "preview.png"))
            self.emit("stage", step="capture", status="done", message="原始截图已保存")
            self.emit("stage", step="rectify", status="working", message="检测完整纸边并进行透视校正…")
        elif event == "prepared":
            self.emit("image", key="rectified", path=str(session.directory / "artwork.png"))
            self.emit("stage", step="rectify", status="done", message="完整校正作品已保存")
        elif event == "generating":
            self.emit("stage", step="generate", status="working", message="正在随机分配画派并生成 AI 图片（最多 10 秒）…")
        elif event == "generated":
            self.emit("stage", step="generate", status="done", message="AI 生成完成")

    def _run_round(self) -> None:
        self.round_started = time.monotonic()
        self.emit("reset")
        stage = "capture"
        try:
            session = self.controller.prepare("direct")
            stage = "generate"
            meta = self.controller.generate(session)
            if meta.get("state") != "COMPLETED":
                error = meta.get("error") or {}
                if error.get("category") == "timeout":
                    message = "AI 生成超时，本轮已结束。可重新按 P 开始下一轮。"
                    self.emit("stage", step="generate", status="error", message=message)
                    self.emit("ai_timeout", message="AI 生成超时，本轮没有完整 AI 图片")
                    self.emit("completed", elapsed=time.monotonic() - self.round_started)
                    self._set_state("READY", message)
                    return
                raise ConfigError(error.get("message", "生成失败，详情见任务记录"))
            stage = "save"
            output = Path(meta["output_path"])
            if not output.is_file():
                raise ConfigError("生成结果文件缺失，请重新处理本轮作品")
            self.emit("stage", step="save", status="done", message="A4 结果已保存")
            self.emit("image", key="ai", path=str(output))
            style = meta.get("generation", {}).get("style_id", "")
            style_name = style
            job_id = meta.get("generation_job_id")
            if job_id:
                try:
                    snapshot = self.controller.generation_store.load(job_id).meta.get("style")
                    if isinstance(snapshot, dict):
                        style_name = next(
                            (value for key in ("name_zh", "name", "id")
                             if isinstance(value := snapshot.get(key), str) and value.strip()),
                            style,
                        )
                except (ConfigError, OSError):
                    logging.warning("无法读取本轮画派名称，显示画派 id", exc_info=True)
            self.emit("style", style_id=style, style_name=style_name)
            if not self.print_enabled:
                self.emit("stage", step="print", status="skipped", message="实体打印已跳过，图片已保存")
                self.emit("print_controls", confirm=False, retry=False)
                self.emit("completed", elapsed=time.monotonic() - self.round_started)
                self._set_state("READY", "本轮图片已保存，P 已重新激活。可放入下一张画作。")
                return
            stage = "print"
            self.pending = {
                "status": "submitting", "created_at": utc_now_iso(),
                "session_dir": str(session.directory), "output_path": str(output),
                "generation_job_id": meta.get("generation_job_id"), "job_id": None,
                "printer": self.printer_name or self.config.get("printer", {}).get("name"),
            }
            # 在提交前持久化；崩溃后只能核实原打印，不能自动重新提交。
            self._save_pending()
            self.emit("stage", step="print", status="working", message="正在提交 A4 打印并监视原作业…")
            settings = self.config.get("printer", {})
            result = self.printer.print_image(
                output, timeout=settings.get("timeout_sec", 180),
                poll_interval=settings.get("poll_interval_sec", 0.5),
                progress=self._print_progress, on_job=self._on_print_job,
            )
            self._print_result(record(result))
        except Exception as exc:
            logging.exception("GUI 工作流失败")
            if stage == "print" and self.pending is not None:
                self._fail_print(exc)
            else:
                self.emit("error", message=str(exc))
                failed = "rectify" if (self.controller.session and self.controller.session.meta.get("capture")) and stage == "capture" else stage
                self.emit("stage", step=failed, status="error", message=str(exc))
                message = "本轮失败，未提交打印。修正问题后可重新按 P。" if self.print_enabled else "本轮失败。修正问题后可重新按 P。"
                self._set_state("READY", message)

    def _on_print_job(self, *args, **kwargs) -> None:
        if args and isinstance(args[0], dict):
            details = args[0]
        elif args:
            details = {"job_id": args[0]}
            if len(args) > 1:
                details["printer"] = args[1]
        else:
            details = kwargs
        job_id = details.get("job_id")
        if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id <= 0:
            raise ConfigError("打印机未返回可靠的打印作业编号；请核实原队列，不会自动重印。")
        if self.pending is None:
            raise ConfigError("打印恢复记录尚未创建，无法登记打印作业")
        self.pending.update(job_id=job_id, status="printing")
        if details.get("printer"):
            self.pending["printer"] = details["printer"]
        self._save_pending()

    def _print_progress(self, message) -> None:
        self.emit("stage", step="print", status="working", message=str(message))

    def _print_result(self, result: dict) -> None:
        self.pending.update(result)
        self._save_pending()
        if result.get("status") == "printed" and result.get("needs_confirmation") is False:
            self._complete_print("driver_reported_printed")
        elif result.get("status") == "spooled" and result.get("needs_confirmation") is True:
            self._set_state("AWAITING_PRINT_CONFIRM", "打印机已接收作业，尚未确认出纸。出纸后点击「确认已出纸」。P 保持锁定。")
            self.emit("stage", step="print", status="working", message=result.get("message", "等待确认出纸"))
            self.emit("print_controls", confirm=True, retry=True)
        else:
            raise ConfigError("打印驱动尚未提供可靠的打印完成状态，请重新检查原作业；P 保持锁定。")

    def _fail_print(self, exc: Exception) -> None:
        message = str(exc)
        if self.pending is not None:
            job_id = getattr(exc, "job_id", None)
            if isinstance(job_id, int) and not isinstance(job_id, bool) and job_id > 0:
                self.pending["job_id"] = job_id
            if getattr(exc, "printer", None):
                self.pending["printer"] = exc.printer
            self.pending.update(status="error", error=message)
            try:
                self._save_pending()
            except Exception as storage_error:
                logging.exception("打印恢复记录写入失败")
                message += f"；打印恢复记录无法保存: {storage_error}。请保持本窗口开启并核实原打印队列。"
        self.emit("error", message=message)
        self.emit("stage", step="print", status="error", message=message)
        self._set_state("PRINT_ERROR", "打印状态尚未解决，P 保持锁定。" + message)
        self.emit("print_controls", confirm=False, retry=True)

    def _save_pending(self) -> None:
        self.pending["updated_at"] = utc_now_iso()
        atomic_write_json(self.pending_path, self.pending)
        directory = Path(self.pending["session_dir"])
        atomic_write_json(directory / "print.json", self.pending)

    def _read_pending(self) -> None:
        if not self.pending_path.exists():
            self.pending = None
            return
        try:
            data = json.loads(self.pending_path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict):
                raise ValueError("打印状态须为对象")
            if any(not isinstance(data.get(key), str) or not data[key].strip()
                   for key in ("session_dir", "output_path", "status")):
                raise ValueError("缺少有效的打印状态、采集目录或输出路径")
            if data["status"] not in {"submitting", "printing", "printed", "spooled", "error", "completed"}:
                raise ValueError("无法识别打印状态")
            job_id = data.get("job_id")
            if job_id is not None and (isinstance(job_id, bool) or not isinstance(job_id, int) or job_id <= 0):
                raise ValueError("打印作业编号无效")
            printer = data.get("printer")
            if printer is not None and (not isinstance(printer, str) or not printer.strip()):
                raise ValueError("打印机名称无效")
            directory = Path(data["session_dir"]).resolve()
            if not directory.is_relative_to((self.root / "runs/phone").resolve()):
                raise ValueError("打印记录关联路径超出采集目录")
            self.pending = None if data["status"] == "completed" else data
        except (OSError, ValueError) as exc:
            raise ConfigError(f"打印恢复记录无法读取，请先核实上一轮作业: {exc}") from exc

    def _restore_images(self) -> None:
        directory = Path(self.pending["session_dir"])
        for key, path in (("original", directory / "preview.png"),
                          ("rectified", directory / "artwork.png"),
                          ("ai", Path(self.pending["output_path"]))):
            if path.is_file():
                self.emit("image", key=key, path=str(path))
        self.emit("session", directory=str(directory), session_id=directory.name)

    def confirm_printed(self) -> bool:
        with self._gate:
            if (
                not self.print_enabled or self.state != "AWAITING_PRINT_CONFIRM" or not self.pending
                or self.pending.get("status") != "spooled"
                or self.pending.get("needs_confirmation") is not True
            ):
                return False
            self._complete_print("operator_confirmed_paper_output")
            return True

    def _complete_print(self, confirmation: str) -> None:
        self.pending.update(status="completed", confirmation=confirmation, completed_at=utc_now_iso())
        self._save_pending()
        self.pending = None
        self.emit("stage", step="print", status="done", message="打印完成")
        self.emit("print_controls", confirm=False, retry=False)
        elapsed = time.monotonic() - self.round_started if self.round_started is not None else None
        self.emit("completed", elapsed=elapsed)
        self._set_state("READY", "本轮打印完成，P 已重新激活。可放入下一张画作。")

    def retry_print_status(self) -> bool:
        if not self.print_enabled:
            return False
        return self._launch(("AWAITING_PRINT_CONFIRM", "PRINT_ERROR"), "PRINTING", "正在查询原打印作业，P 保持锁定…", self._resume_print)

    def _resume_print(self) -> None:
        try:
            if not self.pending or not self.pending.get("job_id"):
                raise ConfigError("没有可靠的打印作业编号，无法自动核实。请联系现场操作人员检查 Windows 原打印队列及实际出纸；不会自动重新打印或解锁 P。")
            from .gui_print import document_name_for

            settings = self.config.get("printer", {})
            result = self.printer.wait_for_job(
                self.pending["job_id"], printer=self.pending.get("printer"),
                timeout=settings.get("timeout_sec", 180),
                poll_interval=settings.get("poll_interval_sec", 0.5), progress=self._print_progress,
                expected_document=document_name_for(Path(self.pending["output_path"])),
            )
            self._print_result(record(result))
        except Exception as exc:
            self._fail_print(exc)

    def get_preview_frame(self):
        """读取独立视频解码线程的最新缓存帧；不执行截图或设备查询。"""
        return self.preview.latest_frame() if self.preview is not None else None

    def close(self) -> bool:
        with self._gate:
            if self.state in ("STARTING", "RUNNING", "PRINTING"):
                return False
            if self.preview is not None:
                self.preview.close()
                self.preview = None
                self._preview_started = False
            if self.camera is not None:
                self.camera.close()
            if self._phone_lock is not None:
                self._phone_lock.__exit__(None, None, None)
                self._phone_lock = None
            self.state = "CLOSED"
        return True
