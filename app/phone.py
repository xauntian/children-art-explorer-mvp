"""手机取景采集、作品确认和本地生成入口。

python -m app.phone --probe
python -m app.phone --mode confirm
python -m app.phone --mode direct
python -m app.phone --mode auto --headless
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .config import load_app_config, project_root_from
from .errors import ConfigError, PipelineError
from .phone_runtime import PhoneRunLock, reconcile_phone_sessions
from .preprocess import encode_png, resize_to_max_side
from .storage import JobStore, atomic_write_bytes, atomic_write_json, new_job_id, utc_now_iso

MODES = ("confirm", "direct", "auto")
EXTRACTION_METHODS = ("checkerboard", "aruco", "paper")
CAPTURE_KEY_MODES = {"p": "direct", "t": "confirm", "r": "confirm"}


def load_phone_config(path: Path | None = None, project_root: Path | None = None) -> dict:
    root = (project_root or project_root_from()).resolve()
    config_path = path or root / "configs" / "phone_capture.json"
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"手机配置无法读取: {config_path}: {exc}") from exc
    if not isinstance(config, dict) or config.get("mode", "confirm") not in MODES:
        raise ConfigError("手机配置必须是对象，mode 必须为 confirm/direct/auto")
    if not isinstance(config.get("phone"), dict):
        raise ConfigError("手机配置缺少 phone 对象")
    if config.get("capture_backend", "adb") not in ("adb", "camera_stream"):
        raise ConfigError("capture_backend 必须为 adb 或 camera_stream")
    if config.get("capture_backend") == "camera_stream" and not isinstance(config.get("stream"), dict):
        raise ConfigError("camera_stream 采集需要 stream 配置对象")
    if not isinstance(config.get("extraction", {}), dict):
        raise ConfigError("extraction 必须为对象")
    if config.get("extraction", {}).get("method", "checkerboard") not in EXTRACTION_METHODS:
        raise ConfigError("extraction.method 必须为 checkerboard、aruco 或 paper")
    if not isinstance(config.get("generation_warmup", False), bool):
        raise ConfigError("generation_warmup 必须为布尔值")
    directory = config.get("capture_runs_dir", "runs/phone")
    if not isinstance(directory, str) or not directory:
        raise ConfigError("capture_runs_dir 必须为非空路径")
    output = (root / directory).resolve()
    if not output.is_relative_to(root / "runs"):
        raise ConfigError("手机采集记录必须保存在项目 runs 目录内")
    return {**config, "project_root": root, "capture_runs_dir": output}


class PhoneSession:
    """前置采集记录；不占用生成状态机，也不提前随机画派。"""

    def __init__(self, directory: Path, mode: str):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.meta: dict[str, Any] = {
            "session_id": directory.name,
            "mode": mode,
            "state": "CAPTURING",
            "created_at": utc_now_iso(),
            "generation_job_id": None,
            "history": [],
            "timings": {},
        }
        self.set_state("CAPTURING")

    @property
    def state(self) -> str:
        return self.meta["state"]

    def save(self) -> None:
        self.meta["updated_at"] = utc_now_iso()
        atomic_write_json(self.directory / "meta.json", self.meta)

    def set_state(self, state: str) -> None:
        self.meta["state"] = state
        self.meta["history"].append({"state": state, "at": utc_now_iso()})
        self.save()

    def fail(self, exc: Exception) -> None:
        self.meta["error"] = (
            exc.to_dict()
            if isinstance(exc, PipelineError)
            else {"category": "phone_error", "message": str(exc)}
        )
        self.set_state("FAILED")


class PhoneController:
    def __init__(self, config: dict, *, camera=None, extractor=None, app_factory=None, observer=None):
        self.config = config
        self.root = config["project_root"]
        self.extraction_method = config.get("extraction", {}).get("method", "checkerboard")
        if self.extraction_method not in EXTRACTION_METHODS:
            raise ConfigError("extraction.method 必须为 checkerboard、aruco 或 paper")
        if camera is None:
            if config.get("capture_backend", "adb") == "camera_stream":
                from .phone_stream import PhoneStreamCapture

                camera = PhoneStreamCapture(self.root, stream=config["stream"], **config["phone"])
            else:
                from .phone_capture import PhoneCapture

                camera = PhoneCapture(self.root, **config["phone"])
        if extractor is None:
            if self.extraction_method == "paper":
                from .paper_extract import extract_paper_artwork

                extractor = extract_paper_artwork
            elif self.extraction_method == "aruco":
                from .fast_artwork_extract import extract_fast_artwork

                extractor = extract_fast_artwork
            else:
                from .artwork_extract import extract_artwork

                extractor = extract_artwork
        self.camera = camera
        self.extractor = extractor
        self.app_factory = app_factory
        self.observer = observer
        self.session: PhoneSession | None = None
        self._lock = threading.Lock()
        self._busy = False
        app_config = load_app_config(self.root)
        self.max_side = app_config.pre_max_input_side
        self.generation_store = JobStore(app_config.runs_dir, app_config.temp_suffix)

    def _notify(self, event: str, session: PhoneSession) -> None:
        if self.observer is not None:
            self.observer(event, session)

    def _enter(self) -> None:
        with self._lock:
            if self._busy:
                raise ConfigError("当前任务正在处理，请等待完成")
            self._busy = True

    def _leave(self) -> None:
        with self._lock:
            self._busy = False

    @staticmethod
    def _save_capture(session: PhoneSession, capture) -> None:
        suffix = capture.meta.get("raw_suffix")
        if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"):
            suffix = ".png" if capture.raw_bytes.startswith(b"\x89PNG\r\n\x1a\n") else ".jpg"
        raw_name = "original" + suffix
        atomic_write_bytes(session.directory / raw_name, capture.raw_bytes)
        session.meta["capture"] = {
            **capture.meta,
            "source": capture.source,
            "raw_file": raw_name,
            "sha256": hashlib.sha256(capture.raw_bytes).hexdigest(),
        }
        atomic_write_bytes(session.directory / "preview.png", encode_png(capture.preview_image))
        session.save()

    def prepare(self, mode: str) -> PhoneSession:
        if mode not in MODES:
            raise ConfigError(f"未知模式: {mode}")
        self._enter()
        session = None
        try:
            active = self.generation_store.find_active()
            if active is not None:
                raise ConfigError(f"已有生成任务 {active.job_id} 正在处理，请先完成该任务")
            if self.session and self.session.state in ("AWAITING_INPUT_CONFIRM", "INPUT_READY"):
                self.session.set_state("SUPERSEDED")
            directory = self.config["capture_runs_dir"] / new_job_id()
            session = self.session = PhoneSession(directory, mode)
            self._notify("capturing", session)
            started = time.monotonic()
            capture = self.camera.capture()
            self._save_capture(session, capture)
            if capture.source == "camera_stream":
                session.meta["camera_zoom"] = capture.meta["camera_zoom"]
                session.meta["camera_zoom_source"] = "camera2_stream_setting"
            session.meta["timings"]["capture_sec"] = round(time.monotonic() - started, 3)
            session.set_state("EXTRACTING")
            self._notify("captured", session)
            started = time.monotonic()
            photo_fallback_elapsed = 0.0
            profile = {k: v for k, v in self.config.get("extraction", {}).items() if k != "method"}
            if profile.get("board_spec_path"):
                profile["board_spec_path"] = str((self.root / profile["board_spec_path"]).resolve())
            session.meta["extraction_method"] = self.extraction_method
            if capture.source in ("camera_photo", "camera_stream"):
                profile = {
                    k: v
                    for k, v in profile.items()
                    if k not in ("ignore_regions", "preview_occlusion_regions")
                }
            from .artwork_extract import ArtworkExtractError

            result = None
            fallback_reason = None
            try:
                result = self.extractor(capture.preview_image, profile=profile)
            except ArtworkExtractError as exc:
                if capture.source not in ("preview_screenshot", "adb_screencap"):
                    raise
                if self.extraction_method in ("aruco", "paper"):
                    # 缺失定位标记或完整纸边时立即失败，不为不可靠几何浪费一次快门。
                    raise
                if not self.config["phone"].get("fallback_photo", True):
                    raise
                fallback_reason = "preview_extraction_unreliable"
                session.meta["preview_extraction_error"] = exc.to_dict()
            if (
                result is not None
                and capture.source in ("preview_screenshot", "adb_screencap")
                and result.meta.get("preview_occlusion")
            ):
                fallback_reason = "camera_ui_overlaps_artwork"
            if fallback_reason:
                # 保留被 UI 遮挡的截图，改用同一 2× 取景的新照片；最多一次回退。
                rejected = dict(session.meta["capture"])
                raw_path = directory / rejected["raw_file"]
                rejected_name = "preview-rejected-raw" + raw_path.suffix
                raw_path.rename(directory / rejected_name)
                (directory / "preview.png").rename(directory / "preview-rejected.png")
                rejected["raw_file"] = rejected_name
                rejected["reason"] = fallback_reason
                session.meta["rejected_preview"] = rejected
                session.meta["capture"] = rejected
                if result is not None:
                    atomic_write_bytes(
                        directory / "preview-rejected-detection.png", encode_png(result.overlay)
                    )
                session.save()
                fallback_started = time.monotonic()
                capture = self.camera.capture_photo(reason=fallback_reason)
                self._save_capture(session, capture)
                photo_fallback_elapsed = time.monotonic() - fallback_started
                session.meta["timings"]["photo_fallback_sec"] = round(photo_fallback_elapsed, 3)
                session.meta["timings"]["capture_sec"] = round(
                    session.meta["timings"]["capture_sec"] + photo_fallback_elapsed, 3
                )
                photo_profile = {
                    k: v
                    for k, v in profile.items()
                    if k not in ("ignore_regions", "preview_occlusion_regions")
                }
                result = self.extractor(capture.preview_image, profile=photo_profile)
            if result.meta.get("preview_occlusion"):
                raise ConfigError("相机界面遮挡了作品，请重新采集")
            for name, image in (
                ("detection.png", result.overlay),
                ("board.png", result.board_image),
                ("mask.png", result.mask),
                ("artwork.png", result.artwork),
            ):
                atomic_write_bytes(directory / name, encode_png(image))
            model_input = resize_to_max_side(result.model_input, self.max_side)
            input_bytes = encode_png(model_input)
            atomic_write_bytes(directory / "input.png", input_bytes)
            session.meta["extraction"] = result.meta
            session.meta["input_sha256"] = hashlib.sha256(input_bytes).hexdigest()
            session.meta["input_size"] = [model_input.shape[1], model_input.shape[0]]
            session.meta["timings"]["extraction_sec"] = round(
                time.monotonic() - started - photo_fallback_elapsed, 3
            )
            session.set_state("AWAITING_INPUT_CONFIRM" if mode == "confirm" else "INPUT_READY")
            self._notify("prepared", session)
            return session
        except Exception as exc:
            if session is not None:
                session.fail(exc)
            raise
        finally:
            self._leave()

    def generate(self, session: PhoneSession) -> dict:
        self._enter()
        try:
            if session is not self.session or session.state not in (
                "AWAITING_INPUT_CONFIRM",
                "INPUT_READY",
            ):
                raise ConfigError("该输入已失效或正在处理，不能重复确认")
            data = (session.directory / "input.png").read_bytes()
            if hashlib.sha256(data).hexdigest() != session.meta["input_sha256"]:
                raise ConfigError("待确认作品文件发生变化，请重新采集")
            session.meta["input_confirmed_at"] = utc_now_iso()
            session.meta["input_confirmation"] = (
                "manual" if session.meta["mode"] == "confirm" else "automatic"
            )
            generation_id = "phone-" + session.directory.name
            session.meta["generation_job_id"] = generation_id
            started = time.monotonic()
            session.set_state("GENERATING")
            self._notify("generating", session)
            if self.app_factory is None:
                from .main import App

                app = App(config=load_app_config(self.root).for_artwork())
            else:
                app = self.app_factory()
            run_options = {}
            if getattr(getattr(app, "config", None), "generation_time_budget_sec", None) is not None:
                run_options["started_at"] = started
            summary = app.pipeline.run(
                source_bytes=data,
                source_name=str(session.directory / "input.png"),
                job_id=generation_id,
                **run_options,
            )
            session.meta["generation"] = summary
            if "generation_deadline" in summary:
                session.meta["generation_deadline"] = summary["generation_deadline"]
            session.meta["timings"]["generation_sec"] = round(time.monotonic() - started, 3)
            if summary.get("state") == "COMPLETED":
                session.meta["output_path"] = summary.get("output_path") or str(
                    self.root / "runs" / generation_id / "output.png"
                )
                session.set_state("COMPLETED")
                self._notify("generated", session)
            else:
                session.meta["error"] = summary.get("error") or {"message": "任务失败，原因已保存"}
                session.set_state("FAILED")
            return session.meta
        except Exception as exc:
            if session is self.session and session.state not in (
                "COMPLETED",
                "CANCELLED",
                "SUPERSEDED",
            ):
                session.fail(exc)
            raise
        finally:
            self._leave()

    def cancel(self) -> None:
        with self._lock:
            if self._busy:
                raise ConfigError("正在处理，不能取消已提交的任务")
            if self.session and self.session.state in ("AWAITING_INPUT_CONFIRM", "INPUT_READY"):
                self.session.set_state("CANCELLED")

    def run_auto(self) -> dict:
        return self.generate(self.prepare("auto"))


class PhoneWindow:
    """简单本地窗口；采集时隐藏自身，避免进入手机拍屏画面。"""

    def __init__(self, controller: PhoneController, mode: str):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("AMCC5000 · 手机作品采集")
        self.root.geometry("1180x820")
        camera_hint = "相机串流已就绪" if controller.config.get("capture_backend") == "camera_stream" else "相机保持 2×"
        self.status = tk.StringVar(value=f"{camera_hint}。P 采集并生成，T 仅采集，Enter 确认生成。")
        self.busy = False
        self.events: queue.Queue = queue.Queue()
        self.photos: dict = {}
        self.held: set[str] = set()
        self.pending_release: dict[str, str] = {}
        bar = ttk.Frame(self.root, padding=12)
        bar.pack(fill="x")
        ttk.Label(bar, text=f"入口: {mode}").pack(side="left", padx=8)
        self.capture_button = ttk.Button(bar, text="采集并生成 / P", command=self.capture)
        self.capture_button.pack(side="left", padx=6)
        self.test_button = ttk.Button(
            bar, text="仅采集 / T", command=lambda: self.capture(mode="confirm")
        )
        self.test_button.pack(side="left", padx=6)
        self.confirm_button = ttk.Button(
            bar, text="确认并生成 / Enter", command=self.confirm, state="disabled"
        )
        self.confirm_button.pack(side="left", padx=6)
        self.retake_button = ttk.Button(
            bar, text="仅重取 / R", command=lambda: self.capture(mode="confirm")
        )
        self.retake_button.pack(side="left", padx=6)
        self.cancel_button = ttk.Button(bar, text="取消 / Esc", command=self.cancel)
        self.cancel_button.pack(side="left", padx=6)
        self.auto_button = ttk.Button(
            bar, text="自动调试一轮", command=lambda: self.capture(auto=True)
        )
        self.auto_button.pack(side="left", padx=6)
        ttk.Label(self.root, textvariable=self.status, padding=(12, 8), wraplength=1130).pack(
            fill="x"
        )
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=8)
        self.panels = {}
        for key, title in (
            ("preview", "相机画面"),
            ("detection", "识别边界"),
            ("artwork", "完整校正作品"),
            ("input", "模型输入"),
            ("output", "生成结果"),
        ):
            frame = ttk.Frame(self.tabs)
            label = ttk.Label(frame, anchor="center")
            label.pack(fill="both", expand=True)
            self.tabs.add(frame, text=title)
            self.panels[key] = label
        self.location = tk.StringVar(value=str(controller.config["capture_runs_dir"]))
        ttk.Label(self.root, textvariable=self.location, padding=12, wraplength=1130).pack(fill="x")
        self.root.bind("<KeyPress>", self._key_down)
        self.root.bind("<KeyRelease>", self._key_up)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._poll)

    def _key_down(self, event) -> None:
        key = event.keysym.lower()
        if key in self.pending_release:
            self.root.after_cancel(self.pending_release.pop(key))
        if key in self.held:
            return
        self.held.add(key)

    def _key_up(self, event) -> None:
        key = event.keysym.lower()
        self.pending_release[key] = self.root.after(80, lambda: self._release(key))

    def _release(self, key: str) -> None:
        self.pending_release.pop(key, None)
        self.held.discard(key)
        # 释放后触发，避免采集隐藏窗口期间收不到 KeyRelease 而卡住 P。
        if key in CAPTURE_KEY_MODES:
            self.capture(mode=CAPTURE_KEY_MODES[key])
        elif key == "return":
            self.confirm()
        elif key == "escape":
            self.cancel()
        elif key == "q":
            self.close()

    def _set_busy(self, value: bool) -> None:
        self.busy = value
        state = "disabled" if value else "normal"
        for button in (
            self.capture_button,
            self.test_button,
            self.retake_button,
            self.auto_button,
            self.cancel_button,
        ):
            button.configure(state=state)
        self.confirm_button.configure(state="disabled")

    def capture(self, auto: bool = False, *, mode: str = "direct") -> None:
        if self.busy:
            return
        self._set_busy(True)
        self.status.set("正在获取新的相机画面…")
        mode = "auto" if auto else mode
        # 本窗口临时隐藏；后面的用户测试图保持原样。
        self.root.withdraw()
        self.root.after(450, lambda: self._start_capture(mode))

    def _start_capture(self, mode: str) -> None:
        def work():
            try:
                session = self.controller.prepare(mode)
                self.events.put(("prepared", session))
                if mode != "confirm":
                    self.controller.generate(session)
                    self.events.put(("finished", session))
            except Exception as exc:
                self.events.put(("error", exc))

        threading.Thread(target=work, daemon=True).start()

    def confirm(self) -> None:
        session = self.controller.session
        if self.busy or not session or session.state != "AWAITING_INPUT_CONFIRM":
            return
        self._set_busy(True)
        self.status.set("已确认输入，正在随机分配画派并生成…")

        def work():
            try:
                self.controller.generate(session)
                self.events.put(("finished", session))
            except Exception as exc:
                self.events.put(("error", exc))

        threading.Thread(target=work, daemon=True).start()

    def cancel(self) -> None:
        if self.busy:
            return
        self.controller.cancel()
        self.confirm_button.configure(state="disabled")
        self.status.set("已取消待确认输入。P 采集并生成，T 仅采集。")

    def _show_image(self, key: str, path: Path) -> None:
        from PIL import Image, ImageTk

        if not path.is_file():
            return
        with Image.open(path) as original:
            img = original.convert("RGBA")
            if key == "artwork":
                # 透明区域只在预览中合成灰底；文件本身保持 alpha。
                background = Image.new("RGBA", img.size, (210, 210, 210, 255))
                background.alpha_composite(img)
                img = background
            img.thumbnail((1100, 580), Image.Resampling.LANCZOS)
            self.photos[key] = ImageTk.PhotoImage(img)
        self.panels[key].configure(image=self.photos[key])

    def _prepared(self, session: PhoneSession) -> None:
        self.root.deiconify()
        for key in ("preview", "detection", "artwork", "input"):
            self._show_image(key, session.directory / (key + ".png"))
        self.location.set(str(session.directory))
        self.tabs.select(2)
        if session.meta["mode"] == "confirm":
            self._set_busy(False)
            self.confirm_button.configure(state="normal")
            self.status.set("请检查抠图和方向。Enter 确认并生成，R 重取画面。")
        else:
            self.status.set("作品已提取，正在生成并保存…")

    def _poll(self) -> None:
        try:
            while True:
                event, value = self.events.get_nowait()
                if event == "prepared":
                    self._prepared(value)
                elif event == "finished":
                    self._set_busy(False)
                    if value.state == "COMPLETED":
                        self._show_image("output", Path(value.meta["output_path"]))
                        self.tabs.select(4)
                        style = value.meta.get("generation", {}).get("style_id", "")
                        self.status.set(f"完成 · {style}。按 P 处理下一件作品。")
                    else:
                        self.status.set("任务失败，原因已保存。按 P 重新开始。")
                elif event == "error":
                    self.root.deiconify()
                    self._set_busy(False)
                    self.status.set(f"失败：{value}")
                    session = self.controller.session
                    if session:
                        self.location.set(str(session.directory))
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def close(self) -> None:
        if self.busy:
            self.status.set("任务正在处理，请等待结束后关闭，避免中断生成。")
            return
        self.controller.cancel()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def read_console_key() -> str:
    """读取终端真实按键，不发送任何桌面输入；长按释放后才触发。"""
    if os.name != "nt":
        return input().strip().lower() or "enter"
    import ctypes
    import msvcrt

    while msvcrt.kbhit():
        msvcrt.getwch()  # 忙碌期间的按键不排队。
    value = msvcrt.getwch()
    if value in ("\x00", "\xe0"):
        msvcrt.getwch()
        return ""
    if value == "\x03":
        raise KeyboardInterrupt
    virtual_key = {"\r": 13, "\x1b": 27}.get(value, ord(value.upper()) if value else 0)
    if virtual_key:
        while ctypes.windll.user32.GetAsyncKeyState(virtual_key) & 0x8000:
            time.sleep(0.02)
    return {"\r": "enter", "\x1b": "escape"}.get(value, value.lower())


class PhoneConsole:
    """默认命令行交互，不打开或操控桌面窗口。"""

    def __init__(self, controller: PhoneController, mode: str, *, key_reader=read_console_key):
        self.controller = controller
        self.mode = mode
        self.key_reader = key_reader

    @staticmethod
    def show_session(session: PhoneSession) -> None:
        print(f"状态: {session.state}", flush=True)
        print(f"记录: {session.directory}", flush=True)
        if (session.directory / "artwork.png").is_file():
            print(f"完整校正作品: {session.directory / 'artwork.png'}", flush=True)
            print(f"模型输入: {session.directory / 'input.png'}", flush=True)
        if session.meta.get("output_path"):
            print(f"最终作品: {session.meta['output_path']}", flush=True)
        if session.meta.get("error"):
            print(json.dumps(session.meta["error"], ensure_ascii=False), flush=True)

    def run(self) -> int:
        if sys.stdin is None or not sys.stdin.isatty():
            raise ConfigError("交互模式需要终端；自动调试请使用 --mode auto --headless")
        camera_hint = "相机实时串流" if self.controller.config.get("capture_backend") == "camera_stream" else "2×"
        print(f"Mate 40 Pro · {camera_hint} · {self.mode}", flush=True)
        print(
            "P 采集并生成，T 仅采集，Enter 确认当前输入并生成，R 仅重取，Esc 取消，Q 退出。",
            flush=True,
        )
        try:
            while True:
                key = self.key_reader()
                session = self.controller.session
                if key == "q":
                    self.controller.cancel()
                    return 0
                try:
                    if key in CAPTURE_KEY_MODES:
                        capture_mode = CAPTURE_KEY_MODES[key]
                        session = self.controller.prepare(capture_mode)
                        self.show_session(session)
                        if capture_mode == "direct":
                            print("正在生成并保存作品…", flush=True)
                            self.controller.generate(session)
                            self.show_session(session)
                        else:
                            print(
                                "Enter 确认此输入并生成，T/R 仅重新采集，P 采集新图并生成。",
                                flush=True,
                            )
                    elif key == "enter":
                        if session is None or session.state != "AWAITING_INPUT_CONFIRM":
                            print("当前没有待确认输入。", flush=True)
                            continue
                        print("正在生成并保存作品…", flush=True)
                        self.controller.generate(session)
                        self.show_session(session)
                    elif key == "escape":
                        self.controller.cancel()
                        print("待确认输入已取消。", flush=True)
                except Exception as exc:
                    logging.exception("手机采集轮次失败")
                    print(f"失败: {exc}", flush=True)
                    if self.controller.session:
                        self.show_session(self.controller.session)
        except KeyboardInterrupt:
            self.controller.cancel()
            return 0


def require_preview_calibration(config: dict) -> None:
    """生产入口必须先限定相机取景区域；probe 仍可采完整截图用于校准。"""
    if config.get("capture_backend") == "camera_stream":
        return
    phone = config["phone"]
    if phone.get("preview_roi") is None or phone.get("preview_screen_size") is None:
        raise ConfigError("请先配置 preview_roi 和 preview_screen_size，避免把相机界面作为作品输入")


def run_phone(controller: PhoneController, mode: str, *, gui: bool = False) -> int:
    """锁覆盖采集、等待确认和生成，旧正式任务只恢复关联，不重新提交。"""
    require_preview_calibration(controller.config)
    with PhoneRunLock(controller.config["capture_runs_dir"]):
        reports = reconcile_phone_sessions(
            controller.config["capture_runs_dir"], controller.generation_store
        )
        blocked = [report for report in reports if report["blocking"]]
        if blocked:
            raise ConfigError("旧采集任务需要处理: " + json.dumps(blocked, ensure_ascii=False))
        if controller.config.get("generation_warmup", False):
            active = controller.generation_store.find_active()
            if active is not None:
                raise ConfigError(f"已有生成任务 {active.job_id} 正在处理，请先完成该任务")
            from .model_warmup import warmup_generation

            print("模型预热中，完成后可拍摄……", flush=True)
            report = warmup_generation(controller.root)
            if report is not None:
                print(f"模型预热完成（{report['total_sec']:.3f} 秒），可拍摄。", flush=True)
        streaming = controller.config.get("capture_backend") == "camera_stream"
        try:
            if streaming:
                print("正在连接手机相机串流……", flush=True)
                controller.camera.start()
                print("相机串流已就绪，按 P 截取新帧并生成。", flush=True)
            if mode == "auto":
                result = controller.run_auto()
                if sys.stdout is not None:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0 if result["state"] == "COMPLETED" else 1
            if gui:
                PhoneWindow(controller, mode).run()
                return 0
            return PhoneConsole(controller, mode).run()
        finally:
            if streaming:
                controller.camera.close()


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Mate 40 Pro 相机串流、作品采集与生成")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--headless", action="store_true", help="auto 模式无窗口运行一轮")
    presentation = parser.add_mutually_exclusive_group()
    presentation.add_argument("--cli", action="store_true", help="命令行按键模式（默认）")
    presentation.add_argument("--gui", action="store_true", help="可选本地窗口；默认不打开")
    parser.add_argument("--probe", action="store_true", help="仅检查设备和预览，不触发快门或生成")
    args = parser.parse_args(argv)
    root = project_root_from()
    log_dir = root / "runs" / "runtime"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / "phone.log", encoding="utf-8", level=logging.INFO)
    try:
        config = load_phone_config(args.config, root)
        controller = PhoneController(config)
        if args.probe:
            report = controller.camera.probe()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report.get("available") else 1
        mode = args.mode or config.get("mode", "confirm")
        if args.headless and mode != "auto":
            raise ConfigError("无窗口入口仅适用于 auto 模式")
        return run_phone(controller, mode, gui=args.gui)
    except Exception as exc:
        logging.exception("手机采集入口失败")
        if sys.stderr is not None:
            print(f"手机采集失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
