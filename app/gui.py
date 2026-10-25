"""AMCC5000 本地桌面控制台：python -m app.gui。"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import project_root_from
from .gui_runtime import GuiRuntime, load_gui_config

BG = "#F4F4EF"
CARD = "#FFFFFF"
INK = "#203B34"
MUTED = "#6C7C75"
GREEN = "#19745B"
PALE = "#E5F1EA"
LINE = "#DDE4DD"
RED = "#B64838"
STEPS = (("capture", "原始采集"), ("rectify", "裁切与校正"),
         ("generate", "AI 生成"), ("save", "结果保存"), ("print", "打印出纸"))


def hong_kong_time():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S")


class StudioWindow:
    def __init__(self, runtime: GuiRuntime):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk, self.runtime = tk, ttk, runtime
        self.root = tk.Tk()
        self.root.title("AMCC5000 · 艺术探索工作台")
        self.root.configure(bg=BG)
        width = min(1360, self.root.winfo_screenwidth() - 80)
        height = min(940, self.root.winfo_screenheight() - 100)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(1060, 730)
        self.root.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.photos = {}
        self.image_paths = {}
        self.ai_notice = None
        self.stage_states = {key: "skipped" if key == "print" and not runtime.print_enabled else "waiting" for key, _ in STEPS}
        self.last_preview_sequence = None
        self.preview_live = False
        self.last_progress = ""
        self.started = None
        self.held = set()
        self.release_after = {}
        self.status = tk.StringVar(value="点击「开始」准备运行环境，然后按 P 处理一张画作。")
        self.location = tk.StringVar(value="每轮图片与记录将保存到本项目的 runs 文件夹")
        self.elapsed = tk.StringVar(value="等待开始")
        self.style_label = tk.StringVar(value="随机画派 · 等待生成")
        self._styles()
        self._build()
        self.root.bind("<KeyPress>", self._key_down)
        self.root.bind("<KeyRelease>", self._key_up)
        self.root.bind("<FocusOut>", self._focus_out)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(80, self._poll)

    def _styles(self):
        style = self.ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Start.TButton", background=GREEN, foreground="white", borderwidth=0,
                        font=("Microsoft YaHei UI", 12, "bold"), padding=(24, 12))
        style.map("Start.TButton", background=[("disabled", "#CCD9D1"), ("active", "#115C47")],
                  foreground=[("disabled", "#74867B")])
        style.configure("Action.TButton", background=PALE, foreground=GREEN, borderwidth=0, padding=(18, 10))
        style.map("Action.TButton", background=[("disabled", "#EDF0EB"), ("active", "#D2E9DA")],
                  foreground=[("disabled", "#9CA89E")])
        style.configure("Small.TButton", background=CARD, foreground=INK, bordercolor=LINE, padding=(12, 7))

    def _label(self, parent, text=None, variable=None, size=10, color=INK, bg=CARD, bold=False, **kwargs):
        return self.tk.Label(parent, text=text, textvariable=variable, fg=color, bg=bg,
                             font=("Microsoft YaHei UI", size, "bold" if bold else "normal"), **kwargs)

    def _build(self):
        tk, ttk = self.tk, self.ttk
        shell = tk.Frame(self.root, bg=BG)
        shell.pack(fill="both", expand=True, padx=26, pady=22)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(3, weight=1)
        header = tk.Frame(shell, bg=BG)
        header.grid(row=0, column=0, sticky="ew")
        title = tk.Frame(header, bg=BG)
        title.pack(side="left", fill="both", expand=True)
        self._label(title, "AMCC5000  /  LOCAL ART STUDIO", size=9, color=GREEN, bg=BG, bold=True).pack(anchor="w")
        self._label(title, "艺术探索工作台", size=25, bg=BG, bold=True).pack(anchor="w", pady=(7, 4))
        self._label(title, "从纸上创作，到一张新的艺术作品。", color=MUTED, bg=BG).pack(anchor="w")

        control = tk.Frame(shell, bg=CARD, padx=18, pady=14, highlightbackground=LINE, highlightthickness=1)
        control.grid(row=1, column=0, sticky="ew", pady=(20, 12))
        self.start_button = ttk.Button(control, text="开始", style="Start.TButton", command=self.start)
        self.start_button.pack(side="left")
        self.run_button = ttk.Button(control, text="采集并运行  /  P", style="Action.TButton", command=self.trigger, state="disabled")
        self.run_button.pack(side="left", padx=(12, 18))
        status_box = tk.Frame(control, bg=CARD)
        status_box.pack(side="left", fill="both", expand=True)
        self.badge = self._label(status_box, "● 尚未启动", color=MUTED, bold=True)
        self.badge.pack(anchor="w")
        self._label(status_box, variable=self.status, color=MUTED, size=9, wraplength=640, justify="left").pack(anchor="w", pady=(5, 0))
        self._label(control, variable=self.elapsed, size=11, color=GREEN, bold=True).pack(side="right", padx=(12, 0))

        dependencies = tk.Frame(shell, bg=BG)
        dependencies.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        self.dependency_labels = {}
        for index, (name, title) in enumerate((("phone", "手机相机"), ("ai", "本地 AI"), ("printer", "打印机"))):
            dependencies.columnconfigure(index, weight=1, uniform="dep")
            box = tk.Frame(dependencies, bg=BG)
            box.grid(row=0, column=index, sticky="ew", padx=(0, 14 if index < 2 else 0))
            self._label(box, title, color=MUTED, bg=BG, size=8).pack(anchor="w")
            text = "○ 已跳过 · 仅保存数字图片" if name == "printer" and not self.runtime.print_enabled else "○ 等待检查"
            label = self._label(box, text, bg=BG, size=9, anchor="w")
            label.pack(fill="x", pady=(3, 0))
            self.dependency_labels[name] = label

        previews = tk.Frame(shell, bg=BG)
        previews.grid(row=3, column=0, sticky="nsew")
        self.panels, self.subtitles, self.panel_titles = {}, {}, {}
        for index, (key, number, title, caption) in enumerate((
            ("original", "01", "手机实时取景", "等待连接手机屏幕视频"),
            ("rectified", "02", "校正后的作品", "保留完整纸面与画作内容"),
            ("ai", "03", "AI 生成结果", "随机画派 · A4 比例输出"),
        )):
            previews.columnconfigure(index, weight=1, uniform="preview")
            previews.rowconfigure(0, weight=1)
            box = tk.Frame(previews, bg=CARD, highlightbackground=LINE, highlightthickness=1)
            box.grid(row=0, column=index, sticky="nsew", padx=(0, 14 if index < 2 else 0))
            top = tk.Frame(box, bg=CARD, padx=16, pady=13)
            top.pack(fill="x")
            self._label(top, number, size=10, color=GREEN, bold=True).pack(side="left", padx=(0, 9))
            title_label = self._label(top, title, size=12, bold=True)
            title_label.pack(side="left")
            self.panel_titles[key] = title_label
            canvas = tk.Canvas(box, bg="#F0F2ED", highlightthickness=0, width=300, height=270)
            canvas.pack(fill="both", expand=True, padx=10)
            canvas.bind("<Configure>", lambda event, k=key: self._render_image(k))
            self.panels[key] = canvas
            subtitle = self._label(box, caption, size=8, color=MUTED, anchor="w")
            subtitle.pack(fill="x", padx=15, pady=(11, 14))
            self.subtitles[key] = subtitle
            self._render_image(key)

        steps = tk.Frame(shell, bg=BG)
        steps.grid(row=4, column=0, sticky="ew", pady=(15, 10))
        self.step_labels = {}
        for index, (key, title) in enumerate(STEPS):
            steps.columnconfigure(index, weight=1, uniform="step")
            word = "已跳过" if key == "print" and not self.runtime.print_enabled else "等待"
            label = self._label(steps, f"{index + 1:02d}   {title}  ·  {word}", bg="#E9ECE5", color=MUTED, size=9, pady=12)
            label.grid(row=0, column=index, sticky="ew", padx=(0, 7 if index < 4 else 0))
            self.step_labels[key] = label

        row = tk.Frame(shell, bg=BG)
        row.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self._label(row, variable=self.style_label, bg=BG, size=9, color=GREEN).pack(side="left")
        self.print_controls = tk.Frame(row, bg=BG)
        self.print_controls.pack(side="right")
        self.retry_button = ttk.Button(self.print_controls, text="重新检查原打印", style="Small.TButton", command=self.retry_status)
        self.confirm_button = ttk.Button(self.print_controls, text="确认已出纸", style="Action.TButton", command=self.confirm_printed)

        self.log = tk.Text(shell, height=4, bg=CARD, fg=MUTED, relief="flat", borderwidth=0,
                           font=("Microsoft YaHei UI", 9), padx=12, pady=9, state="disabled", wrap="word")
        self.log.grid(row=6, column=0, sticky="ew")
        footer = tk.Frame(shell, bg=BG)
        footer.grid(row=7, column=0, sticky="ew", pady=(8, 0))
        self.path_label = self._label(footer, variable=self.location, size=8, color=MUTED, bg=BG, anchor="w", cursor="hand2")
        self.path_label.pack(side="left", fill="x", expand=True)
        self.path_label.bind("<Button-1>", lambda event: self._open_output())
        self._label(footer, "手机保持原生相机取景", size=8, color=MUTED, bg=BG).pack(side="right")
        self._append_log("点击开始后自动检查依赖、连接设备并预热模型。" + ("实体打印已关闭。" if not self.runtime.print_enabled else ""))

    def _append_log(self, message):
        if message == self.last_progress:
            return
        self.last_progress = message
        self.log.configure(state="normal")
        self.log.insert("end", f"{hong_kong_time()}  {message}\n")
        if int(self.log.index("end-1c").split(".")[0]) > 200:
            self.log.delete("1.0", "30.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self):
        if self.runtime.start():
            self.start_button.configure(state="disabled", text="准备中…")
            self.root.focus_set()

    def trigger(self):
        if self.runtime.trigger():
            self.run_button.configure(state="disabled")
            self.root.focus_set()

    def confirm_printed(self):
        try:
            self.runtime.confirm_printed()
        except Exception as exc:
            self.status.set(f"打印完成记录未能保存，P 保持锁定：{exc}")
            self._append_log(str(exc))

    def retry_status(self):
        if self.runtime.retry_print_status():
            self.confirm_button.configure(state="disabled")
            self.retry_button.configure(state="disabled")

    def _key_down(self, event):
        if event.keysym.lower() != "p":
            return
        key = "p"
        if key in self.release_after:
            self.root.after_cancel(self.release_after.pop(key))
        if key in self.held:
            return "break"
        self.held.add(key)
        # 仅按下边沿触发；忙碌时忽略，并且不排入下一轮。
        if self.runtime.state == "READY":
            self.trigger()
        return "break"

    def _key_up(self, event):
        if event.keysym.lower() == "p":
            self.release_after["p"] = self.root.after(80, self._release_p)

    def _release_p(self):
        callback = self.release_after.pop("p", None)
        if callback is not None:
            self.root.after_cancel(callback)
        self.held.discard("p")

    def _focus_out(self, event):
        # 丢失焦点时清掉长按状态；新的按下仍须通过运行时互斥。
        if event.widget is self.root:
            self._release_p()

    def _render_image(self, key):
        from PIL import Image, ImageTk
        if key == "original" and self._refresh_preview(force=True):
            return
        canvas = self.panels.get(key)
        if canvas is None:
            return
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 20 or height < 20:
            return
        canvas.delete("all")
        path = self.image_paths.get(key)
        if not path:
            if key == "ai" and self.ai_notice:
                canvas.create_text(width / 2, height / 2, text=self.ai_notice, fill=RED,
                                   width=max(100, width - 30), font=("Microsoft YaHei UI", 11))
                return
            placeholder = {"original": "等待手机实时视频", "rectified": "等待纸面校正", "ai": "等待 AI 生成"}[key]
            canvas.create_text(width / 2, height / 2, text=placeholder, fill="#9AA79C", font=("Microsoft YaHei UI", 11))
            return
        try:
            with Image.open(path) as source:
                im = source.convert("RGB")
            im.thumbnail((width - 18, height - 18), Image.Resampling.LANCZOS)
            self.photos[key] = ImageTk.PhotoImage(im, master=self.root)
            canvas.create_image(width / 2, height / 2, image=self.photos[key])
        except (OSError, ValueError) as exc:
            canvas.create_text(width / 2, height / 2, text="预览暂不可用", fill=RED)
            logging.warning("GUI 图片预览失败: %s", exc)

    def _render_preview(self, frame):
        from PIL import Image, ImageTk
        canvas = self.panels["original"]
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 20 or height < 20:
            return
        image = Image.fromarray(frame.image[:, :, ::-1])
        if self.runtime.preview_roi is not None:
            left, top, right, bottom = self.runtime.preview_roi
            image = image.crop((round(left * image.width), round(top * image.height),
                                round(right * image.width), round(bottom * image.height)))
        image.thumbnail((width - 18, height - 18), Image.Resampling.LANCZOS)
        self.photos["original"] = ImageTk.PhotoImage(image, master=self.root)
        canvas.delete("all")
        canvas.create_image(width / 2, height / 2, image=self.photos["original"])
        self.panel_titles["original"].configure(text="手机实时取景")
        self.subtitles["original"].configure(text="实时视频 · 按 P 保存本轮原图")
        self.last_preview_sequence = frame.sequence
        self.preview_live = True

    def _refresh_preview(self, force=False):
        """仅显示视频线程已解码的新帧，不查询手机或排队截图。"""
        if self.runtime.state not in ("NEW", "STARTING", "READY", "ERROR") and "original" in self.image_paths:
            return False
        frame = self.runtime.get_preview_frame()
        if frame is None:
            if self.preview_live:
                self.panels["original"].delete("all")
                self.subtitles["original"].configure(text="视频暂不可用 · 请检查手机连接")
                self.preview_live = False
            return False
        if force or not self.preview_live or frame.sequence != self.last_preview_sequence:
            self._render_preview(frame)
        return True

    def _stage(self, data):
        key, status = data["step"], data["status"]
        if key not in self.step_labels:
            return
        self.stage_states[key] = status
        title = dict(STEPS)[key]
        index = [item[0] for item in STEPS].index(key) + 1
        word, color, bg = {"waiting": ("等待", MUTED, "#E9ECE5"),
                           "skipped": ("已跳过", MUTED, "#E9ECE5"),
                           "working": ("进行中", GREEN, "#DDEFE3"),
                           "done": ("完成", GREEN, PALE),
                           "error": ("失败", RED, "#F8E8E2")}[status]
        self.step_labels[key].configure(text=f"{index:02d}   {title}  ·  {word}", fg=color, bg=bg)
        if data.get("message"):
            self.status.set(data["message"])
            self._append_log(data["message"])

    def _handle(self, kind, data):
        if kind == "status":
            state = data["state"]
            ready = state == "READY"
            self.status.set(data["message"])
            self.run_button.configure(state="normal" if ready else "disabled")
            self.start_button.configure(state="normal" if state in ("NEW", "ERROR") else "disabled",
                                        text="重新准备" if state == "ERROR" else "开始" if state == "NEW" else "准备中…" if state == "STARTING" else "系统已启动")
            badge = {"READY": "● P 已激活", "ERROR": "● 准备未完成", "STARTING": "● 正在准备",
                     "RUNNING": "● 正在处理 · P 已锁定", "PRINTING": "● 查询打印 · P 已锁定",
                     "PRINT_ERROR": "● 打印待处理 · P 已锁定",
                     "AWAITING_PRINT_CONFIRM": "● 等待出纸确认 · P 已锁定"}.get(state, "● 尚未启动")
            self.badge.configure(text=badge, fg=GREEN if ready else RED if state in ("ERROR", "PRINT_ERROR") else MUTED)
            self._append_log(data["message"])
        elif kind == "dependency":
            label = self.dependency_labels[data["name"]]
            prefix = {"ready": "●", "working": "◌", "error": "!"}.get(data["status"], "○")
            text = data["message"].replace("\n", " ")
            label.configure(text=f"{prefix} {text[:52]}", fg=RED if data["status"] == "error" else GREEN if data["status"] == "ready" else MUTED)
            self._append_log(data["message"])
        elif kind == "stage":
            self._stage(data)
        elif kind == "reset":
            self.image_paths.clear()
            self.ai_notice = None
            self.subtitles["ai"].configure(text="随机画派 · A4 比例输出")
            self.started = time.monotonic()
            self.style_label.set("随机画派 · 等待生成")
            self.confirm_button.pack_forget()
            self.retry_button.pack_forget()
            for key, _ in STEPS:
                self._stage({"step": key, "status": "skipped" if key == "print" and not self.runtime.print_enabled else "waiting"})
            for key in self.panels:
                self._render_image(key)
        elif kind == "image":
            self.image_paths[data["key"]] = data["path"]
            if data["key"] == "original":
                self.preview_live = False
                self.panel_titles["original"].configure(text="本轮原始取景")
                self.subtitles["original"].configure(text="已保存本轮原图 · 处理中保持显示")
            elif data["key"] == "ai":
                self.ai_notice = None
                self.subtitles["ai"].configure(text="AI 图片已保存 · A4 比例输出")
            self._render_image(data["key"])
        elif kind == "style":
            self.style_label.set("本轮画派 · " + (data.get("style_name") or data["style_id"]))
        elif kind == "ai_timeout":
            self.image_paths.pop("ai", None)
            self.ai_notice = data["message"]
            self._render_image("ai")
            self.subtitles["ai"].configure(text="本轮已结束 · 可按 P 重新开始")
        elif kind == "session":
            self.location.set(data["directory"])
        elif kind == "print_controls":
            for button, field in ((self.retry_button, "retry"), (self.confirm_button, "confirm")):
                if data.get(field):
                    button.configure(state="normal")
                    button.pack(side="left", padx=(8, 0))
                else:
                    button.pack_forget()
        elif kind == "completed":
            self.started = None
            self.elapsed.set(f"本轮 {data['elapsed']:.1f}s" if data.get("elapsed") is not None else "本轮完成")
        elif kind == "error":
            self.started = None
            self.elapsed.set("需要处理")
            self._append_log(data["message"])

    def _poll(self):
        try:
            for _ in range(100):
                kind, data = self.runtime.events.get_nowait()
                self._handle(kind, data)
        except queue.Empty:
            pass
        if self.started is not None:
            self.elapsed.set(f"本轮 {time.monotonic() - self.started:.1f}s")
        self._refresh_preview()
        self.root.after(80, self._poll)

    def _open_output(self):
        path = Path(self.location.get())
        if path.is_dir() and os.name == "nt":
            os.startfile(str(path))

    def close(self):
        if self.runtime.close():
            self.root.destroy()
        else:
            self.status.set("当前操作尚未结束，请等待后关闭。")

    def run(self):
        self.root.mainloop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AMCC5000 桌面工作台")
    parser.add_argument("--config", type=Path, help="GUI 配置路径")
    args = parser.parse_args(argv)
    root = project_root_from()
    logs = root / "runs/runtime"
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=logs / "gui.log", encoding="utf-8", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        runtime = GuiRuntime(root, load_gui_config(root, args.config))
        StudioWindow(runtime).run()
        return 0
    except Exception as exc:
        logging.exception("桌面入口失败")
        import tkinter as tk
        from tkinter import messagebox
        root_window = tk.Tk()
        root_window.withdraw()
        messagebox.showerror("艺术探索工作台启动失败", f"{exc}\n\n详情：{logs / 'gui.log'}")
        root_window.destroy()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
