"""从 CLI 在指定显示器展示采集底板，只控制本工具自己的窗口。

只读勘查：
    python -m tools.show_capture_board --list-monitors
展示与 CLI 调整：
    python -m tools.show_capture_board --monitor 1 --scale .8 --y-offset -50 \
        --control-file runs/runtime/board-display.json
控制文件示例：{"scale": 0.8, "y_offset": -50, "close": false}。
scale 相对于将整张图片完整放入显示器的比例；offset 使用屏幕物理像素。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
from ctypes import wintypes
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _windows_api():
    if os.name != "nt":
        raise RuntimeError("显示器定位目前仅支持 Windows")
    return ctypes.WinDLL("user32", use_last_error=True)


def _set_process_dpi() -> None:
    """仅设置本进程 DPI 坐标，不修改 Windows 全局显示设置。"""
    user32 = _windows_api()
    try:
        function = user32.SetProcessDpiAwarenessContext
        function.argtypes = [ctypes.c_void_p]
        function.restype = wintypes.BOOL
        function(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
    except AttributeError:
        try:
            ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass


def _rect(rect) -> dict[str, int]:
    return {name: int(getattr(rect, name)) for name in ("left", "top", "right", "bottom")}


def list_monitors() -> list[dict[str, Any]]:
    """只读枚举显示器；index 是本次 Windows 枚举顺序。"""
    user32 = _windows_api()

    class MonitorInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )
    user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(MonitorInfo)]
    user32.GetMonitorInfoW.restype = wintypes.BOOL
    user32.EnumDisplayMonitors.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        callback_type,
        wintypes.LPARAM,
    ]
    user32.EnumDisplayMonitors.restype = wintypes.BOOL
    monitors = []

    @callback_type
    def collect(handle, _hdc, _bounds, _data):
        info = MonitorInfo()
        info.cbSize = ctypes.sizeof(info)
        if not user32.GetMonitorInfoW(handle, ctypes.byref(info)):
            return False
        bounds = _rect(info.rcMonitor)
        monitors.append(
            {
                "index": len(monitors),
                "name": str(info.szDevice),
                "primary": bool(info.dwFlags & 1),
                "rect": bounds,
                "work_rect": _rect(info.rcWork),
                "width": bounds["right"] - bounds["left"],
                "height": bounds["bottom"] - bounds["top"],
            }
        )
        return True

    if not user32.EnumDisplayMonitors(None, None, collect, 0) or not monitors:
        raise ctypes.WinError(ctypes.get_last_error())
    return monitors


def _monitor_for_rect(bounds: dict[str, int], monitors: list[dict]) -> int | None:
    overlap = []
    for monitor in monitors:
        rect = monitor["rect"]
        width = max(0, min(bounds["right"], rect["right"]) - max(bounds["left"], rect["left"]))
        height = max(0, min(bounds["bottom"], rect["bottom"]) - max(bounds["top"], rect["top"]))
        overlap.append((width * height, monitor["index"]))
    area, index = max(overlap)
    return index if area else None


def photos_windows(monitors: list[dict]) -> dict[str, Any]:
    """只查询 Photos / Microsoft.Photos 进程主窗口，不枚举其他应用的标题。"""
    script = """
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$rows = @(Get-Process -Name 'Photos','Microsoft.Photos' -ErrorAction SilentlyContinue |
  Select-Object Id,ProcessName,@{Name='hwnd';Expression={$_.MainWindowHandle.ToInt64()}})
ConvertTo-Json -InputObject $rows -Compress
"""
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8-sig",
            errors="replace",
            timeout=5,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if completed.returncode:
            return {"windows": [], "note": "Photos 进程只读查询未成功"}
        processes = json.loads(completed.stdout.strip() or "[]")
        if isinstance(processes, dict):
            processes = [processes]
        user32 = _windows_api()
        user32.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowRect.restype = wintypes.BOOL
        user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
        user32.IsWindowVisible.restype = wintypes.BOOL
        windows = []
        for process in processes:
            handle = int(process.get("hwnd") or 0)
            if not handle:
                continue
            bounds = wintypes.RECT()
            if user32.GetWindowRect(ctypes.c_void_p(handle), ctypes.byref(bounds)):
                rect = _rect(bounds)
                windows.append(
                    {
                        "pid": int(process["Id"]),
                        "process_name": process["ProcessName"],
                        "rect": rect,
                        "visible": bool(user32.IsWindowVisible(ctypes.c_void_p(handle))),
                        "monitor_index": _monitor_for_rect(rect, monitors),
                    }
                )
        return {
            "windows": windows,
            "note": "仅查询 Photos / Microsoft.Photos 主窗口，未读取其他窗口标题",
        }
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"windows": [], "note": f"Photos 定位不可用: {exc}"}


def _scale(value: Any) -> float:
    parsed = float(value)
    if isinstance(value, bool) or not math.isfinite(parsed) or not 0.05 <= parsed <= 4:
        raise ValueError("scale 必须为 0.05 到 4 之间的数字")
    return parsed


def _offset(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("offset 必须是整数物理像素")
    return value


class BoardViewer:
    def __init__(self, args, monitor: dict):
        # 只在真实展示入口导入 Tk；--list-monitors 不创建任何窗口。
        import tkinter as tk

        from PIL import Image, ImageTk

        self.args = args
        self.monitor = monitor
        self.Image = Image
        self.ImageTk = ImageTk
        self.pid = os.getpid()
        self.image_path = args.image.resolve()
        with Image.open(self.image_path) as opened:
            opened.load()
            self.source_image = opened.convert("RGBA")
        self.control_file = args.control_file.resolve() if args.control_file else None
        self.state_file = (
            args.state_file or ROOT / "runs" / "runtime" / f"board-display-state-{self.pid}.json"
        ).resolve()
        self.display = {
            "scale": _scale(args.scale),
            "x_offset": args.x_offset,
            "y_offset": args.y_offset,
        }
        self.control_stamp = None
        self.last_control_error = None
        self.status = "starting"
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title(f"AMCC5000 capture board {self.pid}")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", not args.no_topmost)
        width, height = monitor["width"], monitor["height"]
        left, top = monitor["rect"]["left"], monitor["rect"]["top"]
        self.root.geometry(f"{width}x{height}{left:+d}{top:+d}")
        self.canvas = tk.Canvas(self.root, bg=args.background, highlightthickness=0, borderwidth=0)
        self.canvas.pack(fill="both", expand=True)
        self.item = self.canvas.create_image(0, 0, anchor="center")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.render()
        self.root.update_idletasks()
        self.root.deiconify()
        self._position_own_window(left, top, width, height)
        self.status = "visible"
        self.save_state()
        print(json.dumps(self.state(), ensure_ascii=False), flush=True)
        self.root.after(200, self.poll_control)

    def _position_own_window(self, left: int, top: int, width: int, height: int) -> None:
        # Win32 确保负坐标显示器也使用绝对坐标；只持有本 Tk 窗口的句柄。
        user32 = _windows_api()
        user32.GetAncestor.argtypes = [ctypes.c_void_p, wintypes.UINT]
        user32.GetAncestor.restype = ctypes.c_void_p
        handle = user32.GetAncestor(ctypes.c_void_p(self.root.winfo_id()), 2)
        user32.SetWindowPos.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        user32.SetWindowPos.restype = wintypes.BOOL
        insert_after = ctypes.c_void_p(-1 if not self.args.no_topmost else 0)
        if not user32.SetWindowPos(handle, insert_after, left, top, width, height, 0x10 | 0x40):
            raise ctypes.WinError(ctypes.get_last_error())

    def render(self) -> None:
        width, height = self.monitor["width"], self.monitor["height"]
        base = min(width / self.source_image.width, height / self.source_image.height)
        factor = base * self.display["scale"]
        target = (
            max(1, round(self.source_image.width * factor)),
            max(1, round(self.source_image.height * factor)),
        )
        image = self.source_image.resize(target, self.Image.Resampling.NEAREST)
        self.photo = self.ImageTk.PhotoImage(image)
        self.canvas.itemconfigure(self.item, image=self.photo)
        self.canvas.coords(
            self.item, width / 2 + self.display["x_offset"], height / 2 + self.display["y_offset"]
        )
        self.rendered_size = list(target)

    def state(self) -> dict:
        return {
            "kind": "capture-board-viewer",
            "pid": self.pid,
            "status": self.status,
            "image": str(self.image_path),
            "monitor": self.monitor,
            "display": self.display,
            "rendered_size": self.rendered_size,
            "control_file": str(self.control_file) if self.control_file else None,
            "state_file": str(self.state_file),
            "updated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(f".{self.pid}.tmp")
        temporary.write_text(
            json.dumps(self.state(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.state_file)

    def poll_control(self) -> None:
        try:
            if self.control_file and self.control_file.is_file():
                info = self.control_file.stat()
                stamp = (info.st_mtime_ns, info.st_size)
                if stamp != self.control_stamp:
                    control = json.loads(self.control_file.read_text(encoding="utf-8-sig"))
                    if not isinstance(control, dict):
                        raise ValueError("control-file 必须是 JSON 对象")
                    if control.get("close") is True:
                        self.close()
                        return
                    display = dict(self.display)
                    if "scale" in control:
                        display["scale"] = _scale(control["scale"])
                    for key in ("x_offset", "y_offset"):
                        if key in control or key.replace("_", "-") in control:
                            display[key] = _offset(
                                control.get(key, control.get(key.replace("_", "-")))
                            )
                    if display != self.display:
                        self.display = display
                        self.render()
                        self.save_state()
                    self.control_stamp = stamp
                    self.last_control_error = None
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            message = str(exc)
            if message != self.last_control_error:
                print(f"control-file 未应用: {message}", file=sys.stderr, flush=True)
                self.last_control_error = message
        self.root.after(200, self.poll_control)

    def close(self) -> None:
        self.status = "closed"
        self.save_state()
        self.root.destroy()

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            if self.status != "closed":
                self.status = "closed"
                self.save_state()


def main(argv: list[str] | None = None) -> int:
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--list-monitors",
        action="store_true",
        help="只读输出显示器与 Photos 主窗口信息，不创建窗口",
    )
    parser.add_argument(
        "--image", type=Path, default=ROOT / "assets" / "capture-board-fast-demo.png"
    )
    parser.add_argument("--monitor", type=int, help="显示器 index；默认主显示器")
    parser.add_argument(
        "--scale", type=float, default=1.0, help="相对完整放入显示器的比例，0.05 到 4"
    )
    parser.add_argument("--x-offset", type=int, default=0)
    parser.add_argument("--y-offset", type=int, default=0)
    parser.add_argument("--background", default="#333333", help="窗口未被图像覆盖的区域颜色")
    parser.add_argument(
        "--control-file", type=Path, help="每 200ms 只读 scale/x_offset/y_offset/close"
    )
    parser.add_argument("--state-file", type=Path, help="保存本 viewer 的 PID、显示器与显示状态")
    parser.add_argument("--no-topmost", action="store_true")
    args = parser.parse_args(argv)
    try:
        _set_process_dpi()
        monitors = list_monitors()
        if args.list_monitors:
            print(
                json.dumps(
                    {"monitors": monitors, "photos": photos_windows(monitors)},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        selected = args.monitor
        if selected is None:
            selected = next(item["index"] for item in monitors if item["primary"])
        monitor = next((item for item in monitors if item["index"] == selected), None)
        if monitor is None:
            raise ValueError(f"不存在 monitor index {selected}；先运行 --list-monitors")
        viewer = BoardViewer(args, monitor)
        viewer.run()
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"底板展示失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
