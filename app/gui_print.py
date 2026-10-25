"""Windows printing for the opt-in desktop GUI, independent of the legacy stub.

One call submits one document. Queue disappearance and JOB_STATUS_COMPLETE do
not prove paper delivery, so they require the GUI's explicit output confirmation.
All methods are blocking and belong on the GUI's worker thread.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

JOB_PAUSED = 0x0001
JOB_ERROR = 0x0002
JOB_DELETING = 0x0004
JOB_OFFLINE = 0x0020
JOB_PAPEROUT = 0x0040
JOB_PRINTED = 0x0080
JOB_DELETED = 0x0100
JOB_BLOCKED = 0x0200
JOB_USER_INTERVENTION = 0x0400
JOB_COMPLETE = 0x1000

_PRINTER_ERRORS = {
    0x00000001: "打印机已暂停",
    0x00000002: "打印机报告错误",
    0x00000004: "打印机正在删除",
    0x00000008: "打印机卡纸",
    0x00000010: "打印机缺纸",
    0x00000020: "打印机等待手动进纸",
    0x00000040: "打印机纸张异常",
    0x00000080: "打印机脱机",
    0x00001000: "打印机不可用",
    0x00040000: "打印机无碳粉/墨水",
    0x00100000: "打印机需要人工处理",
    0x00200000: "打印机盖未关闭",
    0x00400000: "打印机内存不足",
    0x00800000: "打印服务器状态未知",
}
_JOB_ERRORS = {
    JOB_PAUSED: "打印作业已暂停",
    JOB_ERROR: "打印作业报告错误",
    JOB_DELETING: "打印作业正在删除",
    JOB_OFFLINE: "打印作业等待脱机打印机",
    JOB_PAPEROUT: "打印作业等待纸张",
    JOB_DELETED: "打印作业已被删除，未确认出纸",
    JOB_BLOCKED: "打印作业被驱动阻塞",
    JOB_USER_INTERVENTION: "打印作业需要人工处理",
}


@dataclass(frozen=True)
class PrinterInfo:
    name: str
    driver: str = ""
    port: str = ""
    status: int = 0
    is_default: bool = False


@dataclass(frozen=True)
class JobSnapshot:
    job_id: int
    status: int = 0
    status_text: str = ""
    pages_printed: int = 0
    total_pages: int = 0
    document: str = ""


@dataclass(frozen=True)
class PrintResult:
    status: str
    printer: str
    job_id: int
    needs_confirmation: bool
    message: str


class PrintError(RuntimeError):
    """A failure never instructs the caller to automatically resubmit a job."""

    def __init__(
        self,
        message: str,
        *,
        printer: str | None = None,
        job_id: int | None = None,
        requires_resolution: bool = False,
    ):
        super().__init__(message)
        self.printer = printer
        self.job_id = job_id
        self.requires_resolution = requires_resolution


class PrintBackend(Protocol):
    def list_printers(self) -> list[PrinterInfo]: ...
    def get_printer(self, printer: str) -> PrinterInfo: ...
    def get_job(self, printer: str, job_id: int) -> JobSnapshot | None: ...
    def submit_image(
        self, printer: str, path: Path, on_job: Callable[[int], None] | None
    ) -> int: ...


def is_virtual_printer(printer: PrinterInfo) -> bool:
    """Exclude known file/fax drivers and file ports before choosing a device."""
    description = f"{printer.name} {printer.driver}".casefold()
    return any(
        word in description
        for word in ("pdf", "fax", "xps", "onenote", "virtual printer", "document writer")
    ) or printer.port.casefold() in {"portprompt:", "file:", "nul:", "nul"}


def document_name_for(path: str | Path) -> str:
    """Stable job identity survives GUI restarts and distinguishes session paths."""
    normalized = os.path.normcase(str(Path(path).resolve()))
    token = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]
    return f"AMCC5000-{token}"


def contain_box(
    image_size: tuple[int, int],
    page_size: tuple[int, int],
    dpi: tuple[int, int] = (1, 1),
) -> tuple[int, ...]:
    """Center with physical aspect preserved, even at e.g. 600 x 1200 DPI."""
    width, height = image_size
    page_width, page_height = page_size
    dpi_x, dpi_y = dpi
    if min(width, height, page_width, page_height, dpi_x, dpi_y) <= 0:
        raise ValueError("图片和打印区域的尺寸必须大于 0")
    scale = min(page_width / (width * dpi_x), page_height / (height * dpi_y))
    target_width = max(1, round(width * scale * dpi_x))
    target_height = max(1, round(height * scale * dpi_y))
    left = (page_width - target_width) // 2
    top = (page_height - target_height) // 2
    return left, top, left + target_width, top + target_height


def _status_error(status: int, labels: dict[int, str]) -> str:
    return "；".join(text for flag, text in labels.items() if status & flag)


class PrinterAdapter:
    def __init__(self, printer_name: str | None = None, *, backend: PrintBackend | None = None):
        self.printer_name = printer_name or None
        self.backend = backend
        self.printer: PrinterInfo | None = None
        self._documents: dict[tuple[str, int], str] = {}

    def _backend(self) -> PrintBackend:
        if self.backend is None:
            self.backend = WindowsPrintBackend()
        return self.backend

    def prepare(self) -> PrinterInfo:
        """Read-only discovery; never opens a print dialog or submits a page."""
        printers = self._backend().list_printers()
        chosen_name = self.printer_name or (self.printer.name if self.printer else None)
        if chosen_name:
            selected = next((p for p in printers if p.name == chosen_name), None)
            if selected is None:
                raise PrintError(f"找不到选定的打印机：{chosen_name}")
            if is_virtual_printer(selected):
                raise PrintError(f"{selected.name} 是虚拟打印机；请连接并选择实体 A4 打印机")
        else:
            physical = [p for p in printers if not is_virtual_printer(p)]
            if not physical:
                names = "、".join(p.name for p in printers) or "无"
                raise PrintError(f"未检测到实体 A4 打印机。当前打印机：{names}；请安装并连接实体打印机")
            default = next((p for p in physical if p.is_default), None)
            if default is not None:
                selected = default
            elif len(physical) == 1:
                selected = physical[0]
            else:
                names = "、".join(p.name for p in physical)
                raise PrintError(
                    f"检测到多台实体打印机（{names}），但 Windows 默认打印机不是实体机。"
                    "请将目标实体机设为 Windows 默认打印机，或在 configs/gui.json 的 printer.name 中指定名称"
                )
        error = _status_error(selected.status, _PRINTER_ERRORS)
        if error:
            raise PrintError(error, printer=selected.name)
        self.printer = selected
        return selected

    def inspect_job(self, job_id: int, *, printer: str | None = None) -> JobSnapshot | None:
        """Read a saved job without creating any new printing work."""
        name = printer or (self.printer.name if self.printer else self.printer_name)
        if not name:
            name = self.prepare().name
        return self._backend().get_job(name, job_id)

    def print_image(
        self,
        path: str | Path,
        *,
        timeout: float = 180.0,
        poll_interval: float = 0.5,
        progress: Callable[[str], None] | None = None,
        on_job: Callable[[int], None] | None = None,
    ) -> PrintResult:
        """Submit exactly once, persist its ID through on_job, then monitor it."""
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("打印超时与轮询间隔必须大于 0")
        image_path = Path(path)
        if not image_path.is_file():
            raise PrintError(f"打印图片不存在：{image_path}")
        printer = self.prepare()
        job_id: int | None = None

        def registered(value: int) -> None:
            nonlocal job_id
            job_id = value
            self._documents[(printer.name, value)] = document_name_for(image_path)
            if on_job:
                on_job(value)

        try:
            if progress:
                progress(f"向 {printer.name} 提交 A4 打印作业")
            job_id = self._backend().submit_image(printer.name, image_path, registered)
        except Exception as exc:
            raise PrintError(
                f"打印提交失败：{exc}；请检查原作业，程序不会自动重复提交",
                printer=printer.name,
                job_id=job_id,
                requires_resolution=job_id is not None,
            ) from exc
        return self.wait_for_job(
            job_id,
            printer=printer.name,
            timeout=timeout,
            poll_interval=poll_interval,
            progress=progress,
            expected_document=document_name_for(image_path),
        )

    def wait_for_job(
        self,
        job_id: int,
        *,
        printer: str | None = None,
        timeout: float = 180.0,
        poll_interval: float = 0.5,
        progress: Callable[[str], None] | None = None,
        expected_document: str | None = None,
    ) -> PrintResult:
        """Resume only the saved printer/job ID; this method never submits."""
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("打印超时与轮询间隔必须大于 0")
        name = printer or (self.printer.name if self.printer else self.printer_name)
        if not name:
            name = self.prepare().name
        expected_document = expected_document or self._documents.get((name, job_id))
        started = time.monotonic()
        previous = None
        while True:
            try:
                state = self._backend().get_printer(name)
                error = _status_error(state.status, _PRINTER_ERRORS)
                job = self._backend().get_job(name, job_id)
                if job is not None and expected_document and job.document and job.document != expected_document:
                    raise PrintError("原打印作业编号已被其他文档使用，不能将另一文档的状态当作本轮完成")
                identity_verified = bool(job is not None and expected_document and job.document == expected_document)
                if job is not None:
                    error = error or _status_error(job.status, _JOB_ERRORS)
                if error:
                    raise PrintError(error)
            except Exception as exc:
                raise PrintError(
                    f"{exc}。打印作业 {job_id} 保持锁定；处理故障后重新检查原作业，勿重复提交",
                    printer=name,
                    job_id=job_id,
                    requires_resolution=True,
                ) from exc
            if job is not None and job.status & JOB_PRINTED and identity_verified:
                return PrintResult(
                    "printed", name, job_id, False,
                    "Windows 打印驱动报告作业已打印完成（非物理出纸传感器确认）",
                )
            if job is None or job.status & JOB_COMPLETE or not identity_verified:
                reason = "作业已离开 Windows 队列" if job is None else "驱动报告已将作业发送到打印机"
                if job is not None and not identity_verified:
                    reason = "缺少原文档标识，无法核实此编号是否仍对应上一轮打印"
                return PrintResult(
                    "spooled", name, job_id, True,
                    f"{reason}，尚无可靠出纸确认；请取到完整纸张后在 GUI 中确认出纸",
                )
            text = job.status_text or f"作业 {job_id} 打印中（{job.pages_printed}/{job.total_pages} 页）"
            if progress and text != previous:
                try:
                    progress(text)
                except Exception as exc:
                    raise PrintError(
                        f"打印状态通知失败：{exc}；请重新检查原打印作业 {job_id}",
                        printer=name,
                        job_id=job_id,
                        requires_resolution=True,
                    ) from exc
                previous = text
            if time.monotonic() - started >= timeout:
                raise PrintError(
                    f"等待打印作业 {job_id} 超时；请检查打印机与原作业，程序不会重复提交",
                    printer=name,
                    job_id=job_id,
                    requires_resolution=True,
                )
            time.sleep(poll_interval)


class WindowsPrintBackend:
    """Small ctypes Winspool/GDI backend; Pillow decodes the source pixels."""

    def __init__(self):
        if os.name != "nt":
            raise PrintError("真实打印需要 Windows 打印后台服务")
        from ctypes import wintypes as wt

        self.wt = wt
        self.spool = ctypes.WinDLL("winspool.drv", use_last_error=True)
        self.gdi = ctypes.WinDLL("gdi32", use_last_error=True)

        class PrinterInfo2(ctypes.Structure):
            _fields_ = [
                (name, wt.LPWSTR) for name in (
                    "pServerName", "pPrinterName", "pShareName", "pPortName", "pDriverName",
                    "pComment", "pLocation",
                )
            ] + [("pDevMode", ctypes.c_void_p)] + [
                (name, wt.LPWSTR) for name in (
                    "pSepFile", "pPrintProcessor", "pDatatype", "pParameters",
                )
            ] + [("pSecurityDescriptor", ctypes.c_void_p)] + [
                (name, wt.DWORD) for name in (
                    "Attributes", "Priority", "DefaultPriority", "StartTime", "UntilTime",
                    "Status", "cJobs", "AveragePPM",
                )
            ]

        class SystemTime(ctypes.Structure):
            _fields_ = [(name, wt.WORD) for name in (
                "year", "month", "weekday", "day", "hour", "minute", "second", "millisecond",
            )]

        class JobInfo1(ctypes.Structure):
            _fields_ = [("JobId", wt.DWORD)] + [
                (name, wt.LPWSTR) for name in (
                    "pPrinterName", "pMachineName", "pUserName", "pDocument", "pDatatype", "pStatus",
                )
            ] + [(name, wt.DWORD) for name in (
                "Status", "Priority", "Position", "TotalPages", "PagesPrinted",
            )] + [("Submitted", SystemTime)]

        class DevMode(ctypes.Structure):
            _fields_ = [("dmDeviceName", wt.WCHAR * 32)] + [
                (name, wt.WORD) for name in (
                    "dmSpecVersion", "dmDriverVersion", "dmSize", "dmDriverExtra",
                )
            ] + [("dmFields", wt.DWORD)] + [
                (name, ctypes.c_short) for name in (
                    "dmOrientation", "dmPaperSize", "dmPaperLength", "dmPaperWidth", "dmScale",
                    "dmCopies", "dmDefaultSource", "dmPrintQuality", "dmColor", "dmDuplex",
                    "dmYResolution", "dmTTOption", "dmCollate",
                )
            ] + [("dmFormName", wt.WCHAR * 32), ("dmLogPixels", wt.WORD)] + [
                (name, wt.DWORD) for name in (
                    "dmBitsPerPel", "dmPelsWidth", "dmPelsHeight", "dmDisplayFlags",
                    "dmDisplayFrequency", "dmICMMethod", "dmICMIntent", "dmMediaType",
                    "dmDitherType", "dmReserved1", "dmReserved2", "dmPanningWidth", "dmPanningHeight",
                )
            ]

        class DocInfo(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_int), ("lpszDocName", wt.LPCWSTR),
                ("lpszOutput", wt.LPCWSTR), ("lpszDatatype", wt.LPCWSTR), ("fwType", wt.DWORD),
            ]

        class BitmapInfoHeader(ctypes.Structure):
            _fields_ = [
                ("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG),
                ("biPlanes", wt.WORD), ("biBitCount", wt.WORD),
                ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD),
                ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG),
                ("biClrUsed", wt.DWORD), ("biClrImportant", wt.DWORD),
            ]

        self.PrinterInfo2, self.JobInfo1 = PrinterInfo2, JobInfo1
        self.DevMode, self.DocInfo = DevMode, DocInfo
        self.BitmapInfoHeader = BitmapInfoHeader
        pointer = ctypes.c_void_p
        self._bind(self.spool, "EnumPrintersW", wt.BOOL, [wt.DWORD, wt.LPWSTR, wt.DWORD, pointer, wt.DWORD, pointer, pointer])
        self._bind(self.spool, "GetDefaultPrinterW", wt.BOOL, [wt.LPWSTR, pointer])
        self._bind(self.spool, "OpenPrinterW", wt.BOOL, [wt.LPWSTR, pointer, pointer])
        self._bind(self.spool, "ClosePrinter", wt.BOOL, [wt.HANDLE])
        self._bind(self.spool, "GetPrinterW", wt.BOOL, [wt.HANDLE, wt.DWORD, pointer, wt.DWORD, pointer])
        self._bind(self.spool, "GetJobW", wt.BOOL, [wt.HANDLE, wt.DWORD, wt.DWORD, pointer, wt.DWORD, pointer])
        self._bind(self.spool, "DocumentPropertiesW", ctypes.c_long, [wt.HWND, wt.HANDLE, wt.LPWSTR, pointer, pointer, wt.DWORD])
        self._bind(self.gdi, "CreateDCW", wt.HDC, [wt.LPCWSTR, wt.LPCWSTR, wt.LPCWSTR, pointer])
        self._bind(self.gdi, "DeleteDC", wt.BOOL, [wt.HDC])
        self._bind(self.gdi, "GetDeviceCaps", ctypes.c_int, [wt.HDC, ctypes.c_int])
        self._bind(self.gdi, "StartDocW", ctypes.c_int, [wt.HDC, pointer])
        self._bind(self.gdi, "SetStretchBltMode", ctypes.c_int, [wt.HDC, ctypes.c_int])
        self._bind(self.gdi, "SetBrushOrgEx", wt.BOOL, [wt.HDC, ctypes.c_int, ctypes.c_int, pointer])
        self._bind(self.gdi, "StretchDIBits", ctypes.c_int, [
            wt.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            pointer, pointer, wt.UINT, wt.DWORD,
        ])
        for method in ("StartPage", "EndPage", "EndDoc", "AbortDoc"):
            self._bind(self.gdi, method, ctypes.c_int, [wt.HDC])

    @staticmethod
    def _bind(library, name, restype, argtypes):
        function = getattr(library, name)
        function.restype, function.argtypes = restype, argtypes

    @staticmethod
    def _win_error(action: str) -> PrintError:
        return PrintError(f"{action}：{ctypes.WinError(ctypes.get_last_error())}")

    def _open(self, printer: str):
        handle = self.wt.HANDLE()
        if not self.spool.OpenPrinterW(printer, ctypes.byref(handle), None):
            raise self._win_error(f"无法打开打印机 {printer}")
        return handle

    def _default_printer(self) -> str | None:
        needed = self.wt.DWORD()
        self.spool.GetDefaultPrinterW(None, ctypes.byref(needed))
        if not needed.value:
            return None
        buffer = ctypes.create_unicode_buffer(needed.value)
        if not self.spool.GetDefaultPrinterW(buffer, ctypes.byref(needed)):
            return None
        return buffer.value

    def _info(self, value, default: str | None = None) -> PrinterInfo:
        return PrinterInfo(
            value.pPrinterName or "", value.pDriverName or "", value.pPortName or "",
            value.Status, value.pPrinterName == default,
        )

    def list_printers(self) -> list[PrinterInfo]:
        needed, count = self.wt.DWORD(), self.wt.DWORD()
        success = self.spool.EnumPrintersW(6, None, 2, None, 0, ctypes.byref(needed), ctypes.byref(count))
        if not needed.value:
            if not success and ctypes.get_last_error() not in (0, 122):
                raise self._win_error("无法枚举 Windows 打印机")
            return []
        buffer = ctypes.create_string_buffer(needed.value)
        if not self.spool.EnumPrintersW(6, None, 2, buffer, needed.value, ctypes.byref(needed), ctypes.byref(count)):
            raise self._win_error("无法枚举 Windows 打印机")
        printers = ctypes.cast(buffer, ctypes.POINTER(self.PrinterInfo2))
        default = self._default_printer()
        return [self._info(printers[index], default) for index in range(count.value)]

    def get_printer(self, printer: str) -> PrinterInfo:
        handle = self._open(printer)
        try:
            needed = self.wt.DWORD()
            self.spool.GetPrinterW(handle, 2, None, 0, ctypes.byref(needed))
            if not needed.value:
                raise self._win_error("无法读取打印机状态")
            buffer = ctypes.create_string_buffer(needed.value)
            if not self.spool.GetPrinterW(handle, 2, buffer, needed.value, ctypes.byref(needed)):
                raise self._win_error("无法读取打印机状态")
            return self._info(ctypes.cast(buffer, ctypes.POINTER(self.PrinterInfo2)).contents)
        finally:
            self.spool.ClosePrinter(handle)

    def get_job(self, printer: str, job_id: int) -> JobSnapshot | None:
        handle = self._open(printer)
        try:
            needed = self.wt.DWORD()
            self.spool.GetJobW(handle, job_id, 1, None, 0, ctypes.byref(needed))
            if not needed.value:
                if ctypes.get_last_error() in (2, 87, 1168):
                    return None
                raise self._win_error(f"无法读取打印作业 {job_id}")
            buffer = ctypes.create_string_buffer(needed.value)
            if not self.spool.GetJobW(handle, job_id, 1, buffer, needed.value, ctypes.byref(needed)):
                if ctypes.get_last_error() in (2, 87, 1168):
                    return None
                raise self._win_error(f"无法读取打印作业 {job_id}")
            value = ctypes.cast(buffer, ctypes.POINTER(self.JobInfo1)).contents
            return JobSnapshot(
                value.JobId, value.Status, value.pStatus or "", value.PagesPrinted,
                value.TotalPages, value.pDocument or "",
            )
        finally:
            self.spool.ClosePrinter(handle)

    def _page_devmode(self, handle, printer: str, landscape: bool):
        size = self.spool.DocumentPropertiesW(None, handle, printer, None, None, 0)
        if size < ctypes.sizeof(self.DevMode):
            raise PrintError("打印驱动未提供可配置的 A4 页面")
        buffer = ctypes.create_string_buffer(size)
        if self.spool.DocumentPropertiesW(None, handle, printer, buffer, None, 2) != 1:
            raise self._win_error("无法读取打印页面设置")
        mode = ctypes.cast(buffer, ctypes.POINTER(self.DevMode)).contents
        mode.dmFields |= 0x00000001 | 0x00000002 | 0x00000010 | 0x00000100
        mode.dmFields &= ~(0x00000004 | 0x00000008 | 0x00010000)
        mode.dmOrientation = 2 if landscape else 1
        mode.dmPaperSize = 9  # DMPAPER_A4
        mode.dmScale = 100
        mode.dmCopies = 1
        if self.spool.DocumentPropertiesW(None, handle, printer, buffer, buffer, 10) != 1:
            raise self._win_error("打印驱动拒绝 A4 页面设置")
        if mode.dmPaperSize != 9 or mode.dmOrientation != (2 if landscape else 1):
            raise PrintError("打印驱动不支持要求的 A4 纸张或页面方向")
        return buffer

    def submit_image(self, printer: str, path: Path, on_job: Callable[[int], None] | None) -> int:
        from PIL import Image, ImageOps

        with Image.open(path) as source:
            source = ImageOps.exif_transpose(source)
            source.load()
            image = Image.new("RGB", source.size, "white")
            if source.mode in ("RGBA", "LA") or "transparency" in source.info:
                rgba = source.convert("RGBA")
                image.paste(rgba, mask=rgba.getchannel("A"))
            else:
                image.paste(source.convert("RGB"))
        handle = self._open(printer)
        dc = None
        started = False
        try:
            devmode = self._page_devmode(handle, printer, image.width > image.height)
            dc = self.gdi.CreateDCW("WINSPOOL", printer, None, devmode)
            if not dc:
                raise self._win_error("无法创建打印设备上下文")
            page_size = (self.gdi.GetDeviceCaps(dc, 8), self.gdi.GetDeviceCaps(dc, 10))
            dpi = (self.gdi.GetDeviceCaps(dc, 88), self.gdi.GetDeviceCaps(dc, 90))
            box = contain_box(image.size, page_size, dpi)
            document = self.DocInfo()
            document.cbSize = ctypes.sizeof(document)
            document.lpszDocName = document_name_for(path)
            job_id = self.gdi.StartDocW(dc, ctypes.byref(document))
            if job_id <= 0:
                raise self._win_error("无法创建 Windows 打印作业")
            started = True
            if on_job:
                on_job(job_id)
            if self.gdi.StartPage(dc) <= 0:
                raise self._win_error("无法开始 A4 打印页")
            self._draw_image(dc, image, box)
            if self.gdi.EndPage(dc) <= 0:
                raise self._win_error("无法提交 A4 打印页")
            if self.gdi.EndDoc(dc) <= 0:
                raise self._win_error("无法完成打印作业提交")
            started = False
            return job_id
        finally:
            if dc:
                if started:
                    self.gdi.AbortDoc(dc)
                self.gdi.DeleteDC(dc)
            self.spool.ClosePrinter(handle)

    def _draw_image(self, dc, image, box: tuple[int, ...]) -> None:
        """Send top-down BGRX pixels and check GDI's scanline result."""
        pixels = image.tobytes("raw", "BGRX")
        header = self.BitmapInfoHeader()
        header.biSize = ctypes.sizeof(header)
        header.biWidth, header.biHeight = image.width, -image.height
        header.biPlanes, header.biBitCount = 1, 32
        header.biSizeImage = len(pixels)
        if not self.gdi.SetStretchBltMode(dc, 4):  # HALFTONE
            raise self._win_error("无法设置打印图像缩放")
        if not self.gdi.SetBrushOrgEx(dc, 0, 0, None):
            raise self._win_error("无法设置打印图像原点")
        left, top, right, bottom = box
        copied = self.gdi.StretchDIBits(
            dc, left, top, right - left, bottom - top, 0, 0, image.width, image.height,
            ctypes.c_char_p(pixels), ctypes.byref(header), 0, 0x00CC0020,
        )
        if copied in (0, -1):
            raise self._win_error("打印图像像素传输失败")
