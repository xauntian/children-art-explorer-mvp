"""米家口袋打印机 1S：单份照片打印、设备完成确认与原作业恢复。"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import time
from pathlib import Path

from PIL import Image, ImageOps

from .records import PrinterInfo, PrintError, PrintResult, document_name_for, document_lock
from .layout import compose_print_image, print_caption
from .records import atomic_write_json, utc_now_iso

PHOTO_SIZE = (1040, 1560)
MAX_JPEG_BYTES = 500 * 1024


class MijiaNotSubmittedError(PrintError):
    """已确认 print_job 尚未尝试，可结束本轮而不创建待核实打印。"""

    safe_to_retry = True


def encode_mijia_image(path: Path, *, style_snapshot: dict | None = None) -> bytes:
    """保留完整成品；横图转为竖向，等比放入照片画布，补白而不裁切。"""
    if style_snapshot is not None:
        canvas = compose_print_image(path, style_snapshot)
        canvas.save(path.with_name(f"{path.stem}-print.png"))
        return _encode_jpeg(canvas)
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source).convert("RGBA")
        image = Image.new("RGB", oriented.size, "white")
        image.paste(oriented, mask=oriented.getchannel("A"))
    if image.width > image.height:
        image = image.transpose(Image.Transpose.ROTATE_90)
    fitted = ImageOps.contain(image, PHOTO_SIZE, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", PHOTO_SIZE, "white")
    canvas.paste(fitted, ((PHOTO_SIZE[0] - fitted.width) // 2,
                         (PHOTO_SIZE[1] - fitted.height) // 2))
    return _encode_jpeg(canvas)


def _encode_jpeg(canvas: Image.Image) -> bytes:
    for quality in range(100, 19, -2):
        output = io.BytesIO()
        canvas.save(output, format="JPEG", quality=quality, subsampling=0,
                    progressive=False)
        data = output.getvalue()
        if len(data) <= MAX_JPEG_BYTES:
            return data
    raise PrintError("照片压缩后仍超过打印机的图片大小限制；未提交打印")


class MijiaPrinterAdapter:
    def __init__(self, address: str, *, receipts_dir: Path, session_factory=None):
        if not isinstance(address, str) or not re.fullmatch(r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", address):
            raise ValueError("printer.address 须为有效的蓝牙 MAC 地址")
        self.address = address.upper()
        self.name = f"米家口袋打印机 1S ({self.address})"
        self.receipts_dir = Path(receipts_dir)
        self.session_factory = session_factory
        self._live_jobs: dict[int, str] = {}

    def _session(self):
        if self.session_factory is None:
            from .transport import MijiaSession

            return MijiaSession(self.address)
        return self.session_factory(self.address)

    def _error(self, message: str, job_id: int | None = None) -> PrintError:
        return PrintError(message, printer=self.name, job_id=job_id,
                          requires_resolution=True)

    @staticmethod
    def _settings(timeout, poll_interval):
        if not all(math.isfinite(x) and x > 0 for x in (timeout, poll_interval)):
            raise ValueError("打印超时与轮询间隔必须大于 0")

    def _device_status(self, session) -> dict:
        status = session.rpc("mixed_status", {})
        if (not isinstance(status, dict) or not isinstance(status.get("category"), str)
                or type(status.get("error")) is not int):
            raise self._error("打印机未返回有效状态，无法确认是否可打印")
        if status["error"] != 0:
            raise self._error(f"打印机报告错误 {status['error']}；请处理设备故障后查询原作业")
        return status

    def prepare(self, *, require_idle=False) -> PrinterInfo:
        try:
            with self._session() as session:
                status = self._device_status(session)
                if require_idle and status["category"] != "idle":
                    raise self._error("米家 1S 尚未空闲，请等待当前打印结束后再拍摄")
        except PrintError:
            raise
        except Exception as exc:
            raise self._error(f"无法连接米家 1S，请确认打印机开机且未被其他应用占用：{exc}") from exc
        return PrinterInfo(self.name, "Mijia 1S Bluetooth SPP", self.address)

    def _receipt_path(self, document: str) -> Path:
        if not re.fullmatch(r"AMCC5000-[0-9a-f]{32}", document):
            raise self._error("缺少可靠的原图片标识，无法核实打印作业")
        return self.receipts_dir / f"{document}.json"

    def _save(self, path: Path, receipt: dict, **fields) -> None:
        receipt.update(fields, updated_at=utc_now_iso())
        atomic_write_json(path, receipt)

    def print_image(self, path: str | Path, *, timeout=180.0, poll_interval=0.5,
                    progress=None, on_job=None, style_snapshot: dict | None = None) -> PrintResult:
        self._settings(timeout, poll_interval)
        image_path = Path(path).resolve()
        document = document_name_for(image_path)
        receipt_path = self._receipt_path(document)
        with document_lock(receipt_path):
            return self._submit(image_path, document, receipt_path, timeout, poll_interval,
                                progress, on_job, style_snapshot)

    def _submit(self, image_path, document, receipt_path, timeout, poll_interval,
                progress, on_job, style_snapshot):
        job_id = None
        submission_attempted = False
        if receipt_path.exists():
            try:
                old = json.loads(receipt_path.read_text(encoding="utf-8"))
                value = old.get("job_id")
                job_id = value if type(value) is int and value > 0 else None
            except (OSError, ValueError, AttributeError):
                pass
            raise self._error("这张成品已有打印记录；请查询原作业，不会重复提交", job_id)
        try:
            data = encode_mijia_image(image_path, style_snapshot=style_snapshot)
            receipt = {"address": self.address, "printer": self.name,
                       "document": document, "image_path": str(image_path),
                       "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                       "jpeg_sha256": hashlib.sha256(data).hexdigest(), "copies": 1,
                       "created_at": utc_now_iso(), "job_id": None}
            if style_snapshot is not None:
                name, description = print_caption(style_snapshot)
                receipt.update(print_name=name, print_description=description,
                               print_layout_path=str(image_path.with_name(f"{image_path.stem}-print.png")))
            with self._session() as session:
                status = self._device_status(session)
                if status["category"] != "idle":
                    raise self._error("打印机尚未空闲，本轮未新建打印任务")
                # 任何请求前落盘；即使设备收到请求而回复丢失，也不自动重发。
                self._save(receipt_path, receipt, phase="creating_job")
                submission_attempted = True
                result = session.rpc("print_job", {"file_size": len(data), "copies": 1,
                                     "channel": 528, "job_type": 0, "code": 0, "job_id": 0})
                if not isinstance(result, dict) or type(result.get("job_id")) is not int or result["job_id"] <= 0:
                    raise self._error("打印机未返回可靠的作业编号；请核实设备，不会自动重印")
                job_id = result["job_id"]
                self._save(receipt_path, receipt, job_id=job_id, phase="job_created")
                if on_job:
                    on_job(job_id)
                self._live_jobs[job_id] = document
                if progress:
                    progress("正在向米家 1S 传输 AI 成品…")
                self._save(receipt_path, receipt, phase="uploading")
                session.send_image(data, job_id)
                self._save(receipt_path, receipt, phase="image_sent")
                time.sleep(1)
                self._save(receipt_path, receipt, phase="confirming")
                session.rpc("confirm_job", [job_id])
                self._save(receipt_path, receipt, phase="confirmed")
                return self._wait(session, receipt_path, receipt, timeout, poll_interval, progress)
        except Exception as exc:
            if not submission_attempted:
                raise MijiaNotSubmittedError(
                    f"米家 1S 尚未提交打印：{exc}；AI 成品已保留",
                    printer=self.name, requires_resolution=False,
                ) from exc
            if receipt_path.exists():
                try:
                    if job_id is not None:
                        self._save(receipt_path, receipt, last_error=str(exc))
                except OSError:
                    pass
            raise self._error(f"米家 1S 打印未完成：{exc}；请检查原作业，不会自动重印", job_id) from exc

    def wait_for_job(self, job_id: int, *, printer=None, timeout=180.0,
                     poll_interval=0.5, progress=None, expected_document=None) -> PrintResult:
        """仅查询已登记的原任务；不重发创建、图片或确认命令。"""
        self._settings(timeout, poll_interval)
        if type(job_id) is not int or job_id <= 0 or (printer is not None and printer != self.name):
            raise self._error("原作业的设备或编号与当前米家打印机不匹配", job_id)
        document = expected_document or self._live_jobs.get(job_id)
        receipt_path = self._receipt_path(document or "")
        with document_lock(receipt_path):
            return self._resume(job_id, document, receipt_path, timeout, poll_interval, progress)

    def _resume(self, job_id, document, receipt_path, timeout, poll_interval, progress):
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt.get("address") != self.address or receipt.get("job_id") != job_id
                    or receipt.get("document") != document or receipt.get("copies") != 1):
                raise ValueError("原任务记录不匹配")
            if receipt.get("phase") not in {"confirming", "confirmed", "complete", "awaiting_confirmation"}:
                raise ValueError("原任务的图片传输或打印确认尚未完成")
            image_path = Path(receipt["image_path"])
            if (document_name_for(image_path) != document
                    or hashlib.sha256(image_path.read_bytes()).hexdigest() != receipt["image_sha256"]):
                raise ValueError("原成品文件已变化，无法核实打印归属")
            with self._session() as session:
                return self._wait(session, receipt_path, receipt, timeout, poll_interval, progress)
        except Exception as exc:
            raise self._error(f"无法核实米家 1S 原作业：{exc}；不会重新提交", job_id) from exc

    def _wait(self, session, receipt_path, receipt, timeout, poll_interval, progress):
        job_id = receipt["job_id"]
        deadline = time.monotonic() + timeout
        previous = None
        while True:
            result = session.rpc("job_info", [job_id])
            if not isinstance(result, list):
                raise self._error("打印机未返回有效的原任务状态", job_id)
            matches = [item for item in result if isinstance(item, dict)
                       and type(item.get("job_id")) is int and item["job_id"] == job_id]
            if len(matches) != 1:
                raise self._error("设备中找不到唯一匹配的原作业；不能将其他任务当作本轮完成", job_id)
            info = matches[0]
            if (type(info.get("job_type")) is not int or info["job_type"] != 0
                    or type(info.get("prt_copies")) is not int
                    or not 0 <= info["prt_copies"] <= 1):
                raise self._error("设备任务类型或份数与本轮单张照片不符", job_id)
            state = info.get("job_state")
            if state in {"aborted", "canceled"}:
                raise self._error(f"打印机报告原任务 {state}，请检查实际出纸和设备", job_id)
            device_status = self._device_status(session)
            if state != previous and progress:
                label = {"downloading": "接收照片", "printing": "正在打印照片",
                         "finished": "正在核实打印完成"}.get(state, "等待打印机处理")
                progress(f"米家 1S：{label}")
                previous = state
            if state == "finished" and info["prt_copies"] == 1 and device_status["category"] == "idle":
                live = self._live_jobs.get(job_id) == receipt["document"]
                self._save(receipt_path, receipt, phase="complete" if live else "awaiting_confirmation",
                           job_state=state, printed_copies=1)
                if live:
                    return PrintResult("printed", self.name, job_id, False,
                                       "米家 1S 确认本轮照片打印完成，已恢复空闲")
                return PrintResult("spooled", self.name, job_id, True,
                                   "设备报告原编号已完成；重启后无法核实编号是否复用，请取到本轮照片后确认已出纸")
            if time.monotonic() >= deadline:
                raise self._error("等待米家 1S 打印完成超时；请查询原作业", job_id)
            time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
