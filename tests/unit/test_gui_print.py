"""No test submits to a Windows printer: the backend is always a fake."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app import gui_print
from app.gui_print import (
    JOB_COMPLETE,
    JOB_DELETED,
    JOB_PAPEROUT,
    JOB_PRINTED,
    JobSnapshot,
    PrinterAdapter,
    PrinterInfo,
    PrintError,
    WindowsPrintBackend,
    contain_box,
    document_name_for,
    is_virtual_printer,
)


class FakeBackend:
    def __init__(self, jobs=None, printers=None):
        self.printers = printers if printers is not None else [PrinterInfo("Physical", "Laser", "USB001", is_default=True)]
        self.jobs = list(jobs) if jobs is not None else [JobSnapshot(42, JOB_PRINTED)]
        self.submissions = []
        self.reads = []
        self.document = None

    def list_printers(self):
        return self.printers

    def get_printer(self, printer):
        self.reads.append(("printer", printer))
        return next(p for p in self.printers if p.name == printer)

    def get_job(self, printer, job_id):
        self.reads.append(("job", printer, job_id))
        job = self.jobs.pop(0) if len(self.jobs) > 1 else self.jobs[0]
        if job is not None and self.document and not job.document:
            job = replace(job, document=self.document)
        return job

    def submit_image(self, printer, path, on_job):
        self.submissions.append((printer, path))
        self.document = document_name_for(path)
        on_job(42)
        return 42


@pytest.fixture
def picture(tmp_path):
    path = tmp_path / "a4.png"
    path.write_bytes(b"backend validates actual image")
    return path


@pytest.mark.parametrize("name,driver,port", [
    ("Microsoft Print to PDF", "Microsoft Print To PDF", "PORTPROMPT:"),
    ("Adobe PDF", "Adobe PDF Converter", "Documents\\*.pdf"),
    ("导出为WPS PDF", "Kingsoft Virtual Printer Driver", "Kingsoft Virtual Printer Port"),
    ("Fax", "Fax", "SHRFAX:"),
    ("XPS Writer", "XPS", "PORTPROMPT:"),
    ("Generic", "Generic", "FILE:"),
])
def test_rejects_virtual_printer(name, driver, port):
    info = PrinterInfo(name, driver, port, is_default=True)
    assert is_virtual_printer(info)
    with pytest.raises(PrintError, match="实体"):
        PrinterAdapter(backend=FakeBackend(printers=[info])).prepare()


def test_prepare_prefers_physical_default_and_never_submits():
    backend = FakeBackend(printers=[
        PrinterInfo("PDF", is_default=True),
        PrinterInfo("First", "Laser", "USB001"),
        PrinterInfo("Second", "Laser", "USB002", is_default=True),
    ])
    assert PrinterAdapter(backend=backend).prepare().name == "Second"
    assert backend.submissions == []


@pytest.mark.parametrize("has_virtual_default", [True, False])
def test_multiple_physical_printers_require_explicit_selection_without_physical_default(has_virtual_default):
    backend = FakeBackend(printers=[
        PrinterInfo("PDF", is_default=has_virtual_default),
        PrinterInfo("First", "Laser", "USB001"),
        PrinterInfo("Second", "Laser", "USB002"),
    ])
    with pytest.raises(PrintError, match="多台实体打印机") as caught:
        PrinterAdapter(backend=backend).prepare()
    assert "configs/gui.json" in str(caught.value)
    assert "printer.name" in str(caught.value)
    assert backend.submissions == []
    assert PrinterAdapter("Second", backend=backend).prepare().name == "Second"


def test_unique_physical_printer_can_be_selected_with_virtual_default():
    backend = FakeBackend(printers=[
        PrinterInfo("PDF", is_default=True),
        PrinterInfo("Physical", "Laser", "USB001"),
    ])
    assert PrinterAdapter(backend=backend).prepare().name == "Physical"
    assert backend.submissions == []


def test_explicit_printer_selection_and_missing_name():
    backend = FakeBackend()
    assert PrinterAdapter("Physical", backend=backend).prepare().name == "Physical"
    with pytest.raises(PrintError, match="找不到"):
        PrinterAdapter("Missing", backend=backend).prepare()


@pytest.mark.parametrize("status,label", [(0x80, "脱机"), (0x10, "缺纸"), (0x08, "卡纸")])
def test_prepare_reports_device_error(status, label):
    backend = FakeBackend(printers=[PrinterInfo("Physical", status=status)])
    with pytest.raises(PrintError, match=label):
        PrinterAdapter(backend=backend).prepare()
    assert backend.submissions == []


def test_tracks_exact_job_and_only_explicit_driver_printed_completes(picture):
    backend = FakeBackend([JobSnapshot(42, pages_printed=0, total_pages=1), JobSnapshot(42, JOB_PRINTED)])
    observed = []
    result = PrinterAdapter(backend=backend).print_image(
        picture, on_job=observed.append, poll_interval=0.001
    )
    assert result.status == "printed"
    assert result.needs_confirmation is False
    assert result.job_id == 42
    assert "驱动报告" in result.message
    assert observed == [42]
    assert len(backend.submissions) == 1
    assert all(read[2] == 42 for read in backend.reads if read[0] == "job")


@pytest.mark.parametrize("job", [None, JobSnapshot(42, JOB_COMPLETE)])
def test_queue_disappearance_or_complete_requires_manual_paper_confirmation(picture, job):
    backend = FakeBackend([job])
    result = PrinterAdapter(backend=backend).print_image(picture)
    assert result.status == "spooled"
    assert result.needs_confirmation is True
    assert len(backend.submissions) == 1


def test_job_error_preserves_id_and_does_not_retry(picture):
    backend = FakeBackend([JobSnapshot(42, JOB_PAPEROUT)])
    with pytest.raises(PrintError, match="等待纸张") as caught:
        PrinterAdapter(backend=backend).print_image(picture)
    assert caught.value.job_id == 42
    assert caught.value.printer == "Physical"
    assert caught.value.requires_resolution is True
    assert len(backend.submissions) == 1


def test_monitor_timeout_preserves_id_and_does_not_retry(picture, monkeypatch):
    backend = FakeBackend([JobSnapshot(42)])
    ticks = iter([0.0, 10.0])
    monkeypatch.setattr(gui_print.time, "monotonic", lambda: next(ticks))
    with pytest.raises(PrintError, match="超时") as caught:
        PrinterAdapter(backend=backend).print_image(picture, timeout=1)
    assert caught.value.job_id == 42
    assert caught.value.requires_resolution
    assert len(backend.submissions) == 1


def test_resume_and_inspect_never_submit():
    backend = FakeBackend([JobSnapshot(42, JOB_PRINTED, document="original-document")])
    adapter = PrinterAdapter(backend=backend)
    assert adapter.inspect_job(42, printer="Physical").job_id == 42
    assert adapter.wait_for_job(42, printer="Physical", expected_document="original-document").status == "printed"
    assert backend.submissions == []


def test_submit_failure_after_job_registration_preserves_pending_id(picture):
    class FailingBackend(FakeBackend):
        def submit_image(self, printer, path, on_job):
            self.submissions.append((printer, path))
            on_job(42)
            raise OSError("EndPage failure")

    backend = FailingBackend()
    with pytest.raises(PrintError, match="不会自动重复提交") as caught:
        PrinterAdapter(backend=backend).print_image(picture)
    assert caught.value.job_id == 42
    assert caught.value.requires_resolution
    assert len(backend.submissions) == 1


def test_missing_image_is_checked_before_printer_access(tmp_path):
    class UnavailableBackend:
        def list_printers(self):
            pytest.fail("must not access printer before validating image")

    with pytest.raises(PrintError, match="不存在"):
        PrinterAdapter(backend=UnavailableBackend()).print_image(tmp_path / "missing.png")


@pytest.mark.parametrize("image,page,expected", [
    ((100, 100), (1000, 1400), (0, 200, 1000, 1200)),
    ((200, 100), (1000, 1400), (0, 450, 1000, 950)),
    ((100, 200), (1400, 1000), (450, 0, 950, 1000)),
])
def test_image_contain_is_centered_without_stretch(image, page, expected):
    assert contain_box(image, page) == expected


def test_invalid_geometry_and_timeouts(picture):
    with pytest.raises(ValueError):
        contain_box((0, 100), (100, 100))
    with pytest.raises(ValueError):
        PrinterAdapter(backend=FakeBackend()).print_image(picture, timeout=0)


def test_printer_selection_stays_fixed_after_default_changes():
    backend = FakeBackend(printers=[PrinterInfo("First", is_default=True), PrinterInfo("Second")])
    adapter = PrinterAdapter(backend=backend)
    assert adapter.prepare().name == "First"
    backend.printers = [PrinterInfo("First"), PrinterInfo("Second", is_default=True)]
    assert adapter.prepare().name == "First"
    backend.printers = [PrinterInfo("Second", is_default=True)]
    with pytest.raises(PrintError, match="First"):
        adapter.prepare()


def test_document_identity_is_stable_and_distinguishes_same_basename(tmp_path):
    first = tmp_path / "round-one" / "output.png"
    second = tmp_path / "round-two" / "output.png"
    assert document_name_for(first) == document_name_for(first)
    assert document_name_for(first) != document_name_for(second)


def test_reused_job_id_from_another_document_stays_locked():
    backend = FakeBackend([JobSnapshot(42, JOB_PRINTED, document="unrelated-document")])
    with pytest.raises(PrintError, match="其他文档") as caught:
        PrinterAdapter(backend=backend).wait_for_job(
            42, printer="Physical", expected_document="original-document"
        )
    assert caught.value.requires_resolution
    assert backend.submissions == []


@pytest.mark.parametrize("expected,actual", [(None, "other-document"), ("original", "")])
def test_unverifiable_document_identity_never_auto_completes(expected, actual):
    backend = FakeBackend([JobSnapshot(42, JOB_PRINTED, document=actual)])
    result = PrinterAdapter(backend=backend).wait_for_job(
        42, printer="Physical", expected_document=expected
    )
    assert result.status == "spooled"
    assert result.needs_confirmation


def test_explicit_deleted_job_is_error_not_printed():
    backend = FakeBackend([JobSnapshot(42, JOB_DELETED, document="original")])
    with pytest.raises(PrintError, match="已被删除") as caught:
        PrinterAdapter(backend=backend).wait_for_job(
            42, printer="Physical", expected_document="original"
        )
    assert caught.value.requires_resolution


def test_non_square_device_dpi_preserves_physical_aspect_ratio():
    left, top, right, bottom = contain_box((210, 297), (4960, 14032), (600, 1200))
    physical_width = (right - left) / 600
    physical_height = (bottom - top) / 1200
    assert physical_width / physical_height == pytest.approx(210 / 297, abs=0.0001)
    assert left == 0
    assert top <= 1


def test_gdi_receives_top_down_bgrx_and_checks_pixel_transfer():
    from PIL import Image

    class Header(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32), ("biHeight", ctypes.c_int32),
            ("biPlanes", ctypes.c_uint16), ("biBitCount", ctypes.c_uint16),
            ("biCompression", ctypes.c_uint32), ("biSizeImage", ctypes.c_uint32),
            ("biXPelsPerMeter", ctypes.c_int32), ("biYPelsPerMeter", ctypes.c_int32),
            ("biClrUsed", ctypes.c_uint32), ("biClrImportant", ctypes.c_uint32),
        ]

    captured = {}

    def draw(*args):
        header = args[10]._obj
        captured["header"] = header
        captured["pixels"] = ctypes.string_at(args[9], header.biSizeImage)
        captured["box"] = args[1:5]
        return 2

    backend = WindowsPrintBackend.__new__(WindowsPrintBackend)
    backend.BitmapInfoHeader = Header
    backend.gdi = SimpleNamespace(
        SetStretchBltMode=lambda *args: 1, SetBrushOrgEx=lambda *args: True,
        StretchDIBits=draw,
    )
    image = Image.new("RGB", (1, 2))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((0, 1), (0, 0, 255))
    backend._draw_image(99, image, (10, 20, 30, 60))
    assert captured["header"].biHeight == -2
    assert captured["header"].biBitCount == 32
    assert captured["pixels"] == b"\x00\x00\xff\x00\xff\x00\x00\x00"
    assert captured["box"] == (10, 20, 20, 40)
    backend.gdi.StretchDIBits = lambda *args: 0
    backend._win_error = lambda action: PrintError(action)
    with pytest.raises(PrintError, match="像素传输失败"):
        backend._draw_image(99, image, (10, 20, 30, 60))
