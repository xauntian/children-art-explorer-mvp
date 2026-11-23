"""The direct Mijia adapter is exercised only through in-memory sessions."""

from __future__ import annotations

import io
import json
import random

import pytest
from PIL import Image, ImageDraw

from mijia1s import printer as mijia_print
from mijia1s.records import PrinterInfo, PrintError, document_name_for
from mijia1s.printer import MijiaNotSubmittedError, MijiaPrinterAdapter

ADDRESS = "02:00:00:00:00:01"
OTHER_ADDRESS = "02:00:00:00:00:02"
JOB_ID = 42


class FakeSession:
    def __init__(self, *, status=None, job=None):
        self.status = status or {"category": "idle", "error": 0}
        self.job = job or {
            "job_id": JOB_ID,
            "job_state": "finished",
            "job_type": 0,
            "prt_copies": 1,
        }
        self.events = []
        self.uploads = []
        self.failures = {}
        self.job_reply = {"job_id": JOB_ID}

    def __enter__(self):
        self.events.append(("enter",))
        if "connect" in self.failures:
            raise self.failures["connect"]
        return self

    def __exit__(self, *_args):
        self.events.append(("exit",))

    def rpc(self, method, params):
        self.events.append(("rpc", method, params))
        if method in self.failures:
            raise self.failures[method]
        if method == "mixed_status":
            return dict(self.status)
        if method == "print_job":
            return self.job_reply
        if method == "confirm_job":
            return {}
        if method == "job_info":
            return [dict(self.job)]
        pytest.fail(f"Unexpected printer method: {method}")

    def send_image(self, data, job_id, progress=None):
        self.events.append(("upload", job_id))
        if "upload" in self.failures:
            raise self.failures["upload"]
        self.uploads.append((bytes(data), job_id))
        if progress is not None:
            progress(1, 1)

    def calls(self, method):
        return [event for event in self.events if event[:2] == ("rpc", method)]


class FakeFactory:
    def __init__(self, session=None):
        self.session = session or FakeSession()
        self.addresses = []

    def __call__(self, address):
        self.addresses.append(address)
        return self.session


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(mijia_print.time, "monotonic", lambda: now[0])

    def sleep(seconds):
        now[0] += max(float(seconds), 0.01)

    monkeypatch.setattr(mijia_print.time, "sleep", sleep)
    return now


@pytest.fixture
def picture(tmp_path):
    path = tmp_path / "confirmed.png"
    Image.new("RGBA", (80, 120), (30, 130, 210, 255)).save(path)
    return path


def make_adapter(tmp_path, session=None, address=ADDRESS):
    factory = FakeFactory(session)
    adapter = MijiaPrinterAdapter(
        address, receipts_dir=tmp_path / "receipts", session_factory=factory
    )
    return adapter, factory


def receipt_for(tmp_path, picture):
    return tmp_path / "receipts" / f"{document_name_for(picture)}.json"


def test_prepare_reads_status_without_creating_or_uploading_a_job(tmp_path):
    adapter, factory = make_adapter(tmp_path)

    assert isinstance(adapter.prepare(), PrinterInfo)

    assert factory.addresses and set(factory.addresses) == {ADDRESS}
    assert factory.session.calls("mixed_status")
    assert not factory.session.calls("print_job")
    assert not factory.session.uploads


@pytest.mark.parametrize("status", [
    {"category": "printing", "error": 0},
    {"category": "idle", "error": 8},
])
def test_busy_or_faulted_printer_cannot_create_a_job(tmp_path, picture, status):
    adapter, factory = make_adapter(tmp_path, FakeSession(status=status))

    with pytest.raises(MijiaNotSubmittedError) as caught:
        adapter.print_image(picture)

    assert caught.value.safe_to_retry is True
    assert caught.value.requires_resolution is False
    assert caught.value.job_id is None
    assert not factory.session.calls("print_job")
    assert not factory.session.uploads
    assert not receipt_for(tmp_path, picture).exists()


@pytest.mark.parametrize("phase", ["connect", "mixed_status"])
def test_connection_or_status_failure_before_create_is_safe_to_retry(
    tmp_path, picture, phase
):
    session = FakeSession()
    session.failures[phase] = ConnectionError("printer powered off")
    adapter, _ = make_adapter(tmp_path, session)

    with pytest.raises(MijiaNotSubmittedError) as caught:
        adapter.print_image(picture)

    assert caught.value.safe_to_retry is True
    assert caught.value.requires_resolution is False
    assert caught.value.job_id is None
    assert not session.calls("print_job")
    assert not session.uploads
    assert not session.calls("confirm_job")
    assert not receipt_for(tmp_path, picture).exists()


def test_failed_initial_receipt_write_cannot_send_a_create_request(
    tmp_path, picture, monkeypatch
):
    adapter, factory = make_adapter(tmp_path)

    def fail_write(*_args, **_kwargs):
        raise OSError("receipt directory is not writable")

    monkeypatch.setattr(mijia_print, "atomic_write_json", fail_write)

    with pytest.raises(MijiaNotSubmittedError) as caught:
        adapter.print_image(picture)

    assert caught.value.safe_to_retry is True
    assert caught.value.requires_resolution is False
    assert caught.value.job_id is None
    assert not factory.session.calls("print_job")
    assert not factory.session.uploads
    assert not factory.session.calls("confirm_job")


def test_print_creates_one_copy_then_records_job_before_callback_and_upload(
    tmp_path, picture, clock
):
    session = FakeSession()
    adapter, _ = make_adapter(tmp_path, session)
    source = picture.read_bytes()

    def on_job(job_id):
        record = json.loads(receipt_for(tmp_path, picture).read_text(encoding="utf-8"))
        assert record["job_id"] == JOB_ID
        assert record["address"] == ADDRESS
        assert record["phase"]
        session.events.append(("on_job", job_id))

    result = adapter.print_image(picture, on_job=on_job)

    assert result.status == "printed"
    assert result.job_id == JOB_ID
    assert result.needs_confirmation is False
    assert len(session.calls("print_job")) == 1
    params = session.calls("print_job")[0][2]
    assert params["copies"] == 1
    assert params["channel"] == 528
    assert params["job_type"] == 0
    assert params["job_id"] == 0
    assert params["file_size"] == len(session.uploads[0][0])
    assert session.events.index(("on_job", JOB_ID)) < session.events.index(("upload", JOB_ID))
    assert session.events.index(("upload", JOB_ID)) < session.events.index(
        ("rpc", "confirm_job", [JOB_ID])
    )
    assert len(session.uploads) == 1
    assert picture.read_bytes() == source
    record = json.loads(receipt_for(tmp_path, picture).read_text(encoding="utf-8"))
    assert record["job_id"] == JOB_ID
    assert record["address"] == ADDRESS
    assert len(record["image_sha256"]) == 64


def test_upload_is_rgb_portrait_jpeg_within_device_size_limit(tmp_path, picture, clock):
    adapter, factory = make_adapter(tmp_path)
    adapter.print_image(picture)

    data, job_id = factory.session.uploads[0]
    assert job_id == JOB_ID
    assert len(data) <= 500 * 1024
    with Image.open(io.BytesIO(data)) as uploaded:
        assert uploaded.format == "JPEG"
        assert uploaded.mode == "RGB"
        assert uploaded.size == (1040, 1560)


def test_real_adapter_uploads_template_and_records_this_style(tmp_path, picture, clock):
    adapter, factory = make_adapter(tmp_path)
    snapshot = {"id": "cubism", "print_name": "Cubism",
                "print_description": "Shapes show the same thing from different sides."}

    assert adapter.print_image(picture, style_snapshot=snapshot).status == "printed"

    data, _ = factory.session.uploads[0]
    with Image.open(io.BytesIO(data)) as uploaded:
        assert uploaded.crop((43, 1420, 997, 1540)).getextrema()[0][0] < 128
    saved = json.loads(receipt_for(tmp_path, picture).read_text(encoding="utf-8"))
    assert saved["print_name"] == snapshot["print_name"]
    assert saved["print_description"] == snapshot["print_description"]
    assert saved["print_layout_path"] == str(picture.with_name("confirmed-print.png"))
    assert picture.with_name("confirmed-print.png").is_file()
    assert len(factory.session.calls("print_job")) == 1


def test_landscape_is_rotated_and_contained_without_removing_source_edges(tmp_path, clock):
    picture = tmp_path / "landscape.png"
    source = Image.new("RGB", (400, 100), (0, 180, 0))
    drawing = ImageDraw.Draw(source)
    drawing.rectangle((0, 0, 79, 99), fill=(230, 0, 0))
    drawing.rectangle((320, 0, 399, 99), fill=(0, 0, 230))
    source.save(picture)
    original = picture.read_bytes()
    adapter, factory = make_adapter(tmp_path)

    adapter.print_image(picture)

    with Image.open(io.BytesIO(factory.session.uploads[0][0])) as uploaded:
        # A quarter-turn makes this a narrow portrait; contain leaves side margins.
        assert min(uploaded.getpixel((20, 780))) > 240
        assert min(uploaded.getpixel((1020, 780))) > 240
        top = uploaded.getpixel((520, 60))
        bottom = uploaded.getpixel((520, 1500))
        assert {max(range(3), key=top.__getitem__), max(range(3), key=bottom.__getitem__)} == {0, 2}
        assert uploaded.getpixel((520, 780))[1] > 140
    assert picture.read_bytes() == original


def test_high_detail_image_is_compressed_below_the_transfer_limit(tmp_path, clock):
    picture = tmp_path / "noise.png"
    Image.frombytes("RGB", (1300, 1800), random.Random(7).randbytes(1300 * 1800 * 3)).save(picture)
    adapter, factory = make_adapter(tmp_path)

    adapter.print_image(picture)

    assert 0 < len(factory.session.uploads[0][0]) <= 500 * 1024


def test_callback_failure_keeps_job_id_and_stops_before_upload_or_confirm(
    tmp_path, picture, clock
):
    adapter, factory = make_adapter(tmp_path)

    def fail_callback(_job_id):
        raise OSError("local pending-job persistence failed")

    with pytest.raises(PrintError) as caught:
        adapter.print_image(picture, on_job=fail_callback)

    assert caught.value.job_id == JOB_ID
    assert caught.value.requires_resolution
    assert len(factory.session.calls("print_job")) == 1
    assert not factory.session.uploads
    assert not factory.session.calls("confirm_job")
    assert json.loads(receipt_for(tmp_path, picture).read_text(encoding="utf-8"))["job_id"] == JOB_ID


def test_upload_failure_preserves_receipt_and_never_confirms(tmp_path, picture, clock):
    session = FakeSession()
    session.failures["upload"] = ConnectionError("link lost")
    adapter, _ = make_adapter(tmp_path, session)

    with pytest.raises(PrintError) as caught:
        adapter.print_image(picture)

    assert caught.value.job_id == JOB_ID
    assert caught.value.requires_resolution
    assert len(session.calls("print_job")) == 1
    assert not session.calls("confirm_job")
    assert json.loads(receipt_for(tmp_path, picture).read_text(encoding="utf-8"))["job_id"] == JOB_ID


def test_lost_create_reply_is_durable_and_cannot_create_another_job_after_restart(
    tmp_path, picture, clock
):
    session = FakeSession()
    session.failures["print_job"] = TimeoutError("reply lost after send")
    adapter, factory = make_adapter(tmp_path, session)

    with pytest.raises(PrintError) as caught:
        adapter.print_image(picture)

    assert caught.value.job_id is None
    assert caught.value.requires_resolution
    assert not getattr(caught.value, "safe_to_retry", False)
    assert receipt_for(tmp_path, picture).exists()
    session.failures.clear()
    restarted = MijiaPrinterAdapter(
        ADDRESS, receipts_dir=tmp_path / "receipts", session_factory=factory
    )
    with pytest.raises(PrintError):
        restarted.print_image(picture)
    assert len(session.calls("print_job")) == 1
    assert not session.uploads
    assert not session.calls("confirm_job")


@pytest.mark.parametrize("job_id", [0, -1, True, "42", None])
def test_invalid_remote_job_id_never_uploads_or_confirms(tmp_path, picture, clock, job_id):
    session = FakeSession()
    session.job_reply = {"job_id": job_id}
    adapter, _ = make_adapter(tmp_path, session)

    with pytest.raises(PrintError):
        adapter.print_image(picture)

    assert len(session.calls("print_job")) == 1
    assert not session.uploads
    assert not session.calls("confirm_job")


@pytest.mark.parametrize("job_change,status", [
    ({"job_id": 99}, {"category": "idle", "error": 0}),
    ({"job_state": "printing"}, {"category": "idle", "error": 0}),
    ({"job_type": 1}, {"category": "idle", "error": 0}),
    ({"prt_copies": 0}, {"category": "idle", "error": 0}),
    ({}, {"category": "printing", "error": 0}),
    ({}, {"category": "idle", "error": 8}),
])
def test_incomplete_or_mismatched_device_evidence_never_reports_printed(
    tmp_path, picture, clock, job_change, status
):
    session = FakeSession()
    adapter, _ = make_adapter(tmp_path, session)

    def on_job(_job_id):
        session.job.update(job_change)
        session.status = status

    try:
        result = adapter.print_image(picture, on_job=on_job, timeout=0.1, poll_interval=0.05)
    except PrintError as error:
        assert error.job_id == JOB_ID
        assert error.requires_resolution
    else:
        assert result.status == "spooled"
        assert result.needs_confirmation
    assert len(session.calls("print_job")) == 1


def test_new_adapter_requires_manual_confirmation_even_when_remote_job_is_finished(
    tmp_path, picture, clock
):
    adapter, factory = make_adapter(tmp_path)
    assert adapter.print_image(picture).status == "printed"
    restarted = MijiaPrinterAdapter(
        ADDRESS, receipts_dir=tmp_path / "receipts", session_factory=factory
    )

    result = restarted.wait_for_job(
        JOB_ID, expected_document=document_name_for(picture), timeout=0.1, poll_interval=0.05
    )

    assert result.job_id == JOB_ID
    assert result.status == "spooled"
    assert result.needs_confirmation
    assert len(factory.session.calls("print_job")) == 1
    assert len(factory.session.uploads) == 1


def test_receipt_cannot_be_reused_for_another_printer(tmp_path, picture, clock):
    adapter, factory = make_adapter(tmp_path)
    adapter.print_image(picture)
    other = MijiaPrinterAdapter(
        OTHER_ADDRESS, receipts_dir=tmp_path / "receipts", session_factory=factory
    )

    with pytest.raises(PrintError):
        other.print_image(picture)

    assert len(factory.session.calls("print_job")) == 1
    assert OTHER_ADDRESS not in factory.addresses


def test_wait_rejects_an_explicitly_different_printer_without_access(tmp_path):
    adapter, factory = make_adapter(tmp_path)

    with pytest.raises(PrintError):
        adapter.wait_for_job(JOB_ID, printer=OTHER_ADDRESS)

    assert not factory.addresses


def test_resume_rejects_changed_source_before_connecting(tmp_path, picture, clock):
    adapter, factory = make_adapter(tmp_path)
    adapter.print_image(picture)
    connections = len(factory.addresses)
    Image.new("RGB", (80, 120), "red").save(picture)

    with pytest.raises(PrintError):
        adapter.wait_for_job(JOB_ID, expected_document=document_name_for(picture))

    assert len(factory.addresses) == connections
    assert len(factory.session.calls("print_job")) == 1


def test_missing_or_invalid_image_is_rejected_before_printer_access(tmp_path):
    adapter, factory = make_adapter(tmp_path)
    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not an image")

    for path in (tmp_path / "missing.png", invalid):
        with pytest.raises(MijiaNotSubmittedError) as caught:
            adapter.print_image(path)
        assert caught.value.safe_to_retry is True
        assert caught.value.requires_resolution is False
        assert caught.value.job_id is None
        assert not receipt_for(tmp_path, path).exists()

    assert not factory.addresses
