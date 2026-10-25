"""Desktop P gating, recovery and video preview without real devices or printing."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from app import gui_runtime
from app.errors import ConfigError
from app.gui_print import PrinterInfo, PrintError, PrintResult
from tests.fixtures import make_app_config


def wait_worker(runtime):
    runtime._worker.join(timeout=3)
    assert not runtime._worker.is_alive(), "mocked workflow did not finish"


def take_events(runtime):
    values = []
    while not runtime.events.empty():
        values.append(runtime.events.get_nowait())
    return values


@pytest.fixture
def runtime_env(tmp_path, monkeypatch):
    make_app_config(tmp_path, checker={"enabled": False})
    phone_config = {
        "project_root": tmp_path,
        "capture_runs_dir": tmp_path / "runs/phone",
        "capture_backend": "camera_stream",
        "phone": {"serial": "same-phone", "fallback_photo": True},
        "extraction": {},
        "generation_warmup": True,
    }
    loader = Mock(return_value=phone_config)
    monkeypatch.setattr(gui_runtime, "load_phone_config", loader)
    reconcile = Mock(return_value=[])
    monkeypatch.setattr(gui_runtime, "reconcile_phone_sessions", reconcile)
    lock = MagicMock()
    lock_factory = Mock(return_value=lock)
    monkeypatch.setattr(gui_runtime, "PhoneRunLock", lock_factory)
    camera = Mock()
    report = {"available": True, "serial": "same-phone"}
    camera.prepare.return_value = report
    preview = Mock()
    preview.latest_frame.return_value = None
    preview_factory = Mock(return_value=preview)
    printer = Mock()
    printer.prepare.return_value = PrinterInfo("Physical A4")
    spooled = PrintResult("spooled", "Physical A4", 17, True, "await paper")
    printed = PrintResult("printed", "Physical A4", 17, False, "driver PRINTED")
    printer.wait_for_job.return_value = printed
    session_dir = tmp_path / "runs/phone/mock-session"
    session_dir.mkdir(parents=True)
    for name in ("preview.png", "artwork.png"):
        (session_dir / name).write_bytes(b"test preview")
    output = tmp_path / "runs/mock-generation/output.png"
    output.parent.mkdir()
    output.write_bytes(b"test output")
    session = SimpleNamespace(directory=session_dir, meta={"capture": {"serial": "same-phone"}})
    meta = {
        "state": "COMPLETED", "output_path": str(output),
        "generation_job_id": "phone-mock-session", "generation": {"style_id": "cubism"},
    }
    controller = Mock()
    controller.session = session
    controller.generation_store.find_active.return_value = None
    factory = Mock(return_value=controller)

    def prepare(_mode):
        observer = factory.call_args.kwargs["observer"]
        for event in ("capturing", "captured", "prepared"):
            observer(event, session)
        return session

    def generate(value):
        assert value is session
        observer = factory.call_args.kwargs["observer"]
        for event in ("generating", "generated"):
            observer(event, session)
        return meta

    controller.prepare.side_effect = prepare
    controller.generate.side_effect = generate

    def print_image(_output, **kwargs):
        kwargs["on_job"](17)
        kwargs["progress"]("printing")
        return spooled

    printer.print_image.side_effect = print_image
    service = Mock(return_value={"reused": True})
    warmup = Mock(return_value={"total_sec": 0.1})
    config = {"printer": {"name": None, "timeout_sec": 1, "poll_interval_sec": 0.01}}

    def build():
        return gui_runtime.GuiRuntime(
            tmp_path, config, camera=camera, printer=printer, service=service,
            warmup=warmup, controller_factory=factory, preview_factory=preview_factory,
        )

    return SimpleNamespace(
        root=tmp_path, runtime=build(), build=build, camera=camera, printer=printer,
        service=service, warmup=warmup, controller=controller, factory=factory,
        phone=phone_config, loader=loader, lock=lock, lock_factory=lock_factory,
        reconcile=reconcile, session=session, meta=meta, output=output,
        spooled=spooled, printed=printed, preview=preview, preview_factory=preview_factory,
    )


def ready(env):
    assert env.runtime.start() is True
    wait_worker(env.runtime)
    assert env.runtime.state == "READY"
    return env.runtime


def test_start_checks_all_dependencies_and_preserves_manual_phone_settings(runtime_env):
    env = runtime_env
    runtime = ready(env)

    assert runtime.start() is False
    assert env.phone["capture_backend"] == "adb"
    assert env.phone["phone"]["fallback_photo"] is False
    assert env.phone["phone"]["serial"] == "same-phone"
    assert "zoom" not in env.phone["phone"]
    assert runtime.printer_name == "Physical A4"
    env.camera.prepare.assert_called_once_with()
    env.preview_factory.assert_called_once_with(env.root, env.phone["phone"], {})
    env.preview.start.assert_called_once_with(serial="same-phone")
    env.printer.prepare.assert_called_once_with()
    env.service.assert_called_once()
    env.warmup.assert_called_once_with(env.root)
    env.controller.prepare.assert_not_called()
    env.controller.generate.assert_not_called()
    env.printer.print_image.assert_not_called()
    dependencies = [data for kind, data in take_events(runtime) if kind == "dependency"]
    assert {data["name"] for data in dependencies if data["status"] == "ready"} == {"phone", "ai", "printer"}


def test_warmup_readiness_reports_actual_completed_style_count(runtime_env):
    env = runtime_env
    env.warmup.return_value = {
        "total_sec": 24.5,
        "style_prewarms": [{"status": "completed"} for _ in range(8)] + [{"status": "skipped"}],
    }

    runtime = ready(env)

    dependencies = [data for kind, data in take_events(runtime) if kind == "dependency"]
    assert dependencies[-1]["message"] == "AI 模型已就绪 · 8 种画派 · 预热 24.5 秒"


@pytest.mark.parametrize("snapshot,expected", [
    ({"name_zh": "波普艺术", "name": "Pop Art", "id": "pop_art"}, "波普艺术"),
    ({"name_zh": "", "name": "Pop Art", "id": "pop_art"}, "Pop Art"),
    ({"id": "pop_art"}, "pop_art"),
])
def test_style_name_comes_from_the_completed_job_snapshot(runtime_env, snapshot, expected):
    env = runtime_env
    env.meta["generation"]["style_id"] = "pop_art"
    env.controller.generation_store.load.return_value = SimpleNamespace(meta={"style": snapshot})
    runtime = ready(env)
    take_events(runtime)

    assert runtime.trigger() is True
    wait_worker(runtime)

    styles = [data for kind, data in take_events(runtime) if kind == "style"]
    assert styles == [{"style_id": "pop_art", "style_name": expected}]
    env.controller.generation_store.load.assert_called_once_with("phone-mock-session")


def test_missing_style_snapshot_keeps_successful_round_and_uses_id(runtime_env):
    env = runtime_env
    env.controller.generation_store.load.side_effect = ConfigError("job snapshot unavailable")
    runtime = ready(env)
    take_events(runtime)

    assert runtime.trigger() is True
    wait_worker(runtime)

    assert runtime.state == "AWAITING_PRINT_CONFIRM"
    styles = [data for kind, data in take_events(runtime) if kind == "style"]
    assert styles == [{"style_id": "cubism", "style_name": "cubism"}]


def test_p_remains_locked_until_spooled_paper_is_confirmed(runtime_env):
    env = runtime_env
    runtime = ready(env)
    entered, release = threading.Event(), threading.Event()

    def printing(_path, **kwargs):
        before = json.loads(runtime.pending_path.read_text(encoding="utf-8"))
        assert before["status"] == "submitting"
        assert before["printer"] == "Physical A4"
        kwargs["on_job"](17)
        saved = json.loads(runtime.pending_path.read_text(encoding="utf-8"))
        assert saved["job_id"] == 17
        assert saved["status"] == "printing"
        entered.set()
        assert release.wait(3)
        return env.spooled

    env.printer.print_image.side_effect = printing
    assert runtime.trigger() is True
    assert entered.wait(3)
    try:
        assert runtime.state == "RUNNING"
        assert runtime.trigger() is False
        assert runtime.confirm_printed() is False
        assert runtime.close() is False
    finally:
        release.set()
    wait_worker(runtime)
    assert runtime.state == "AWAITING_PRINT_CONFIRM"
    assert runtime.trigger() is False
    assert runtime.pending["job_id"] == 17
    assert runtime.confirm_printed() is True
    assert runtime.state == "READY"
    assert runtime.pending is None
    assert json.loads(runtime.pending_path.read_text(encoding="utf-8"))["status"] == "completed"
    assert runtime.confirm_printed() is False
    assert runtime.trigger() is True
    wait_worker(runtime)
    assert env.controller.generate.call_count == 2
    assert env.printer.print_image.call_count == 2
    env.warmup.assert_called_once()


def test_driver_printed_completes_round_and_reactivates_p(runtime_env):
    env = runtime_env
    runtime = ready(env)
    env.printer.print_image.side_effect = None
    env.printer.print_image.return_value = env.printed

    assert runtime.trigger() is True
    wait_worker(runtime)

    assert runtime.state == "READY"
    assert runtime.pending is None
    saved = json.loads(runtime.pending_path.read_text(encoding="utf-8"))
    assert saved["confirmation"] == "driver_reported_printed"
    events = take_events(runtime)
    assert any(kind == "completed" for kind, _data in events)
    assert {data["key"] for kind, data in events if kind == "image"} == {"original", "rectified", "ai"}
    assert any(kind == "stage" and data["step"] == "print" and data["status"] == "done" for kind, data in events)


def test_print_error_keeps_p_locked_and_stores_original_job_identity(runtime_env):
    env = runtime_env
    runtime = ready(env)
    env.printer.print_image.side_effect = PrintError(
        "paper out", printer="Original Printer", job_id=42, requires_resolution=True,
    )

    assert runtime.trigger() is True
    wait_worker(runtime)

    assert runtime.state == "PRINT_ERROR"
    assert runtime.trigger() is False
    assert runtime.confirm_printed() is False
    assert runtime.pending["job_id"] == 42
    assert runtime.pending["printer"] == "Original Printer"
    saved = json.loads(runtime.pending_path.read_text(encoding="utf-8"))
    assert saved["job_id"] == 42
    controls = [data for kind, data in take_events(runtime) if kind == "print_controls"]
    assert controls[-1] == {"confirm": False, "retry": True}


def test_pending_restart_never_resubmits_and_queries_matching_document(runtime_env):
    from app.gui_print import document_name_for

    env = runtime_env
    runtime = ready(env)
    assert runtime.trigger() is True
    wait_worker(runtime)
    assert runtime.close() is True
    env.printer.print_image.reset_mock()
    env.controller.generate.reset_mock()
    env.warmup.reset_mock()
    restored = env.build()

    assert restored.start() is True
    wait_worker(restored)

    assert restored.state == "PRINT_ERROR"
    assert restored.trigger() is False
    assert restored.confirm_printed() is False
    env.printer.print_image.assert_not_called()
    env.controller.generate.assert_not_called()
    env.warmup.assert_not_called()
    assert restored.retry_print_status() is True
    wait_worker(restored)
    assert restored.state == "READY"
    call = env.printer.wait_for_job.call_args
    assert call.args == (17,)
    assert call.kwargs["printer"] == "Physical A4"
    assert call.kwargs["expected_document"] == document_name_for(env.output)
    env.printer.print_image.assert_not_called()


def test_submitting_crash_with_unknown_job_keeps_lock_and_gives_recovery_guidance(runtime_env):
    env = runtime_env
    runtime = ready(env)
    env.printer.print_image.side_effect = PrintError("submission uncertain")
    assert runtime.trigger() is True
    wait_worker(runtime)

    assert runtime.retry_print_status() is True
    wait_worker(runtime)

    assert runtime.state == "PRINT_ERROR"
    assert runtime.confirm_printed() is False
    assert runtime.trigger() is False
    env.printer.wait_for_job.assert_not_called()
    errors = [data["message"] for kind, data in take_events(runtime) if kind == "error"]
    assert any("现场操作人员" in message and "Windows" in message for message in errors)
    assert env.printer.print_image.call_count == 1


def test_pending_record_failure_never_prints_or_strands_worker(runtime_env, monkeypatch):
    env = runtime_env
    runtime = ready(env)
    monkeypatch.setattr(gui_runtime, "atomic_write_json", Mock(side_effect=OSError("disk full")))

    assert runtime.trigger() is True
    wait_worker(runtime)

    assert runtime.state == "PRINT_ERROR"
    assert runtime.trigger() is False
    assert runtime.confirm_printed() is False
    env.printer.print_image.assert_not_called()
    assert any(kind == "error" and "恢复记录无法保存" in data["message"] for kind, data in take_events(runtime))


def test_print_job_callback_failure_still_saves_id_from_printer_error(runtime_env, monkeypatch):
    env = runtime_env
    runtime = ready(env)
    real_write = gui_runtime.atomic_write_json

    def write(path, data):
        if data.get("status") == "printing":
            raise OSError("intermittent disk error")
        return real_write(path, data)

    monkeypatch.setattr(gui_runtime, "atomic_write_json", write)

    def printer(_path, **kwargs):
        try:
            kwargs["on_job"](81)
        except OSError as exc:
            raise PrintError("callback failed", printer="Physical A4", job_id=81) from exc

    env.printer.print_image.side_effect = printer
    assert runtime.trigger() is True
    wait_worker(runtime)

    assert runtime.state == "PRINT_ERROR"
    assert runtime.pending["job_id"] == 81
    assert json.loads(runtime.pending_path.read_text(encoding="utf-8"))["job_id"] == 81


@pytest.mark.parametrize("dependency", ["phone", "printer", "ai"])
def test_start_failure_still_checks_other_dependencies_and_can_retry(runtime_env, dependency):
    env = runtime_env
    selected = {"phone": env.camera.prepare, "printer": env.printer.prepare, "ai": env.service}[dependency]
    original = selected.return_value
    selected.side_effect = RuntimeError(f"{dependency} missing")
    assert env.runtime.start() is True
    wait_worker(env.runtime)

    assert env.runtime.state == "ERROR"
    assert env.runtime.trigger() is False
    env.camera.prepare.assert_called_once()
    env.printer.prepare.assert_called_once()
    env.service.assert_called_once()
    env.warmup.assert_not_called()
    selected.side_effect = None
    selected.return_value = original
    assert env.runtime.start() is True
    wait_worker(env.runtime)
    assert env.runtime.state == "READY"
    env.lock.__enter__.assert_called_once()


def test_failed_lock_acquisition_is_retried_and_never_bypassed(runtime_env):
    env = runtime_env
    env.lock.__enter__.side_effect = [ConfigError("another instance"), env.lock]
    assert env.runtime.start() is True
    wait_worker(env.runtime)
    assert env.runtime.state == "ERROR"
    assert env.runtime._phone_lock is None
    env.camera.prepare.assert_not_called()

    assert env.runtime.start() is True
    wait_worker(env.runtime)

    assert env.runtime.state == "READY"
    assert env.lock.__enter__.call_count == 2


def test_active_generation_blocks_startup_without_resubmitting(runtime_env):
    env = runtime_env
    env.controller.generation_store.find_active.return_value = SimpleNamespace(job_id="old-job")
    assert env.runtime.start() is True
    wait_worker(env.runtime)

    assert env.runtime.state == "ERROR"
    env.controller.generate.assert_not_called()
    env.printer.print_image.assert_not_called()
    env.service.assert_not_called()


def test_failed_phone_report_blocks_preview_start_and_ready(runtime_env):
    env = runtime_env
    env.camera.prepare.return_value = {"available": False, "serial": None, "error": "locked"}
    assert env.runtime.start() is True
    wait_worker(env.runtime)

    assert env.runtime.state == "ERROR"
    env.preview_factory.assert_not_called()
    env.preview.start.assert_not_called()
    env.warmup.assert_not_called()
    assert any(kind == "dependency" and data["name"] == "phone" and data["status"] == "error"
               for kind, data in take_events(env.runtime))


def test_missing_generation_output_never_submits_print_and_allows_new_capture(runtime_env):
    env = runtime_env
    runtime = ready(env)
    env.output.unlink()
    assert runtime.trigger() is True
    wait_worker(runtime)

    assert runtime.state == "READY"
    assert runtime.pending is None
    env.printer.print_image.assert_not_called()


def test_generation_failure_never_submits_print_and_allows_new_capture(runtime_env):
    env = runtime_env
    runtime = ready(env)
    env.controller.generate.side_effect = None
    env.controller.generate.return_value = {"state": "FAILED", "error": {"message": "generation failed"}}

    assert runtime.trigger() is True
    wait_worker(runtime)

    assert runtime.state == "READY"
    assert runtime.pending is None
    env.printer.print_image.assert_not_called()


def test_warmup_failure_blocks_p_and_can_retry(runtime_env):
    env = runtime_env
    env.warmup.side_effect = RuntimeError("model load failed")
    assert env.runtime.start() is True
    wait_worker(env.runtime)
    assert env.runtime.state == "ERROR"
    assert env.runtime.trigger() is False

    env.warmup.side_effect = None
    assert env.runtime.start() is True
    wait_worker(env.runtime)

    assert env.runtime.state == "READY"
    env.lock.__enter__.assert_called_once()


@pytest.mark.parametrize("invalid", [
    {"status": "unknown"}, {"output_path": None}, {"job_id": True},
    {"job_id": -1}, {"printer": 42}, {"session_dir": "outside"},
])
def test_corrupt_recovery_record_blocks_start_without_reprint(runtime_env, invalid):
    env = runtime_env
    pending = {
        "status": "spooled", "session_dir": str(env.session.directory),
        "output_path": str(env.output), "job_id": 17, "printer": "Physical A4",
        **invalid,
    }
    env.runtime.pending_path.parent.mkdir(parents=True, exist_ok=True)
    env.runtime.pending_path.write_text(json.dumps(pending), encoding="utf-8")

    assert env.runtime.start() is True
    wait_worker(env.runtime)

    assert env.runtime.state == "ERROR"
    assert env.runtime.trigger() is False
    env.controller.generate.assert_not_called()
    env.printer.print_image.assert_not_called()


def test_restart_after_completed_print_does_not_reprint(runtime_env):
    env = runtime_env
    runtime = ready(env)
    assert runtime.trigger() is True
    wait_worker(runtime)
    assert runtime.confirm_printed() is True
    assert runtime.close() is True
    env.printer.print_image.reset_mock()
    restored = env.build()

    assert restored.start() is True
    wait_worker(restored)

    assert restored.state == "READY"
    assert restored.pending is None
    env.printer.print_image.assert_not_called()


def test_capture_event_only_publishes_saved_image_without_querying_phone(runtime_env):
    env = runtime_env
    runtime = ready(env)
    take_events(runtime)
    env.camera.reset_mock()
    env.preview.reset_mock()

    runtime._phone_event("captured", env.session)

    events = take_events(runtime)
    assert any(kind == "image" and data == {
        "key": "original", "path": str(env.session.directory / "preview.png"),
    } for kind, data in events)
    assert env.camera.mock_calls == []
    assert env.preview.mock_calls == []


def test_preview_starts_after_phone_prepare_and_before_warmup(runtime_env):
    env = runtime_env
    order = []
    report = env.camera.prepare.return_value
    env.camera.prepare.side_effect = lambda: order.append("phone") or report
    env.preview.start.side_effect = lambda **_kwargs: order.append("video")
    env.warmup.side_effect = lambda _root: order.append("warmup")

    ready(env)

    assert order == ["phone", "video", "warmup"]
    assert env.runtime.preview is env.preview


def test_start_retry_reuses_existing_preview_without_starting_another_stream(runtime_env):
    env = runtime_env
    env.warmup.side_effect = RuntimeError("model load failed")
    assert env.runtime.start() is True
    wait_worker(env.runtime)
    assert env.runtime.state == "ERROR"

    env.warmup.side_effect = None
    ready(env)

    env.preview_factory.assert_called_once()
    env.preview.start.assert_called_once_with(serial="same-phone")
    env.preview.close.assert_not_called()


def test_preview_start_failure_blocks_ready_and_can_retry(runtime_env):
    env = runtime_env
    env.preview.start.side_effect = RuntimeError("video decoder unavailable")

    assert env.runtime.start() is True
    wait_worker(env.runtime)

    assert env.runtime.state == "ERROR"
    assert env.runtime.preview is None
    env.preview.close.assert_called_once_with()
    env.service.assert_called_once()
    env.warmup.assert_not_called()
    assert env.runtime.trigger() is False

    env.preview.start.side_effect = None
    ready(env)

    assert env.preview_factory.call_count == 2
    assert env.preview.start.call_count == 2


@pytest.mark.parametrize("state", ["READY", "RUNNING", "PRINTING", "AWAITING_PRINT_CONFIRM", "ERROR"])
def test_preview_frame_reads_only_video_cache_even_during_a_round(runtime_env, state):
    env = runtime_env
    runtime = ready(env)
    runtime.state = state
    frame = SimpleNamespace(sequence=7, image=object())
    env.preview.latest_frame.return_value = frame
    env.camera.reset_mock()
    env.preview.reset_mock()

    assert runtime.get_preview_frame() is frame

    env.preview.latest_frame.assert_called_once_with()
    assert env.camera.mock_calls == []
    env.controller.prepare.assert_not_called()
    env.controller.generate.assert_not_called()


def test_preview_frame_is_empty_before_start_without_opening_devices(runtime_env):
    env = runtime_env

    assert env.runtime.get_preview_frame() is None

    env.preview_factory.assert_not_called()
    assert env.camera.mock_calls == []


def test_close_stops_video_before_closing_phone(runtime_env):
    env = runtime_env
    runtime = ready(env)
    resources = Mock()
    resources.attach_mock(env.preview.close, "preview_close")
    resources.attach_mock(env.camera.close, "camera_close")

    assert runtime.close() is True

    assert [call[0] for call in resources.mock_calls] == ["preview_close", "camera_close"]
    assert runtime.state == "CLOSED"


@pytest.mark.parametrize("bad_config", [
    [], {"printer": []}, {"preview": []},
    {"printer": {"poll_interval_sec": 0}}, {"printer": {"name": ""}},
])
def test_gui_config_rejects_invalid_values(tmp_path, bad_config):
    path = tmp_path / "gui.json"
    path.write_text(json.dumps(bad_config), encoding="utf-8")

    with pytest.raises(ConfigError):
        gui_runtime.load_gui_config(tmp_path, path)
