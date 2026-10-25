"""Image-only GUI rounds never require or alter physical print jobs."""

from __future__ import annotations

import json
import threading
from unittest.mock import Mock

import pytest

from app import gui_print, gui_runtime
from app.errors import ConfigError
from tests.unit.test_gui_runtime import runtime_env as runtime_env_fixture
from tests.unit.test_gui_runtime import take_events, wait_worker

runtime_env = runtime_env_fixture


def disabled_runtime(env, *, printer=True):
    return gui_runtime.GuiRuntime(
        env.root,
        {"printer": {"enabled": False, "timeout_sec": 1, "poll_interval_sec": 0.01}},
        camera=env.camera,
        printer=env.printer if printer else None,
        service=env.service,
        warmup=env.warmup,
        controller_factory=env.factory,
        preview_factory=env.preview_factory,
    )


def start_ready(runtime):
    assert runtime.start() is True
    wait_worker(runtime)
    assert runtime.state == "READY"


def test_disabled_startup_skips_printer_and_recovery_checks(runtime_env, monkeypatch):
    env = runtime_env
    runtime = disabled_runtime(env)
    read_pending = Mock(side_effect=AssertionError("must not inspect print recovery"))
    monkeypatch.setattr(runtime, "_read_pending", read_pending)

    start_ready(runtime)

    assert runtime.print_enabled is False
    env.camera.prepare.assert_called_once_with()
    env.preview.start.assert_called_once_with(serial="same-phone")
    env.service.assert_called_once()
    env.warmup.assert_called_once_with(env.root)
    assert env.printer.mock_calls == []
    read_pending.assert_not_called()
    events = take_events(runtime)
    startup = next(data for kind, data in events if kind == "status")
    assert "打印机" not in startup["message"]
    printer_events = [
        data for kind, data in events if kind == "dependency" and data["name"] == "printer"
    ]
    assert printer_events
    assert {data["status"] for data in printer_events} == {"skipped"}


def test_disabled_startup_does_not_construct_printer_adapter(runtime_env, monkeypatch):
    env = runtime_env
    adapter = Mock(side_effect=AssertionError("must not construct a printer adapter"))
    monkeypatch.setattr(gui_print, "PrinterAdapter", adapter)
    runtime = disabled_runtime(env, printer=False)

    start_ready(runtime)

    adapter.assert_not_called()
    assert runtime.printer is None


def test_disabled_round_outputs_all_images_and_reactivates_p(runtime_env, monkeypatch):
    env = runtime_env
    runtime = disabled_runtime(env)
    write_pending = Mock(side_effect=AssertionError("must not write print recovery"))
    monkeypatch.setattr(gui_runtime, "atomic_write_json", write_pending)
    start_ready(runtime)
    take_events(runtime)
    entered, release = threading.Event(), threading.Event()
    generate = env.controller.generate.side_effect

    def blocked_generate(session):
        entered.set()
        assert release.wait(3)
        return generate(session)

    env.controller.generate.side_effect = blocked_generate
    assert runtime.trigger() is True
    assert entered.wait(3)
    try:
        assert runtime.state == "RUNNING"
        assert runtime.trigger() is False
    finally:
        release.set()
    wait_worker(runtime)

    assert runtime.state == "READY"
    assert runtime.pending is None
    assert not runtime.pending_path.exists()
    assert not (env.session.directory / "print.json").exists()
    assert env.printer.mock_calls == []
    write_pending.assert_not_called()
    events = take_events(runtime)
    assert {data["key"] for kind, data in events if kind == "image"} == {
        "original", "rectified", "ai",
    }
    assert any(kind == "completed" for kind, _data in events)
    assert any(
        kind == "stage" and data["step"] == "print" and data["status"] == "skipped"
        for kind, data in events
    )
    ready = [data for kind, data in events if kind == "status" and data["state"] == "READY"][-1]
    assert "保存" in ready["message"]
    assert "P" in ready["message"] and "重新激活" in ready["message"]
    assert not any(
        kind == "print_controls" and (data["confirm"] or data["retry"])
        for kind, data in events
    )

    assert runtime.trigger() is True
    wait_worker(runtime)

    assert runtime.state == "READY"
    assert env.controller.generate.call_count == 2
    assert env.printer.mock_calls == []
    write_pending.assert_not_called()


@pytest.mark.parametrize("corrupt", [False, True])
def test_disabled_round_preserves_existing_pending_records(runtime_env, corrupt):
    env = runtime_env
    runtime = disabled_runtime(env)
    pending = {
        "status": "spooled", "session_dir": str(env.session.directory),
        "output_path": str(env.output), "job_id": 17, "printer": "Physical A4",
        "needs_confirmation": True,
    }
    original = b"{broken recovery JSON" if corrupt else json.dumps(pending).encode("utf-8")
    runtime.pending_path.parent.mkdir(parents=True, exist_ok=True)
    runtime.pending_path.write_bytes(original)
    session_record = env.session.directory / "print.json"
    session_record.write_bytes(original)

    start_ready(runtime)
    assert runtime.pending is None
    assert runtime.trigger() is True
    wait_worker(runtime)

    assert runtime.state == "READY"
    assert runtime.pending is None
    assert runtime.pending_path.read_bytes() == original
    assert session_record.read_bytes() == original
    assert env.printer.mock_calls == []
    assert runtime.close() is True

    if not corrupt:
        restored = env.build()
        assert restored.start() is True
        wait_worker(restored)
        assert restored.state == "PRINT_ERROR"
        assert restored.pending["job_id"] == 17
        assert restored.trigger() is False
        env.printer.print_image.assert_not_called()


@pytest.mark.parametrize("state", ["AWAITING_PRINT_CONFIRM", "PRINT_ERROR"])
def test_disabled_print_confirmation_and_retry_cannot_touch_old_jobs(runtime_env, state):
    env = runtime_env
    runtime = disabled_runtime(env)
    runtime.state = state
    runtime.pending = {"status": "spooled", "job_id": 17, "needs_confirmation": True}
    original_pending = runtime.pending.copy()

    assert runtime.confirm_printed() is False
    assert runtime.retry_print_status() is False

    assert runtime.state == state
    assert runtime.pending == original_pending
    assert runtime._worker is None
    assert env.printer.mock_calls == []
    assert not runtime.pending_path.exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_gui_config_accepts_boolean_printer_enabled(tmp_path, enabled):
    path = tmp_path / "gui.json"
    path.write_text(json.dumps({"printer": {"enabled": enabled}}), encoding="utf-8")

    config = gui_runtime.load_gui_config(tmp_path, path)

    assert config["printer"]["enabled"] is enabled


@pytest.mark.parametrize("enabled", [None, 0, 1, "false", "true", [], {}])
def test_gui_config_rejects_non_boolean_printer_enabled(tmp_path, enabled):
    path = tmp_path / "gui.json"
    path.write_text(json.dumps({"printer": {"enabled": enabled}}), encoding="utf-8")

    with pytest.raises(ConfigError):
        gui_runtime.load_gui_config(tmp_path, path)


def test_omitted_printer_enabled_keeps_existing_print_workflow(runtime_env):
    assert runtime_env.runtime.print_enabled is True
