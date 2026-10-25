"""CLI、入口锁和恢复集成；采集与模型适配器均为注入的模拟对象。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import phone, states
from app.artwork_extract import ArtworkExtractError
from app.errors import CaptureError, ConfigError
from app.phone_runtime import PhoneRunLock
from app.storage import atomic_write_json
from tests.unit.test_phone_controller import _patch_paper_extractor
from tests.unit.test_phone_controller import controller_env as controller_env


def _calibrate(env):
    env.config["phone"].update(preview_roi=[0.0, 0.0, 1.0, 1.0], preview_screen_size=[144, 96])


def _terminal(monkeypatch):
    monkeypatch.setattr(phone.sys, "stdin", SimpleNamespace(isatty=lambda: True))


def _forbid_window(*args, **kwargs):
    raise AssertionError("CLI must not construct a desktop window")


def _main_with_controller(monkeypatch, env):
    monkeypatch.setattr(phone, "project_root_from", lambda: env.config["project_root"])
    monkeypatch.setattr(phone, "load_phone_config", lambda *args: env.config)
    monkeypatch.setattr(phone, "PhoneController", lambda config: env.controller)
    monkeypatch.setattr(phone, "PhoneWindow", _forbid_window)


@pytest.mark.parametrize("missing", ["preview_roi", "preview_screen_size"])
def test_uncalibrated_entry_stops_before_lock_capture_and_model(controller_env, missing):
    env = controller_env
    _calibrate(env)
    env.config["phone"][missing] = None
    with pytest.raises(ConfigError, match="preview_roi|preview_screen_size"):
        phone.run_phone(env.controller, "auto")
    assert env.camera.calls == 0
    assert env.extractor.calls == env.pipeline.calls == []
    assert env.controller.session is None
    assert not (env.config["capture_runs_dir"] / ".phone-run.lock").exists()


def test_probe_is_allowed_without_calibration_and_never_runs_capture_pipeline(
    controller_env, monkeypatch, capsys
):
    env = controller_env
    env.config["generation_warmup"] = True
    _main_with_controller(monkeypatch, env)
    probes = []

    def probe():
        probes.append(True)
        return {"available": True, "source": "preview_screenshot"}

    monkeypatch.setattr(env.camera, "probe", probe, raising=False)
    monkeypatch.setattr(phone, "run_phone", _forbid_window)
    assert phone.main(["--probe"]) == 0
    assert probes == [True]
    assert '"available": true' in capsys.readouterr().out
    assert env.camera.calls == 0
    assert env.factory_calls == env.pipeline.calls == []


def test_existing_os_lock_rejects_entry_before_capture(controller_env):
    env = controller_env
    _calibrate(env)
    with PhoneRunLock(env.config["capture_runs_dir"]):
        with pytest.raises(ConfigError, match="实例|锁"):
            phone.run_phone(env.controller, "auto")
    assert env.camera.calls == 0
    assert env.factory_calls == env.pipeline.calls == []
    assert env.controller.session is None


def test_enabled_startup_warmup_holds_phone_lock_and_precedes_capture(
    controller_env, monkeypatch, capsys
):
    from app import model_warmup

    env = controller_env
    _calibrate(env)
    env.config["generation_warmup"] = True
    calls = []

    def warmup(root):
        assert root == env.config["project_root"]
        assert env.camera.calls == 0
        assert env.controller.session is None
        with pytest.raises(ConfigError, match="实例|锁"):
            with PhoneRunLock(env.config["capture_runs_dir"]):
                pass
        calls.append(True)
        return {"total_sec": 2.345}

    monkeypatch.setattr(model_warmup, "warmup_generation", warmup)
    assert phone.run_phone(env.controller, "auto") == 0
    assert calls == [True]
    assert env.camera.calls == 1
    assert "模型预热完成（2.345 秒）" in capsys.readouterr().out


def test_failed_startup_warmup_prevents_capture_and_releases_lock(controller_env, monkeypatch):
    from app import model_warmup

    env = controller_env
    _calibrate(env)
    env.config["generation_warmup"] = True

    def warmup(root):
        raise ConfigError("模型预热失败")

    monkeypatch.setattr(model_warmup, "warmup_generation", warmup)
    with pytest.raises(ConfigError, match="模型预热失败"):
        phone.run_phone(env.controller, "auto")
    assert env.camera.calls == 0
    assert env.factory_calls == env.pipeline.calls == []
    with PhoneRunLock(env.config["capture_runs_dir"]):
        pass


def test_active_job_blocks_startup_warmup(controller_env, monkeypatch):
    from app import model_warmup

    env = controller_env
    _calibrate(env)
    env.config["generation_warmup"] = True
    env.controller.generation_store.create(job_id="active-before-warmup")
    monkeypatch.setattr(model_warmup, "warmup_generation", _forbid_window)
    with pytest.raises(ConfigError, match="已有生成任务"):
        phone.run_phone(env.controller, "auto")
    assert env.camera.calls == 0


@pytest.mark.parametrize("mode", ["auto", "confirm"])
def test_stream_starts_after_warmup_before_auto_or_console_and_closes(
    controller_env, monkeypatch, mode
):
    from app import model_warmup

    env = controller_env
    env.config.update(capture_backend="camera_stream", stream={"zoom": 1.5}, generation_warmup=True)
    events = []

    def warmup(root):
        events.append("warmup")
        return {"total_sec": 0.1}

    def start():
        assert events == ["warmup"]
        assert env.camera.calls == 0
        assert env.controller.session is None
        events.append("start")

    def run_console():
        assert events == ["warmup", "start"]
        events.append("console")
        return 0

    monkeypatch.setattr(model_warmup, "warmup_generation", warmup)
    monkeypatch.setattr(env.camera, "start", start, raising=False)
    monkeypatch.setattr(env.camera, "close", lambda: events.append("close"), raising=False)
    env.camera.on_capture = lambda: events.append("capture")
    monkeypatch.setattr(phone, "PhoneConsole", lambda *args: SimpleNamespace(run=run_console))
    assert phone.run_phone(env.controller, mode) == 0
    assert events == ["warmup", "start", "capture" if mode == "auto" else "console", "close"]


@pytest.mark.parametrize("failure_stage", ["start", "capture", "console"])
def test_stream_closes_on_start_capture_or_console_error_and_releases_lock(
    controller_env, monkeypatch, failure_stage
):
    env = controller_env
    env.config.update(capture_backend="camera_stream", stream={"zoom": 1.5})
    events = []

    def start():
        events.append("start")
        if failure_stage == "start":
            raise CaptureError("stream startup failed")

    def run_console():
        events.append("console")
        raise CaptureError("stream console failed")

    monkeypatch.setattr(env.camera, "start", start, raising=False)
    monkeypatch.setattr(env.camera, "close", lambda: events.append("close"), raising=False)
    monkeypatch.setattr(phone, "PhoneConsole", lambda *args: SimpleNamespace(run=run_console))
    if failure_stage == "capture":
        env.camera.error = CaptureError("stream capture failed")
    mode = "confirm" if failure_stage == "console" else "auto"
    with pytest.raises(CaptureError, match="stream"):
        phone.run_phone(env.controller, mode)
    assert events[0] == "start"
    assert events[-1] == "close"
    assert events.count("close") == 1
    assert env.factory_calls == env.pipeline.calls == []
    with PhoneRunLock(env.config["capture_runs_dir"]):
        pass


def test_stream_probe_never_starts_camera_or_model_warmup(controller_env, monkeypatch):
    from app import model_warmup

    env = controller_env
    env.config.update(capture_backend="camera_stream", stream={"zoom": 1.5}, generation_warmup=True)
    _main_with_controller(monkeypatch, env)
    probes = []

    def probe():
        probes.append(True)
        return {"available": True, "source": "camera_stream"}

    monkeypatch.setattr(env.camera, "probe", probe, raising=False)
    monkeypatch.setattr(env.camera, "start", _forbid_window, raising=False)
    monkeypatch.setattr(model_warmup, "warmup_generation", _forbid_window)
    assert phone.main(["--probe"]) == 0
    assert probes == [True]
    assert env.camera.calls == 0
    assert env.factory_calls == env.pipeline.calls == []


def test_active_linked_generation_blocks_startup_without_resubmitting(controller_env):
    env = controller_env
    _calibrate(env)
    directory = env.config["capture_runs_dir"] / "old-session"
    atomic_write_json(
        directory / "meta.json",
        {
            "session_id": "old-session",
            "state": "GENERATING",
            "generation_job_id": "phone-old-session",
            "history": [],
        },
    )
    store = env.controller.generation_store
    job = store.create(job_id="phone-old-session")
    job.meta["state"] = states.GENERATING
    job.save()
    original_frontend = (directory / "meta.json").read_bytes()
    original_formal = (job.dir / "meta.json").read_bytes()
    with pytest.raises(ConfigError, match="旧采集任务|resume"):
        phone.run_phone(env.controller, "auto")
    assert env.camera.calls == 0
    assert env.factory_calls == env.pipeline.calls == []
    assert (directory / "meta.json").read_bytes() == original_frontend
    assert (job.dir / "meta.json").read_bytes() == original_formal
    with PhoneRunLock(env.config["capture_runs_dir"]):
        pass


def test_unrelated_active_job_blocks_before_capture_but_is_not_recovered(controller_env):
    env = controller_env
    _calibrate(env)
    job = env.controller.generation_store.create(job_id="ordinary-active-job")
    original = (job.dir / "meta.json").read_bytes()
    with pytest.raises(ConfigError, match="已有生成任务"):
        phone.run_phone(env.controller, "auto")
    assert env.camera.calls == 0
    assert env.factory_calls == env.pipeline.calls == []
    assert env.controller.session is None
    assert (job.dir / "meta.json").read_bytes() == original


def test_auto_runs_one_controller_chain_without_terminal_keys_or_window(
    controller_env, monkeypatch, capsys
):
    env = controller_env
    _calibrate(env)
    monkeypatch.setattr(phone.sys, "stdin", None)
    monkeypatch.setattr(phone, "PhoneWindow", _forbid_window)
    monkeypatch.setattr(phone, "PhoneConsole", _forbid_window)
    assert phone.run_phone(env.controller, "auto") == 0
    session = env.controller.session
    assert session.state == "COMPLETED"
    assert env.camera.calls == len(env.extractor.calls) == len(env.pipeline.calls) == 1
    assert env.pipeline.calls[0]["source_bytes"] == (session.directory / "input.png").read_bytes()
    assert env.pipeline.calls[0]["job_id"] == session.meta["generation_job_id"]
    assert '"state": "COMPLETED"' in capsys.readouterr().out


def test_startup_cancels_interrupted_input_before_accepting_one_new_auto_capture(controller_env):
    env = controller_env
    _calibrate(env)
    old = env.config["capture_runs_dir"] / "old-input"
    atomic_write_json(
        old / "meta.json",
        {
            "session_id": "old-input",
            "state": "AWAITING_INPUT_CONFIRM",
            "generation_job_id": None,
            "history": [],
        },
    )
    assert phone.run_phone(env.controller, "auto") == 0
    assert json.loads((old / "meta.json").read_text(encoding="utf-8"))["state"] == "FAILED"
    assert env.camera.calls == len(env.pipeline.calls) == 1
    assert env.controller.session.directory != old


def test_main_defaults_to_console_and_does_not_construct_window(controller_env, monkeypatch):
    env = controller_env
    _calibrate(env)
    _terminal(monkeypatch)
    _main_with_controller(monkeypatch, env)
    selected = []
    real_console = phone.PhoneConsole

    def console(controller, mode):
        selected.append(mode)
        return real_console(controller, mode, key_reader=lambda: "q")

    monkeypatch.setattr(phone, "PhoneConsole", console)
    assert phone.main([]) == 0
    assert selected == ["confirm"]
    assert env.camera.calls == 0
    assert env.pipeline.calls == []


@pytest.mark.parametrize("entry_mode", ["confirm", "direct"])
def test_t_captures_only_in_either_entry_and_enter_submits_once(
    controller_env, monkeypatch, capsys, entry_mode
):
    env = controller_env
    _terminal(monkeypatch)

    def keys():
        yield "t"
        assert env.controller.session.state == "AWAITING_INPUT_CONFIRM"
        assert env.controller.session.meta["mode"] == "confirm"
        assert env.camera.calls == 1
        assert env.pipeline.calls == env.factory_calls == []
        yield "enter"
        assert env.controller.session.state == "COMPLETED"
        assert len(env.pipeline.calls) == 1
        yield "enter"
        assert len(env.pipeline.calls) == 1
        yield "q"

    sequence = keys()
    assert (
        phone.PhoneConsole(env.controller, entry_mode, key_reader=lambda: next(sequence)).run() == 0
    )
    session = env.controller.session
    output = capsys.readouterr().out
    assert str(session.directory / "artwork.png") in output
    assert str(session.directory / "input.png") in output
    assert len(env.pipeline.calls) == 1
    assert session.meta["input_confirmation"] == "manual"
    assert session.state == "COMPLETED"


@pytest.mark.parametrize("entry_mode", ["confirm", "direct"])
def test_p_runs_the_full_pipeline_in_either_entry_without_enter(
    controller_env, monkeypatch, capsys, entry_mode
):
    env = controller_env
    _terminal(monkeypatch)

    def keys():
        yield "p"
        assert env.controller.session.state == "COMPLETED"
        assert env.controller.session.meta["mode"] == "direct"
        assert len(env.pipeline.calls) == 1
        yield "enter"
        yield "q"

    sequence = keys()
    assert (
        phone.PhoneConsole(env.controller, entry_mode, key_reader=lambda: next(sequence)).run() == 0
    )
    assert env.camera.calls == len(env.pipeline.calls) == 1
    assert env.controller.session.meta["input_confirmation"] == "automatic"
    assert env.controller.session.state == "COMPLETED"
    output = capsys.readouterr().out
    assert "正在生成并保存作品" in output
    assert "自动检查" not in output


def test_failed_console_task_does_not_claim_an_automatic_review_rejection(
    controller_env, monkeypatch, capsys
):
    env = controller_env
    _terminal(monkeypatch)
    monkeypatch.setattr(env.pipeline, "run", lambda **kwargs: {"state": "FAILED"})
    sequence = iter(["p", "q"])
    assert (
        phone.PhoneConsole(env.controller, "confirm", key_reader=lambda: next(sequence)).run() == 0
    )
    assert env.controller.session.state == "FAILED"
    assert env.controller.session.meta["error"] == {"message": "任务失败，原因已保存"}
    output = capsys.readouterr().out
    assert "任务失败，原因已保存" in output
    assert "自动检查" not in output


@pytest.mark.parametrize("entry_mode", ["confirm", "direct"])
def test_console_retake_and_escape_do_not_submit_cancelled_inputs(
    controller_env, monkeypatch, entry_mode
):
    env = controller_env
    _terminal(monkeypatch)

    def keys():
        yield "t"
        original = env.controller.session
        yield "r"
        assert original.state == "SUPERSEDED"
        assert env.controller.session is not original
        assert env.controller.session.state == "AWAITING_INPUT_CONFIRM"
        assert env.controller.session.meta["mode"] == "confirm"
        assert env.factory_calls == env.pipeline.calls == []
        yield "escape"
        assert env.controller.session.state == "CANCELLED"
        yield "enter"
        yield "q"

    sequence = keys()
    assert (
        phone.PhoneConsole(env.controller, entry_mode, key_reader=lambda: next(sequence)).run() == 0
    )
    assert env.camera.calls == 2
    assert env.factory_calls == env.pipeline.calls == []


@pytest.mark.parametrize("entry_mode", ["confirm", "direct"])
def test_p_supersedes_a_capture_waiting_for_enter_and_generates_only_the_new_input(
    controller_env, monkeypatch, entry_mode
):
    env = controller_env
    _terminal(monkeypatch)

    def keys():
        yield "t"
        pending = env.controller.session
        assert env.pipeline.calls == []
        yield "p"
        assert pending.state == "SUPERSEDED"
        assert env.controller.session.state == "COMPLETED"
        assert env.controller.session is not pending
        assert env.pipeline.calls[0]["job_id"] == env.controller.session.meta["generation_job_id"]
        assert pending.meta["generation_job_id"] is None
        yield "enter"
        yield "q"

    sequence = keys()
    assert (
        phone.PhoneConsole(env.controller, entry_mode, key_reader=lambda: next(sequence)).run() == 0
    )
    assert env.camera.calls == 2
    assert len(env.factory_calls) == len(env.pipeline.calls) == 1


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("p", ("capture", "direct")),
        ("t", ("capture", "confirm")),
        ("r", ("capture", "confirm")),
        ("return", ("confirm",)),
        ("escape", ("cancel",)),
        ("q", ("close",)),
    ],
)
def test_optional_window_shortcuts_match_console_without_opening_a_window(key, expected):
    window = phone.PhoneWindow.__new__(phone.PhoneWindow)
    window.pending_release = {key: "scheduled"}
    window.held = {key}
    calls = []
    window.capture = lambda *, mode: calls.append(("capture", mode))
    window.confirm = lambda: calls.append(("confirm",))
    window.cancel = lambda: calls.append(("cancel",))
    window.close = lambda: calls.append(("close",))
    window._release(key)
    assert calls == [expected]
    assert window.pending_release == {}
    assert window.held == set()


def test_interactive_console_requires_terminal_before_accepting_input(controller_env, monkeypatch):
    env = controller_env
    monkeypatch.setattr(phone.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(ConfigError, match="终端"):
        phone.PhoneConsole(env.controller, "confirm", key_reader=_forbid_window).run()
    assert env.camera.calls == 0


@pytest.mark.parametrize("extraction_ok", [True, False])
@pytest.mark.parametrize("method", ["aruco", "paper"])
def test_auto_cli_builds_quad_extractor_without_device_window_or_periodic_grid_fallback(
    controller_env, monkeypatch, capsys, extraction_ok, method
):
    env = controller_env
    _calibrate(env)
    env.config["extraction"] = {
        "method": method,
        "rotation_degrees": 270,
    }
    if method == "aruco":
        env.config["extraction"]["board_spec_path"] = "assets/capture-board-fast.json"
    extraction_calls = []
    created = []
    actual_controller = phone.PhoneController

    def fast(image, *, profile):
        extraction_calls.append(profile)
        assert "method" not in profile
        if method == "aruco":
            assert (
                Path(profile["board_spec_path"])
                == env.config["project_root"] / "assets" / "capture-board-fast.json"
            )
        if not extraction_ok:
            raise ArtworkExtractError("Input quadrilateral must be visible")
        return env.extractor.result

    def controller(config):
        instance = actual_controller(config)
        created.append(instance)
        return instance

    def app(*, config):
        return env.controller.app_factory()

    monkeypatch.setattr(phone, "project_root_from", lambda: env.config["project_root"])
    monkeypatch.setattr(phone, "load_phone_config", lambda *args: env.config)
    monkeypatch.setattr(phone, "PhoneController", controller)
    monkeypatch.setattr("app.phone_capture.PhoneCapture", lambda *args, **kwargs: env.camera)
    if method == "paper":
        _patch_paper_extractor(monkeypatch, fast)
    else:
        monkeypatch.setattr("app.fast_artwork_extract.extract_fast_artwork", fast)
    monkeypatch.setattr("app.artwork_extract.extract_artwork", _forbid_window)
    monkeypatch.setattr("app.main.App", app)
    monkeypatch.setattr(env.camera, "capture_photo", _forbid_window, raising=False)
    monkeypatch.setattr(phone, "PhoneWindow", _forbid_window)
    monkeypatch.setattr(phone, "PhoneConsole", _forbid_window)
    monkeypatch.setattr(phone.sys, "stdin", None)

    assert phone.main(["--mode", "auto", "--headless"]) == (0 if extraction_ok else 1)
    assert len(created) == env.camera.calls == len(extraction_calls) == 1
    session = created[0].session
    assert session.meta["extraction_method"] == method
    if extraction_ok:
        assert session.state == "COMPLETED"
        assert len(env.factory_calls) == len(env.pipeline.calls) == 1
        assert (
            env.pipeline.calls[0]["source_bytes"] == (session.directory / "input.png").read_bytes()
        )
    else:
        assert session.state == "FAILED"
        assert env.factory_calls == env.pipeline.calls == []
        assert not (session.directory / "input.png").exists()
        assert "Input quadrilateral" in capsys.readouterr().err
