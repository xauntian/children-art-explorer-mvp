"""采集确认与自动生成的边界测试；不用手机、UI 或真实模型。"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest

from app.artwork_extract import ArtworkExtractError, ArtworkExtractionResult
from app.errors import CaptureError, ConfigError, GenerationError, PreprocessError
from app.phone import PhoneController, load_phone_config
from app.phone_capture import PhoneCaptureResult
from tests.fixtures import make_app_config, solid_png


def _extracted_artwork() -> ArtworkExtractionResult:
    # 透明区域中的 RGB 仍有颜色，便于发现误存为不透明图片或错误合成。
    artwork = np.full((96, 96, 4), (12, 50, 200, 0), dtype=np.uint8)
    artwork[20:76, 20:76, 3] = 255
    model_input = np.full((96, 96, 3), 255, dtype=np.uint8)
    model_input[20:76, 20:76] = artwork[20:76, 20:76, :3]
    board = np.full((128, 128, 3), 150, dtype=np.uint8)
    return ArtworkExtractionResult(
        board_image=board,
        artwork=artwork,
        rgba=artwork[:, :, [2, 1, 0, 3]],
        model_input=model_input,
        overlay=board.copy(),
        mask=artwork[:, :, 3].copy(),
        board_mask=np.full((128, 128), 255, dtype=np.uint8),
        board_corners=[[0, 0], [127, 0], [127, 127], [0, 127]],
        artwork_corners=None,
        meta={"fixture": "transparent artwork"},
    )


class FakeCamera:
    def __init__(self):
        raw = solid_png(144, 96)
        preview = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        self.result = PhoneCaptureResult(raw, "adb_screencap", preview, {"serial": "SIMULATED"})
        self.calls = 0
        self.error = None
        self.on_capture = None

    def capture(self):
        self.calls += 1
        if self.on_capture:
            self.on_capture()
        if self.error:
            raise self.error
        return self.result


class FakeExtractor:
    def __init__(self):
        self.result = _extracted_artwork()
        self.calls = []
        self.error = None

    def __call__(self, image, *, profile):
        self.calls.append({"image": image, "profile": profile})
        if self.error:
            raise self.error
        return self.result


class FakePipeline:
    def __init__(self, root: Path):
        self.root = root
        self.calls = []
        self.state = "COMPLETED"
        self.error = None
        self.on_run = None

    def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.on_run:
            self.on_run()
        if self.error:
            raise self.error
        directory = self.root / "runs" / kwargs["job_id"]
        directory.mkdir(parents=True, exist_ok=True)
        if self.state == "COMPLETED":
            (directory / "output.png").write_bytes(solid_png())
            return {
                "job_id": kwargs["job_id"],
                "state": self.state,
                "style_id": "cubism",
                "output_path": str(directory / "output.png"),
            }
        # 合法质量检查不通过：保留候选，不产生已确认输出。
        (directory / "attempt_2.png").write_bytes(solid_png())
        return {
            "job_id": kwargs["job_id"],
            "state": self.state,
            "error": {"category": "quality_check_failed", "message": "检查未通过"},
        }


@pytest.fixture
def controller_env(tmp_path):
    make_app_config(tmp_path)
    config = {
        "mode": "confirm",
        "project_root": tmp_path,
        "capture_runs_dir": tmp_path / "runs" / "phone",
        "phone": {},
        "extraction": {"board_expected_ratio": 1.0},
    }
    camera = FakeCamera()
    extractor = FakeExtractor()
    pipeline = FakePipeline(tmp_path)
    factory_calls = []

    def factory():
        factory_calls.append(True)
        return SimpleNamespace(pipeline=pipeline)

    controller = PhoneController(config, camera=camera, extractor=extractor, app_factory=factory)
    return SimpleNamespace(
        controller=controller,
        config=config,
        camera=camera,
        extractor=extractor,
        pipeline=pipeline,
        factory_calls=factory_calls,
    )


def _saved_meta(session):
    return json.loads((session.directory / "meta.json").read_text(encoding="utf-8"))


def test_confirm_prepare_stops_before_model_and_preserves_distinct_image_assets(controller_env):
    env = controller_env
    session = env.controller.prepare("confirm")

    assert session.state == "AWAITING_INPUT_CONFIRM"
    assert env.camera.calls == len(env.extractor.calls) == 1
    assert env.factory_calls == env.pipeline.calls == []
    assert session.meta["generation_job_id"] is None
    assert "generation" not in session.meta
    assert not (session.directory / "output.png").exists()

    raw = session.directory / "original.png"
    assert raw.read_bytes() == env.camera.result.raw_bytes
    assert not (session.directory / "original.jpg").exists()
    assert session.meta["capture"]["raw_file"] == "original.png"
    artwork = cv2.imread(str(session.directory / "artwork.png"), cv2.IMREAD_UNCHANGED)
    model_input = cv2.imread(str(session.directory / "input.png"), cv2.IMREAD_UNCHANGED)
    assert artwork.shape[2] == 4
    assert artwork[0, 0, 3] == 0
    assert artwork[48, 48, 3] == 255
    assert model_input.shape[2] == 3
    assert model_input[0, 0].tolist() == [255, 255, 255]
    assert model_input[48, 48].tolist() == [12, 50, 200]
    assert _saved_meta(session)["state"] == "AWAITING_INPUT_CONFIRM"


def test_direct_input_is_automatically_confirmed_on_generation(controller_env):
    env = controller_env
    session = env.controller.prepare("direct")
    assert session.state == "INPUT_READY"
    result = env.controller.generate(session)
    assert result["state"] == "COMPLETED"
    assert result["input_confirmation"] == "automatic"
    assert len(env.factory_calls) == len(env.pipeline.calls) == 1
    assert Path(result["output_path"]).is_file()


def test_run_auto_submits_exactly_the_saved_input_without_manual_confirmation(controller_env):
    env = controller_env
    result = env.controller.run_auto()
    session = env.controller.session
    assert result["state"] == "COMPLETED"
    assert result["mode"] == "auto"
    assert result["input_confirmation"] == "automatic"
    assert env.camera.calls == len(env.extractor.calls) == len(env.pipeline.calls) == 1
    assert len(env.factory_calls) == 1
    submission = env.pipeline.calls[0]
    assert submission["source_bytes"] == (session.directory / "input.png").read_bytes()
    assert submission["job_id"] == result["generation_job_id"]
    # 画派随机分配属于既有流水线；前置确认不得提前强制指定画派。
    assert "style_id" not in submission
    assert _saved_meta(session)["state"] == "COMPLETED"


def test_manual_confirmation_is_recorded_only_when_user_submits(controller_env):
    env = controller_env
    session = env.controller.prepare("confirm")
    assert "input_confirmed_at" not in session.meta
    result = env.controller.generate(session)
    assert result["input_confirmation"] == "manual"
    assert result["input_confirmed_at"]
    assert len(env.pipeline.calls) == 1


def test_retake_supersedes_old_session_without_invalidating_new_one(controller_env):
    env = controller_env
    old = env.controller.prepare("confirm")
    current = env.controller.prepare("confirm")
    assert old.state == "SUPERSEDED"
    assert current.state == "AWAITING_INPUT_CONFIRM"
    assert old.directory != current.directory
    with pytest.raises(ConfigError, match="失效|重复确认"):
        env.controller.generate(old)
    assert old.state == _saved_meta(old)["state"] == "SUPERSEDED"
    assert current.state == "AWAITING_INPUT_CONFIRM"
    assert env.factory_calls == env.pipeline.calls == []
    env.controller.generate(current)
    assert len(env.pipeline.calls) == 1


def test_duplicate_confirmation_never_resubmits_or_changes_completed_to_failed(controller_env):
    env = controller_env
    session = env.controller.prepare("confirm")
    env.controller.generate(session)
    saved = _saved_meta(session)
    with pytest.raises(ConfigError, match="失效|重复确认"):
        env.controller.generate(session)
    assert len(env.factory_calls) == len(env.pipeline.calls) == 1
    assert session.state == "COMPLETED"
    assert _saved_meta(session) == saved
    assert all(event["state"] != "FAILED" for event in session.meta["history"])


def test_cancelled_session_cannot_submit_but_can_be_replaced(controller_env):
    env = controller_env
    session = env.controller.prepare("confirm")
    env.controller.cancel()
    with pytest.raises(ConfigError, match="失效|重复确认"):
        env.controller.generate(session)
    assert session.state == _saved_meta(session)["state"] == "CANCELLED"
    assert env.factory_calls == env.pipeline.calls == []
    replacement = env.controller.prepare("confirm")
    assert replacement is not session
    assert replacement.state == "AWAITING_INPUT_CONFIRM"


def test_changed_input_hash_is_rejected_before_model_construction(controller_env):
    env = controller_env
    session = env.controller.prepare("confirm")
    (session.directory / "input.png").write_bytes(solid_png(value=90))
    with pytest.raises(ConfigError, match="发生变化"):
        env.controller.generate(session)
    assert session.state == "FAILED"
    assert env.factory_calls == env.pipeline.calls == []
    assert "output_path" not in session.meta
    assert not (session.directory / "output.png").exists()


@pytest.mark.parametrize("stage", ["capture", "extraction"])
def test_capture_or_extraction_failure_never_enters_generation(controller_env, stage):
    env = controller_env
    if stage == "capture":
        env.camera.error = CaptureError("模拟无设备")
    else:
        env.extractor.error = PreprocessError("模拟无法定位作品")
    with pytest.raises((CaptureError, PreprocessError)):
        env.controller.run_auto()
    session = env.controller.session
    assert session.state == _saved_meta(session)["state"] == "FAILED"
    assert env.factory_calls == env.pipeline.calls == []
    assert "output_path" not in session.meta
    assert not (session.directory / "input.png").exists()
    assert not (session.directory / "output.png").exists()
    if stage == "capture":
        assert env.extractor.calls == []
    else:
        # 识别失败仍保留真实采集的原始 PNG，便于重新校准。
        assert (session.directory / "original.png").read_bytes() == env.camera.result.raw_bytes


@pytest.mark.parametrize("failure", ["technical", "quality"])
def test_generation_failure_keeps_input_without_publishing_final_output(controller_env, failure):
    env = controller_env
    session = env.controller.prepare("confirm")
    if failure == "technical":
        env.pipeline.error = GenerationError("模拟本地模型不可用")
        with pytest.raises(GenerationError):
            env.controller.generate(session)
    else:
        env.pipeline.state = "FAILED"
        env.controller.generate(session)
    persisted = _saved_meta(session)
    assert session.state == persisted["state"] == "FAILED"
    assert "output_path" not in persisted
    assert (session.directory / "original.png").is_file()
    assert (session.directory / "artwork.png").is_file()
    assert (session.directory / "input.png").is_file()
    assert not (session.directory / "output.png").exists()
    assert not (
        env.config["project_root"] / "runs" / session.meta["generation_job_id"] / "output.png"
    ).exists()
    assert len(env.factory_calls) == len(env.pipeline.calls) == 1


def test_busy_capture_and_generation_reject_other_actions_without_corrupting_active_task(
    controller_env,
):
    env = controller_env
    rejected_states = []

    def during_active_operation():
        session = env.controller.session
        state = session.state
        for action in (
            lambda: env.controller.prepare("confirm"),
            lambda: env.controller.generate(session),
            env.controller.cancel,
        ):
            with pytest.raises(ConfigError, match="处理|等待"):
                action()
        assert session.state == state
        rejected_states.append(state)

    env.camera.on_capture = during_active_operation
    env.pipeline.on_run = during_active_operation
    result = env.controller.run_auto()
    assert result["state"] == "COMPLETED"
    assert rejected_states == ["CAPTURING", "GENERATING"]
    assert env.camera.calls == len(env.pipeline.calls) == 1
    # 忙碌标志在流程结束后释放，可以处理下一件作品。
    env.camera.on_capture = None
    assert env.controller.prepare("confirm").state == "AWAITING_INPUT_CONFIRM"


def test_default_app_receives_artwork_config_to_prevent_second_border_crop(
    controller_env, monkeypatch
):
    env = controller_env
    make_app_config(
        env.config["project_root"],
        preprocess={"frame_detect": True, "border_trim_ratio": 0.06, "crop_ratios": [0.1] * 4},
    )
    created_configs = []

    def fake_app(*, config):
        created_configs.append(config)
        return SimpleNamespace(pipeline=env.pipeline)

    monkeypatch.setattr("app.main.App", fake_app)
    env.controller.app_factory = None
    env.controller.run_auto()
    assert len(created_configs) == 1
    adapted = created_configs[0]
    assert adapted.pre_frame_detect is False
    assert adapted.pre_border_trim_ratio == 0
    assert all(value == 0 for value in adapted.pre_crop_ratios)


def _write_config(tmp_path, payload, *, encoding="utf-8"):
    path = tmp_path / "phone.json"
    path.write_text(json.dumps(payload), encoding=encoding)
    return path


def test_phone_config_loads_bom_and_resolves_runs_directory(tmp_path):
    path = _write_config(tmp_path, {"phone": {}, "mode": "confirm"}, encoding="utf-8-sig")
    loaded = load_phone_config(path, tmp_path)
    assert loaded["project_root"] == tmp_path.resolve()
    assert loaded["capture_runs_dir"] == tmp_path.resolve() / "runs" / "phone"
    assert loaded["phone"] == {}


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"phone": None},
        {"phone": {}, "mode": "unknown"},
        {"phone": {}, "extraction": []},
        {"phone": {}, "capture_runs_dir": ""},
        {"phone": {}, "capture_runs_dir": "../outside"},
        {"phone": {}, "capture_runs_dir": "runs/../../outside"},
    ],
)
def test_invalid_phone_config_fails_before_device_access(tmp_path, payload):
    path = _write_config(tmp_path, payload)
    with pytest.raises(ConfigError):
        load_phone_config(path, tmp_path)


def test_absolute_capture_output_outside_runs_is_rejected(tmp_path):
    path = _write_config(tmp_path, {"phone": {}, "capture_runs_dir": str(tmp_path / "outside")})
    with pytest.raises(ConfigError, match="runs"):
        load_phone_config(path, tmp_path)


@pytest.mark.parametrize("exists", [True, False])
def test_missing_or_invalid_json_is_config_error(tmp_path, exists):
    path = tmp_path / "phone.json"
    if exists:
        path.write_text("{invalid", encoding="utf-8")
    with pytest.raises(ConfigError, match="无法读取"):
        load_phone_config(path, tmp_path)


def test_ui_occluded_screenshot_falls_back_once_and_preserves_both_originals(controller_env):
    env = controller_env
    env.camera.result = replace(env.camera.result, source="preview_screenshot")
    env.config["extraction"]["preview_occlusion_regions"] = [[0.3, 0.8, 0.8, 1.0]]
    env.config["extraction"]["ignore_regions"] = [[10, 80, 60, 95]]
    screenshot = env.camera.result.raw_bytes
    photo = PhoneCaptureResult(
        solid_png(180, 120, value=91),
        "camera_photo",
        np.full((120, 180, 3), 91, np.uint8),
        {"raw_suffix": ".png", "fallback_reason": "camera_ui_overlaps_artwork"},
    )
    fallback_calls = []

    def fallback(*, reason):
        fallback_calls.append(reason)
        return photo

    env.camera.capture_photo = fallback
    extraction_calls = []

    def extract(image, *, profile):
        extraction_calls.append(profile)
        result = _extracted_artwork()
        result.meta["preview_occlusion"] = len(extraction_calls) == 1
        return result

    env.controller.extractor = extract
    session = env.controller.prepare("confirm")
    assert session.state == "AWAITING_INPUT_CONFIRM"
    assert fallback_calls == ["camera_ui_overlaps_artwork"]
    assert (session.directory / "preview-rejected-raw.png").read_bytes() == screenshot
    assert (session.directory / "original.png").read_bytes() == photo.raw_bytes
    assert (session.directory / "preview-rejected.png").is_file()
    assert (session.directory / "preview-rejected-detection.png").is_file()
    assert session.meta["capture"]["source"] == "camera_photo"
    assert session.meta["rejected_preview"]["raw_file"] == "preview-rejected-raw.png"
    assert "preview_occlusion_regions" in extraction_calls[0]
    assert "preview_occlusion_regions" not in extraction_calls[1]
    assert "ignore_regions" not in extraction_calls[1]
    assert "preview_occlusion_regions" in env.config["extraction"]
    assert env.pipeline.calls == []


def test_ui_fallback_failure_keeps_rejected_raw_and_never_generates(controller_env):
    env = controller_env
    env.camera.result = replace(env.camera.result, source="preview_screenshot")
    env.extractor.result.meta["preview_occlusion"] = True

    def fallback(*, reason):
        raise CaptureError("快门无新照片")

    env.camera.capture_photo = fallback
    with pytest.raises(CaptureError, match="快门"):
        env.controller.run_auto()
    session = env.controller.session
    assert session.state == "FAILED"
    assert (
        session.directory / "preview-rejected-raw.png"
    ).read_bytes() == env.camera.result.raw_bytes
    assert session.meta["capture"]["raw_file"] == "preview-rejected-raw.png"
    assert not (session.directory / "input.png").exists()
    assert env.factory_calls == env.pipeline.calls == []


def test_native_photo_does_not_use_screenshot_occlusion_coordinates(controller_env):
    env = controller_env
    env.camera.result = replace(env.camera.result, source="camera_photo")
    env.config["extraction"]["preview_occlusion_regions"] = [[0, 0, 1, 1]]
    env.config["extraction"]["ignore_regions"] = [[0, 0, 10, 10]]
    env.controller.prepare("direct")
    assert "preview_occlusion_regions" not in env.extractor.calls[0]["profile"]
    assert "ignore_regions" not in env.extractor.calls[0]["profile"]
    assert len(env.extractor.calls) == 1


def test_stream_uses_full_camera_frame_and_actual_zoom_without_screenshot_masks(controller_env):
    env = controller_env
    env.camera.result = replace(
        env.camera.result,
        source="camera_stream",
        meta={"camera_zoom": 1.5, "raw_suffix": ".png", "frame_age_ms": 12.3},
    )
    env.config["phone"]["preview_roi"] = [0.4, 0.4, 0.6, 0.6]
    env.config["phone"]["preview_screen_size"] = [1344, 2772]
    env.config["extraction"].update(
        preview_occlusion_regions=[[0, 0, 1, 1]], ignore_regions=[[0, 0, 10, 10]]
    )
    session = env.controller.prepare("confirm")
    assert session.state == "AWAITING_INPUT_CONFIRM"
    assert env.extractor.calls[0]["image"] is env.camera.result.preview_image
    assert env.extractor.calls[0]["image"].shape == (96, 144, 3)
    assert "preview_occlusion_regions" not in env.extractor.calls[0]["profile"]
    assert "ignore_regions" not in env.extractor.calls[0]["profile"]
    assert session.meta["camera_zoom"] == 1.5
    assert session.meta["camera_zoom_source"] == "camera2_stream_setting"
    assert session.meta["capture"]["source"] == "camera_stream"
    assert session.meta["capture"]["frame_age_ms"] == 12.3
    assert (session.directory / "original.png").read_bytes() == env.camera.result.raw_bytes
    assert "rejected_preview" not in session.meta
    assert env.factory_calls == env.pipeline.calls == []


def test_stream_extraction_failure_cannot_fall_back_to_native_photo_or_dcim(controller_env):
    env = controller_env
    env.config["phone"]["fallback_photo"] = True
    env.camera.result = replace(
        env.camera.result, source="camera_stream", meta={"camera_zoom": 1.5}
    )
    env.extractor.error = ArtworkExtractError("stream artwork edge is missing")

    def forbidden_photo(**kwargs):
        pytest.fail("camera stream must never trigger a native shutter or search DCIM")

    env.camera.capture_photo = forbidden_photo
    with pytest.raises(ArtworkExtractError, match="edge is missing"):
        env.controller.prepare("direct")
    assert env.camera.calls == len(env.extractor.calls) == 1
    assert env.controller.session.state == "FAILED"
    assert "rejected_preview" not in env.controller.session.meta
    assert "photo_fallback_sec" not in env.controller.session.meta["timings"]
    assert env.factory_calls == env.pipeline.calls == []


def test_stream_backend_constructs_stream_camera_with_configured_options(
    controller_env, monkeypatch
):
    env = controller_env
    payload = {
        "phone": {"serial": "SIMULATED"},
        "capture_backend": "camera_stream",
        "stream": {"zoom": 1.5, "size": [2048, 1536]},
    }
    config = load_phone_config(_write_config(env.config["project_root"], payload), env.controller.root)
    calls = []

    def factory(root, **kwargs):
        calls.append((root, kwargs))
        return env.camera

    module = ModuleType("app.phone_stream")
    module.PhoneStreamCapture = factory
    monkeypatch.setitem(sys.modules, "app.phone_stream", module)
    controller = PhoneController(config, extractor=env.extractor, app_factory=env.controller.app_factory)
    assert controller.camera is env.camera
    assert calls == [(env.controller.root, {"stream": payload["stream"], **payload["phone"]})]


def test_unreliable_preview_extraction_uses_one_new_photo_and_keeps_error_evidence(controller_env):
    env = controller_env
    env.camera.result = replace(env.camera.result, source="preview_screenshot")
    screenshot = env.camera.result.raw_bytes
    calls = []
    photo = replace(env.camera.result, source="camera_photo", raw_bytes=solid_png(value=90))

    def capture_photo(*, reason):
        calls.append(reason)
        return photo

    env.camera.capture_photo = capture_photo
    extraction_calls = []

    def extract(image, *, profile):
        extraction_calls.append(profile)
        if len(extraction_calls) == 1:
            raise ArtworkExtractError("预览棋盘周期不可靠")
        return _extracted_artwork()

    env.controller.extractor = extract
    result = env.controller.run_auto()
    session = env.controller.session
    assert result["state"] == "COMPLETED"
    assert calls == ["preview_extraction_unreliable"]
    assert (session.directory / "preview-rejected-raw.png").read_bytes() == screenshot
    assert (session.directory / "original.png").read_bytes() == photo.raw_bytes
    assert "周期不可靠" in result["preview_extraction_error"]["message"]
    assert not (session.directory / "preview-rejected-detection.png").exists()
    assert len(extraction_calls) == 2
    assert len(env.pipeline.calls) == 1


def test_native_extraction_failure_after_preview_fallback_cannot_take_another_photo(controller_env):
    env = controller_env
    env.camera.result = replace(env.camera.result, source="preview_screenshot")
    env.extractor.error = ArtworkExtractError("棋盘证据不足")
    calls = []

    def capture_photo(*, reason):
        calls.append(reason)
        return replace(env.camera.result, source="camera_photo")

    env.camera.capture_photo = capture_photo
    with pytest.raises(ArtworkExtractError, match="证据不足"):
        env.controller.run_auto()
    assert calls == ["preview_extraction_unreliable"]
    assert len(env.extractor.calls) == 2
    assert env.controller.session.state == "FAILED"
    assert env.factory_calls == env.pipeline.calls == []


def test_disabled_photo_fallback_preserves_preview_extraction_failure(controller_env):
    env = controller_env
    env.camera.result = replace(env.camera.result, source="preview_screenshot")
    env.config["phone"]["fallback_photo"] = False
    env.extractor.error = ArtworkExtractError("预览周期不可靠")

    def forbidden_photo(**kwargs):
        pytest.fail("disabled fallback must not trigger a shutter")

    env.camera.capture_photo = forbidden_photo
    with pytest.raises(ArtworkExtractError, match="周期不可靠"):
        env.controller.run_auto()
    assert env.controller.session.state == "FAILED"
    assert len(env.extractor.calls) == 1
    assert env.factory_calls == env.pipeline.calls == []


def _aruco_config(env):
    return {
        **env.config,
        "extraction": {
            "method": "aruco",
            "board_spec_path": "assets/capture-board-fast.json",
            "rotation_degrees": 270,
            "max_detection_side": 1400,
            "max_board_side": 1024,
        },
    }


def _paper_config(env):
    return {
        **env.config,
        "extraction": {"method": "paper", "rotation_degrees": 270},
    }


def _patch_paper_extractor(monkeypatch, extractor):
    # 测试的是 controller 默认 import/dispatch；CV 实现与依赖由其独立测试负责。
    module = ModuleType("app.paper_extract")
    module.extract_paper_artwork = extractor
    monkeypatch.setitem(sys.modules, "app.paper_extract", module)


def test_paper_config_loader_and_default_dispatch_keep_detector_profile_clean(
    controller_env, monkeypatch
):
    env = controller_env
    payload = {"phone": {}, "extraction": {"method": "paper", "rotation_degrees": 270}}
    config = load_phone_config(
        _write_config(env.config["project_root"], payload), env.config["project_root"]
    )
    calls = []

    def paper(image, *, profile):
        calls.append(profile)
        assert profile == {"rotation_degrees": 270}
        return _extracted_artwork()

    def wrong_route(*args, **kwargs):
        pytest.fail("paper mode must not dispatch to either board extractor")

    _patch_paper_extractor(monkeypatch, paper)
    monkeypatch.setattr("app.fast_artwork_extract.extract_fast_artwork", wrong_route)
    monkeypatch.setattr("app.artwork_extract.extract_artwork", wrong_route)
    controller = PhoneController(config, camera=env.camera, app_factory=env.controller.app_factory)
    session = controller.prepare("confirm")
    assert session.state == "AWAITING_INPUT_CONFIRM"
    assert session.meta["extraction_method"] == "paper"
    assert len(calls) == 1
    assert config["extraction"] == payload["extraction"]
    assert env.factory_calls == env.pipeline.calls == []


def test_aruco_default_extractor_dispatch_removes_method_and_resolves_board_spec(
    controller_env, monkeypatch
):
    env = controller_env
    config = _aruco_config(env)
    calls = []

    def fast(image, *, profile):
        calls.append(profile)
        assert "method" not in profile
        assert Path(profile["board_spec_path"]).is_absolute()
        assert (
            Path(profile["board_spec_path"])
            == env.config["project_root"] / "assets" / "capture-board-fast.json"
        )
        return _extracted_artwork()

    def wrong_route(*args, **kwargs):
        pytest.fail("ArUco mode must not run the periodic-checkerboard extractor")

    monkeypatch.setattr("app.fast_artwork_extract.extract_fast_artwork", fast)
    monkeypatch.setattr("app.artwork_extract.extract_artwork", wrong_route)
    controller = PhoneController(config, camera=env.camera, app_factory=env.controller.app_factory)
    session = controller.prepare("confirm")
    assert session.state == "AWAITING_INPUT_CONFIRM"
    assert session.meta["extraction_method"] == "aruco"
    assert len(calls) == 1
    assert calls[0]["rotation_degrees"] == 270
    assert config["extraction"]["method"] == "aruco"
    assert config["extraction"]["board_spec_path"] == "assets/capture-board-fast.json"
    assert env.factory_calls == env.pipeline.calls == []


@pytest.mark.parametrize("method", [None, "checkerboard"])
def test_default_and_explicit_legacy_configs_keep_the_legacy_extractor(
    controller_env, monkeypatch, method
):
    env = controller_env
    config = {**env.config, "extraction": {"board_expected_ratio": 1.0}}
    if method is not None:
        config["extraction"]["method"] = method
    calls = []

    def legacy(image, *, profile):
        calls.append(profile)
        assert "method" not in profile
        assert profile["board_expected_ratio"] == 1.0
        return _extracted_artwork()

    monkeypatch.setattr("app.artwork_extract.extract_artwork", legacy)
    monkeypatch.setattr(
        "app.fast_artwork_extract.extract_fast_artwork",
        lambda *args, **kwargs: pytest.fail("legacy configuration must not call fast extraction"),
    )
    controller = PhoneController(config, camera=env.camera, app_factory=env.controller.app_factory)
    controller.prepare("confirm")
    assert controller.session.meta["extraction_method"] == "checkerboard"
    assert len(calls) == 1
    assert env.factory_calls == env.pipeline.calls == []


@pytest.mark.parametrize("source", ["preview_screenshot", "adb_screencap", "camera_photo"])
@pytest.mark.parametrize("method", ["aruco", "paper"])
def test_quad_location_failure_never_takes_another_photo_or_generates(
    controller_env, monkeypatch, source, method
):
    env = controller_env
    env.camera.result = replace(env.camera.result, source=source)
    extraction_calls = []

    def fast(image, *, profile):
        extraction_calls.append(profile)
        raise ArtworkExtractError("Full input quadrilateral is not visible")

    def forbidden_photo(**kwargs):
        pytest.fail("Missing input geometry must fail immediately without taking a photo")

    if method == "paper":
        _patch_paper_extractor(monkeypatch, fast)
    else:
        monkeypatch.setattr("app.fast_artwork_extract.extract_fast_artwork", fast)
    env.camera.capture_photo = forbidden_photo
    controller = PhoneController(
        _paper_config(env) if method == "paper" else _aruco_config(env),
        camera=env.camera,
        app_factory=env.controller.app_factory,
    )
    with pytest.raises(ArtworkExtractError, match="quadrilateral"):
        controller.run_auto()
    assert controller.session.state == "FAILED"
    assert controller.session.meta["extraction_method"] == method
    assert (
        controller.session.directory / "original.png"
    ).read_bytes() == env.camera.result.raw_bytes
    assert not (controller.session.directory / "input.png").exists()
    assert env.camera.calls == len(extraction_calls) == 1
    assert env.factory_calls == env.pipeline.calls == []


@pytest.mark.parametrize("method", ["unknown", "ARUCO", None])
def test_unknown_extraction_method_is_rejected_by_loader_and_controller_before_capture(
    controller_env, method
):
    env = controller_env
    config = _aruco_config(env)
    config["extraction"]["method"] = method
    path = _write_config(
        env.config["project_root"], {"phone": {}, "extraction": {"method": method}}
    )
    with pytest.raises(ConfigError, match="extraction.method"):
        load_phone_config(path, env.config["project_root"])
    with pytest.raises(ConfigError, match="extraction.method"):
        PhoneController(config, camera=env.camera, extractor=env.extractor)
    assert env.camera.calls == 0
    assert env.extractor.calls == env.factory_calls == env.pipeline.calls == []


@pytest.mark.parametrize("native_result", ["success", "extraction_failure", "still_occluded"])
@pytest.mark.parametrize("method", ["aruco", "paper"])
def test_quad_ui_overlap_allows_only_one_photo_and_never_reuses_preview_occlusion_boxes(
    controller_env, monkeypatch, native_result, method
):
    env = controller_env
    config = _paper_config(env) if method == "paper" else _aruco_config(env)
    config["extraction"]["preview_occlusion_regions"] = [[0.3, 0.8, 0.8, 1.0]]
    env.camera.result = replace(env.camera.result, source="preview_screenshot")
    screenshot = env.camera.result.raw_bytes
    photo = replace(env.camera.result, source="camera_photo", raw_bytes=solid_png(value=90))
    photos = []
    profiles = []

    def capture_photo(*, reason):
        photos.append(reason)
        return photo

    def fast(image, *, profile):
        profiles.append(profile)
        assert "method" not in profile
        if method == "aruco":
            assert Path(profile["board_spec_path"]).is_absolute()
        result = _extracted_artwork()
        if len(profiles) == 1:
            result.meta["preview_occlusion"] = True
        elif native_result == "extraction_failure":
            raise ArtworkExtractError("Native photo still lacks a marker")
        elif native_result == "still_occluded":
            result.meta["preview_occlusion"] = True
        return result

    env.camera.capture_photo = capture_photo
    if method == "paper":
        _patch_paper_extractor(monkeypatch, fast)
    else:
        monkeypatch.setattr("app.fast_artwork_extract.extract_fast_artwork", fast)
    controller = PhoneController(config, camera=env.camera, app_factory=env.controller.app_factory)
    if native_result == "success":
        assert controller.run_auto()["state"] == "COMPLETED"
        assert len(env.pipeline.calls) == 1
    else:
        with pytest.raises((ArtworkExtractError, ConfigError)):
            controller.run_auto()
        assert controller.session.state == "FAILED"
        assert env.factory_calls == env.pipeline.calls == []
        assert not (controller.session.directory / "input.png").exists()
    session = controller.session
    assert photos == ["camera_ui_overlaps_artwork"]
    assert len(profiles) == 2
    assert "preview_occlusion_regions" in profiles[0]
    assert "preview_occlusion_regions" not in profiles[1]
    assert config["extraction"]["preview_occlusion_regions"] == [[0.3, 0.8, 0.8, 1.0]]
    assert (session.directory / "preview-rejected-raw.png").read_bytes() == screenshot
    assert (session.directory / "original.png").read_bytes() == photo.raw_bytes
