"""Calibration records must distinguish independent axes, errors and artifacts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.checker import COMPARISON_VERSION, CheckResult
from app.config import load_app_config
from app.errors import CheckTechnicalError
from app.preprocess import prepare_model_input
from tools import calibrate_checker as tool


def preprocessed(name):
    config = load_app_config(tool.ROOT)
    return prepare_model_input(
        tool.ROOT / tool.FIXTURES / "originals" / name, max_input_side=config.pre_max_input_side,
        frame_detect=config.pre_frame_detect, border_trim_ratio=config.pre_border_trim_ratio,
        crop_ratios=config.pre_crop_ratios, min_area_ratio=config.pre_min_area_ratio,
        max_area_ratio=config.pre_max_area_ratio, approx_epsilon_ratio=config.pre_approx_epsilon_ratio,
    )[0].image


@pytest.mark.parametrize("case,name", [
    ("remove-square", tool.SHAPES), ("swap-square-circle", tool.SHAPES),
    ("move-sun", tool.HOUSE),
])
def test_real_fixture_perturbations_preserve_whole_crops_and_other_pixels(case, name):
    original = preprocessed(name)
    before = original.copy()
    candidate, details = tool.perturb(original, case)
    assert np.array_equal(original, before)
    assert candidate.shape == original.shape
    touched = np.zeros(original.shape[:2], dtype=bool)
    sources = details["source_rectangles_xyxy"]
    destinations = details["destination_rectangles_xyxy"]
    for x0, y0, x1, y1 in sources + destinations:
        touched[y0:y1, x0:x1] = True
    assert np.array_equal(candidate[~touched], original[~touched])
    for source, destination in zip(sources, destinations, strict=bool(destinations)):
        a, b, c, d = source
        x, y, z, t = destination
        assert np.array_equal(original[b:d, a:c], candidate[y:t, x:z])
    if case == "remove-square":
        a, b, c, d = sources[0]
        assert cv2.cvtColor(candidate[b:d, a:c], cv2.COLOR_BGR2HSV)[:, :, 1].max() < 60
    assert not np.array_equal(original, candidate)


def test_relocate_rejects_clipping_and_overwriting_subjects():
    image = np.full((100, 100, 3), 245, np.uint8)
    image[10:30, 10:30] = (0, 0, 255)
    image[60:80, 60:80] = (255, 0, 0)
    with pytest.raises(ValueError, match="crop"):
        tool.relocate(image, [(10, 10, 30, 30)], [(1, 1)])
    with pytest.raises(ValueError, match="cover"):
        tool.relocate(image, [(10, 10, 30, 30)], [(70, 70)])


def test_prepare_only_creates_eleven_hashed_cases_without_model_calls(monkeypatch, tmp_path):
    class ForbiddenChecker:
        def __init__(self, **kwargs):
            raise AssertionError("prepare-only must not contact a model")
    monkeypatch.setattr(tool, "RecordingChecker", ForbiddenChecker)
    out = tmp_path / "prepared"
    assert tool.main(["--out", str(out), "--prepare-only"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 3
    assert manifest["content_policy"] == tool.CONTENT_POLICY
    assert len(manifest["cases"]) == 11
    for case in manifest["cases"]:
        assert case["content_policy"] == tool.CONTENT_POLICY
        assert tool.sha256((out / case["original_path"]).read_bytes()) == case["original_sha256"]
        assert tool.sha256((out / case["candidate_path"]).read_bytes()) == case["candidate_sha256"]
        assert tool.sha256(tool.Path(case["source_path"]).read_bytes()) == case["source_sha256"]
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["checked"] == 0
    assert report["summary"]["content"]["accuracy"] is None
    assert report["summary"]["style"]["labelled"] == 7
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert "N/A" in markdown
    assert tool.CONTENT_POLICY["version"] in markdown
    assert "Expected theme association" in markdown
    by_case = {row["case_id"]: row for row in manifest["cases"]}
    for case_id in ("added-tree", "remove-square", "swap-square-circle", "move-sun"):
        assert by_case[case_id]["expected_content"] is True
    assert by_case["added-tree"]["known_reject"] is False
    for case_id in ("remove-square", "swap-square-circle", "move-sun"):
        assert by_case[case_id]["expected_style"] == "mismatch"
        assert by_case[case_id]["known_reject"] is True
    assert by_case["unrelated"]["expected_content"] is False
    assert by_case["unrelated"]["known_reject"] is True


def test_textured_house_case_preserves_historical_run_source(tmp_path):
    source = tool.ROOT / tool.GENERATED / "house_textured.png"
    original_bytes = source.read_bytes()
    out = tmp_path / "textured"
    assert tool.main(["--out", str(out), "--cases", "house-textured", "--prepare-only"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    assert case["expected_content"] is True
    assert case["expected_style"] == "match"
    assert case["known_reject"] is False
    assert case["candidate_source_path"] == str(source)
    assert case["candidate_source_sha256"] == tool.sha256(original_bytes)
    assert (out / case["candidate_path"]).read_bytes() == original_bytes
    assert source.read_bytes() == original_bytes


def test_existing_output_rejected_before_configuration_or_model(monkeypatch, tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(tool, "load_app_config", lambda _: pytest.fail("must reject before loading"))
    with pytest.raises(FileExistsError):
        tool.main(["--out", str(tmp_path)])
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("cases", [["missing"], ["identity,identity"]])
def test_unknown_or_duplicate_case_ids_rejected(tmp_path, cases):
    out = tmp_path / "bad"
    with pytest.raises(SystemExit):
        tool.main(["--out", str(out), "--cases", *cases])
    assert not out.exists()


def test_num_predict_override_is_recorded_without_changing_config(tmp_path):
    config_before = (tool.ROOT / "configs/app.json").read_bytes()
    out = tmp_path / "tokens"
    assert tool.main(["--out", str(out), "--cases", "identity", "--prepare-only",
                      "--num-predict", "768"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checker"]["options"]["num_predict"] == 768
    assert (tool.ROOT / "configs/app.json").read_bytes() == config_before


@pytest.mark.parametrize("value", ["0", "-1"])
def test_num_predict_rejects_nonpositive_values(tmp_path, value):
    out = tmp_path / "bad-tokens"
    with pytest.raises(SystemExit):
        tool.main(["--out", str(out), "--num-predict", value])
    assert not out.exists()


@pytest.mark.parametrize("mode", ["pairwise", "two_stage", "comparison"])
def test_runs_use_configured_dispatch_with_overrides_and_full_response(monkeypatch, tmp_path, mode):
    calls = []

    class FakeChecker:
        def __init__(self, **kwargs):
            assert kwargs["mode"] == mode
            assert kwargs["model"] == "test-model"
            assert kwargs["options"]["num_gpu"] == 0
            self.calls = []

        def check_pair(self, **kwargs):
            calls.append(kwargs)
            self.calls.append({"response": "complete-response"})
            result = CheckResult("mismatch", True, "identity", raw_text="complete-raw")
            if mode == "comparison":
                result.extra["comparison_board"] = tool.build_comparison_board(
                    kwargs["original_bytes"], kwargs["candidate_bytes"])[1]
            return result

    monkeypatch.setattr(tool, "RecordingChecker", FakeChecker)
    out = tmp_path / mode
    assert tool.main(["--out", str(out), "--model", "test-model", "--mode", mode,
                      "--cases", "identity,tone-only"]) == 0
    assert len(calls) == 2
    assert calls[0]["original_bytes"] == calls[0]["candidate_bytes"]
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["content_policy"] == tool.CONTENT_POLICY
    assert report["summary"]["content"]["correct"] == 2
    assert report["summary"]["style"]["correct"] == 2
    assert all(c["raw_text"] == "complete-raw" for c in report["cases"])
    assert all(c["model_responses"] == [{"response": "complete-response"}] for c in report["cases"])
    if mode == "comparison":
        assert report["checker"]["comparison_version"] == COMPARISON_VERSION
        assert COMPARISON_VERSION in (out / "report.md").read_text(encoding="utf-8")
        for case in report["cases"]:
            board = case["comparison_board"]
            assert tool.sha256((out / board["path"]).read_bytes()) == board["sha256"]
            assert board["sha256"] == case["actual"]["comparison_board"]["sha256"]


def test_comparison_prepare_only_saves_actual_board_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(tool, "RecordingChecker", lambda **kwargs: pytest.fail("no model calls"))
    out = tmp_path / "board"
    assert tool.main(["--out", str(out), "--cases", "identity", "--mode", "comparison",
                      "--prepare-only"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    expected, metadata = tool.build_comparison_board(
        (out / case["original_path"]).read_bytes(), (out / case["candidate_path"]).read_bytes())
    assert (out / case["comparison_board"]["path"]).read_bytes() == expected
    assert case["comparison_board"]["sha256"] == metadata["sha256"]


def test_comparison_board_mismatch_is_recorded_as_error(monkeypatch, tmp_path):
    class FakeChecker:
        def __init__(self, **kwargs):
            self.calls = []

        def check_pair(self, **kwargs):
            return CheckResult("match", True, "wrong board", extra={"comparison_board": {"sha256": "wrong"}})
    monkeypatch.setattr(tool, "RecordingChecker", FakeChecker)
    out = tmp_path / "mismatch"
    assert tool.main(["--out", str(out), "--cases", "identity", "--mode", "comparison"]) == 1
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["errors"] == 1
    assert report["cases"][0]["error"]["type"] == "CheckTechnicalError"


@pytest.mark.parametrize("payload", [
    {"style_match": "match", "content_preserved": "true", "reason": "bad"},
    {"style_match": "yes", "content_preserved": True, "reason": "bad"},
])
def test_invalid_core_payload_is_error_not_a_quality_rejection(monkeypatch, tmp_path, payload):
    class FakeChecker:
        def __init__(self, **kwargs):
            self.calls = []

        def check_pair(self, **kwargs):
            return SimpleNamespace(to_dict=lambda: payload, raw_text="bad-raw")
    monkeypatch.setattr(tool, "RecordingChecker", FakeChecker)
    out = tmp_path / "invalid"
    assert tool.main(["--out", str(out), "--cases", "identity"]) == 1
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["errors"] == 1
    assert report["summary"]["content"]["evaluated"] == 0
    assert report["summary"]["known_reject"]["evaluated"] == 0
    assert "actual" not in report["cases"][0]


def test_statistics_score_axes_independently_and_do_not_hide_errors():
    def row(content, style, actual=None, error=False):
        return {"expected_content": content, "expected_style": style,
                "known_reject": content is False or style == "mismatch",
                "status": "error" if error else "checked", "actual": actual}
    rows = [
        row(True, "mismatch", {"content_preserved": True, "style_match": "match", "passed": True}),
        row(False, None, {"content_preserved": True, "style_match": "mismatch", "passed": False}),
        row(True, None, {"content_preserved": None, "style_match": "uncertain", "passed": False}),
        row(False, None, error=True),
    ]
    result = tool.summarize(rows)
    assert result["content"] == {"labelled": 4, "evaluated": 3, "correct": 1,
                                  "accuracy": 1 / 3, "abstained": 1, "errors": 1}
    assert result["style"]["evaluated"] == 1
    assert result["style"]["correct"] == 0
    assert result["known_reject"] == {"total": 3, "evaluated": 2, "false_accepts": 1, "errors": 1}


def test_recording_keeps_complete_response_before_validation_error(monkeypatch):
    raw = {"message": {"content": '{"style_match":"bad"}'}}
    monkeypatch.setattr(tool.VisionChecker, "_post", lambda *args: raw)
    checker = tool.RecordingChecker(base_url="http://127.0.0.1:1", model="fake", mode="pairwise")
    with pytest.raises(CheckTechnicalError):
        checker.check_pair(b"original", b"candidate", "prompt", "style", "features", "rules")
    assert checker.calls == [{"endpoint": "/api/chat", "response": raw,
                              "messages": [{"role": "user", "content": "prompt", "image_count": 2}],
                              "prompt": None, "image_count": 0}]
