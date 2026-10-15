"""Comparison checker contracts, recorded schema compatibility and failure boundaries."""

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import requests

from app import states
from app.checker import (
    COMPARISON_CONTENT_PROMPT,
    COMPARISON_OBSERVATION_PROMPT,
    COMPARISON_VERSION,
    VisionChecker,
    _validate_axis_payload,
    _validate_comparison_content,
    _validate_comparison_observation,
    build_comparison_board,
)
from app.errors import (
    CheckTechnicalError,
    ConfigError,
    PipelineError,
    ServiceUnavailableError,
    TimeoutError_,
)
from tests.fixtures import make_app_config, marker_drawing_png, solid_png

CONTENT = {
    "original_subjects": ["house", {"name": "sun", "position": "upper right"}],
    "candidate_subjects": ["house", {"name": "sun", "position": "upper right"}],
    "missing_subjects": [], "added_subjects": [], "layout_changed": False,
    "content_preserved": True, "reason": "The house and sunlight retain the original home theme.",
}
OBSERVATION = {
    "original_treatment": "marker outlines", "candidate_treatment": "expressive paint",
    "transformation": "technique", "reason": "The visible marks changed.",
}
STYLE = {"style_match": "match", "reason": "Visible expressive brushwork."}
HISTORICAL_RESPONSES = json.loads(
    (Path(__file__).parent / "fixtures/comparison_responses.json").read_text(encoding="utf-8")
)


def configured_checker(monkeypatch, content=None, observation=None, style=None):
    checker = VisionChecker("http://127.0.0.1:1", "local-model", mode="comparison")
    calls = []
    replies = iter(json.dumps(item) for item in (
        CONTENT if content is None else content,
        OBSERVATION if observation is None else observation,
        STYLE if style is None else style,
    ))

    def respond(prompt, images):
        calls.append((prompt, images))
        return next(replies)

    monkeypatch.setattr(checker, "_call", respond)
    return checker, calls


def check(checker):
    return checker.check_pair(
        marker_drawing_png(), solid_png(), "legacy check prompt",
        "Fauvism", "expressive brushwork", "retain a recognisable theme", attempt=2, duration_sec=3.5,
    )


def test_comparison_board_is_deterministic_and_preserves_orientation():
    original = solid_png(width=80, height=160)
    candidate = solid_png(width=160, height=80, value=100)
    data, metadata = build_comparison_board(original, candidate)
    same_data, same_metadata = build_comparison_board(original, candidate)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert data == same_data and metadata == same_metadata
    assert image.shape == (776, 1088, 3)
    assert metadata["sha256"] == hashlib.sha256(data).hexdigest()
    assert metadata["original_sha256"] == hashlib.sha256(original).hexdigest()
    assert metadata["candidate_sha256"] == hashlib.sha256(candidate).hexdigest()
    assert metadata["panels"] == [
        {"label": "A - ORIGINAL", "source_size": [80, 160], "image_rect": [101, 60, 350, 700]},
        {"label": "B - CANDIDATE", "source_size": [160, 80], "image_rect": [552, 280, 520, 260]},
    ]
    assert image[650, 250].tolist() == [200, 200, 200]
    assert image[500, 900].tolist() == [100, 100, 100]


@pytest.mark.parametrize("bad", [b"", b"not a picture", None, "not bytes"])
@pytest.mark.parametrize("role", ["original", "candidate"])
def test_invalid_board_input_is_technical_failure(bad, role):
    images = {"original_bytes": solid_png(), "candidate_bytes": solid_png()}
    images[f"{role}_bytes"] = bad
    with pytest.raises(CheckTechnicalError):
        build_comparison_board(**images)


def test_comparison_uses_one_board_two_visual_calls_and_independent_style(monkeypatch):
    checker, calls = configured_checker(monkeypatch)
    result = check(checker)
    assert result.passed is True
    assert result.attempt == 2 and result.duration_sec == 3.5
    assert len(calls) == 3
    assert calls[0][0] == COMPARISON_CONTENT_PROMPT
    assert calls[1][0] == COMPARISON_OBSERVATION_PROMPT
    assert len(calls[0][1]) == 1 and calls[0][1] == calls[1][1]
    assert calls[2][1] == []
    assert "Fauvism" not in calls[0][0] and "Fauvism" not in calls[1][0]
    assert "Fauvism" in calls[2][0]
    evidence = result.to_dict()
    assert evidence["comparison_board"]["sha256"] == hashlib.sha256(calls[0][1][0]).hexdigest()
    assert evidence["mode"] == "comparison" and evidence["comparison_version"]
    assert evidence["comparison_prompts"] == dict(zip(("content", "observation", "style"),
                                                     (call[0] for call in calls), strict=True))
    assert json.loads(evidence["comparison_raw"]["content"]) == CONTENT
    assert json.loads(result.raw_text) == evidence["comparison_raw"]


@pytest.mark.parametrize("field,value", [
    ("added_subjects", ["tree"]), ("missing_subjects", [{"type": "sun"}]),
    ("layout_changed", True),
])
def test_element_changes_are_diagnostic_not_a_theme_veto(monkeypatch, field, value):
    content = {**CONTENT, field: value}
    checker, _ = configured_checker(monkeypatch, content=content)
    result = check(checker)
    assert result.content_preserved is True and result.passed
    assert result.extra["comparison_content"][field] == value
    assert result.extra["verdict_guards"] == []
    assert result.extra["comparison_content"]["content_preserved"] is True


def test_related_theme_can_pass_with_omissions_additions_and_rearrangement(pipeline_env, monkeypatch):
    pipeline, store, generator, _, _ = pipeline_env()
    content = {
        **CONTENT,
        "candidate_subjects": ["house", "garden"],
        "missing_subjects": ["sun"], "added_subjects": ["garden"], "layout_changed": True,
        "reason": "The home theme remains recognisable through the house despite the new garden and layout.",
    }
    checker, calls = configured_checker(monkeypatch, content=content)
    pipeline.checker = checker
    summary = pipeline.run(marker_drawing_png(), "photo.png")
    job = store.load(summary["job_id"])
    assert summary["state"] == states.COMPLETED
    assert len(generator.calls) == 1 and len(calls) == 3
    evidence = job.read_check_result(1)
    assert evidence["passed"] is True
    assert evidence["comparison_content"] == content
    assert evidence["comparison_version"] == "comparison-v3-style-first"
    assert job.output_path.exists()


def test_wholly_unrelated_candidate_cannot_pass_on_style_alone(monkeypatch):
    content = {
        **CONTENT, "candidate_subjects": ["underwater fish"],
        "missing_subjects": ["house", "sun"], "added_subjects": ["underwater fish"],
        "layout_changed": True, "content_preserved": False,
        "reason": "An underwater scene has no recognisable connection to the original home theme.",
    }
    checker, _ = configured_checker(monkeypatch, content=content)
    result = check(checker)
    assert result.style_match == "match"
    assert result.content_preserved is False and not result.passed
    assert result.extra["comparison_content"] == content


@pytest.mark.parametrize("transformation,expected", [
    ("none", "mismatch"), ("colour_only", "mismatch"), ("uncertain", "uncertain"),
])
def test_observed_transformation_vetoes_style_match(monkeypatch, transformation, expected):
    checker, _ = configured_checker(monkeypatch, observation={
        **OBSERVATION, "transformation": transformation,
    })
    result = check(checker)
    assert result.style_match == expected and not result.passed
    assert f"transformation={transformation}" in result.reason
    assert f"style_match={expected}" in result.reason
    assert result.extra["style_judgement"]["style_match"] == "match"


@pytest.mark.parametrize("preserved", [False, None])
def test_content_false_or_unknown_is_never_promoted(monkeypatch, preserved):
    checker, _ = configured_checker(monkeypatch, content={**CONTENT, "content_preserved": preserved})
    result = check(checker)
    assert result.content_preserved is preserved and not result.passed


@pytest.mark.parametrize("component,field,value,expected_calls", [
    ("content", "content_preserved", "true", 1),
    ("content", "content_preserved", 1, 1),
    ("content", "layout_changed", 0, 1),
    ("content", "layout_changed", "false", 1),
    ("content", "added_subjects", "none", 1),
    ("content", "missing_subjects", None, 1),
    ("content", "original_subjects", [1], 1),
    ("content", "candidate_subjects", [{"name": 12, "position": "upper right"}], 1),
    ("content", "original_subjects", [{"name": "", "parts": ["rays"]}], 1),
    ("content", "original_subjects", [{"type": "house", "position": float("nan")}], 1),
    ("content", "candidate_subjects", [{}], 1),
    ("content", "candidate_subjects", [{"position": "upper right"}], 1),
    ("content", "reason", "", 1),
    ("observation", "transformation", "color_only", 2),
    ("observation", "transformation", True, 2),
    ("observation", "original_treatment", [], 2),
    ("observation", "candidate_treatment", None, 2),
    ("observation", "reason", "", 2),
    ("style", "style_match", True, 3),
    ("style", "style_match", "MATCH", 3),
    ("style", "reason", "", 3),
])
def test_invalid_comparison_fields_stop_immediately(monkeypatch, component, field, value, expected_calls):
    payloads = {"content": copy.deepcopy(CONTENT), "observation": dict(OBSERVATION), "style": dict(STYLE)}
    payloads[component][field] = value
    checker, calls = configured_checker(monkeypatch, **payloads)
    with pytest.raises(CheckTechnicalError) as error:
        check(checker)
    assert len(calls) == expected_calls
    assert error.value.details["comparison_component"] == component
    assert len(error.value.details["comparison_raw"]) == expected_calls


@pytest.mark.parametrize("component,field", [
    ("content", field) for field in CONTENT
] + [("observation", field) for field in OBSERVATION] + [("style", "style_match"), ("style", "reason")])
def test_required_comparison_fields_cannot_be_missing(monkeypatch, component, field):
    payloads = {"content": dict(CONTENT), "observation": dict(OBSERVATION), "style": dict(STYLE)}
    payloads[component].pop(field)
    checker, _ = configured_checker(monkeypatch, **payloads)
    with pytest.raises(CheckTechnicalError):
        check(checker)


@pytest.mark.parametrize("row", HISTORICAL_RESPONSES, ids=lambda row: row["case"])
def test_historical_responses_remain_parseable_without_relabelling(row):
    # Parse the recorded responses as historical evidence, not a fresh run of the
    # current checker or current quality labels. The old false verdicts stay false.
    content = _validate_comparison_content(json.dumps(row["content"]))
    observation = _validate_comparison_observation(json.dumps(row["observation"]))
    style_raw = json.dumps(row["style"])
    style = _validate_axis_payload(style_raw, "style_match")
    assert content == row["content"]
    assert observation == row["observation"]
    assert style["style_match"] == row["style"]["style_match"]
    assert json.loads(style["raw"]) == row["style"]


@pytest.mark.parametrize("colours", ["green and red", ["green", "red"], {"roof": "green"}])
def test_subject_auxiliary_descriptions_do_not_require_fixed_field_names(monkeypatch, colours):
    subject = {"type": "house", "colors": colours, "observed_details": {
        "bounds": [0.1, 0.2, 0.4, 0.5], "partly_occluded": False, "unknown_detail": None,
    }}
    content = {**CONTENT, "original_subjects": [subject], "missing_subjects": [subject]}
    checker, _ = configured_checker(monkeypatch, content=content)
    result = check(checker)
    assert result.extra["comparison_content"]["original_subjects"] == [subject]
    assert result.content_preserved is True and result.passed
    assert result.extra["comparison_content"]["missing_subjects"] == [subject]


def test_pipeline_does_not_retry_comparison_schema_failure(pipeline_env, monkeypatch):
    pipeline, store, generator, _, _ = pipeline_env()
    checker, calls = configured_checker(monkeypatch, content={**CONTENT, "layout_changed": "false"})
    pipeline.checker = checker
    summary = pipeline.run(marker_drawing_png(), "photo.png")
    job = store.load(summary["job_id"])
    assert summary["state"] == states.FAILED
    assert len(generator.calls) == 1 and len(calls) == 1
    assert job.read_check_result(1)["technical_error"]["details"]["comparison_component"] == "content"
    assert not job.output_path.exists()


def test_comparison_truncation_keeps_completed_evidence_and_stops(pipeline_env, monkeypatch):
    pipeline, store, generator, _, _ = pipeline_env()
    checker = VisionChecker("http://127.0.0.1:1", "local-model", mode="comparison")
    pipeline.checker = checker
    calls = []

    def respond(endpoint, body):
        calls.append(endpoint)
        if len(calls) == 1:
            return {"message": {"content": json.dumps(CONTENT)}, "done_reason": "stop"}
        return {
            "message": {"content": '{"original_treatment":"marker'},
            "done_reason": "length", "eval_count": 768,
        }

    monkeypatch.setattr(checker, "_post", respond)
    summary = pipeline.run(marker_drawing_png(), "photo.png")
    job = store.load(summary["job_id"])
    error = job.read_check_result(1)["technical_error"]
    assert summary["state"] == states.FAILED
    assert len(calls) == 2 and len(generator.calls) == 1
    assert not job.output_path.exists() and not job.attempt_path(2).exists()
    assert error["details"]["comparison_component"] == "observation"
    assert error["details"]["done_reason"] == "length"
    assert error["details"]["raw_response"] == '{"original_treatment":"marker'
    assert json.loads(error["details"]["comparison_raw"]["content"]) == CONTENT


@pytest.mark.parametrize("failed_call", [2, 3])
@pytest.mark.parametrize("failure,category", [
    (requests.Timeout("checker timed out"), "timeout"),
    (requests.ConnectionError("checker disconnected"), "service_unavailable"),
])
def test_comparison_transport_failure_preserves_prior_evidence_and_category(
    pipeline_env, monkeypatch, failed_call, failure, category,
):
    pipeline, store, generator, _, _ = pipeline_env()
    checker = VisionChecker("http://127.0.0.1:1", "local-model", mode="comparison")
    pipeline.checker = checker
    calls = []

    def respond(url, **kwargs):
        calls.append(kwargs["json"]["messages"][0]["content"])
        if len(calls) == failed_call:
            raise failure
        payload = (CONTENT, OBSERVATION)[len(calls) - 1]
        response = Mock(status_code=200)
        response.json.return_value = {"message": {"content": json.dumps(payload)}}
        return response

    monkeypatch.setattr(checker.session, "post", respond)
    summary = pipeline.run(marker_drawing_png(), "photo.png")
    job = store.load(summary["job_id"])
    error = job.read_check_result(1)["technical_error"]
    details = error["details"]
    assert summary["state"] == states.FAILED
    assert len(calls) == failed_call and len(generator.calls) == 1
    assert not job.output_path.exists() and not job.attempt_path(2).exists()
    assert error["category"] == category
    assert details["mode"] == "comparison"
    assert details["comparison_version"] == COMPARISON_VERSION
    assert details["comparison_component"] == ("observation" if failed_call == 2 else "style")
    assert list(details["comparison_prompts"].values()) == calls
    assert len(details["comparison_raw"]) == failed_call - 1
    assert json.loads(details["comparison_raw"]["content"]) == CONTENT
    if failed_call == 3:
        assert json.loads(details["comparison_raw"]["observation"]) == OBSERVATION
    assert details["comparison_board"]["width"] == 1088
    assert job.meta["errors"][-1]["details"] == details
    assert job.meta["attempts"][0]["error"]["details"] == details


@pytest.mark.parametrize("error_type", [PipelineError, CheckTechnicalError, ServiceUnavailableError, TimeoutError_])
@pytest.mark.parametrize("details", [None, {}])
def test_empty_error_details_keep_original_format(error_type, details):
    error = error_type("failure", stage=states.CHECKING, details=details)
    assert error.to_dict() == {
        "category": error_type.category, "stage": states.CHECKING, "message": "failure",
    }


@pytest.mark.parametrize("mode", ["comparison", "two_stage", "pairwise"])
def test_config_accepts_supported_modes(tmp_path, mode):
    assert make_app_config(tmp_path, checker={"mode": mode}).checker_mode == mode


def test_config_and_checker_reject_unknown_mode(tmp_path):
    with pytest.raises(ConfigError):
        make_app_config(tmp_path, checker={"mode": "typo"})
    with pytest.raises(CheckTechnicalError):
        VisionChecker("http://127.0.0.1:1", "model", mode="typo")


@pytest.mark.parametrize("requested,installed,available", [
    ("qwen3-vl:8b-instruct", ["qwen3-vl:8b"], False),
    ("qwen3-vl:8b", ["qwen3-vl:8b-instruct"], False),
    ("qwen3-vl:8b-instruct", ["qwen3-vl:8b-instruct"], True),
    ("qwen3-vl", ["qwen3-vl:8b"], False),
    ("qwen3-vl", ["qwen3-vl:latest"], True),
    ("qwen3-vl:latest", ["qwen3-vl"], True),
    ("localhost:11434/team/model", ["localhost:11434/team/model:latest"], True),
])
def test_model_verification_requires_exact_tag(monkeypatch, requested, installed, available):
    checker = VisionChecker("http://127.0.0.1:1", requested)
    monkeypatch.setattr(checker, "list_models", lambda: installed)
    if available:
        checker.verify_model()
    else:
        with pytest.raises(ServiceUnavailableError):
            checker.verify_model()
