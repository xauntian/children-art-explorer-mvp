"""Truncated local-model responses remain failures with auditable evidence."""

import json

import pytest

from app import states
from app.checker import VisionChecker
from app.errors import CheckTechnicalError
from tests.fixtures import marker_drawing_png


def truncated_payload(api, text):
    payload = {
        "done_reason": "length",
        "model": "local-model:7b",
        "eval_count": 384,
        "prompt_eval_count": 812,
        "thinking": "private reasoning must not be saved",
        "images": ["image-base64-must-not-be-saved"],
        "context": [1, 2, 3],
    }
    if api == "chat":
        payload["message"] = {
            "content": text,
            "thinking": "private message reasoning",
            "images": ["message-image-base64"],
        }
    else:
        payload["response"] = text
    return payload


def expected_details(text):
    result = {
        "done_reason": "length",
        "model": "local-model:7b",
        "eval_count": 384,
        "prompt_eval_count": 812,
    }
    if isinstance(text, str):
        result["raw_response"] = text
    return result


@pytest.mark.parametrize("api", ["chat", "generate"])
@pytest.mark.parametrize("text", ["partial response " * 300, "", None])
def test_truncation_records_only_available_response_and_safe_metadata(monkeypatch, api, text):
    checker = VisionChecker("http://127.0.0.1:1", "local-model:7b", api=api)
    monkeypatch.setattr(checker, "_post", lambda *args: truncated_payload(api, text))

    with pytest.raises(CheckTechnicalError) as error:
        checker._call("private request", [b"private image bytes"])

    assert error.value.stage == states.CHECKING
    assert error.value.to_dict()["details"] == expected_details(text)


def test_technical_error_without_details_keeps_original_serialization():
    error = CheckTechnicalError("bad JSON", stage=states.CHECKING)
    assert error.to_dict() == {
        "category": "check_technical_error", "stage": "CHECKING", "message": "bad JSON"
    }


def test_truncation_does_not_copy_nested_or_invalid_metadata(monkeypatch):
    checker = VisionChecker("http://127.0.0.1:1", "local-model:7b")
    monkeypatch.setattr(checker, "_post", lambda *args: {
        "done_reason": "length", "message": {"content": {"images": ["private"]}},
        "model": {"thinking": "private"}, "eval_count": True,
        "prompt_eval_count": {"images": ["private"]},
    })
    with pytest.raises(CheckTechnicalError) as error:
        checker._call("describe", [b"image"])
    assert error.value.to_dict()["details"] == {"done_reason": "length"}


@pytest.mark.parametrize("api", ["chat", "generate"])
def test_pipeline_persists_truncation_evidence_without_retry(pipeline_env, monkeypatch, api):
    pipeline, store, generator, _, _ = pipeline_env()
    checker = VisionChecker("http://127.0.0.1:1", "local-model:7b", api=api)
    pipeline.checker = checker
    text = "partial image description " * 200
    requests = []

    def respond(endpoint, body):
        requests.append(endpoint)
        return truncated_payload(api, text)

    monkeypatch.setattr(checker, "_post", respond)
    summary = pipeline.run(marker_drawing_png(), "photo.jpg")
    job = store.load(summary["job_id"])
    check = job.read_check_result(1)
    metadata = json.loads(job.meta_path.read_text(encoding="utf-8"))

    assert summary["state"] == states.FAILED
    assert summary["attempt_count"] == 1
    assert len(generator.calls) == 1
    assert requests == [f"/api/{api}"]
    assert not job.output_path.exists()
    assert not job.attempt_path(2).exists()
    assert check["technical_error"]["details"] == expected_details(text)
    assert metadata["errors"][-1]["details"] == expected_details(text)
    assert metadata["attempts"][0]["error"]["details"] == expected_details(text)
    assert summary["error"]["details"] == expected_details(text)
    assert "passed" not in check


@pytest.mark.parametrize("api", ["chat", "generate"])
def test_completed_response_still_returns_content(monkeypatch, api):
    checker = VisionChecker("http://127.0.0.1:1", "local-model:7b", api=api)
    payload = truncated_payload(api, "complete content")
    payload["done_reason"] = "stop"
    monkeypatch.setattr(checker, "_post", lambda *args: payload)
    assert checker._call("describe", [b"image"]) == "complete content"
