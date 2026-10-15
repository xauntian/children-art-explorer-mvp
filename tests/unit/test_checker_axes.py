"""Evidence isolation and strict results for the independent checking axes."""

import json
from pathlib import Path

import pytest

from app.checker import VisionChecker
from app.errors import CheckTechnicalError

ORIGINAL = """**STYLE:** ORIGINAL_MARKER_EVIDENCE

### SUBJECTS:
- one flower, with a stem and petals

**LAYOUT**:
- flower centred at the top, stem below

COLOURS: ORIGINAL_PALETTE_EVIDENCE
"""
CANDIDATE = """STYLE: CANDIDATE_DABS_EVIDENCE
SUBJECTS: one flower with petals and stem
LAYOUT: flower centred at the top, stem below
COLOURS: CANDIDATE_PALETTE_EVIDENCE
"""


def judge(monkeypatch, content, style, candidate=CANDIDATE):
    checker = VisionChecker("http://127.0.0.1:1", "model")
    calls = []
    responses = iter([json.dumps(content), json.dumps(style)])

    def respond(prompt, images):
        calls.append((prompt, images))
        return next(responses)

    monkeypatch.setattr(checker, "_call", respond)
    result = checker._judge_descriptions(
        ORIGINAL, candidate, "Impressionism", "broken dabs", "retain a recognisable theme",
    )
    return result, calls


def test_judgement_axes_receive_separate_evidence_and_preserve_raw(monkeypatch):
    content = {"content_preserved": True, "reason": "same flower and stem"}
    style = {"style_match": "match", "reason": "visible broken dabs"}
    result, calls = judge(monkeypatch, content, style)
    content_prompt, images = calls[0]
    assert images == []
    assert "flower" in content_prompt and "stem" in content_prompt
    assert "Impressionism" not in content_prompt
    assert "ORIGINAL_MARKER_EVIDENCE" not in content_prompt
    assert "CANDIDATE_DABS_EVIDENCE" not in content_prompt
    assert "PALETTE_EVIDENCE" not in content_prompt
    style_prompt, images = calls[1]
    assert images == []
    assert "Impressionism" in style_prompt
    assert "CANDIDATE_DABS_EVIDENCE" in style_prompt
    assert "ORIGINAL_MARKER_EVIDENCE" in style_prompt
    assert "flower" not in style_prompt and "stem" not in style_prompt
    assert result.passed is True
    assert json.loads(result.extra["content_judgement"]["raw"]) == content
    assert json.loads(result.extra["style_judgement"]["raw"]) == style


def test_two_stage_accepts_theme_with_merged_repositioned_and_added_elements(monkeypatch):
    candidate = """STYLE: CANDIDATE_DABS_EVIDENCE
SUBJECTS: a bouquet merging the original flower with new buds
LAYOUT: flowers spread across the image, with altered sizes and positions
COLOURS: CANDIDATE_PALETTE_EVIDENCE
"""
    content = {"content_preserved": True, "reason": "The flower theme remains recognisable."}
    result, calls = judge(monkeypatch, content, {
        "style_match": "match", "reason": "Visible broken dabs build the forms.",
    }, candidate=candidate)
    assert result.passed
    assert "not one-to-one object correspondence" in calls[0][0]
    assert "Omitting or merging elements, adding theme-related elements" in calls[0][0]
    assert "wholly unrelated" in calls[0][0]
    assert "bouquet" in calls[0][0]
    assert result.extra["content_judgement"]["reason"] == content["reason"]


def test_two_stage_describes_theme_without_requiring_exhaustive_object_matching(monkeypatch):
    checker = VisionChecker("http://127.0.0.1:1", "model", mode="two_stage")
    calls = []
    replies = iter([
        ORIGINAL, CANDIDATE,
        '{"content_preserved":true,"reason":"recognisable flower theme"}',
        '{"style_match":"match","reason":"broken colour dabs"}',
    ])

    def respond(prompt, images):
        calls.append((prompt, images))
        return next(replies)

    monkeypatch.setattr(checker, "_call", respond)
    result = checker.check_pair(
        b"original", b"candidate", "unused pairwise prompt", "Impressionism", "broken dabs",
        "retain a recognisable theme",
    )
    assert result.passed
    assert calls[0][0] == calls[1][0]
    assert "theme and recognisable visual" in calls[0][0]
    assert "without requiring an exhaustive object inventory" in calls[0][0]
    assert [images for _, images in calls] == [[b"original"], [b"candidate"], [], []]


@pytest.mark.parametrize("content", [False, None])
def test_style_match_does_not_override_failed_or_uncertain_content(monkeypatch, content):
    result, _ = judge(
        monkeypatch,
        {"content_preserved": content, "reason": "not preserved", "style_match": "match"},
        {"style_match": "match", "reason": "paint", "content_preserved": True},
    )
    assert result.content_preserved is content
    assert result.passed is False


@pytest.mark.parametrize("style", ["mismatch", "uncertain"])
def test_preserved_content_does_not_override_style_failure(monkeypatch, style):
    result, _ = judge(
        monkeypatch,
        {"content_preserved": True, "reason": "same subjects", "style_match": "match"},
        {"style_match": style, "reason": "marker"},
    )
    assert result.style_match == style
    assert result.passed is False


@pytest.mark.parametrize("bad", ["true", 1, "false", {}, []])
def test_content_axis_rejects_coerced_values(monkeypatch, bad):
    with pytest.raises(CheckTechnicalError):
        judge(
            monkeypatch,
            {"content_preserved": bad, "reason": "r"},
            {"style_match": "match", "reason": "r"},
        )


@pytest.mark.parametrize("bad", [True, 1, "MATCH", "yes", None])
def test_style_axis_rejects_invalid_values(monkeypatch, bad):
    with pytest.raises(CheckTechnicalError):
        judge(
            monkeypatch,
            {"content_preserved": True, "reason": "r"},
            {"style_match": bad, "reason": "r"},
        )


@pytest.mark.parametrize("axis", ["content_preserved", "style_match"])
@pytest.mark.parametrize("missing", ["axis", "reason"])
def test_axis_requires_its_own_result_and_reason(monkeypatch, axis, missing):
    content = {"content_preserved": True, "reason": "r"}
    style = {"style_match": "match", "reason": "r"}
    payload = content if axis == "content_preserved" else style
    payload.pop(axis if missing == "axis" else "reason")
    with pytest.raises(CheckTechnicalError, match="缺少字段"):
        judge(monkeypatch, content, style)


def test_unlabelled_description_does_not_fall_back_to_mixed_evidence(monkeypatch):
    checker = VisionChecker("http://127.0.0.1:1", "model")
    monkeypatch.setattr(checker, "_call", lambda *args: pytest.fail("must not call model"))
    with pytest.raises(CheckTechnicalError, match="标签"):
        checker._judge_descriptions(
            "a colourful painting of a flower", CANDIDATE, "Fauvism", "vivid", "layout",
        )


@pytest.mark.parametrize("content,style,reason,passed", [
    (True, "match", "The flower theme remains despite merged forms and new buds.", True),
    (False, "match", "The underwater scene is wholly unrelated to the flower drawing.", False),
    (None, "match", "No reliable evidence of a theme connection.", False),
    (True, "uncertain", "Insufficient evidence of the requested painting technique.", False),
    (True, "mismatch", "The same marks remain with only a hue change.", False),
])
def test_pairwise_uses_style_first_theme_policy_and_strict_verdicts(
    monkeypatch, content, style, reason, passed,
):
    checker = VisionChecker("http://127.0.0.1:1", "model", mode="pairwise")
    prompt = (Path(__file__).resolve().parents[2] / "prompts/check.txt").read_text(encoding="utf-8")
    for key, value in {
        "style_name": "Impressionism", "style_features": "broken colour dabs",
        "preservation_rules": "retain a recognisable theme",
    }.items():
        prompt = prompt.replace("{" + key + "}", value)
    payload = {"content_preserved": content, "style_match": style, "reason": reason}
    calls = []

    def respond(actual_prompt, images):
        calls.append((actual_prompt, images))
        return json.dumps(payload)

    monkeypatch.setattr(checker, "_call", respond)
    result = checker.check_pair(
        b"original", b"candidate", prompt, "Impressionism", "broken colour dabs",
        "retain a recognisable theme",
    )
    assert calls == [(prompt, [b"original", b"candidate"])]
    assert "Stylisation takes priority" in prompt
    assert "theme association, not exact object preservation" in prompt
    assert "Omitting or merging elements, adding theme-related elements" in prompt
    assert "wholly unrelated" in prompt and "mere hue change" in prompt
    assert result.content_preserved is content
    assert result.style_match == style and result.passed is passed


@pytest.mark.parametrize("field,value", [("content_preserved", "true"), ("style_match", "MATCH")])
def test_pairwise_theme_policy_does_not_relax_json_types(monkeypatch, field, value):
    checker = VisionChecker("http://127.0.0.1:1", "model", mode="pairwise")
    payload = {"content_preserved": True, "style_match": "match", "reason": "theme connected"}
    payload[field] = value
    monkeypatch.setattr(checker, "_call", lambda *args: json.dumps(payload))
    with pytest.raises(CheckTechnicalError):
        checker.check_pair(b"original", b"candidate", "theme check", "Fauvism", "paint", "theme")
