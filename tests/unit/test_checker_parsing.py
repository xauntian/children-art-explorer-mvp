"""检查结果解析与严格通过条件测试。

要求：字段缺失、类型错误、非法枚举或无法解析均视为技术错误；
不得将字符串 "true"、数字 1 或模型声称的置信度转换成通过；
uncertain、false、null 一律不通过。
"""

from __future__ import annotations

import pytest

from app.checker import (
    CheckResult,
    extract_json_object,
    validate_check_payload,
)
from app.errors import CheckTechnicalError

VALID = {"style_match": "match", "content_preserved": True, "reason": "可见的非自然色块"}


# ---------------------------------------------------------------- 通过条件
def test_strict_pass_requires_match_and_true():
    result = validate_check_payload(dict(VALID))
    assert result.passed is True


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"style_match": "mismatch", "content_preserved": True, "reason": "r"}, False),
        ({"style_match": "uncertain", "content_preserved": True, "reason": "r"}, False),
        ({"style_match": "match", "content_preserved": False, "reason": "r"}, False),
        ({"style_match": "match", "content_preserved": None, "reason": "r"}, False),
        ({"style_match": "mismatch", "content_preserved": False, "reason": "r"}, False),
        ({"style_match": "uncertain", "content_preserved": None, "reason": "r"}, False),
    ],
)
def test_non_pass_combinations(payload, expected):
    assert validate_check_payload(payload).passed is expected


def test_uncertain_is_never_a_pass():
    """uncertain 必须被规则拒绝，不能当作默认通过。"""
    result = validate_check_payload(
        {"style_match": "uncertain", "content_preserved": True, "reason": "证据不足"}
    )
    assert result.passed is False


# ---------------------------------------------------------------- 非法类型
@pytest.mark.parametrize("bad_value", ["true", "True", 1, 0, "yes", [], {}])
def test_content_preserved_rejects_non_boolean(bad_value):
    """字符串 "true"、数字 1、置信度等都不能转换成通过。"""
    payload = {"style_match": "match", "content_preserved": bad_value, "reason": "r"}
    with pytest.raises(CheckTechnicalError):
        validate_check_payload(payload)


def test_content_preserved_accepts_json_false_and_null():
    assert validate_check_payload(
        {"style_match": "match", "content_preserved": False, "reason": "r"}
    ).content_preserved is False
    assert validate_check_payload(
        {"style_match": "match", "content_preserved": None, "reason": "r"}
    ).content_preserved is None


@pytest.mark.parametrize("bad_style", ["MATCH", "Match", "yes", 1, None, True, "passed", ""])
def test_style_match_rejects_invalid_enum(bad_style):
    with pytest.raises(CheckTechnicalError):
        validate_check_payload(
            {"style_match": bad_style, "content_preserved": True, "reason": "r"}
        )


def test_reason_must_be_string():
    for bad in (None, 1, True, [], {}):
        with pytest.raises(CheckTechnicalError):
            validate_check_payload(
                {"style_match": "match", "content_preserved": True, "reason": bad}
            )


def test_reason_may_be_empty_string_but_must_be_present():
    # 空字符串仍是字符串（字段合法）；缺失才是技术错误
    assert validate_check_payload(
        {"style_match": "match", "content_preserved": True, "reason": ""}
    ).reason == ""
    with pytest.raises(CheckTechnicalError, match="缺少字段"):
        validate_check_payload({"style_match": "match", "content_preserved": True})


@pytest.mark.parametrize("missing", ["style_match", "content_preserved", "reason"])
def test_missing_fields_are_technical_errors(missing):
    payload = dict(VALID)
    payload.pop(missing)
    with pytest.raises(CheckTechnicalError, match="缺少字段"):
        validate_check_payload(payload)


def test_non_object_payload_is_technical_error():
    for bad in ([], "match", 1, None, True):
        with pytest.raises(CheckTechnicalError):
            validate_check_payload(bad)


def test_extra_fields_are_preserved_but_do_not_affect_pass():
    payload = dict(VALID)
    payload["confidence"] = 0.4
    payload["notes"] = "extra"
    result = validate_check_payload(payload)
    assert result.passed is True
    assert result.extra["confidence"] == 0.4


def test_confidence_alone_never_passes():
    """模型只给置信度、没有合法字段时不得通过。"""
    with pytest.raises(CheckTechnicalError):
        validate_check_payload({"confidence": 0.99})


# ---------------------------------------------------------------- JSON 提取
def test_extract_plain_json():
    assert extract_json_object('{"style_match": "match"}') == {"style_match": "match"}


def test_extract_json_inside_code_fence():
    text = '```json\n{"style_match": "match", "content_preserved": true, "reason": "ok"}\n```'
    assert extract_json_object(text)["content_preserved"] is True


def test_extract_json_with_surrounding_prose():
    text = 'Here is the result:\n{"style_match": "mismatch", "content_preserved": false, "reason": "no"}\nHope this helps.'
    payload = extract_json_object(text)
    assert payload["style_match"] == "mismatch"


def test_extract_json_with_trailing_text_after_object():
    text = '{"style_match": "match", "content_preserved": true, "reason": "a {b} c"} done'
    assert extract_json_object(text)["style_match"] == "match"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I cannot determine the style.",
        "{style_match: match}",
        '{"style_match": "match",}',
        "{'style_match': 'match'}",
        "[1, 2, 3]",
    ],
)
def test_unparseable_responses_are_technical_errors(text):
    with pytest.raises(CheckTechnicalError):
        extract_json_object(text)


# ---------------------------------------------------------------- CheckResult
def test_check_result_to_dict_contains_pass_flag():
    result = CheckResult(style_match="match", content_preserved=True, reason="ok", attempt=1)
    payload = result.to_dict()
    assert payload["passed"] is True
    assert payload["attempt"] == 1
    assert payload["style_match"] == "match"


def test_check_result_to_dict_keeps_extra_evidence():
    """两阶段描述与对照结果必须落盘，便于事后抽查误判。"""
    result = CheckResult(
        style_match="mismatch",
        content_preserved=False,
        reason="no style",
        extra={
            "mode": "two_stage",
            "original_description": "a house drawing",
            "candidate_description": "an abstract painting",
            "control": {"style_match": "mismatch"},
        },
    )
    payload = result.to_dict()
    assert payload["mode"] == "two_stage"
    assert payload["original_description"] == "a house drawing"
    assert payload["candidate_description"] == "an abstract painting"
    assert payload["control"]["style_match"] == "mismatch"


def test_extra_fields_cannot_override_core_verdict():
    """extra 不允许覆盖核心字段，避免伪造通过。"""
    result = CheckResult(
        style_match="mismatch",
        content_preserved=False,
        reason="no",
        extra={"passed": True, "style_match": "match", "content_preserved": True},
    )
    payload = result.to_dict()
    assert payload["passed"] is False
    assert payload["style_match"] == "mismatch"
    assert payload["content_preserved"] is False


def test_check_result_with_null_content_is_not_passed():
    result = CheckResult(style_match="match", content_preserved=None, reason="unsure")
    assert result.passed is False
