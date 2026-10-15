"""本地视觉模型调用与检查结果解析。

检查模型使用原图与候选图的证据，并清楚区分两张图的角色。
风格化优先；content_preserved 表示主题或可辨认视觉线索的关联，不要求逐物保留。
通过条件极严：style_match 必须为 "match" 且 content_preserved 必须是 JSON 的 true。
字符串 "true"、数字 1、模型自称的置信度一律不转换为通过。
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import requests

from .errors import (
    CheckTechnicalError,
    PipelineError,
    ServiceUnavailableError,
    TimeoutError_,
)

STYLE_MATCH_VALUES = ("match", "mismatch", "uncertain")
REQUIRED_FIELDS = ("style_match", "content_preserved", "reason")
COMPARISON_VERSION = "comparison-v3-style-first"


def model_is_available(requested: str, installed: list[str]) -> bool:
    """精确匹配模型标签；未指定标签时仅接受 latest，不匹配任意变体。"""
    def canonical(name: str) -> str:
        # A registry may contain a port; only a suffix on the model name is a tag.
        return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"

    return canonical(requested) in {canonical(name) for name in installed if isinstance(name, str)}

# Keep visual observation independent of the requested style.
COMPARISON_CONTENT_PROMPT = """This is a comparison board. LEFT panel A is ORIGINAL; RIGHT panel B is CANDIDATE.
Ignore labels and separator. Judge whether the candidate has a recognisable connection to the original's
theme or some of its visual cues. Stylisation takes priority over preserving every element.
The legacy field content_preserved means theme association, not one-to-one object correspondence.
Omitting or merging elements, adding theme-related elements, and changing counts, proportions, positions
or composition are allowed. Materials, colours, outlines and the background may be fully reinterpreted.
Return content_preserved=true when the theme or some recognisable visual cues still connect the images.
Return false when the candidate is wholly unrelated; return null when the evidence is insufficient.
Do not infer a connection merely because a candidate was requested or because both images are paintings.
Return JSON with original_subjects (array), candidate_subjects (array), missing_subjects (array),
added_subjects (array), layout_changed (boolean or null), content_preserved (true/false/null), and reason
(a short non-empty string naming the original theme or cues and the evidence for or against a connection).
The subject lists and layout_changed are diagnostic observations only. Missing or added elements and
layout changes are not rejection criteria. Group parts under their parent object; do not count brushwork
as a new subject. Do not judge painting style."""

COMPARISON_OBSERVATION_PROMPT = """Compare the visible rendering in LEFT panel A (original) and RIGHT panel B (candidate).
Ignore labels and separator. Describe the actual treatment of edges and the INSIDE of shapes in both panels.
Has the way of making marks or constructing forms changed, or have only the colours changed?
Do not guess an art-historical movement. Do not assume the right image was successfully transformed.
Reply ONLY JSON: original_treatment (string), candidate_treatment (string),
transformation (exactly one of none, colour_only, technique, uncertain), reason (string).
Use none if both pictures are unchanged. Use colour_only if the same marks remain and only their hues differ.
Use technique only for a visible change of mark-making, surface treatment or construction of forms."""


def build_comparison_board(original_bytes: bytes, candidate_bytes: bytes) -> tuple[bytes, dict]:
    """建立固定 A/B 单图及可落盘的复现证据；不裁切任一张输入图。"""
    import cv2
    import numpy as np

    images = []
    for role, data in (("original", original_bytes), ("candidate", candidate_bytes)):
        if not isinstance(data, bytes) or not data:
            raise CheckTechnicalError(f"比较图的 {role} 图像为空或类型非法", stage="CHECKING")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise CheckTechnicalError(f"比较图的 {role} 图像无法解码", stage="CHECKING")
        images.append(image)

    panel_w, panel_h, margin, header = 520, 700, 16, 44
    canvas = np.full(
        (panel_h + header + 2 * margin, 2 * panel_w + 3 * margin, 3), 232, np.uint8
    )
    placements = []
    for index, image in enumerate(images):
        ratio = min(panel_w / image.shape[1], panel_h / image.shape[0])
        size = (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio)))
        resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        x = margin + index * (panel_w + margin) + (panel_w - size[0]) // 2
        y = margin + header + (panel_h - size[1]) // 2
        canvas[y:y + size[1], x:x + size[0]] = resized
        label = ("A - ORIGINAL", "B - CANDIDATE")[index]
        cv2.putText(
            canvas, label, (margin + index * (panel_w + margin), 40),
            cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2,
        )
        placements.append({
            "label": label, "source_size": [image.shape[1], image.shape[0]],
            "image_rect": [x, y, size[0], size[1]],
        })
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise CheckTechnicalError("比较图编码失败", stage="CHECKING")
    data = encoded.tobytes()
    return data, {
        "version": "comparison-board-v1", "width": canvas.shape[1], "height": canvas.shape[0],
        "panel_width": panel_w, "panel_height": panel_h, "margin": margin, "header": header,
        "sha256": hashlib.sha256(data).hexdigest(),
        "original_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "panels": placements,
    }


def _comparison_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CheckTechnicalError(f"比较结果 {field_name} 必须是非空字符串", stage="CHECKING")


def _validate_subject_list(value: Any, field_name: str) -> None:
    """接受已实测的字符串/结构化主体；保留原结构，不猜测或强制转换类型。"""
    if not isinstance(value, list):
        raise CheckTechnicalError(f"比较结果 {field_name} 必须是数组", stage="CHECKING")
    identity_keys = {"name", "type", "object_type", "object_name", "shape", "subject", "object", "label"}

    def validate_description(item: Any, path: str) -> None:
        # Auxiliary evidence is not a decision field. The prompt does not constrain
        # its vocabulary or whether colours/positions are strings, arrays or objects.
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float) and math.isfinite(item):
            return
        if isinstance(item, list):
            for nested in item:
                validate_description(nested, path)
            return
        if isinstance(item, dict):
            for key, nested in item.items():
                _comparison_text(key, f"{path} key")
                validate_description(nested, f"{path}.{key}")
            return
        raise CheckTechnicalError(f"比较结果 {path} 必须是有效 JSON 描述", stage="CHECKING")

    for index, subject in enumerate(value):
        path = f"{field_name}[{index}]"
        if isinstance(subject, str):
            _comparison_text(subject, path)
            continue
        if not isinstance(subject, dict) or not identity_keys.intersection(subject):
            raise CheckTechnicalError(f"比较结果 {path} 必须是字符串或带主体标识的对象", stage="CHECKING")
        for key, item in subject.items():
            _comparison_text(key, f"{path} key")
            if key in identity_keys:
                _comparison_text(item, f"{path}.{key}")
            else:
                validate_description(item, f"{path}.{key}")


def _validate_comparison_content(raw_text: str) -> dict:
    payload = extract_json_object(raw_text)
    fields = (
        "original_subjects", "candidate_subjects", "missing_subjects", "added_subjects",
        "layout_changed", "content_preserved", "reason",
    )
    missing = [key for key in fields if key not in payload]
    if missing:
        raise CheckTechnicalError(f"比较结果缺少字段: {', '.join(missing)}", stage="CHECKING")
    for key in fields[:4]:
        _validate_subject_list(payload[key], key)
    for key in ("layout_changed", "content_preserved"):
        if payload[key] is not None and type(payload[key]) is not bool:
            raise CheckTechnicalError(f"比较结果 {key} 必须是布尔值或 null", stage="CHECKING")
    _comparison_text(payload["reason"], "reason")
    return {key: payload[key] for key in fields}


def _validate_comparison_observation(raw_text: str) -> dict:
    payload = extract_json_object(raw_text)
    fields = ("original_treatment", "candidate_treatment", "transformation", "reason")
    for key in fields:
        if key not in payload:
            raise CheckTechnicalError(f"渲染观察缺少字段: {key}", stage="CHECKING")
        _comparison_text(payload[key], key)
    if payload["transformation"] not in ("none", "colour_only", "technique", "uncertain"):
        raise CheckTechnicalError("渲染观察 transformation 枚举非法", stage="CHECKING")
    return {key: payload[key] for key in fields}


@dataclass
class CheckResult:
    """一次合法检查结果的解析产物。"""

    style_match: str
    content_preserved: bool | None
    reason: str
    raw_text: str = ""
    attempt: int | None = None
    duration_sec: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """严格通过条件：match 且 content_preserved is True。"""
        return self.style_match == "match" and self.content_preserved is True

    def to_dict(self) -> dict:
        payload = {
            "style_match": self.style_match,
            "content_preserved": self.content_preserved,
            "reason": self.reason,
            "passed": self.passed,
        }
        if self.attempt is not None:
            payload["attempt"] = self.attempt
        if self.duration_sec is not None:
            payload["duration_sec"] = round(self.duration_sec, 3)
        # 保留额外判定依据（两阶段描述、对照结果、模型多余字段），便于事后抽查误判
        for key, value in self.extra.items():
            if key not in payload:
                payload[key] = value
        return payload


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_object(text: str) -> dict:
    """从模型文本中取出 JSON 对象。

    依次尝试：整体解析 → 去掉 Markdown 代码围栏 → 扫描第一个完整 JSON 对象。
    全部失败视为技术错误。
    """
    if not isinstance(text, str) or not text.strip():
        raise CheckTechnicalError("视觉模型返回空响应", stage="CHECKING")

    candidates: list[str] = [text.strip(), _strip_code_fence(text)]

    decoder = json.JSONDecoder()
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

    # 扫描文本中第一个可解析的 JSON 对象
    for candidate in candidates:
        index = 0
        while True:
            start = candidate.find("{", index)
            if start < 0:
                break
            try:
                parsed, end = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                index = start + 1
                continue
            if isinstance(parsed, dict):
                return parsed
            index = start + 1

    raise CheckTechnicalError(
        f"视觉模型响应不是合法 JSON 对象: {text[:300]}", stage="CHECKING"
    )


def validate_check_payload(payload: Any) -> CheckResult:
    """校验检查 JSON 的字段与值；任何非法情况都视为技术错误。"""
    if not isinstance(payload, dict):
        raise CheckTechnicalError(
            f"检查结果必须是 JSON 对象，实际为 {type(payload).__name__}", stage="CHECKING"
        )

    missing = [name for name in REQUIRED_FIELDS if name not in payload]
    if missing:
        raise CheckTechnicalError(
            f"检查结果缺少字段: {', '.join(missing)}", stage="CHECKING"
        )

    style_match = payload["style_match"]
    if not isinstance(style_match, str):
        raise CheckTechnicalError(
            f"style_match 必须是字符串，实际为 {type(style_match).__name__}", stage="CHECKING"
        )
    if style_match not in STYLE_MATCH_VALUES:
        raise CheckTechnicalError(
            f"style_match 取值非法: {style_match!r}（仅接受 {', '.join(STYLE_MATCH_VALUES)}）",
            stage="CHECKING",
        )

    content_preserved = payload["content_preserved"]
    if content_preserved is not None and not isinstance(content_preserved, bool):
        # 不接受字符串 "true"、数字 1 等
        raise CheckTechnicalError(
            "content_preserved 必须是 JSON 的 true、false 或 null，实际为 "
            f"{type(content_preserved).__name__}({content_preserved!r})",
            stage="CHECKING",
        )

    reason = payload["reason"]
    if not isinstance(reason, str):
        raise CheckTechnicalError(
            f"reason 必须是字符串，实际为 {type(reason).__name__}", stage="CHECKING"
        )

    extra = {
        key: value
        for key, value in payload.items()
        if key not in REQUIRED_FIELDS
    }
    return CheckResult(
        style_match=style_match,
        content_preserved=content_preserved,
        reason=reason,
        extra=extra,
    )


def _description_sections(description: str) -> dict[str, str]:
    """提取带标签的描述；兼容空行、多行列表及 Markdown 标题。"""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in description.splitlines():
        # 标签可为 **SUBJECTS:** 或 ### SUBJECTS:；不改写正文的对象描述。
        label_line = line.strip().lstrip("#-* ")
        match = re.match(
            r"^(STYLE|SUBJECTS|LAYOUT|COLOURS)\s*\*{0,2}\s*:\s*\*{0,2}(.*)$",
            label_line, re.IGNORECASE,
        )
        if match:
            current = match.group(1).upper()
            sections.setdefault(current, []).append(match.group(2).strip())
        elif current:
            sections[current].append(line.strip())
    result = {key: "\n".join(lines).strip() for key, lines in sections.items()}
    missing = [key for key in ("STYLE", "SUBJECTS", "LAYOUT") if not result.get(key)]
    if missing:
        raise CheckTechnicalError(
            f"图像描述缺少有效标签内容: {', '.join(missing)}", stage="CHECKING"
        )
    return result


def _validate_axis_payload(raw_text: str, axis: str) -> dict:
    """复用严格字段校验，但每次调用只接纳其负责的判定轴。"""
    payload = extract_json_object(raw_text)
    missing = [key for key in (axis, "reason") if key not in payload]
    if missing:
        raise CheckTechnicalError(
            f"检查结果缺少字段: {', '.join(missing)}", stage="CHECKING"
        )
    combined = {
        "style_match": "uncertain",
        "content_preserved": None,
        "reason": payload["reason"],
        axis: payload[axis],
    }
    validate_check_payload(combined)
    # 保留模型原始响应供抽查，但另一判定轴的多余字段绝不参与最终合并。
    return {axis: payload[axis], "reason": payload["reason"], "raw": raw_text}


class VisionChecker:
    """通过本地 Ollama 兼容服务调用视觉语言模型。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api: str = "chat",
        connect_timeout: float = 5.0,
        request_timeout: float = 300.0,
        keep_alive: str | None = None,
        options: dict | None = None,
        session: requests.Session | None = None,
        mode: str = "two_stage",
        control: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api = api
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self.keep_alive = keep_alive
        self.options = options or {}
        self.session = session or requests.Session()
        if mode not in ("two_stage", "pairwise", "comparison"):
            raise CheckTechnicalError(f"未知检查方式: {mode}", stage="CHECKING")
        self.mode = mode
        self.control = control

    # ---- 健康检查 ----
    def list_models(self) -> list[str]:
        url = f"{self.base_url}/api/tags"
        try:
            response = self.session.get(url, timeout=self.connect_timeout)
        except requests.RequestException as exc:
            raise ServiceUnavailableError(
                f"本地视觉模型服务不可用: {url} ({exc})", stage="CHECKING"
            ) from exc
        if response.status_code != 200:
            raise ServiceUnavailableError(
                f"本地视觉模型服务 /api/tags 返回 {response.status_code}", stage="CHECKING"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ServiceUnavailableError("本地视觉模型服务响应不是 JSON", stage="CHECKING") from exc
        return [m.get("name", "") for m in payload.get("models", []) if isinstance(m, dict)]

    def verify_model(self) -> None:
        names = self.list_models()
        if model_is_available(self.model, names):
            return
        raise ServiceUnavailableError(
            f"本地视觉模型 '{self.model}' 未安装。当前可用: {names or '（无）'}",
            stage="CHECKING",
        )

    # ---- 调用 ----
    def _encode(self, data: bytes) -> str:
        if not data:
            raise CheckTechnicalError("传给视觉模型的图像为空", stage="CHECKING")
        return base64.b64encode(data).decode("ascii")

    def _post(self, endpoint: str, body: dict) -> dict:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.post(url, json=body, timeout=self.request_timeout)
        except requests.Timeout as exc:
            raise TimeoutError_(
                f"视觉模型调用超时（{self.request_timeout}s）", stage="CHECKING"
            ) from exc
        except requests.RequestException as exc:
            raise ServiceUnavailableError(
                f"视觉模型服务不可用: {url} ({exc})", stage="CHECKING"
            ) from exc
        if response.status_code != 200:
            raise ServiceUnavailableError(
                f"视觉模型服务 {endpoint} 返回 {response.status_code}: {response.text[:300]}",
                stage="CHECKING",
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ServiceUnavailableError(
                f"视觉模型服务 {endpoint} 响应不是 JSON", stage="CHECKING"
            ) from exc

    def _call_chat(self, prompt: str, images: list[bytes]) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [self._encode(img) for img in images],
                }
            ],
            "stream": False,
            "options": self.options,
        }
        if self.keep_alive is not None:
            body["keep_alive"] = self.keep_alive
        payload = self._post("/api/chat", body)
        message = payload.get("message")
        if payload.get("done_reason") == "length":
            self._raise_truncated_response(
                payload, message.get("content") if isinstance(message, dict) else None
            )
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise CheckTechnicalError(
                f"视觉模型 chat 响应缺少 message.content: {str(payload)[:300]}",
                stage="CHECKING",
            )
        return message["content"]

    def _call_generate(self, prompt: str, images: list[bytes]) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "images": [self._encode(img) for img in images],
            "stream": False,
            "options": self.options,
        }
        if self.keep_alive is not None:
            body["keep_alive"] = self.keep_alive
        payload = self._post("/api/generate", body)
        response_text = payload.get("response")
        if payload.get("done_reason") == "length":
            self._raise_truncated_response(payload, response_text)
        if not isinstance(response_text, str):
            raise CheckTechnicalError(
                f"视觉模型 generate 响应缺少 response 字段: {str(payload)[:300]}",
                stage="CHECKING",
            )
        return response_text

    @staticmethod
    def _raise_truncated_response(payload: dict, response_text: Any) -> None:
        """保留截断证据；只取允许字段，避免存入图像、thinking 或请求内容。"""
        details = {}
        if isinstance(response_text, str):
            details["raw_response"] = response_text
        for key in ("done_reason", "model"):
            if isinstance(payload.get(key), str):
                details[key] = payload[key]
        for key in ("eval_count", "prompt_eval_count"):
            if type(payload.get(key)) is int:
                details[key] = payload[key]
        raise CheckTechnicalError(
            "视觉模型响应达到长度上限，不能使用截断的检查依据",
            stage="CHECKING",
            details=details,
        )

    def _call(self, prompt: str, images: list[bytes]) -> str:
        return (
            self._call_chat(prompt, images)
            if self.api == "chat"
            else self._call_generate(prompt, images)
        )

    # ---- 多图能力自检 ----
    def verify_multi_image_support(self, probe_bytes: bytes) -> dict:
        """自检视觉模型能否真正同时比较两张图。

        做法：把同一张图分别作为「第一张」和「第二张」提交，并附带一张不同的
        对照图。若模型在 A/A 和 A/B 情形下不能正确区分，双图检查不可信；
        此测试不能确定模型内部忽略了哪张图。

        返回包含 supported / aa_answer / ab_answer 的字典，供 doctor 与配置校验使用。
        """
        import cv2
        import numpy as np

        array = cv2.imdecode(np.frombuffer(probe_bytes, np.uint8), cv2.IMREAD_COLOR)
        if array is None:
            raise CheckTechnicalError("多图自检探针图像无法解码", stage="CHECKING")
        different = cv2.bitwise_not(array)
        ok, buffer = cv2.imencode(".png", different)
        if not ok:
            raise CheckTechnicalError("多图自检对照图编码失败", stage="CHECKING")
        other_bytes = buffer.tobytes()

        question = (
            "You are given two images: IMAGE ONE and IMAGE TWO. "
            "Answer with a single word first: SAME if they show the same picture, "
            "DIFFERENT if they show different pictures."
        )
        aa = self._call(question, [probe_bytes, probe_bytes]).strip().upper()
        ab = self._call(question, [probe_bytes, other_bytes]).strip().upper()
        aa_says_same = aa.startswith("SAME") or aa.startswith("IDENTICAL")
        ab_says_different = ab.startswith("DIFFERENT")
        return {
            "supported": bool(aa_says_same and ab_says_different),
            "aa_answer": aa[:80],
            "ab_answer": ab[:80],
        }

    def _judge_descriptions(
        self,
        original_desc: str,
        candidate_desc: str,
        style_name: str,
        style_features: str,
        preservation_rules: str,
    ) -> CheckResult:
        """分别判定主题关联与风格，避免一个判定轴替代另一个判定轴。"""
        original = _description_sections(original_desc)
        candidate = _description_sections(candidate_desc)

        def evidence(parts: dict[str, str], fields: tuple[str, ...]) -> str:
            return "\n".join(f"{key}: {parts[key]}" for key in fields if parts.get(key))

        content_prompt = (
            "Compare ONLY the theme and recognisable visual cues in these two image "
            "descriptions. Do not evaluate painting style, colour or surface treatment.\n"
            f"Theme association guidance: {preservation_rules}\n\n"
            "ORIGINAL content evidence:\n" + evidence(original, ("SUBJECTS", "LAYOUT"))
            + "\n\nCANDIDATE content evidence:\n"
            + evidence(candidate, ("SUBJECTS", "LAYOUT")) + "\n\n"
            "Stylisation takes priority over preserving every element. The legacy field "
            "content_preserved means theme association, not one-to-one object correspondence. "
            "Omitting or merging elements, adding theme-related elements, changing counts, "
            "proportions, positions or composition, and repainting the material or background "
            "are allowed. Do not reject merely for any such changes.\n"
            "Return true when the original theme or some recognisable visual cues connect "
            "the images. Return false when the candidate is wholly unrelated. Return null "
            "when the descriptions do not provide enough evidence to decide. A detail omitted "
            "from a description is not proof that it is absent from the image. Do not infer "
            "a connection merely because both images are paintings.\n"
            "Reply with exactly one JSON object containing content_preserved (JSON true, "
            "false or null, never a string) and reason (a short string identifying the "
            "theme or cues and the evidence for or against a connection)."
        )
        content_raw = self._call(content_prompt, []).strip()
        content = _validate_axis_payload(content_raw, "content_preserved")

        style_prompt = (
            "Evaluate ONLY the candidate's visible painting style from these image "
            "descriptions. Object identity and content preservation cannot establish a "
            "style match.\n"
            f"Requested style: {style_name}\n"
            f"Expected visible features: {style_features}\n\n"
            "ORIGINAL style evidence:\n" + evidence(original, ("STYLE", "COLOURS"))
            + "\n\nCANDIDATE style evidence:\n"
            + evidence(candidate, ("STYLE", "COLOURS")) + "\n\n"
            "Return match only when the candidate evidence clearly supports the requested "
            "painting features and a stylistic transformation from the original. Bold "
            "colours or geometric subject identities alone do not establish an art "
            "movement. Return mismatch for clearly different painting features or no "
            "visible stylistic transformation, including unchanged marker rendering or "
            "a mere hue change with the same marks. "
            "Return uncertain if the description does not provide enough evidence. "
            "Do not assume an image has the requested style merely because it was "
            "requested.\n"
            "Reply with exactly one JSON object containing style_match (exactly the JSON "
            "string match, mismatch or uncertain) and reason (a short string naming the "
            "visible stylistic evidence). Do not judge subject preservation."
        )
        style_raw = self._call(style_prompt, []).strip()
        style = _validate_axis_payload(style_raw, "style_match")
        result = validate_check_payload({
            "content_preserved": content["content_preserved"],
            "style_match": style["style_match"],
            "reason": f"Content: {content['reason']} Style: {style['reason']}",
        })
        result.raw_text = json.dumps(
            {"content_judgement": content_raw, "style_judgement": style_raw},
            ensure_ascii=False,
        )
        result.extra.update({
            "content_judgement": content,
            "style_judgement": style,
        })
        return result

    def check_comparison(
        self,
        original_bytes: bytes,
        candidate_bytes: bytes,
        style_name: str,
        style_features: str,
        attempt: int | None = None,
        duration_sec: float | None = None,
    ) -> CheckResult:
        """单张 A/B 图分别检查主题关联和渲染变化，再独立粗判画派。"""
        board, board_metadata = build_comparison_board(original_bytes, candidate_bytes)
        raw: dict[str, str] = {}
        prompts: dict[str, str] = {}
        component = "content"
        try:
            prompts[component] = COMPARISON_CONTENT_PROMPT
            raw[component] = self._call(COMPARISON_CONTENT_PROMPT, [board])
            content = _validate_comparison_content(raw[component])
            component = "observation"
            prompts[component] = COMPARISON_OBSERVATION_PROMPT
            raw[component] = self._call(COMPARISON_OBSERVATION_PROMPT, [board])
            observation = _validate_comparison_observation(raw[component])
            style_prompt = f"""Do a coarse style-feature check using these independently observed rendering facts.
Requested style: {style_name}. Expected visual features: {style_features}.
Observations: {json.dumps(observation)}
Do not infer an absent feature from the requested style name. Match only if the observed candidate
technique clearly supports the requested style. A mere hue change or unchanged rendering is mismatch.
Use uncertain when the facts do not support a decision. This is not an art-historical authentication.
Return JSON with style_match (match/mismatch/uncertain) and reason (string)."""
            component = "style"
            prompts[component] = style_prompt
            raw[component] = self._call(style_prompt, [])
            style = _validate_axis_payload(raw[component], "style_match")
            _comparison_text(style["reason"], "reason")
        except PipelineError as exc:
            # Preserve the original technical-error category, including transport
            # failures, plus completed evidence and every submitted prompt.
            exc.details = {
                **(exc.details or {}), "mode": "comparison",
                "comparison_version": COMPARISON_VERSION, "comparison_component": component,
                "comparison_board": board_metadata, "comparison_raw": dict(raw),
                "comparison_prompts": dict(prompts),
            }
            raise

        preserved = content["content_preserved"]
        verdict = style["style_match"]
        guards = []
        # Element inventories and layout changes remain audit evidence; the content
        # verdict now measures theme association rather than exact preservation.
        transformation = observation["transformation"]
        if transformation in ("none", "colour_only", "uncertain"):
            verdict = "uncertain" if transformation == "uncertain" else "mismatch"
            guards.append(f"Style guard: transformation={transformation}; style_match={verdict}.")
        reason = (
            f"Content: {content['reason']} Style: {style['reason']} "
            f"Rendering: {observation['reason']}"
        )
        if guards:
            reason += " " + " ".join(guards)
        result = validate_check_payload({
            "content_preserved": preserved, "style_match": verdict, "reason": reason,
        })
        result.raw_text = json.dumps(raw, ensure_ascii=False)
        result.attempt = attempt
        result.duration_sec = duration_sec
        result.extra.update({
            "mode": "comparison", "comparison_version": COMPARISON_VERSION,
            "comparison_board": board_metadata,
            "comparison_content": content, "comparison_observation": observation,
            "style_judgement": style, "comparison_raw": raw, "verdict_guards": guards,
            "comparison_prompts": dict(prompts),
        })
        return result

    # ---- 两阶段检查（兼容路径）----
    def check_two_stage(
        self,
        original_bytes: bytes,
        candidate_bytes: bytes,
        style_name: str,
        style_features: str,
        preservation_rules: str,
        attempt: int | None = None,
        duration_sec: float | None = None,
        control: bool = False,
    ) -> CheckResult:
        """先分别描述两张图，再基于描述判断风格与主题关联。

        本机模型的双图能力自检未通过，因此分别读取图像，再独立判定两个轴。
        此方法依赖单图描述的准确性；描述仍可能遗漏或误认可见内容。
        """
        describe = (
            "Describe only visible evidence in this image in at most 150 words. "
            "Use exactly these four labelled lines, one per line:\n"
            "STYLE: separately describe the visible treatment of outer boundaries, "
            "interior fills and background. State the actual marks and surface texture "
            "you can see; do not infer the whole medium from outlines alone or guess an "
            "art-historical school.\n"
            "SUBJECTS: describe the visible subject matter, theme and recognisable visual "
            "cues across the whole image. Group parts under their parent object. Put "
            "interior patterns and colour details under STYLE and COLOURS; do not treat "
            "decorative marks as new objects.\n"
            "LAYOUT: briefly describe the overall arrangement and relationships between "
            "the visible forms, without requiring an exhaustive object inventory.\n"
            "COLOURS: briefly list the main colours. Do not invent unseen objects."
        )
        original_desc = self._call(describe, [original_bytes]).strip()
        candidate_desc = self._call(describe, [candidate_bytes]).strip()
        if not original_desc or not candidate_desc:
            raise CheckTechnicalError("视觉模型未能给出图像描述", stage="CHECKING")

        result = self._judge_descriptions(
            original_desc, candidate_desc, style_name, style_features, preservation_rules,
        )

        # Optional diagnostic control: evaluate the original as its own candidate.
        # This has the same independent judgement path as the real candidate.
        control_result: dict | None = None
        if control:
            control_check = self._judge_descriptions(
                original_desc, original_desc, style_name, style_features, preservation_rules,
            )
            control_result = control_check.to_dict()

        result.attempt = attempt
        result.duration_sec = duration_sec
        result.extra.update(
            {
                "mode": "two_stage",
                "original_description": original_desc,
                "candidate_description": candidate_desc,
            }
        )
        if control_result is not None:
            result.extra["control"] = control_result
        return result

    def check_pair(
        self,
        original_bytes: bytes,
        candidate_bytes: bytes,
        prompt: str,
        style_name: str,
        style_features: str,
        preservation_rules: str,
        attempt: int | None = None,
        duration_sec: float | None = None,
    ) -> CheckResult:
        """按配置执行 comparison、two_stage 或 pairwise 检查。"""
        if self.mode == "comparison":
            return self.check_comparison(
                original_bytes=original_bytes,
                candidate_bytes=candidate_bytes,
                style_name=style_name,
                style_features=style_features,
                attempt=attempt,
                duration_sec=duration_sec,
            )
        if self.mode == "two_stage":
            return self.check_two_stage(
                original_bytes=original_bytes,
                candidate_bytes=candidate_bytes,
                style_name=style_name,
                style_features=style_features,
                preservation_rules=preservation_rules,
                attempt=attempt,
                duration_sec=duration_sec,
                control=self.control,
            )
        return self.check(
            original_bytes=original_bytes,
            candidate_bytes=candidate_bytes,
            prompt=prompt,
            attempt=attempt,
            duration_sec=duration_sec,
        )

    def check(
        self,
        original_bytes: bytes,
        candidate_bytes: bytes,
        prompt: str,
        attempt: int | None = None,
        duration_sec: float | None = None,
    ) -> CheckResult:
        """同时提交原图与候选图，返回解析后的检查结果。

        图像顺序固定为「原图、候选图」，与检查 Prompt 中的说明一致。
        """
        raw_text = self._call(prompt, [original_bytes, candidate_bytes])
        payload = extract_json_object(raw_text)
        result = validate_check_payload(payload)
        result.raw_text = raw_text
        result.attempt = attempt
        result.duration_sec = duration_sec
        result.extra["mode"] = "pairwise"
        return result
