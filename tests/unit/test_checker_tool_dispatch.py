"""Checker audit tools must exercise the configured production dispatch path."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.checker import CheckResult
from app.config import load_app_config
from tools import checker_audit, compare_checkers


@pytest.mark.parametrize("mode", ["two_stage", "pairwise", "comparison"])
@pytest.mark.parametrize("module", [checker_audit, compare_checkers])
def test_tools_route_through_check_pair_with_actual_prompt(monkeypatch, tmp_path, mode, module):
    config = load_app_config(module.ROOT)
    config.checker_mode = mode
    config.checker_control = True
    config.load_prompts = lambda: ("", "style={style_name}; features={style_features}; rules={preservation_rules}")
    monkeypatch.setattr(module, "load_app_config", lambda _: config)
    image = np.full((64, 64, 3), 245, dtype=np.uint8)
    monkeypatch.setattr(module, "load_image", lambda _: image)
    monkeypatch.setattr(module, "preprocess", lambda *args, **kwargs: SimpleNamespace(image=image))
    calls = []
    constructors = []

    class Checker:
        def __init__(self, **kwargs):
            constructors.append(kwargs)

        def verify_model(self):
            pass

        def check_pair(self, **kwargs):
            calls.append(kwargs)
            return CheckResult("mismatch", True, "unchanged treatment")

        def check_two_stage(self, *args, **kwargs):
            raise AssertionError("must not bypass configured dispatch")

        def check(self, *args, **kwargs):
            raise AssertionError("must not bypass configured dispatch")

    monkeypatch.setattr(module, "VisionChecker", Checker)
    if module is checker_audit:
        argv = ["--photo", "input.jpg", "--style", "fauvism", "--out", str(tmp_path / "audit")]
        report_path = tmp_path / "audit/input_fauvism_audit.json"
        expected_calls = 4
    else:
        candidate = tmp_path / "candidate.png"
        candidate.write_bytes(b"test-candidate")
        report_path = tmp_path / "compare.json"
        argv = ["--original", "input.jpg", "--candidates", str(candidate), "--style", "fauvism",
                "--models", "model-a,model-b", "--control", "--out", str(report_path)]
        expected_calls = 2
    assert module.main(argv) == 0
    assert len(calls) == expected_calls
    style = config.load_styles().resolve("fauvism")
    expected_prompt = f"style={style.name}; features={style.style_features}; rules={style.preservation_rules}"
    for call in calls:
        assert call["prompt"] == expected_prompt
        assert call["style_name"] == style.name
        assert call["preservation_rules"] == style.preservation_rules
        assert "control" not in call  # Configured on the checker, not on its dispatch call.
    assert all(item["mode"] == mode and item["control"] is True for item in constructors)
    records = json.loads(report_path.read_text(encoding="utf-8"))
    assert all(item["mode"] == mode for item in records)
