"""评估报告必须可追溯且不能覆盖原始证据；不调用任何模型服务。"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import load_app_config
from tools import evaluate_styles as evaluation


def test_existing_report_is_rejected_before_app_construction(tmp_path, monkeypatch):
    report = tmp_path / "report.md"
    report.write_text("historical evidence", encoding="utf-8")

    def must_not_construct_app():
        pytest.fail("an existing report must be rejected before creating App")

    monkeypatch.setattr(evaluation, "App", must_not_construct_app)
    with pytest.raises(SystemExit) as exc:
        evaluation.main(["--inputs", str(tmp_path), "--report", str(report)])
    assert exc.value.code == 2
    assert report.read_text(encoding="utf-8") == "historical evidence"


def test_report_formats_must_not_share_a_path(tmp_path):
    path = tmp_path / "report"
    with pytest.raises(ValueError, match="不同路径"):
        evaluation.validate_report_paths([path, path.parent / "." / path.name])


def test_report_created_after_preflight_is_not_overwritten(tmp_path):
    path = tmp_path / "report.json"
    evaluation.validate_report_paths([path])
    path.write_text("earlier run", encoding="utf-8")
    with pytest.raises(FileExistsError):
        evaluation.write_new_report(path, "new run")
    assert path.read_text(encoding="utf-8") == "earlier run"


@pytest.mark.parametrize("uses_unet", [True, False])
def test_report_identifies_model_used_by_workflow(tmp_path, uses_unet):
    config = load_app_config(Path(__file__).resolve().parents[2])
    app = SimpleNamespace(
        config=config,
        generator=SimpleNamespace(uses_unet_loader=lambda: uses_unet),
    )
    source = tmp_path / "input.jpg"
    source.write_bytes(b"test image bytes")
    metadata = evaluation.evaluation_metadata(app, [source], "development", "synthetic")
    expected = config.comfy_unet_name if uses_unet else config.comfy_checkpoint
    assert metadata["generation_model"] == expected
    assert metadata["checkpoint"] == (None if uses_unet else expected)
    assert metadata["unet_name"] == (expected if uses_unet else None)
    assert len(metadata["workflow_sha256"]) == 64
    assert metadata["dataset"]["role"] == "development"
    assert len(metadata["dataset"]["inputs"][0]["sha256"]) == 64
    assert metadata["dataset"]["independence"] == "not_verified"


def test_report_shows_generator_and_dataset_role_without_claiming_validation():
    summary = {
        "evaluated_at": "2026-10-14T00:00:00Z", "inputs_dir": "samples/heldout",
        "per_style": 5, "total_duration_sec": 10, "checkpoint": None,
        "generation_model": "Qwen-edit.gguf", "workflow_file": "qwen.json",
        "controlnet": None, "checker_model": "local-vlm", "checker_mode": "two_stage",
        "max_input_side": 768, "styles": {}, "records": [], "print_placeholder": {},
        "dataset": {"role": "development", "note": "already used for tuning"},
    }
    report = evaluation.render_report(summary)
    assert "生成模型：`Qwen-edit.gguf`" in report
    assert "输入用途：`development`" in report
    assert "独立性未经脚本验证" in report
    assert "already used for tuning" in report


def test_negative_input_limit_is_rejected_before_app_construction(monkeypatch):
    monkeypatch.setattr(evaluation, "App", lambda: pytest.fail("must not construct App"))
    with pytest.raises(SystemExit) as exc:
        evaluation.main(["--inputs", "samples/debug", "--per-style", "-1"])
    assert exc.value.code == 2
