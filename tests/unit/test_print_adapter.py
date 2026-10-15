"""打印占位接口测试。

要求：默认流程不调用打印接口；单独调用只返回 not_connected；
不导入打印驱动、不访问打印机、不提交系统打印任务，也没有打印副作用。
"""

from __future__ import annotations

import sys

from app import print_adapter
from app.print_adapter import export_to_print, is_print_enabled

FORBIDDEN_MODULES = ("win32print", "win32api", "cups", "pyprint", "subprocess")


def test_placeholder_returns_not_connected():
    result = export_to_print("job-123", "runs/job-123/output.png")
    assert result["job_id"] == "job-123"
    assert result["status"] == "not_connected"
    assert isinstance(result["message"], str) and result["message"]
    assert result["print_enabled"] is False


def test_placeholder_accepts_optional_profile():
    result = export_to_print("job-1", "out.png", profile={"paper": "A4", "dpi": 300})
    assert result["status"] == "not_connected"
    assert result["profile"] == {"paper": "A4", "dpi": 300}


def test_placeholder_does_not_require_existing_file():
    """占位接口不做真实打印，因此不应因文件不存在而抛错。"""
    result = export_to_print("job-404", "does/not/exist.png")
    assert result["status"] == "not_connected"


def test_placeholder_does_not_import_print_drivers():
    for name in FORBIDDEN_MODULES:
        assert name not in sys.modules or name == "subprocess", (
            f"打印占位接口不得导入 {name}"
        )


def test_placeholder_never_reports_printed_status():
    result = export_to_print("job-9", "x.png")
    assert result["status"] != "printed"
    assert "printed" not in str(result["status"]).lower()


def test_is_print_enabled_reads_config_flag():
    class Cfg:
        print_enabled = False

    assert is_print_enabled(Cfg()) is False

    class CfgOn:
        print_enabled = True

    assert is_print_enabled(CfgOn()) is True


def test_default_config_keeps_print_disabled():
    from pathlib import Path

    from app.config import load_app_config

    root = Path(__file__).resolve().parents[2]
    config = load_app_config(root)
    assert config.print_enabled is False, "本轮默认 print_enabled 必须为 false"


def test_normal_pipeline_never_calls_print(pipeline_env, monkeypatch):
    """默认流程结束后不得调用打印接口。"""
    calls: list = []
    monkeypatch.setattr(
        print_adapter, "export_to_print", lambda *a, **k: calls.append((a, k))
    )

    pipeline, store, _, _, _ = pipeline_env(
        [{"style_match": "match", "content_preserved": True, "reason": "ok"}]
    )
    from tests.fixtures import marker_drawing_png

    summary = pipeline.run(marker_drawing_png(), "photo.jpg")
    assert summary["state"] == "COMPLETED"
    assert calls == [], "正常运行不得调用打印接口"


def test_pipeline_does_not_import_print_adapter_at_runtime():
    """主流程模块不导入打印适配器，避免任何隐式打印副作用。"""
    import app.pipeline as pipeline_module

    source = open(pipeline_module.__file__, encoding="utf-8").read()
    assert "print_adapter" not in source
    assert "export_to_print" not in source
