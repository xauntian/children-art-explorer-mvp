"""AI timeout is a software notice, never a fabricated image or print job."""

from __future__ import annotations

from unittest.mock import Mock

from app.gui import StudioWindow
from tests.unit.test_gui_runtime import (
    ready,
    take_events,
    wait_worker,
)
from tests.unit.test_gui_runtime import (
    runtime_env as runtime_env,
)
from tests.unit.test_gui_window import window as window


def test_ai_timeout_keeps_capture_results_and_reactivates_p_without_saving_or_printing(runtime_env):
    env = runtime_env
    runtime = ready(env)
    take_events(runtime)
    paths = [env.session.directory / name for name in ("preview.png", "artwork.png")]
    original_bytes = {path: path.read_bytes() for path in [*paths, env.output]}

    def timeout(session):
        assert session is env.session
        env.factory.call_args.kwargs["observer"]("generating", session)
        return {
            "state": "FAILED",
            # Even a stale path must never be mistaken for this round's AI result.
            "output_path": str(env.output),
            "error": {"category": "timeout", "message": "generation exceeded 10 seconds"},
        }

    env.controller.generate.side_effect = timeout
    assert runtime.trigger() is True
    wait_worker(runtime)

    assert runtime.state == "READY"
    assert runtime.pending is None
    assert not runtime.pending_path.exists()
    env.printer.print_image.assert_not_called()
    env.printer.wait_for_job.assert_not_called()
    events = take_events(runtime)
    assert {data["key"] for kind, data in events if kind == "image"} == {
        "original", "rectified",
    }
    assert any(
        kind == "ai_timeout" and "没有完整 AI 图片" in data["message"]
        for kind, data in events
    )
    assert any(
        kind == "stage" and data["step"] == "generate" and data["status"] == "error"
        for kind, data in events
    )
    assert not any(
        kind == "stage" and data["step"] in {"save", "print"} for kind, data in events
    )
    assert any(kind == "completed" for kind, _data in events)
    assert {path: path.read_bytes() for path in original_bytes} == original_bytes

    assert runtime.trigger() is True
    wait_worker(runtime)
    assert env.controller.prepare.call_count == 2
    assert env.controller.generate.call_count == 2
    assert runtime.state == "READY"
    env.printer.print_image.assert_not_called()


def test_timeout_notice_removes_stale_ai_preview_and_preserves_other_images(window, monkeypatch):
    from PIL import Image, ImageDraw

    save = Mock()
    draw = Mock()
    monkeypatch.setattr(Image.Image, "save", save)
    monkeypatch.setattr(ImageDraw, "Draw", draw)
    window.image_paths.update(
        original="original.png", rectified="artwork.png", ai="previous-ai.png",
    )
    message = "AI 生成超时，本轮没有完整 AI 图片"

    window._handle("ai_timeout", {"message": message})

    assert window.image_paths == {"original": "original.png", "rectified": "artwork.png"}
    assert window.ai_notice == message
    assert "可按 P" in window.subtitles["ai"].options["text"]
    window._render_image.assert_called_once_with("ai")
    assert window.panels["original"].items == []
    assert window.panels["rectified"].items == []
    save.assert_not_called()
    draw.assert_not_called()


def test_timeout_notice_survives_ai_panel_redraw_without_loading_or_creating_an_image(
    window, monkeypatch,
):
    from PIL import Image, ImageTk

    open_image = Mock()
    save = Mock()
    photo = Mock()
    monkeypatch.setattr(Image, "open", open_image)
    monkeypatch.setattr(Image.Image, "save", save)
    monkeypatch.setattr(ImageTk, "PhotoImage", photo)
    message = "AI 生成超时，本轮没有完整 AI 图片"
    window._handle("ai_timeout", {"message": message})

    for _ in range(2):
        StudioWindow._render_image(window, "ai")
        items = window.panels["ai"].items
        assert len(items) == 1
        assert items[0][0] == "text"
        assert items[0][2]["text"] == message

    assert "ai" not in window.image_paths
    open_image.assert_not_called()
    save.assert_not_called()
    photo.assert_not_called()


def test_next_round_reset_clears_timeout_notice_and_restores_default_subtitle(window):
    window._handle("ai_timeout", {"message": "本轮 AI 超时"})
    window._render_image.reset_mock()

    window._handle("reset", {})

    assert not window.ai_notice
    assert window.image_paths == {}
    assert window.subtitles["ai"].options["text"] == "随机画派 · A4 比例输出"
    assert {item.args[0] for item in window._render_image.call_args_list} == {
        "original", "rectified", "ai",
    }


def test_successful_ai_result_clears_previous_timeout_notice_and_updates_subtitle(window):
    window._handle("ai_timeout", {"message": "本轮 AI 超时"})
    window._render_image.reset_mock()

    window._handle("image", {"key": "ai", "path": "new-ai.png"})

    assert not window.ai_notice
    assert window.image_paths["ai"] == "new-ai.png"
    assert "AI 图片已保存" in window.subtitles["ai"].options["text"]
    window._render_image.assert_called_once_with("ai")
