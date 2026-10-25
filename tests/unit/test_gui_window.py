"""桌面窗口行为测试：模拟 root / widgets / runtime，不启动设备或打印。"""

from __future__ import annotations

import queue
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.gui import RED, STEPS, StudioWindow, hong_kong_time
from app.gui_video import PreviewFrame


class Variable:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class Widget:
    def __init__(self, **options):
        self.options = dict(options)
        self.visible = False
        self.items = []

    def configure(self, **options):
        self.options.update(options)

    def pack(self, **options):
        self.visible = True
        self.pack_options = options

    def pack_forget(self):
        self.visible = False

    def delete(self, *args):
        self.items.clear()

    def create_text(self, *args, **kwargs):
        self.items.append(("text", args, kwargs))

    def create_image(self, *args, **kwargs):
        self.items.append(("image", args, kwargs))

    def winfo_width(self):
        return self.options.get("width", 300)

    def winfo_height(self):
        return self.options.get("height", 270)


class Root:
    def __init__(self):
        self.pending = {}
        self.cancelled = []
        self.counter = 0
        self.focus_set = Mock()
        self.destroy = Mock()

    def after(self, delay, callback):
        self.counter += 1
        name = f"after-{self.counter}"
        self.pending[name] = (delay, callback)
        return name

    def after_cancel(self, name):
        self.cancelled.append(name)
        self.pending.pop(name, None)

    def fire(self, name):
        _, callback = self.pending.pop(name)
        callback()


@pytest.fixture
def window():
    runtime = SimpleNamespace(
        state="READY",
        print_enabled=True,
        preview_roi=None,
        config={},
        events=queue.Queue(),
        start=Mock(return_value=True),
        trigger=Mock(return_value=True),
        confirm_printed=Mock(),
        retry_print_status=Mock(return_value=True),
        get_preview_frame=Mock(return_value=None),
        close=Mock(return_value=True),
    )
    # 绕过 Tk 构造，只测试真实窗口方法的事件与控件行为。
    instance = StudioWindow.__new__(StudioWindow)
    instance.runtime = runtime
    instance.root = Root()
    instance.held = set()
    instance.release_after = {}
    instance.status = Variable()
    instance.elapsed = Variable()
    instance.location = Variable()
    instance.style_label = Variable()
    instance.started = None
    instance.last_preview_sequence = None
    instance.preview_live = False
    instance.last_progress = ""
    instance.photos = {}
    instance.image_paths = {}
    instance.start_button = Widget(state="normal", text="开始")
    instance.run_button = Widget(state="normal")
    instance.confirm_button = Widget()
    instance.retry_button = Widget()
    instance.badge = Widget()
    instance.panels = {key: Widget() for key in ("original", "rectified", "ai")}
    instance.panel_titles = {key: Widget() for key in instance.panels}
    instance.subtitles = {key: Widget() for key in instance.panels}
    instance.step_labels = {key: Widget() for key, _ in STEPS}
    instance.stage_states = dict.fromkeys(instance.step_labels, "waiting")
    instance.dependency_labels = {key: Widget() for key in ("phone", "ai", "printer")}
    instance._append_log = Mock()
    instance._render_image = Mock()
    return instance


def key(name="p", widget=None):
    return SimpleNamespace(keysym=name, widget=widget)


@pytest.mark.parametrize("name", ["p", "P"])
def test_lower_and_uppercase_p_trigger_one_ready_round(window, name):
    assert window._key_down(key(name)) == "break"
    window.runtime.trigger.assert_called_once_with()
    assert window.run_button.options["state"] == "disabled"
    assert window.held == {"p"}


@pytest.mark.parametrize("name", ["r", "Return", "space", "Escape"])
def test_unrelated_keys_do_not_start_a_round(window, name):
    assert window._key_down(key(name)) is None
    assert window._key_up(key(name)) is None
    window.runtime.trigger.assert_not_called()
    assert window.root.pending == {}


def test_held_p_autorepeat_does_not_start_more_than_one_round(window):
    for name in ("p", "p", "P", "p", "P"):
        window._key_down(key(name))
    window.runtime.trigger.assert_called_once_with()


def test_key_release_press_autorepeat_cancels_release_and_remains_held(window):
    window._key_down(key())
    window._key_up(key())
    release_id = window.release_after["p"]
    window._key_down(key("P"))
    assert release_id in window.root.cancelled
    assert release_id not in window.root.pending
    assert window.release_after == {}
    assert window.held == {"p"}
    window.runtime.trigger.assert_called_once_with()


def test_new_physical_press_after_release_starts_next_ready_round(window):
    window._key_down(key())
    window._key_up(key())
    window.root.fire(window.release_after["p"])
    assert window.held == set()
    assert window.release_after == {}
    window._key_down(key("P"))
    assert window.runtime.trigger.call_count == 2


@pytest.mark.parametrize("state", ["NEW", "STARTING", "RUNNING", "PRINTING", "PRINT_ERROR", "AWAITING_PRINT_CONFIRM", "ERROR"])
def test_busy_or_unprepared_p_is_ignored_and_never_queued_for_ready(window, state):
    window.runtime.state = state
    window._key_down(key())
    window.runtime.trigger.assert_not_called()
    window.runtime.state = "READY"
    window._handle("status", {"state": "READY", "message": "打印完成，P 已激活"})
    for _ in range(3):
        window._key_down(key("P"))
    window.runtime.trigger.assert_not_called()
    window._key_up(key())
    window.root.fire(window.release_after["p"])
    window._key_down(key())
    window.runtime.trigger.assert_called_once_with()


def test_runtime_guard_wins_over_stale_enabled_button(window):
    window.run_button.configure(state="normal")
    window.runtime.state = "RUNNING"
    window._key_down(key())
    window.runtime.trigger.assert_not_called()


def test_rejected_runtime_trigger_has_no_local_unlock_or_extra_action(window):
    window.runtime.trigger.return_value = False
    window.run_button.configure(state="disabled")
    window.trigger()
    assert window.run_button.options["state"] == "disabled"
    window.root.focus_set.assert_not_called()


def test_start_delegates_preparation_without_running_pipeline(window):
    window.start()
    window.runtime.start.assert_called_once_with()
    window.runtime.trigger.assert_not_called()
    assert window.start_button.options == {"state": "disabled", "text": "准备中…"}
    window.root.focus_set.assert_called_once_with()


def test_rejected_start_does_not_change_button(window):
    window.runtime.start.return_value = False
    window.start()
    assert window.start_button.options == {"state": "normal", "text": "开始"}
    window.root.focus_set.assert_not_called()


@pytest.mark.parametrize("state,run_enabled,start_enabled,badge_text", [
    ("NEW", False, True, "尚未启动"),
    ("STARTING", False, False, "正在准备"),
    ("READY", True, False, "P 已激活"),
    ("RUNNING", False, False, "P 已锁定"),
    ("PRINTING", False, False, "P 已锁定"),
    ("PRINT_ERROR", False, False, "P 已锁定"),
    ("AWAITING_PRINT_CONFIRM", False, False, "P 已锁定"),
    ("ERROR", False, True, "准备未完成"),
])
def test_status_controls_buttons_and_explains_p_lock(window, state, run_enabled, start_enabled, badge_text):
    window._handle("status", {"state": state, "message": f"状态：{state}"})
    assert window.run_button.options["state"] == ("normal" if run_enabled else "disabled")
    assert window.start_button.options["state"] == ("normal" if start_enabled else "disabled")
    assert window.status.get() == f"状态：{state}"
    assert badge_text in window.badge.options["text"]
    if state in ("ERROR", "PRINT_ERROR"):
        assert window.badge.options["fg"] == RED


@pytest.mark.parametrize("status,word", [("waiting", "等待"), ("working", "进行中"), ("done", "完成"), ("error", "失败")])
def test_stage_updates_current_step_and_progress_message(window, status, word):
    window._handle("stage", {"step": "rectify", "status": status, "message": "正在校正纸面"})
    assert window.stage_states["rectify"] == status
    assert "裁切与校正" in window.step_labels["rectify"].options["text"]
    assert word in window.step_labels["rectify"].options["text"]
    assert window.status.get() == "正在校正纸面"
    window._append_log.assert_called_with("正在校正纸面")


def test_image_events_bind_the_three_actual_output_paths(window):
    for name, path in (("original", "raw.png"), ("rectified", "corrected.png"), ("ai", "result.png")):
        window._handle("image", {"key": name, "path": path})
    assert window.image_paths == {"original": "raw.png", "rectified": "corrected.png", "ai": "result.png"}
    assert [call.args[0] for call in window._render_image.call_args_list] == ["original", "rectified", "ai"]


def test_reset_clears_previous_round_images_steps_and_print_controls(window):
    window.image_paths = {"original": "old.png", "ai": "old-ai.png"}
    window.confirm_button.pack()
    window.retry_button.pack()
    window.stage_states = dict.fromkeys(window.stage_states, "done")
    window._handle("reset", {})
    assert window.image_paths == {}
    assert set(window.stage_states.values()) == {"waiting"}
    assert window.started is not None
    assert "等待生成" in window.style_label.get()
    assert window.confirm_button.visible is False
    assert window.retry_button.visible is False
    assert window._render_image.call_count == 3


@pytest.mark.parametrize("retry,confirm", [(False, False), (True, False), (False, True), (True, True)])
def test_print_controls_only_show_authorized_actions(window, retry, confirm):
    window._handle("print_controls", {"retry": retry, "confirm": confirm})
    assert window.retry_button.visible is retry
    assert window.confirm_button.visible is confirm
    for button, visible in ((window.retry_button, retry), (window.confirm_button, confirm)):
        if visible:
            assert button.options["state"] == "normal"


def test_hiding_print_controls_removes_previously_visible_actions(window):
    window._handle("print_controls", {"retry": True, "confirm": True})
    window._handle("print_controls", {})
    assert window.confirm_button.visible is False
    assert window.retry_button.visible is False


def test_retry_queries_original_print_and_disables_controls_until_result(window):
    window._handle("print_controls", {"retry": True, "confirm": True})
    window.retry_status()
    window.runtime.retry_print_status.assert_called_once_with()
    window.runtime.trigger.assert_not_called()
    window.runtime.confirm_printed.assert_not_called()
    assert window.confirm_button.options["state"] == "disabled"
    assert window.retry_button.options["state"] == "disabled"


def test_confirm_paper_output_delegates_to_runtime_and_does_not_generate(window):
    window.confirm_printed()
    window.runtime.confirm_printed.assert_called_once_with()
    window.runtime.trigger.assert_not_called()


def test_failed_completion_record_keeps_explicit_locked_message(window):
    window.runtime.confirm_printed.side_effect = OSError("disk full")
    window.confirm_printed()
    assert "P 保持锁定" in window.status.get()
    assert "disk full" in window.status.get()
    window._append_log.assert_called_with("disk full")


def test_completion_stops_timer_but_unlocking_still_requires_runtime_ready(window):
    window.runtime.state = "AWAITING_PRINT_CONFIRM"
    window.run_button.configure(state="disabled")
    window.started = 1
    window._handle("completed", {"elapsed": 19.25})
    assert window.started is None
    assert window.elapsed.get() == "本轮 19.2s"
    assert window.run_button.options["state"] == "disabled"
    window._key_down(key())
    window.runtime.trigger.assert_not_called()


def test_dependency_error_is_visible_and_logged(window):
    window._handle("dependency", {"name": "printer", "status": "error", "message": "打印机离线"})
    assert window.dependency_labels["printer"].options["text"] == "! 打印机离线"
    assert window.dependency_labels["printer"].options["fg"] == RED
    window._append_log.assert_called_with("打印机离线")


@pytest.fixture
def video_frame(window, monkeypatch):
    pixels = np.full((120, 160, 3), (10, 20, 30), dtype=np.uint8)
    frame = PreviewFrame(image=pixels, sequence=7, age_ms=20)
    window.runtime.get_preview_frame.return_value = frame
    monkeypatch.setattr("PIL.ImageTk.PhotoImage", Mock(return_value="mock-video-photo"))
    return frame


def test_live_preview_displays_cached_frames_without_rendering_duplicates(window, video_frame):
    assert window._refresh_preview() is True
    assert window._refresh_preview() is True

    assert window.runtime.get_preview_frame.call_count == 2
    assert window.panels["original"].items == [
        ("image", (150.0, 135.0), {"image": "mock-video-photo"}),
    ]
    assert window.photos["original"] == "mock-video-photo"
    assert window.last_preview_sequence == video_frame.sequence
    assert window.preview_live is True
    assert "实时" in window.panel_titles["original"].options["text"]
    from PIL import ImageTk
    ImageTk.PhotoImage.assert_called_once()


def test_video_render_uses_same_viewfinder_roi_and_correct_rgb_colors(window, video_frame):
    from PIL import ImageTk

    window.runtime.preview_roi = [0.25, 0.25, 0.75, 0.75]
    window._render_preview(video_frame)

    image = ImageTk.PhotoImage.call_args.args[0]
    assert image.size == (80, 60)
    assert image.getpixel((0, 0)) == (30, 20, 10)
    assert ImageTk.PhotoImage.call_args.kwargs["master"] is window.root


def test_lost_video_clears_old_live_frame(window, video_frame):
    assert window._refresh_preview() is True
    window.runtime.get_preview_frame.return_value = None

    assert window._refresh_preview() is False

    assert window.panels["original"].items == []
    assert window.preview_live is False
    assert "视频暂不可用" in window.subtitles["original"].options["text"]


def test_running_round_freezes_saved_original_then_ready_resumes_video(window, video_frame):
    window._refresh_preview()
    window.runtime.state = "RUNNING"
    window._handle("image", {"key": "original", "path": "round-original.png"})
    window.runtime.get_preview_frame.reset_mock()

    assert window._refresh_preview() is False
    window.runtime.get_preview_frame.assert_not_called()
    assert window.preview_live is False
    assert window.image_paths["original"] == "round-original.png"
    assert "本轮" in window.panel_titles["original"].options["text"]

    window.runtime.state = "READY"
    window._handle("status", {"state": "READY", "message": "图片已保存，P 已重新激活"})
    assert window._refresh_preview() is True
    assert window.preview_live is True
    assert window.image_paths["original"] == "round-original.png"


def test_video_remains_live_while_waiting_for_capture_and_preserves_other_results(window, video_frame):
    window.runtime.state = "RUNNING"
    window._handle("image", {"key": "rectified", "path": "corrected.png"})
    window._handle("image", {"key": "ai", "path": "result.png"})

    assert window._refresh_preview() is True

    assert window.image_paths == {"rectified": "corrected.png", "ai": "result.png"}
    assert [call.args[0] for call in window._render_image.call_args_list] == ["rectified", "ai"]
    assert "rectified" not in window.photos
    assert "ai" not in window.photos


def test_poll_processes_events_on_ui_thread_and_schedules_next_poll(window, monkeypatch):
    monkeypatch.setattr("app.gui.time.monotonic", lambda: 10)
    window.runtime.events.put(("status", {"state": "READY", "message": "系统就绪"}))
    window.runtime.events.put(("style", {"style_id": "cubism", "style_name": "立体主义"}))
    window.runtime.events.put(("session", {"directory": "runs/phone/session-1"}))
    window._poll()
    assert window.status.get() == "系统就绪"
    assert window.style_label.get() == "本轮画派 · 立体主义"
    assert window.location.get() == "runs/phone/session-1"
    window.runtime.get_preview_frame.assert_called_once_with()
    assert len(window.root.pending) == 1
    assert next(iter(window.root.pending.values()))[0] == 80


@pytest.mark.parametrize("data,expected", [
    ({"style_id": "pop_art", "style_name": "波普艺术"}, "本轮画派 · 波普艺术"),
    ({"style_id": "pop_art", "style_name": ""}, "本轮画派 · pop_art"),
    ({"style_id": "unregistered_style"}, "本轮画派 · unregistered_style"),
])
def test_style_display_uses_event_name_and_accepts_any_style_id(window, data, expected):
    window._handle("style", data)

    assert window.style_label.get() == expected


@pytest.mark.parametrize("allowed", [True, False])
def test_close_respects_runtime_busy_guard(window, allowed):
    window.runtime.close.return_value = allowed
    window.close()
    window.runtime.close.assert_called_once_with()
    if allowed:
        window.root.destroy.assert_called_once_with()
    else:
        window.root.destroy.assert_not_called()
        assert "当前操作尚未结束" in window.status.get()


def test_focus_change_to_child_widget_does_not_reset_held_p(window):
    window._key_down(key())
    window._focus_out(key(widget=Widget()))
    assert window.held == {"p"}


def test_focus_out_cancels_old_release_before_busy_key_is_held_again(window):
    window.runtime.state = "RUNNING"
    window._key_down(key())
    window._key_up(key())
    old_release = window.release_after["p"]
    window._focus_out(key(widget=window.root))
    window._key_down(key())
    # 回焦后仍按住的 P 不应被旧定时器当作已松开，更不能等 READY 后启动。
    if old_release in window.root.pending:
        window.root.fire(old_release)
    window.runtime.state = "READY"
    window._key_down(key())
    window.runtime.trigger.assert_not_called()


def test_visible_clock_explicitly_uses_hong_kong_timezone(monkeypatch):
    calls = []

    class FixedDateTime:
        @staticmethod
        def now(zone):
            calls.append(zone)
            return datetime(2026, 10, 25, 9, 2, 3, tzinfo=zone)

    monkeypatch.setattr("app.gui.datetime", FixedDateTime)
    assert hong_kong_time() == "09:02:03"
    assert calls[0].utcoffset(None) == timedelta(hours=8)


def test_log_uses_hong_kong_clock_and_deduplicates_progress(window, monkeypatch):
    monkeypatch.setattr("app.gui.hong_kong_time", lambda: "09:02:03")
    window.log = SimpleNamespace(configure=Mock(), insert=Mock(), index=Mock(return_value="2.0"), delete=Mock(), see=Mock())
    StudioWindow._append_log(window, "相机已就绪")
    StudioWindow._append_log(window, "相机已就绪")
    window.log.insert.assert_called_once_with("end", "09:02:03  相机已就绪\n")
    assert window.log.configure.call_args.kwargs == {"state": "disabled"}
