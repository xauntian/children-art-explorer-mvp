"""任务文件与状态持久化测试。

要求：文件隔离、原子写入、重启不重复提交、状态记录完整。
"""

from __future__ import annotations

import json

import pytest

from app import states
from app.errors import ConfigError
from app.storage import (
    JobStore,
    atomic_write_bytes,
    atomic_write_json,
    new_job_id,
    validate_job_id,
)


def test_new_job_id_is_unique_and_valid():
    ids = {new_job_id() for _ in range(50)}
    assert len(ids) == 50
    for jid in ids:
        assert validate_job_id(jid) == jid


@pytest.mark.parametrize(
    "bad",
    ["", "..", "../etc/passwd", "a/b", "a\\b", "a b", "-leading", "x" * 100, "任务"],
)
def test_invalid_job_ids_rejected(bad):
    with pytest.raises(ConfigError, match="非法 job_id"):
        validate_job_id(bad)


def test_create_builds_expected_directory(tmp_path):
    store = JobStore(tmp_path / "runs")
    job = store.create(style={"id": "fauvism"}, config_snapshot={"max_attempts": 2})

    assert job.dir.is_dir()
    assert job.meta_path.is_file()
    meta = json.loads(job.meta_path.read_text(encoding="utf-8"))
    assert meta["state"] == states.CAPTURED
    assert meta["style"]["id"] == "fauvism"
    assert meta["attempt_count"] == 0
    assert meta["errors"] == []


def test_create_refuses_to_overwrite_existing_job(tmp_path):
    store = JobStore(tmp_path / "runs")
    store.create(job_id="fixed-1")
    with pytest.raises(ConfigError, match="拒绝覆盖"):
        store.create(job_id="fixed-1")


def test_atomic_write_leaves_no_temp_file(tmp_path):
    target = tmp_path / "x.png"
    atomic_write_bytes(target, b"data")
    assert target.read_bytes() == b"data"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_retries_on_transient_permission_error(tmp_path, monkeypatch):
    """云盘同步客户端短暂占用文件时，应退避重试而不是直接失败。"""
    import os as os_module

    target = tmp_path / "y.json"
    real_replace = os_module.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(5, "拒绝访问。")
        return real_replace(src, dst)

    monkeypatch.setattr(os_module, "replace", flaky_replace)
    atomic_write_json(target, {"ok": True}, retry_delay=0.01)

    assert calls["n"] == 3, "应在前两次失败后重试成功"
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_keeps_temp_file_and_raises_after_exhausting_retries(tmp_path, monkeypatch):
    """重试耗尽后必须抛出明确错误，并保留临时文件用于排查。"""
    import os as os_module

    target = tmp_path / "z.json"

    def always_denied(src, dst):
        raise PermissionError(5, "拒绝访问。")

    monkeypatch.setattr(os_module, "replace", always_denied)
    with pytest.raises(OSError, match="原子替换失败"):
        atomic_write_json(target, {"ok": True}, retry_delay=0.01)

    assert not target.exists(), "替换失败时不得生成目标文件"
    temps = list(tmp_path.glob("*.tmp"))
    assert temps, "应保留临时文件用于排查"


def test_atomic_write_json_is_valid_json(tmp_path):
    target = tmp_path / "m.json"
    atomic_write_json(target, {"a": 1, "中文": "值"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "中文": "值"}


def test_atomic_write_replaces_previous_content(tmp_path):
    target = tmp_path / "x.bin"
    atomic_write_bytes(target, b"first-version")
    atomic_write_bytes(target, b"second")
    assert target.read_bytes() == b"second"


def test_state_history_is_recorded(tmp_path):
    store = JobStore(tmp_path / "runs")
    job = store.create()
    job.set_state(states.STYLE_ASSIGNED)
    job.set_state(states.GENERATING)

    meta = json.loads(job.meta_path.read_text(encoding="utf-8"))
    recorded = [entry["state"] for entry in meta["state_history"]]
    assert recorded == [states.CAPTURED, states.STYLE_ASSIGNED, states.GENERATING]


def test_illegal_state_transition_is_rejected(tmp_path):
    store = JobStore(tmp_path / "runs")
    job = store.create()
    with pytest.raises(ConfigError, match="非法状态流转"):
        job.set_state(states.COMPLETED)


def test_attempt_records_are_persisted_before_generation(tmp_path):
    store = JobStore(tmp_path / "runs")
    job = store.create()
    record = job.begin_attempt(1, 777, {"steps": 30, "denoise": 0.65})
    assert record["started_at"]

    reloaded = store.load(job.job_id)
    attempts = reloaded.attempts()
    assert len(attempts) == 1
    assert attempts[0]["seed"] == 777
    assert attempts[0]["params"]["denoise"] == 0.65
    assert reloaded.meta["attempt_count"] == 1


def test_finish_attempt_updates_record(tmp_path):
    store = JobStore(tmp_path / "runs")
    job = store.create()
    job.begin_attempt(1, 1, {})
    job.finish_attempt(1, generate_sec=1.5, check_passed=False)

    reloaded = store.load(job.job_id)
    assert reloaded.attempt_record(1)["generate_sec"] == 1.5
    assert reloaded.attempt_record(1)["check_passed"] is False


def test_finish_attempt_requires_existing_record(tmp_path):
    store = JobStore(tmp_path / "runs")
    job = store.create()
    with pytest.raises(ConfigError, match="缺少第 2 次尝试记录"):
        job.finish_attempt(2, note="x")


def test_error_recording_and_failure(tmp_path):
    store = JobStore(tmp_path / "runs")
    job = store.create()
    job.set_state(states.STYLE_ASSIGNED)

    from app.errors import ServiceUnavailableError

    job.fail(ServiceUnavailableError("ComfyUI 掉线", stage=states.GENERATING))

    reloaded = store.load(job.job_id)
    assert reloaded.state == states.FAILED
    errors = reloaded.meta["errors"]
    assert errors[-1]["category"] == "service_unavailable"
    assert errors[-1]["stage"] == states.GENERATING
    assert "ComfyUI 掉线" in errors[-1]["message"]


def test_two_jobs_have_separate_directories(tmp_path):
    store = JobStore(tmp_path / "runs")
    a = store.create(job_id="job-a")
    b = store.create(job_id="job-b")
    a.save_output(b"aaa")
    b.save_output(b"bbb")

    assert a.dir != b.dir
    assert a.output_path.read_bytes() == b"aaa"
    assert b.output_path.read_bytes() == b"bbb"


def test_find_active_and_terminal(tmp_path):
    store = JobStore(tmp_path / "runs")
    store.create(job_id="job-active")
    done = store.create(job_id="job-done")
    done.set_state(states.STYLE_ASSIGNED)
    done.set_state(states.GENERATING)
    done.set_state(states.CHECKING)
    done.set_state(states.CONFIRMED)
    done.set_state(states.COMPLETED)

    assert store.find_active().job_id == "job-active"
    assert store.find_completed_ids() == {"job-done"}
    assert {j.job_id for j in store.find_incomplete()} == {"job-active"}


def test_load_missing_job_raises(tmp_path):
    store = JobStore(tmp_path / "runs")
    with pytest.raises(ConfigError, match="任务记录不存在"):
        store.load("does-not-exist")


def test_list_jobs_skips_corrupt_records(tmp_path):
    store = JobStore(tmp_path / "runs")
    good = store.create(job_id="good-job")
    broken_dir = store.runs_dir / "broken-job"
    broken_dir.mkdir()
    (broken_dir / "meta.json").write_text("{ not json", encoding="utf-8")

    listed = {job.job_id for job in store.list_jobs()}
    assert good.job_id in listed
    assert "broken-job" not in listed


def test_original_and_input_files_are_stored(tmp_path):
    store = JobStore(tmp_path / "runs")
    job = store.create(job_id="files-job")
    job.save_original(b"jpeg-bytes")
    job.save_input(b"png-bytes")

    assert job.original_path.name == "original.jpg"
    assert job.input_path.name == "input.png"
    assert job.original_path.read_bytes() == b"jpeg-bytes"
    assert job.input_path.read_bytes() == b"png-bytes"


def test_check_result_files_named_per_attempt(tmp_path):
    store = JobStore(tmp_path / "runs")
    job = store.create(job_id="check-job")
    job.save_check_result(1, {"style_match": "mismatch"})
    job.save_check_result(2, {"style_match": "match"})

    assert job.check_path(1).name == "check_1.json"
    assert job.check_path(2).name == "check_2.json"
    assert job.read_check_result(1)["style_match"] == "mismatch"
    assert job.read_check_result(3) is None
