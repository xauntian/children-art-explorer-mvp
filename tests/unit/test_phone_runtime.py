"""真实 OS 互斥与只读正式任务的前置恢复测试，不访问手机或模型。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import states
from app.errors import ConfigError
from app.phone_runtime import PhoneRunLock, reconcile_phone_sessions
from app.storage import JobStore, atomic_write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _child(code: str, directory: Path):
    return subprocess.run(
        [sys.executable, "-B", "-c", code, str(directory)],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=15,
    )


LOCK_CHILD = """
import sys
from app.errors import ConfigError
from app.phone_runtime import PhoneRunLock
try:
    with PhoneRunLock(sys.argv[1]):
        pass
except ConfigError:
    sys.exit(23)
"""


def test_lock_is_nonblocking_across_processes_and_reusable_after_release(tmp_path):
    with PhoneRunLock(tmp_path):
        blocked = _child(LOCK_CHILD, tmp_path)
        assert blocked.returncode == 23, blocked.stderr
    acquired = _child(LOCK_CHILD, tmp_path)
    assert acquired.returncode == 0, acquired.stderr
    assert (tmp_path / ".phone-run.lock").is_file()


def test_os_releases_lock_when_process_exits_without_context_cleanup(tmp_path):
    exited = _child(
        "import os,sys; from app.phone_runtime import PhoneRunLock; "
        "lock=PhoneRunLock(sys.argv[1]); lock.__enter__(); os._exit(0)",
        tmp_path,
    )
    assert exited.returncode == 0, exited.stderr
    with PhoneRunLock(tmp_path):
        pass


def test_exception_releases_lock_and_reentry_is_rejected(tmp_path):
    lock = PhoneRunLock(tmp_path)
    with pytest.raises(RuntimeError):
        with lock:
            with pytest.raises(ConfigError):
                lock.__enter__()
            raise RuntimeError("interrupted")
    with PhoneRunLock(tmp_path):
        pass


def _session(capture_dir: Path, state: str, session_id="session-1", **extra):
    directory = capture_dir / session_id
    directory.mkdir(parents=True)
    payload = {
        "session_id": session_id,
        "state": state,
        "generation_job_id": None,
        "history": [{"state": state, "at": "original"}],
        **extra,
    }
    atomic_write_json(directory / "meta.json", payload)
    return directory


def _read(directory: Path):
    return json.loads((directory / "meta.json").read_text(encoding="utf-8"))


def _job(store: JobStore, state: str, *, job_id="phone-session-1", output=False):
    job = store.create(job_id=job_id)
    job.meta["state"] = state
    if state == states.FAILED:
        job.meta["errors"] = [{"category": "quality_check_failed", "message": "no style match"}]
    job.save()
    if output:
        job.output_path.write_bytes(b"confirmed-output")
    return job


@pytest.mark.parametrize(
    "state", ["CAPTURING", "EXTRACTING", "AWAITING_INPUT_CONFIRM", "INPUT_READY"]
)
def test_interrupted_input_is_failed_without_touching_files_or_generating(tmp_path, state):
    capture = tmp_path / "phone"
    directory = _session(capture, state)
    (directory / "input.png").write_bytes(b"preserve-input")
    store = JobStore(tmp_path / "jobs")
    reports = reconcile_phone_sessions(capture, store)
    assert reports[0]["action"] == "interrupted_failed"
    assert reports[0]["blocking"] is False
    assert _read(directory)["state"] == "FAILED"
    assert (directory / "input.png").read_bytes() == b"preserve-input"
    assert store.list_jobs() == []
    previous = (directory / "meta.json").read_bytes()
    assert reconcile_phone_sessions(capture, store) == []
    assert (directory / "meta.json").read_bytes() == previous


def test_completed_job_reconciles_frontend_without_changing_formal_job(tmp_path):
    capture = tmp_path / "phone"
    directory = _session(capture, "GENERATING", generation_job_id="phone-session-1")
    store = JobStore(tmp_path / "jobs")
    job = _job(store, states.COMPLETED, output=True)
    original = (job.dir / "meta.json").read_bytes()
    reports = reconcile_phone_sessions(capture, store)
    assert reports[0]["action"] == "reconciled_completed"
    assert reports[0]["blocking"] is False
    recovered = _read(directory)
    assert recovered["state"] == "COMPLETED"
    assert recovered["output_path"] == str(job.output_path)
    assert recovered["generation_job_id"] == job.job_id
    assert (job.dir / "meta.json").read_bytes() == original


def test_failed_job_copies_failure_and_does_not_change_formal_job(tmp_path):
    capture = tmp_path / "phone"
    directory = _session(capture, "GENERATING", generation_job_id="phone-session-1")
    store = JobStore(tmp_path / "jobs")
    job = _job(store, states.FAILED)
    original = (job.dir / "meta.json").read_bytes()
    reports = reconcile_phone_sessions(capture, store)
    assert reports[0]["action"] == "reconciled_failed"
    assert _read(directory)["error"]["message"] == "no style match"
    assert _read(directory)["state"] == "FAILED"
    assert (job.dir / "meta.json").read_bytes() == original


@pytest.mark.parametrize("state", states.ACTIVE_STATES)
def test_active_formal_job_still_blocks_and_both_records_remain_unchanged(tmp_path, state):
    capture = tmp_path / "phone"
    directory = _session(capture, "GENERATING", generation_job_id="phone-session-1")
    store = JobStore(tmp_path / "jobs")
    job = _job(store, state)
    frontend_before = (directory / "meta.json").read_bytes()
    formal_before = (job.dir / "meta.json").read_bytes()
    reports = reconcile_phone_sessions(capture, store)
    assert reports[0]["action"] == "generation_active"
    assert reports[0]["blocking"] is True
    assert reports[0]["generation_state"] == state
    assert (directory / "meta.json").read_bytes() == frontend_before
    assert (job.dir / "meta.json").read_bytes() == formal_before


@pytest.mark.parametrize("missing", ["job", "output", "link"])
def test_missing_job_output_or_valid_link_cannot_publish_completion(tmp_path, missing):
    capture = tmp_path / "phone"
    link = None if missing == "link" else "phone-session-1"
    directory = _session(capture, "GENERATING", generation_job_id=link)
    store = JobStore(tmp_path / "jobs")
    if missing == "output":
        _job(store, states.COMPLETED)
    reports = reconcile_phone_sessions(capture, store)
    assert reports[0]["action"] == "interrupted_failed"
    assert _read(directory)["state"] == "FAILED"
    assert "output_path" not in _read(directory)


@pytest.mark.parametrize("state", ["COMPLETED", "FAILED", "CANCELLED", "SUPERSEDED"])
def test_terminal_frontend_is_never_rewritten(tmp_path, state):
    capture = tmp_path / "phone"
    directory = _session(capture, state)
    previous = (directory / "meta.json").read_bytes()
    assert reconcile_phone_sessions(capture, JobStore(tmp_path / "jobs")) == []
    assert (directory / "meta.json").read_bytes() == previous


def test_unrelated_formal_jobs_are_not_scanned_or_modified(tmp_path):
    capture = tmp_path / "phone"
    directory = _session(capture, "GENERATING", generation_job_id="ordinary-job")
    store = JobStore(tmp_path / "jobs")
    unrelated = _job(store, states.GENERATING, job_id="ordinary-job")
    original = (unrelated.dir / "meta.json").read_bytes()
    reports = reconcile_phone_sessions(capture, store)
    assert reports[0]["reason"] == "invalid_generation_link"
    assert _read(directory)["state"] == "FAILED"
    assert (unrelated.dir / "meta.json").read_bytes() == original


@pytest.mark.parametrize("contents", ["{broken", '{"session_id":"session-1","state":"UNKNOWN"}'])
def test_corrupt_records_are_reported_and_preserved(tmp_path, contents):
    capture = tmp_path / "phone"
    directory = capture / "session-1"
    directory.mkdir(parents=True)
    path = directory / "meta.json"
    path.write_text(contents, encoding="utf-8")
    reports = reconcile_phone_sessions(capture, JobStore(tmp_path / "jobs"))
    assert reports[0]["action"] == "read_error"
    assert reports[0]["blocking"] is True
    assert path.read_text(encoding="utf-8") == contents


def test_missing_capture_directory_and_calibration_folder_are_ignored(tmp_path):
    capture = tmp_path / "phone"
    store = JobStore(tmp_path / "jobs")
    assert reconcile_phone_sessions(capture, store) == []
    assert not capture.exists()
    (capture / "calibration").mkdir(parents=True)
    assert reconcile_phone_sessions(capture, store) == []
