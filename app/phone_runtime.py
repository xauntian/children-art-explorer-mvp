"""手机入口的进程互斥与前置会话恢复；不调用手机或模型。"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from typing import BinaryIO

from . import states
from .errors import ConfigError
from .storage import JobStore, atomic_write_json, utc_now_iso

_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "SUPERSEDED"}
_BEFORE_GENERATION = {"CAPTURING", "EXTRACTING", "AWAITING_INPUT_CONFIRM", "INPUT_READY"}


class PhoneRunLock:
    """整个手机入口持有的非阻塞 OS 文件锁；关闭或进程退出时自动释放。

    锁文件保留在目录中，不能通过删除文件或检查旧 PID 判断锁是否有效。
    """

    def __init__(self, capture_runs_dir: str | Path):
        self.path = Path(capture_runs_dir) / ".phone-run.lock"
        self._file: BinaryIO | None = None

    def __enter__(self) -> PhoneRunLock:
        if self._file is not None:
            raise ConfigError("手机采集锁不能重复进入")
        handle = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b")
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if handle is not None:
                handle.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise ConfigError("已有手机采集实例正在运行，请先结束该实例") from exc
            raise ConfigError(f"无法取得手机采集锁: {self.path}: {exc}") from exc
        self._file = handle
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        handle, self._file = self._file, None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _update_session(
    meta_path: Path, meta: dict, report: dict, target: str, *, error: dict | None = None
) -> None:
    timestamp = utc_now_iso()
    history = meta.setdefault("history", [])
    if not isinstance(history, list):
        raise ValueError("前置会话 history 不是列表")
    meta["state"] = target
    meta["updated_at"] = timestamp
    history.append({"state": target, "at": timestamp, "note": "startup_reconciliation"})
    meta["recovery"] = {
        "at": timestamp,
        "action": report["action"],
        "previous_state": report["previous_state"],
        "reason": report.get("reason"),
    }
    if error is not None:
        meta["error"] = error
        meta.pop("output_path", None)
    else:
        meta.pop("error", None)
    atomic_write_json(meta_path, meta)
    report["state"] = target


def _fail_session(meta_path: Path, meta: dict, report: dict, reason: str, message: str) -> None:
    report.update(action="interrupted_failed", reason=reason)
    _update_session(
        meta_path,
        meta,
        report,
        "FAILED",
        error={
            "category": "phone_interrupted",
            "stage": report["previous_state"],
            "message": message,
        },
    )


def reconcile_phone_sessions(capture_dir: str | Path, generation_store: JobStore) -> list[dict]:
    """在持有 PhoneRunLock 时恢复前置会话，正式任务始终只读。

    每项报告含 session_id、previous_state、state、action、blocking 和
    generation_job_id。blocking=True 表示应先解决旧任务或损坏记录；本函数
    不重新生成、不调用模型、不扫描或修改未关联的一般任务。
    """
    directory = Path(capture_dir)
    if not directory.exists():
        return []
    reports: list[dict] = []
    for session_dir in sorted(directory.iterdir()):
        if not session_dir.is_dir():
            continue
        meta_path = session_dir / "meta.json"
        if not meta_path.is_file():
            continue
        report = {
            "session_id": session_dir.name,
            "previous_state": None,
            "state": None,
            "generation_job_id": None,
            "action": "read_error",
            "blocking": False,
        }
        try:
            if not session_dir.resolve().is_relative_to(directory.resolve()):
                raise ValueError("前置会话目录指向采集目录之外")
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            if not isinstance(meta, dict) or meta.get("session_id") != session_dir.name:
                raise ValueError("前置会话编号与目录不一致")
            previous = meta.get("state")
            if not isinstance(previous, str):
                raise ValueError("前置会话缺少合法状态")
            report.update(
                previous_state=previous,
                state=previous,
                generation_job_id=meta.get("generation_job_id"),
            )
            if previous in _TERMINAL:
                continue
            if previous in _BEFORE_GENERATION:
                _fail_session(
                    meta_path,
                    meta,
                    report,
                    "input_interrupted",
                    "前置采集或输入确认已中断，请重新采集。",
                )
            elif previous == "GENERATING":
                job_id = meta.get("generation_job_id")
                if job_id != "phone-" + session_dir.name:
                    _fail_session(
                        meta_path,
                        meta,
                        report,
                        "invalid_generation_link",
                        "前置会话的生成任务关联缺失或无效。",
                    )
                elif not generation_store.exists(job_id):
                    _fail_session(
                        meta_path,
                        meta,
                        report,
                        "generation_job_missing",
                        "关联生成任务不存在，请重新采集。",
                    )
                else:
                    job = generation_store.load(job_id)
                    report["generation_state"] = job.state
                    if job.state == states.COMPLETED and job.output_path.is_file():
                        report.update(action="reconciled_completed", reason="generation_completed")
                        meta["generation"] = job.to_summary()
                        meta["output_path"] = str(job.output_path)
                        _update_session(meta_path, meta, report, "COMPLETED")
                    elif job.state == states.FAILED:
                        report.update(action="reconciled_failed", reason="generation_failed")
                        meta["generation"] = job.to_summary()
                        errors = job.meta.get("errors") or []
                        error = (
                            errors[-1]
                            if errors and isinstance(errors[-1], dict)
                            else {
                                "category": "generation_failed",
                                "message": "关联生成任务已失败。",
                            }
                        )
                        _update_session(meta_path, meta, report, "FAILED", error=error)
                    elif states.is_active(job.state):
                        report.update(
                            action="generation_active",
                            blocking=True,
                            reason="existing_resume_required",
                            message="关联生成任务仍处于活动状态；请使用现有 app.main resume 入口处理，不重新提交。",
                        )
                    elif job.state == states.COMPLETED:
                        _fail_session(
                            meta_path,
                            meta,
                            report,
                            "generation_output_missing",
                            "关联生成任务标记完成，但最终输出不存在；未重新提交生成。",
                        )
                    else:
                        raise ValueError(f"关联生成任务状态无效: {job.state}")
            else:
                raise ValueError(f"未知的前置会话状态: {previous}")
        except Exception as exc:
            report.update(action="read_error", blocking=True, error=str(exc))
        reports.append(report)
    return reports
