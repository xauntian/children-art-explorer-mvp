"""任务文件与状态持久化。

每个任务使用唯一 job_id，输出目录为 runs/{job_id}/：

    original.jpg      原始照片
    input.png         预处理后的模型输入
    attempt_1.png     第一次生成候选
    check_1.json      第一次检查结果
    attempt_2.png     仅在发生第二次生成时保存
    check_2.json      对应第二次检查
    output.png        仅保存自动确认通过的结果
    meta.json         状态、配置、各阶段时间和错误记录

所有写入采用「临时文件 + 原子替换」，避免半写入文件被当作成功结果。
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import states
from .errors import ConfigError

JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

ORIGINAL_NAME = "original.jpg"
INPUT_NAME = "input.png"
OUTPUT_NAME = "output.png"
META_NAME = "meta.json"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def new_job_id(now: datetime | None = None) -> str:
    """唯一任务编号：时间戳 + 短随机后缀。"""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def attempt_image_name(attempt: int) -> str:
    return f"attempt_{attempt}.png"


def check_result_name(attempt: int) -> str:
    return f"check_{attempt}.json"


def validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
        raise ConfigError(f"非法 job_id: {job_id!r}")
    return job_id


def atomic_write_bytes(
    path: Path,
    data: bytes,
    temp_suffix: str = ".tmp",
    retries: int = 6,
    retry_delay: float = 0.15,
) -> None:
    """先写临时文件再原子替换，避免半写入文件。

    输出目录可能位于云盘同步目录中，同步客户端会在上传期间短暂占用文件，
    导致 os.replace 抛出 PermissionError（WinError 5）。这里做有限次退避重试，
    并且只在替换成功后删除临时文件——绝不退化为「直接写目标文件」，
    否则半写入文件可能被当作成功结果。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + temp_suffix)

    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())

    delay = retry_delay
    last_error: OSError | None = None
    for attempt in range(retries + 1):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:  # 同步客户端或杀毒软件短暂占用
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 2.0)

    # 替换失败：保留临时文件以便排查，并抛出明确错误
    raise OSError(
        f"原子替换失败（{retries + 1} 次尝试）: {tmp} -> {path}；"
        f"临时文件已保留。原因: {last_error}"
    ) from last_error


def atomic_write_text(
    path: Path,
    text: str,
    temp_suffix: str = ".tmp",
    retries: int = 6,
    retry_delay: float = 0.15,
) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), temp_suffix, retries, retry_delay)


def atomic_write_json(
    path: Path,
    obj: Any,
    temp_suffix: str = ".tmp",
    retries: int = 6,
    retry_delay: float = 0.15,
) -> None:
    atomic_write_text(
        path,
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        temp_suffix,
        retries,
        retry_delay,
    )


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class Job:
    """单个任务的记录与文件访问。"""

    def __init__(self, store: JobStore, job_id: str, meta: dict) -> None:
        self.store = store
        self.job_id = validate_job_id(job_id)
        self.meta = meta

    # ---- 路径 ----
    @property
    def dir(self) -> Path:
        return self.store.job_dir(self.job_id)

    @property
    def original_path(self) -> Path:
        return self.dir / ORIGINAL_NAME

    @property
    def input_path(self) -> Path:
        return self.dir / INPUT_NAME

    @property
    def output_path(self) -> Path:
        return self.dir / OUTPUT_NAME

    @property
    def meta_path(self) -> Path:
        return self.dir / META_NAME

    def attempt_path(self, attempt: int) -> Path:
        return self.dir / attempt_image_name(attempt)

    def check_path(self, attempt: int) -> Path:
        return self.dir / check_result_name(attempt)

    # ---- 状态 ----
    @property
    def state(self) -> str:
        return self.meta.get("state", "")

    def set_state(self, target: str) -> None:
        current = self.state
        states.assert_transition(current, target)
        self.meta["state"] = target
        self.meta.setdefault("state_history", []).append(
            {"state": target, "at": utc_now_iso()}
        )
        self.save()

    # ---- 持久化 ----
    def save(self) -> None:
        self.meta["updated_at"] = utc_now_iso()
        atomic_write_json(self.meta_path, self.meta, self.store.temp_suffix)

    def save_original(self, data: bytes) -> None:
        atomic_write_bytes(self.original_path, data, self.store.temp_suffix)

    def save_input(self, data: bytes) -> None:
        atomic_write_bytes(self.input_path, data, self.store.temp_suffix)

    def save_attempt(self, attempt: int, data: bytes) -> None:
        atomic_write_bytes(self.attempt_path(attempt), data, self.store.temp_suffix)

    def save_output(self, data: bytes) -> None:
        atomic_write_bytes(self.output_path, data, self.store.temp_suffix)

    def save_check_result(self, attempt: int, payload: Any) -> None:
        atomic_write_json(self.check_path(attempt), payload, self.store.temp_suffix)

    def read_check_result(self, attempt: int) -> Any | None:
        path = self.check_path(attempt)
        if not path.is_file():
            return None
        return read_json(path)

    # ---- 尝试记录 ----
    def attempts(self) -> list[dict]:
        return self.meta.setdefault("attempts", [])

    def begin_attempt(self, attempt: int, seed: int, params: dict) -> dict:
        """进入本地模型调用前先保存任务与尝试信息。"""
        record = {
            "attempt": attempt,
            "seed": seed,
            "params": params,
            "started_at": utc_now_iso(),
            "finished_at": None,
            "generate_sec": None,
            "check_sec": None,
            "check_result": None,
            "check_passed": None,
            "error": None,
        }
        self.attempts().append(record)
        self.meta["attempt_count"] = len(self.attempts())
        self.save()
        return record

    def attempt_record(self, attempt: int) -> dict | None:
        for record in self.attempts():
            if record.get("attempt") == attempt:
                return record
        return None

    def finish_attempt(self, attempt: int, **fields: Any) -> dict:
        record = self.attempt_record(attempt)
        if record is None:
            raise ConfigError(f"任务 {self.job_id} 缺少第 {attempt} 次尝试记录")
        record.update(fields)
        record["finished_at"] = record.get("finished_at") or utc_now_iso()
        self.save()
        return record

    # ---- 错误 ----
    def record_error(self, error: Any) -> None:
        payload = error.to_dict() if hasattr(error, "to_dict") else {"message": str(error)}
        self.meta.setdefault("errors", []).append({"at": utc_now_iso(), **payload})
        self.save()

    def fail(self, error: Any, stage: str | None = None) -> None:
        """记录错误并进入 FAILED（若仍在活动状态）。"""
        if hasattr(error, "stage") and stage is None:
            stage = getattr(error, "stage", None)
        self.record_error(error)
        if not states.is_terminal(self.state):
            self.set_state(states.FAILED)
        self.meta["failed_stage"] = stage or self.meta.get("failed_stage")
        self.save()

    # ---- 只读视图 ----
    def to_summary(self) -> dict:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "style_id": (self.meta.get("style") or {}).get("id"),
            "created_at": self.meta.get("created_at"),
            "updated_at": self.meta.get("updated_at"),
            "attempt_count": self.meta.get("attempt_count", 0),
            "has_output": self.output_path.is_file(),
        }


class JobStore:
    """runs/ 目录的读写管理。"""

    def __init__(self, runs_dir: Path, temp_suffix: str = ".tmp") -> None:
        self.runs_dir = Path(runs_dir)
        self.temp_suffix = temp_suffix
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        return self.runs_dir / validate_job_id(job_id)

    def create(
        self,
        style: dict | None = None,
        config_snapshot: dict | None = None,
        job_id: str | None = None,
    ) -> Job:
        """建立新任务目录与初始记录（不覆盖已有任务文件）。"""
        jid = validate_job_id(job_id) if job_id else new_job_id()
        directory = self.job_dir(jid)
        if directory.exists() and any(directory.iterdir()):
            raise ConfigError(f"任务目录已存在且非空，拒绝覆盖: {directory}")
        directory.mkdir(parents=True, exist_ok=True)

        now = utc_now_iso()
        meta: dict[str, Any] = {
            "job_id": jid,
            "state": states.CAPTURED,
            "created_at": now,
            "updated_at": now,
            "state_history": [{"state": states.CAPTURED, "at": now}],
            "style": style,
            "config": config_snapshot or {},
            "attempts": [],
            "attempt_count": 0,
            "errors": [],
            "timings": {},
            "print": {"called": False},
        }
        job = Job(self, jid, meta)
        job.save()
        return job

    def exists(self, job_id: str) -> bool:
        return (self.job_dir(job_id) / META_NAME).is_file()

    def load(self, job_id: str) -> Job:
        path = self.job_dir(job_id) / META_NAME
        if not path.is_file():
            raise ConfigError(f"任务记录不存在: {path}")
        meta = read_json(path)
        if not isinstance(meta, dict):
            raise ConfigError(f"任务记录不是 JSON 对象: {path}")
        return Job(self, job_id, meta)

    def list_jobs(self) -> list[Job]:
        jobs: list[Job] = []
        if not self.runs_dir.is_dir():
            return jobs
        for entry in sorted(self.runs_dir.iterdir()):
            if not entry.is_dir():
                continue
            if not (entry / META_NAME).is_file():
                continue
            try:
                jobs.append(self.load(entry.name))
            except (ConfigError, json.JSONDecodeError, OSError):
                continue
        return jobs

    def find_active(self) -> Job | None:
        """返回第一件处于活动状态的任务（用于串行与重启恢复）。"""
        for job in self.list_jobs():
            if states.is_active(job.state):
                return job
        return None

    def find_incomplete(self) -> list[Job]:
        """非终态任务：重启后需要处理（查询进度或判定失败）。"""
        return [job for job in self.list_jobs() if not states.is_terminal(job.state)]

    def find_completed_ids(self) -> set[str]:
        return {job.job_id for job in self.list_jobs() if job.state == states.COMPLETED}
