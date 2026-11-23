"""Print results, atomic receipts and per-document locking."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

@dataclass(frozen=True)
class PrinterInfo:
    name: str
    driver: str = ""
    port: str = ""
    status: int = 0
    is_default: bool = False


@dataclass(frozen=True)
class PrintResult:
    status: str
    printer: str
    job_id: int
    needs_confirmation: bool
    message: str


class PrintError(RuntimeError):
    """A failure never instructs the caller to automatically resubmit a job."""

    def __init__(
        self,
        message: str,
        *,
        printer: str | None = None,
        job_id: int | None = None,
        requires_resolution: bool = False,
    ):
        super().__init__(message)
        self.printer = printer
        self.job_id = job_id
        self.requires_resolution = requires_resolution


def document_name_for(path: str | Path) -> str:
    """Stable job identity survives GUI restarts and distinguishes session paths."""
    normalized = os.path.normcase(str(Path(path).resolve()))
    token = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]
    return f"AMCC5000-{token}"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def document_lock(receipt_path: Path):
    """Serialize callers using the same receipt directory and artwork path.

    The stable lock file stays on disk. The OS releases its lock when the
    process exits, including after a crash.
    """
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = receipt_path.with_suffix(".lock")
    with lock_path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise PrintError("Another process is handling this artwork.", requires_resolution=True) from exc
        yield
