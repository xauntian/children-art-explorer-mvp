"""Supervise only generation RPC in a disposable process with an absolute deadline.

The caller retains ownership of Job metadata and final outputs. The child writes
only its private scratch result, so killing it cannot leave a late Job writer.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import time
import uuid
from pathlib import Path

from . import errors
from .errors import ConfigError, GenerationError, PipelineError, TimeoutError_
from .storage import atomic_write_bytes


def _error_payload(exc: PipelineError) -> dict:
    """Keep control messages small enough for a single pipe write."""
    payload = exc.to_dict()
    payload["message"] = str(payload["message"])[:1500]
    try:
        encoded = json.dumps(payload, ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        encoded = ""
    if not encoded or len(encoded.encode("utf-8")) > 4000:
        payload["details"] = {"worker_details_truncated": True}
        payload["message"] = str(payload["message"])[:400]
    return payload


def _restore_error(payload: dict) -> PipelineError:
    category = payload.get("category")
    error_type = next(
        (
            value for value in vars(errors).values()
            if isinstance(value, type) and issubclass(value, PipelineError)
            and value.category == category
        ),
        GenerationError,
    )
    return error_type(
        str(payload.get("message", "生成子进程失败")),
        stage=payload.get("stage") or "GENERATING",
        details=payload.get("details"),
    )


def _generate_worker(connection, root: str, result_path: str, deadline: float, kwargs: dict):
    """Run no Pipeline operations and never touch a formal Job directory."""
    try:
        from .config import load_app_config
        from .main import App

        if time.monotonic() >= deadline:
            raise TimeoutError_("AI 生成已到达本轮截止时间", stage="GENERATING")
        app = App(config=load_app_config(Path(root)).for_artwork())
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError_("AI 生成已到达本轮截止时间", stage="GENERATING")
        with app.comfy_client.deadline(
            remaining, on_prompt_id=lambda prompt_id: connection.send(("prompt_id", prompt_id)),
        ):
            data, prompt_id = app.generator.generate(**kwargs)
        if time.monotonic() >= deadline:
            raise TimeoutError_("AI 生成已到达本轮截止时间", stage="GENERATING")
        atomic_write_bytes(Path(result_path), data, retries=0)
        connection.send(("ready", prompt_id))
    except PipelineError as exc:
        connection.send(("error", _error_payload(exc)))
    except Exception as exc:
        connection.send(("error", _error_payload(GenerationError(
            f"生成子进程异常: {type(exc).__name__}: {exc}", stage="GENERATING",
        ))))
    finally:
        connection.close()


class GuardedGenerator:
    """Delegate configuration methods and strictly supervise the generation call."""

    def __init__(self, delegate, root: Path, *, deadline: float, finalize_margin_sec: float = 0.2):
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
            raise ConfigError("AI 截止时间必须是有限的 monotonic 时间")
        if (
            isinstance(finalize_margin_sec, bool)
            or not isinstance(finalize_margin_sec, (int, float))
            or not math.isfinite(finalize_margin_sec)
            or finalize_margin_sec < 0
        ):
            raise ConfigError("AI 结果保存预留时间必须是有限的非负秒数")
        self.delegate = delegate
        self.root = Path(root).resolve()
        self.deadline = float(deadline)
        self.finalize_margin_sec = float(finalize_margin_sec)
        self.created_at = time.monotonic()

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def _timeout(self, prompt_id: str | None) -> TimeoutError_:
        return TimeoutError_(
            "AI 生成已到达本轮截止时间，已停止等待本轮结果",
            stage="GENERATING",
            details={
                "prompt_id": prompt_id,
                "deadline_sec": max(0.0, self.deadline - self.created_at),
                "deadline_monotonic": self.deadline,
            },
        )

    @staticmethod
    def _stop(process) -> bool:
        """Bound teardown and never wait for an unresponsive RPC worker."""
        if process is None or process.pid is None:
            return True
        if process.is_alive():
            process.terminate()
        process.join(timeout=0.03)
        if process.is_alive():
            process.kill()
            process.join(timeout=0.03)
        stopped = not process.is_alive()
        if stopped:
            process.close()
        return stopped

    @staticmethod
    def _cleanup(scratch: Path) -> None:
        # These are the only files this invocation can create. Never recursively
        # remove a directory, follow other paths, or delete another round's data.
        for name in ("result.png", "result.png.tmp"):
            try:
                (scratch / name).unlink(missing_ok=True)
            except OSError:
                pass
        try:
            scratch.rmdir()
        except OSError:
            pass

    def generate(self, **kwargs) -> tuple[bytes, str]:
        expires_at = self.deadline - self.finalize_margin_sec
        if time.monotonic() >= expires_at:
            raise self._timeout(None)
        scratch = self.root / "runs/runtime/generation-guard" / uuid.uuid4().hex
        scratch.mkdir(parents=True, exist_ok=False)
        result_path = scratch / "result.png"
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = None
        prompt_id = None
        delivered = False
        timeout_error = None
        try:
            if time.monotonic() >= expires_at:
                raise self._timeout(None)
            process = context.Process(
                target=_generate_worker,
                args=(child, str(self.root), str(result_path), expires_at, kwargs),
                daemon=True,
                name="amcc-generation-rpc",
            )
            process.start()
            child.close()
            while True:
                remaining = expires_at - time.monotonic()
                if remaining <= 0:
                    raise self._timeout(prompt_id)
                if not parent.poll(min(0.04, remaining)):
                    if not process.is_alive() and not parent.poll(0):
                        raise GenerationError("生成子进程未返回结果即退出", stage="GENERATING")
                    continue
                try:
                    kind, value = parent.recv()
                except EOFError as exc:
                    raise GenerationError("生成子进程未返回结果即断开", stage="GENERATING") from exc
                if kind == "prompt_id":
                    prompt_id = value
                if time.monotonic() >= expires_at:
                    raise self._timeout(prompt_id)
                if kind == "prompt_id":
                    continue
                if kind == "error":
                    raise _restore_error(value)
                if kind != "ready":
                    raise GenerationError("生成子进程返回了无效消息", stage="GENERATING")
                prompt_id = value
                try:
                    data = result_path.read_bytes()
                except OSError as exc:
                    raise GenerationError("生成子进程结果文件不可读取", stage="GENERATING") from exc
                if time.monotonic() >= expires_at:
                    raise self._timeout(prompt_id)
                if not data:
                    raise GenerationError("生成子进程结果文件为空", stage="GENERATING")
                delivered = True
                return data, prompt_id
        except TimeoutError_ as exc:
            timeout_error = exc
            raise
        finally:
            child.close()
            stopped = self._stop(process)
            # Drain only small prompt-id notices so a submission racing timeout
            # can still be cancelled. A late ready result is never delivered.
            if not delivered:
                for _ in range(8):
                    try:
                        if not parent.poll(0):
                            break
                        kind, value = parent.recv()
                    except (EOFError, OSError):
                        break
                    if kind == "prompt_id":
                        prompt_id = value
                if prompt_id:
                    try:
                        self.delegate.client.cancel_prompt_async(prompt_id)
                    except Exception:
                        pass
                if timeout_error is not None:
                    details = dict(timeout_error.details or {})
                    details["prompt_id"] = prompt_id
                    timeout_error.details = details
            parent.close()
            if stopped:
                self._cleanup(scratch)
