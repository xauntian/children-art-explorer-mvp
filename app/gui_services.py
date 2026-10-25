"""Readiness and hidden local ComfyUI startup for the desktop interface.

These checks never submit an image, generate a warmup image, or print. Existing
services are reused and are never stopped or reconfigured by this module.
"""

from __future__ import annotations

import math
import os
import socket
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import requests

from .config import AppConfig, load_app_config
from .errors import ConfigError, PipelineError, ServiceUnavailableError, TimeoutError_
from .main import App

DEFAULT_COMFY_PYTHON = "comfyui/.venv/Scripts/python.exe"


def _positive_number(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"GUI 配置 {name} 必须是数字")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ConfigError(f"GUI 配置 {name} 必须是{'非负' if allow_zero else '正'}数")
    return number


def _runtime_path(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigError(f"GUI 配置 {name} 必须是非空路径")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _has_listener(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _log_tail(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 3000))
            return handle.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _verify_workflow(app: App) -> dict:
    """Check actual server nodes and every enabled style without queueing work."""
    client = app.comfy_client
    try:
        response = client.session.get(
            f"{client.base_url}/object_info", timeout=client.request_timeout
        )
        response.raise_for_status()
        available = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ServiceUnavailableError(f"无法读取 ComfyUI 工作流节点: {exc}") from exc
    if not isinstance(available, dict):
        raise ServiceUnavailableError("ComfyUI /object_info 返回的节点列表结构异常")

    # Model checks use the same route-aware validation as the CLI doctor.
    app.generator.verify_checkpoint()
    app.generator.verify_controlnet()
    required: set[str] = set()
    for style in app.registry.enabled:
        workflow = app.generator.build_workflow(
            input_image_name="gui-readiness-only.png",
            positive_prompt="readiness check",
            width=768,
            height=1088,
            seed=0,
            params=app.config.sampler_for(style.id),
        )
        required.update(node["class_type"] for node in workflow.values())
    missing = sorted(required - available.keys())
    if missing:
        raise ConfigError("ComfyUI 缺少当前工作流必需的节点: " + ", ".join(missing))
    return {
        "file": str(app.config.comfy_workflow_path),
        "nodes": sorted(required),
        "binding": app.generator.binding,
    }


def ensure_generation_service(
    root: str | Path,
    app_config: AppConfig | None,
    runtime_config: dict,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Reuse a ready service, or start configured local ComfyUI without a window.

    ``runtime_config['comfyui']`` accepts ``python``, ``directory``,
    ``startup_timeout_sec`` (default 120), and ``reserve_vram`` (default 1.0).
    The application configuration remains the source of the endpoint, workflow,
    models, prompts and styles. Automatic startup accepts only plain HTTP on
    localhost/127.0.0.1; a remote endpoint can only be reused if already ready.
    """
    root = Path(root).resolve()
    config = app_config or load_app_config(root)
    if not isinstance(runtime_config, dict):
        raise ConfigError("GUI runtime 配置必须是 JSON 对象")
    service_config = runtime_config.get("comfyui", {})
    if not isinstance(service_config, dict):
        raise ConfigError("GUI 配置 comfyui 必须是 JSON 对象")
    timeout = _positive_number(
        service_config.get("startup_timeout_sec", 120), "comfyui.startup_timeout_sec"
    )
    reserve = _positive_number(
        service_config.get("reserve_vram", 1.0), "comfyui.reserve_vram", allow_zero=True
    )
    emit = progress or (lambda _message: None)
    app = App(config=config, project_root=root)
    emit("正在检查 AI 生成服务…")
    started = False
    process = None
    log_path = None
    try:
        stats = app.comfy_client.system_stats()
    except ServiceUnavailableError as unavailable:
        address = urlsplit(config.comfy_base_url)
        try:
            port = address.port or 80
        except ValueError as exc:
            raise ConfigError(f"ComfyUI 服务地址端口无效: {config.comfy_base_url}") from exc
        if (
            address.scheme != "http"
            or address.hostname not in {"127.0.0.1", "localhost"}
            or address.username is not None
            or address.password is not None
            or address.path not in {"", "/"}
            or address.query
            or address.fragment
        ):
            raise ServiceUnavailableError(
                f"ComfyUI 不可用: {config.comfy_base_url}；仅允许自动启动本机 "
                "http://127.0.0.1 或 http://localhost 服务，请检查指定服务。"
            ) from unavailable
        if _has_listener(address.hostname, port):
            raise ServiceUnavailableError(
                f"ComfyUI 地址 {config.comfy_base_url} 的端口已被使用，但服务尚未就绪。"
                "请等待现有服务启动完成，或检查该服务日志后重试。"
            ) from unavailable

        python = _runtime_path(
            root, service_config.get("python", DEFAULT_COMFY_PYTHON), "comfyui.python"
        )
        directory = _runtime_path(
            root, service_config.get("directory", "comfyui"), "comfyui.directory"
        )
        if not python.is_file():
            raise ConfigError(f"ComfyUI Python 运行环境不存在: {python}") from unavailable
        main_script = directory / "main.py"
        if not main_script.is_file():
            raise ConfigError(f"ComfyUI 启动文件不存在: {main_script}") from unavailable
        runtime_dir = config.runs_dir / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        log_path = runtime_dir / f"comfyui-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.log"
        command = [
            str(python), str(main_script),
            "--listen", "127.0.0.1", "--port", str(port),
            "--reserve-vram", str(reserve), "--disable-auto-launch",
        ]
        emit("正在启动本机 ComfyUI，首次准备可能需要一些时间…")
        try:
            with log_path.open("ab", buffering=0) as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=str(directory),
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
                    close_fds=True,
                )
        except OSError as exc:
            raise ServiceUnavailableError(
                f"ComfyUI 启动失败: {exc}；日志: {log_path}"
            ) from exc
        started = True
        deadline = time.monotonic() + timeout
        last_error: PipelineError = unavailable
        while True:
            exit_code = process.poll()
            if exit_code is not None:
                tail = _log_tail(log_path)
                raise ServiceUnavailableError(
                    f"ComfyUI 启动后退出（退出码 {exit_code}）。日志: {log_path}"
                    + (f"\n{tail}" if tail else "")
                ) from unavailable
            try:
                stats = app.comfy_client.system_stats()
                break
            except ServiceUnavailableError as exc:
                last_error = exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError_(
                    f"等待 ComfyUI 启动超过 {timeout:g} 秒。日志: {log_path}。"
                    f"服务进程保留运行，可待其完成启动后重试。最近错误: {last_error.message}"
                ) from last_error
            time.sleep(min(1.0, remaining))

    if not isinstance(stats, dict):
        raise ServiceUnavailableError("ComfyUI /system_stats 返回的服务信息结构异常")
    emit("正在检查工作流节点和模型权重…")
    workflow_report = _verify_workflow(app)
    if config.checker_enabled:
        from .checker import model_is_available

        emit("正在检查必需的图像检查服务…")
        models = app.checker.list_models()
        if not model_is_available(config.checker_model, models):
            raise ConfigError(f"图像检查服务缺少必需模型: {config.checker_model}")
    emit("AI 生成服务已就绪")
    return {
        "base_url": config.comfy_base_url,
        "reused": not started,
        "started": started,
        "pid": process.pid if process is not None else None,
        "log_path": str(log_path) if log_path is not None else None,
        "workflow": workflow_report,
    }
