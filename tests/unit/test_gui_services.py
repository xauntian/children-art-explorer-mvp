"""GUI service preparation is read-only unless local ComfyUI needs startup."""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import gui_services
from app.errors import ConfigError, ServiceUnavailableError, TimeoutError_
from app.generator import ComfyClient, ComfyGenerator, load_workflow
from tests.fixtures import make_app_config, make_registry


@pytest.fixture
def service_env(tmp_path, monkeypatch):
    config = make_app_config(
        tmp_path,
        comfy={"base_url": "http://127.0.0.1:8189"},
        checker={"enabled": False},
    )
    client = Mock(spec=ComfyClient)
    client.base_url = config.comfy_base_url
    client.request_timeout = config.comfy_request_timeout
    client.system_stats.return_value = {"devices": [], "system": {}}
    client.available_checkpoints.return_value = [config.comfy_checkpoint]
    nodes = load_workflow(config.comfy_workflow_path)
    object_info = {node["class_type"]: {} for node in nodes.values()}
    response = Mock()
    response.json.return_value = object_info
    client.session = Mock()
    client.session.get.return_value = response
    generator = ComfyGenerator(
        client=client,
        workflow_nodes=nodes,
        checkpoint=config.comfy_checkpoint,
        negative_prompt=config.negative_prompt,
    )
    app = SimpleNamespace(
        config=config, comfy_client=client, generator=generator,
        registry=make_registry(), checker=Mock(),
    )
    factory = Mock(return_value=app)
    monkeypatch.setattr(gui_services, "App", factory)
    listener = Mock(return_value=False)
    monkeypatch.setattr(gui_services, "_has_listener", listener)
    process = Mock()
    process.pid = 123
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr(gui_services.subprocess, "Popen", popen)
    python = tmp_path / "runtime-python.exe"
    python.write_bytes(b"unused mock executable")
    comfy_dir = tmp_path / "comfy-runtime"
    comfy_dir.mkdir()
    (comfy_dir / "main.py").write_text("# mocked", encoding="utf-8")
    runtime = {"comfyui": {"python": str(python), "directory": str(comfy_dir)}}
    return SimpleNamespace(
        root=tmp_path, config=config, client=client, app=app, factory=factory,
        object_info=object_info, listener=listener, process=process, popen=popen,
        python=python, directory=comfy_dir, runtime=runtime,
    )


def test_reuses_healthy_service_and_checks_workflow_without_generation(service_env):
    env = service_env
    messages = []

    report = gui_services.ensure_generation_service(
        env.root, env.config, {}, progress=messages.append
    )

    assert report["reused"] is True
    assert report["started"] is False
    assert report["pid"] is None
    assert report["log_path"] is None
    assert "KSampler" in report["workflow"]["nodes"]
    assert messages[-1] == "AI 生成服务已就绪"
    env.factory.assert_called_once_with(config=env.config, project_root=env.root)
    env.client.available_checkpoints.assert_called_once_with()
    env.client.session.get.assert_called_once_with(
        "http://127.0.0.1:8189/object_info", timeout=1.0
    )
    env.popen.assert_not_called()
    env.listener.assert_not_called()
    env.client.queue_prompt.assert_not_called()
    env.client.upload_image.assert_not_called()
    env.app.checker.list_models.assert_not_called()


def test_loads_project_config_when_not_provided(service_env, monkeypatch):
    env = service_env
    loader = Mock(return_value=env.config)
    monkeypatch.setattr(gui_services, "load_app_config", loader)

    gui_services.ensure_generation_service(env.root, None, {})

    loader.assert_called_once_with(env.root)


def test_starts_local_service_hidden_then_checks_nodes_and_weights(service_env):
    env = service_env
    env.client.system_stats.side_effect = [
        ServiceUnavailableError("offline"), {"devices": []},
    ]

    report = gui_services.ensure_generation_service(env.root, env.config, env.runtime)

    assert report["started"] is True
    assert report["reused"] is False
    assert report["pid"] == 123
    assert str(env.root / "runs" / "runtime") in report["log_path"]
    command = env.popen.call_args.args[0]
    assert command == [
        str(env.python), str(env.directory / "main.py"),
        "--listen", "127.0.0.1", "--port", "8189",
        "--reserve-vram", "1.0", "--disable-auto-launch",
    ]
    options = env.popen.call_args.kwargs
    assert options["cwd"] == str(env.directory)
    assert options["stdin"] == subprocess.DEVNULL
    assert options["stderr"] == subprocess.STDOUT
    expected_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    assert options["creationflags"] == expected_flags
    assert options["stdout"].name == report["log_path"]
    env.process.terminate.assert_not_called()
    env.process.kill.assert_not_called()
    env.client.queue_prompt.assert_not_called()


def test_missing_workflow_node_blocks_readiness_without_restarting(service_env):
    env = service_env
    del env.object_info["VAEEncode"]

    with pytest.raises(ConfigError, match="VAEEncode"):
        gui_services.ensure_generation_service(env.root, env.config, env.runtime)

    env.popen.assert_not_called()


def test_missing_weight_blocks_readiness_without_restarting(service_env):
    env = service_env
    env.client.available_checkpoints.return_value = []

    with pytest.raises(ConfigError, match="test.safetensors"):
        gui_services.ensure_generation_service(env.root, env.config, env.runtime)

    env.popen.assert_not_called()


@pytest.mark.parametrize("endpoint", [
    "http://192.168.1.10:8189", "http://example.com:8189",
    "http://127.0.0.1:8189/unexpected", "https://127.0.0.1:8189",
    "http://user:secret@127.0.0.1:8189", "http://127.0.0.2:8189",
])
def test_never_auto_starts_for_remote_or_unsupported_endpoint(service_env, endpoint):
    env = service_env
    env.config.comfy_base_url = endpoint
    env.client.system_stats.side_effect = ServiceUnavailableError("offline")

    with pytest.raises(ServiceUnavailableError, match="仅允许自动启动本机"):
        gui_services.ensure_generation_service(env.root, env.config, env.runtime)

    env.popen.assert_not_called()
    env.listener.assert_not_called()


def test_existing_unhealthy_listener_is_never_replaced(service_env):
    env = service_env
    env.client.system_stats.side_effect = ServiceUnavailableError("starting")
    env.listener.return_value = True

    with pytest.raises(ServiceUnavailableError, match="端口已被使用"):
        gui_services.ensure_generation_service(env.root, env.config, env.runtime)

    env.popen.assert_not_called()


def test_missing_runtime_is_reported_before_attempted_start(service_env):
    env = service_env
    env.python.unlink()
    env.client.system_stats.side_effect = ServiceUnavailableError("offline")

    with pytest.raises(ConfigError, match="Python 运行环境不存在"):
        gui_services.ensure_generation_service(env.root, env.config, env.runtime)

    env.popen.assert_not_called()


def test_early_process_exit_reports_log_and_does_not_retry(service_env):
    env = service_env
    env.client.system_stats.side_effect = ServiceUnavailableError("offline")
    env.process.poll.return_value = 2

    with pytest.raises(ServiceUnavailableError, match="退出码 2.*日志"):
        gui_services.ensure_generation_service(env.root, env.config, env.runtime)

    env.popen.assert_called_once()
    env.process.kill.assert_not_called()


def test_startup_timeout_keeps_process_and_reports_log(service_env, monkeypatch):
    env = service_env
    env.client.system_stats.side_effect = ServiceUnavailableError("offline")
    monkeypatch.setattr(gui_services.time, "monotonic", Mock(side_effect=[0, 121]))

    with pytest.raises(TimeoutError_, match="120 秒.*日志"):
        gui_services.ensure_generation_service(env.root, env.config, env.runtime)

    env.process.terminate.assert_not_called()
    env.process.kill.assert_not_called()


def test_spawn_failure_is_readable(service_env):
    env = service_env
    env.client.system_stats.side_effect = ServiceUnavailableError("offline")
    env.popen.side_effect = OSError("executable denied")

    with pytest.raises(ServiceUnavailableError, match="启动失败.*executable denied"):
        gui_services.ensure_generation_service(env.root, env.config, env.runtime)


def test_enabled_checker_is_required_and_no_generation_is_submitted(service_env):
    env = service_env
    env.config.checker_enabled = True
    env.app.checker.list_models.return_value = []

    with pytest.raises(ConfigError, match="test-vlm"):
        gui_services.ensure_generation_service(env.root, env.config, env.runtime)

    env.app.checker.list_models.assert_called_once_with()
    env.client.queue_prompt.assert_not_called()
    env.popen.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("startup_timeout_sec", 0), ("startup_timeout_sec", True),
    ("startup_timeout_sec", float("inf")), ("reserve_vram", -1),
])
def test_invalid_runtime_numeric_values_fail_early(service_env, field, value):
    env = service_env
    env.runtime["comfyui"][field] = value

    with pytest.raises(ConfigError, match=field):
        gui_services.ensure_generation_service(env.root, env.config, env.runtime)

    env.client.system_stats.assert_not_called()
    env.popen.assert_not_called()
