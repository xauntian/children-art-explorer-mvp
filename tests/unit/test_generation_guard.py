"""Generation supervision tests use disposable workers, never a model or service."""

from __future__ import annotations

import multiprocessing
import pickle
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import generation_guard
from app.errors import ConfigError, GenerationError, ServiceUnavailableError, TimeoutError_
from app.generation_guard import GuardedGenerator

PROMPT_ID = "e5661ef5-5abe-463f-b6d8-af313988c2d0"


def _success_worker(connection, root, result_path, deadline, kwargs):
    connection.send(("prompt_id", PROMPT_ID))
    Path(result_path).write_bytes(b"image-bytes" * 200_000)
    connection.send(("ready", PROMPT_ID))
    connection.close()


def _slow_worker(connection, root, result_path, deadline, kwargs):
    connection.send(("prompt_id", PROMPT_ID))
    time.sleep(5)
    Path(root, "late-output.png").write_bytes(b"must never be written")
    connection.send(("ready", PROMPT_ID))
    connection.close()


def _error_worker(connection, root, result_path, deadline, kwargs):
    connection.send(("error", {
        "category": "service_unavailable", "message": "server unavailable",
        "stage": "GENERATING", "details": {"status": 503},
    }))
    connection.close()


def _empty_worker(connection, root, result_path, deadline, kwargs):
    Path(result_path).write_bytes(b"")
    connection.send(("ready", PROMPT_ID))
    connection.close()


def _crash_worker(connection, root, result_path, deadline, kwargs):
    connection.close()


@pytest.fixture
def delegate():
    return SimpleNamespace(client=Mock(), workflow_nodes={"model": "delegated"})


def test_delegate_configuration_methods_without_calling_delegate_generate(tmp_path, delegate):
    delegate.uses_unet_loader = Mock(return_value=True)
    guard = GuardedGenerator(delegate, tmp_path, deadline=time.monotonic() + 2)
    assert guard.workflow_nodes is delegate.workflow_nodes
    assert guard.uses_unet_loader() is True


@pytest.mark.parametrize("deadline,margin", [(float("nan"), 0.2), (True, 0.2), (4, -1), (4, float("inf"))])
def test_rejects_invalid_timing_configuration(tmp_path, delegate, deadline, margin):
    with pytest.raises(ConfigError):
        GuardedGenerator(delegate, tmp_path, deadline=deadline, finalize_margin_sec=margin)


def test_expired_budget_never_spawns_a_worker(tmp_path, delegate, monkeypatch):
    context = Mock()
    monkeypatch.setattr(generation_guard.multiprocessing, "get_context", context)
    guard = GuardedGenerator(delegate, tmp_path, deadline=time.monotonic() - 1)
    with pytest.raises(TimeoutError_) as caught:
        guard.generate(input_bytes=b"input")
    assert caught.value.details["prompt_id"] is None
    context.assert_not_called()
    delegate.client.cancel_prompt_async.assert_not_called()


def test_returns_large_result_via_private_file_and_cleans_only_own_files(tmp_path, delegate, monkeypatch):
    monkeypatch.setattr(generation_guard, "_generate_worker", _success_worker)
    other_round = tmp_path / "runs/runtime/generation-guard/another-round"
    other_round.mkdir(parents=True)
    sentinel = other_round / "result.png"
    sentinel.write_bytes(b"other round")
    guard = GuardedGenerator(delegate, tmp_path, deadline=time.monotonic() + 4)
    data, prompt_id = guard.generate(input_bytes=b"input")
    assert data == b"image-bytes" * 200_000
    assert prompt_id == PROMPT_ID
    assert sentinel.read_bytes() == b"other round"
    assert list(other_round.parent.iterdir()) == [other_round]
    delegate.client.cancel_prompt_async.assert_not_called()


def test_slow_worker_is_terminated_within_budget_and_cannot_write_late_output(tmp_path, delegate, monkeypatch):
    monkeypatch.setattr(generation_guard, "_generate_worker", _slow_worker)
    previous_pids = {child.pid for child in multiprocessing.active_children()}
    started = time.monotonic()
    guard = GuardedGenerator(delegate, tmp_path, deadline=started + 1.5)
    with pytest.raises(TimeoutError_) as caught:
        guard.generate(input_bytes=b"input")
    assert time.monotonic() - started < 1.7
    assert caught.value.details["prompt_id"] == PROMPT_ID
    delegate.client.cancel_prompt_async.assert_called_once_with(PROMPT_ID)
    assert {child.pid for child in multiprocessing.active_children()} <= previous_pids
    assert not (tmp_path / "late-output.png").exists()
    assert not list((tmp_path / "runs/runtime/generation-guard").iterdir())


def test_worker_technical_failure_preserves_error_category(tmp_path, delegate, monkeypatch):
    monkeypatch.setattr(generation_guard, "_generate_worker", _error_worker)
    guard = GuardedGenerator(delegate, tmp_path, deadline=time.monotonic() + 4)
    with pytest.raises(ServiceUnavailableError, match="server unavailable") as caught:
        guard.generate(input_bytes=b"input")
    assert caught.value.details == {"status": 503}
    assert not list((tmp_path / "runs/runtime/generation-guard").iterdir())


@pytest.mark.parametrize("worker", [_empty_worker, _crash_worker])
def test_empty_output_or_worker_exit_is_a_generation_error(tmp_path, delegate, monkeypatch, worker):
    monkeypatch.setattr(generation_guard, "_generate_worker", worker)
    guard = GuardedGenerator(delegate, tmp_path, deadline=time.monotonic() + 4)
    with pytest.raises(GenerationError):
        guard.generate(input_bytes=b"input")


def test_output_that_arrives_after_deadline_is_never_delivered(tmp_path, delegate, monkeypatch):
    monkeypatch.setattr(generation_guard, "_generate_worker", _success_worker)
    original_read = Path.read_bytes

    def slow_read(path):
        data = original_read(path)
        if path.name == "result.png":
            time.sleep(0.25)
            monkeypatch.setattr(generation_guard.time, "monotonic", lambda: guard.deadline)
        return data

    monkeypatch.setattr(Path, "read_bytes", slow_read)
    guard = GuardedGenerator(delegate, tmp_path, deadline=time.monotonic() + 4)
    with pytest.raises(TimeoutError_):
        guard.generate(input_bytes=b"input")
    delegate.client.cancel_prompt_async.assert_called_once_with(PROMPT_ID)


def test_worker_error_control_message_stays_small():
    payload = generation_guard._error_payload(GenerationError(
        "error " * 3000, stage="GENERATING", details={"large": "中" * 10000},
    ))
    assert len(pickle.dumps(("error", payload))) < 4096
    restored = generation_guard._restore_error(payload)
    assert isinstance(restored, GenerationError)
    assert restored.details == {"worker_details_truncated": True}


def test_actual_worker_only_calls_generator_and_writes_private_result(tmp_path, monkeypatch):
    from app import config, main

    calls = []
    connection = Mock()
    generator = Mock()
    generator.generate.return_value = (b"generated image", PROMPT_ID)

    @contextmanager
    def deadline(seconds, *, on_prompt_id):
        assert 0 < seconds <= 3
        calls.append("deadline")
        on_prompt_id(PROMPT_ID)
        yield

    client = SimpleNamespace(deadline=deadline)
    app = SimpleNamespace(generator=generator, comfy_client=client)
    app_factory = Mock(return_value=app)
    config_value = Mock()
    monkeypatch.setattr(config, "load_app_config", Mock(return_value=config_value))
    monkeypatch.setattr(main, "App", app_factory)
    result = tmp_path / "private-result.png"
    kwargs = {"input_bytes": b"input", "seed": 123}

    generation_guard._generate_worker(
        connection, str(tmp_path), str(result), time.monotonic() + 3, kwargs,
    )

    assert calls == ["deadline"]
    app_factory.assert_called_once_with(config=config_value.for_artwork.return_value)
    generator.generate.assert_called_once_with(**kwargs)
    assert connection.send.call_args_list[0].args == (("prompt_id", PROMPT_ID),)
    assert connection.send.call_args_list[-1].args == (("ready", PROMPT_ID),)
    assert result.read_bytes() == b"generated image"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["private-result.png"]
    connection.close.assert_called_once()


def test_actual_worker_preserves_technical_error_in_control_message(tmp_path, monkeypatch):
    from app import config, main

    connection = Mock()
    monkeypatch.setattr(config, "load_app_config", Mock(side_effect=ConfigError("bad model")))
    app_factory = Mock()
    monkeypatch.setattr(main, "App", app_factory)
    generation_guard._generate_worker(
        connection, str(tmp_path), str(tmp_path / "private-result.png"), time.monotonic() + 3, {},
    )
    kind, payload = connection.send.call_args.args[0]
    assert kind == "error"
    assert payload["category"] == "config_error"
    assert payload["message"] == "bad model"
    app_factory.assert_not_called()
    assert not list(tmp_path.iterdir())
