"""Generation deadlines share one budget and only cancel their own prompt."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from app import generator as module
from app.errors import TimeoutError_
from app.generator import ComfyClient, ComfyGenerator


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=100.0, sleeps=[])

    def sleep(seconds):
        state.sleeps.append(seconds)
        state.now += seconds

    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=lambda: state.now, sleep=sleep),
    )
    return state


def response(payload=None, content=b"generated image"):
    result = Mock(status_code=200, content=content, text="")
    result.json.return_value = payload
    return result


def make_generator(client):
    return ComfyGenerator(
        client,
        {
            "1": {"class_type": "LoadImage", "inputs": {"image": "__input_image__"}},
            "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "m"}},
            "3": {
                "class_type": "VAEEncode", "inputs": {"pixels": ["1", 0], "vae": ["2", 2]},
            },
            "4": {"class_type": "KSampler", "inputs": {"latent_image": ["3", 0]}},
            "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0]}},
        },
        "m", "",
    )


def run_generation(generator):
    return generator.generate(
        b"original", "original.png", "paint", 512, 512, 1,
        {"steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple"},
    )


def prepare_requests(client, clock, *, download_seconds=2.0):
    timeouts = []
    prompt_ids = []

    def post(url, **kwargs):
        timeouts.append(kwargs["timeout"])
        clock.now += 2
        if url.endswith("/upload/image"):
            return response({"name": "original.png"})
        prompt_id = kwargs["json"]["prompt_id"]
        prompt_ids.append(prompt_id)
        return response({"prompt_id": prompt_id})

    def get(url, **kwargs):
        timeouts.append(kwargs["timeout"])
        if "/history/" in url:
            clock.now += 3
            return response({
                prompt_ids[0]: {
                    "status": {"completed": True},
                    "outputs": {"5": {"images": [{"filename": "out.png"}]}},
                },
            })
        clock.now += download_seconds
        return response()

    client.session.post.side_effect = post
    client.session.get.side_effect = get
    return timeouts, prompt_ids


def test_generation_shares_budget_across_upload_queue_history_and_download(clock):
    client = ComfyClient("http://mock", session=Mock())
    deadlines, ids = prepare_requests(client, clock)
    registered = Mock()
    cleanup = client.cancel_prompt_async = Mock()

    with client.deadline(10, on_prompt_id=registered):
        data, prompt_id = run_generation(make_generator(client))

    assert data == b"generated image"
    assert prompt_id == ids[0]
    assert [timeout.total for timeout in deadlines] == [10, 8, 6, 3]
    assert [timeout.connect_timeout for timeout in deadlines] == [5, 5, 5, 3]
    registered.assert_called_once_with(prompt_id)
    cleanup.assert_not_called()
    assert client._active_deadline() is None


def test_late_download_cannot_return_success_and_cancels_only_its_prompt(clock):
    client = ComfyClient("http://mock", session=Mock())
    _, ids = prepare_requests(client, clock, download_seconds=4)
    cleanup = client.cancel_prompt_async = Mock()

    with pytest.raises(TimeoutError_) as error, client.deadline(10):
        run_generation(make_generator(client))

    assert error.value.details == {"prompt_id": ids[0], "deadline_seconds": 10}
    cleanup.assert_called_once_with(ids[0])


def test_prompt_id_is_registered_before_a_stalled_submit_returns(clock):
    client = ComfyClient("http://mock", session=Mock())
    registered = []
    cleanup = client.cancel_prompt_async = Mock()

    def submit(_url, **kwargs):
        assert registered == [kwargs["json"]["prompt_id"]]
        clock.now += 10
        raise requests.Timeout("submission timed out")

    client.session.post.side_effect = submit

    with pytest.raises(TimeoutError_) as error, client.deadline(10, on_prompt_id=registered.append):
        client.queue_prompt({}, "client")

    assert error.value.details["prompt_id"] == registered[0]
    cleanup.assert_called_once_with(registered[0])


def test_polling_sleeps_only_for_remaining_budget_and_never_polls_after_expiry(clock):
    client = ComfyClient("http://mock", session=Mock(), poll_interval=20)
    client.session.get.return_value = response({})
    cleanup = client.cancel_prompt_async = Mock()

    with pytest.raises(TimeoutError_), client.deadline(10):
        client.wait_for("my-prompt")

    assert clock.sleeps == [10]
    assert client.session.get.call_count == 1
    cleanup.assert_called_once_with("my-prompt")


def test_expiry_before_upload_does_not_contact_server_or_cancel_other_jobs(clock):
    client = ComfyClient("http://mock", session=Mock())
    cleanup = client.cancel_prompt_async = Mock()

    with pytest.raises(TimeoutError_) as error, client.deadline(10):
        clock.now += 10
        client.upload_image(b"image", "image.png")

    client.session.post.assert_not_called()
    cleanup.assert_not_called()
    assert error.value.details["prompt_id"] is None


def test_deadline_state_is_thread_local_and_legacy_calls_keep_scalar_timeouts(clock):
    import threading

    client = ComfyClient("http://mock", session=Mock(), request_timeout=30)
    client.session.post.return_value = response({"name": "image.png"})
    errors = []

    def upload_in_other_thread():
        try:
            client.upload_image(b"image", "image.png")
        except Exception as exc:  # pragma: no cover - makes thread failure visible
            errors.append(exc)

    with client.deadline(10):
        worker = threading.Thread(target=upload_in_other_thread)
        worker.start()
        worker.join(timeout=1)

    assert not worker.is_alive()
    assert not errors
    assert client.session.post.call_args.kwargs["timeout"] == 30


def test_deadline_restores_after_failure_and_nested_context_cannot_extend_it(clock):
    client = ComfyClient("http://mock", session=Mock())
    with pytest.raises(TimeoutError_), client.deadline(10):
        clock.now += 2
        with client.deadline(100):
            assert client._active_deadline().expires_at == 110
            clock.now = 110
            client.system_stats()

    assert client._active_deadline() is None
    client.session.get.assert_not_called()


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_invalid_total_budget_is_rejected_without_network(seconds):
    client = ComfyClient("http://mock", session=Mock())
    with pytest.raises(ValueError), client.deadline(seconds):
        client.upload_image(b"image", "image.png")
    client.session.post.assert_not_called()


@pytest.mark.parametrize("cancelled", [True, False])
def test_cancellation_uses_only_atomic_job_api_and_independent_session(monkeypatch, cancelled):
    active_session = Mock()
    cleanup_session = Mock()
    cleanup_session.post.return_value = response({"cancelled": cancelled})
    monkeypatch.setattr(module.requests, "Session", Mock(return_value=cleanup_session))
    client = ComfyClient("http://mock", session=active_session)

    client.cancel_prompt("owned-id")

    cleanup_session.post.assert_called_once_with(
        "http://mock/api/jobs/owned-id/cancel", timeout=(2, 2),
    )
    active_session.post.assert_not_called()
    cleanup_session.close.assert_called_once_with()


@pytest.mark.parametrize("failure", [requests.ConnectionError("offline"), requests.Timeout("late")])
def test_cancellation_request_failure_closes_session_without_unsafe_fallback(monkeypatch, failure):
    cleanup_session = Mock()
    cleanup_session.post.side_effect = failure
    monkeypatch.setattr(module.requests, "Session", Mock(return_value=cleanup_session))
    client = ComfyClient("http://mock", connect_timeout=0.5, session=Mock())
    client.cancel_prompt("owned-id")
    cleanup_session.post.assert_called_once_with(
        "http://mock/api/jobs/owned-id/cancel", timeout=(0.5, 0.5),
    )
    cleanup_session.close.assert_called_once_with()


def test_cancellation_http_error_never_falls_back_to_global_interrupt(monkeypatch):
    cleanup_session = Mock()
    cleanup_session.post.return_value = Mock(status_code=503)
    monkeypatch.setattr(module.requests, "Session", Mock(return_value=cleanup_session))
    client = ComfyClient("http://mock", session=Mock())
    client.cancel_prompt("owned-id")
    cleanup_session.post.assert_called_once_with(
        "http://mock/api/jobs/owned-id/cancel", timeout=(2, 2),
    )
    cleanup_session.close.assert_called_once_with()


def test_missing_prompt_id_does_not_create_a_cancellation_session(monkeypatch):
    create_session = Mock()
    monkeypatch.setattr(module.requests, "Session", create_session)
    client = ComfyClient("http://mock", session=Mock())
    client.cancel_prompt("")
    create_session.assert_not_called()


def test_async_cancellation_does_not_wait_for_cleanup(monkeypatch):
    worker = Mock()
    spawn = Mock(return_value=worker)
    monkeypatch.setattr(module.threading, "Thread", spawn)
    client = ComfyClient("http://mock", session=Mock())
    client.cancel_prompt = Mock()

    assert client.cancel_prompt_async("owned-id") is worker

    assert spawn.call_args.kwargs == {
        "target": client.cancel_prompt, "args": ("owned-id",),
        "name": "comfy-prompt-cleanup", "daemon": True,
    }
    worker.start.assert_called_once_with()
    worker.join.assert_not_called()
    client.cancel_prompt.assert_not_called()


def test_legacy_queue_submission_does_not_change_server_assigned_id():
    client = ComfyClient("http://mock", session=Mock())
    client.session.post.return_value = response({"prompt_id": "server-id"})
    assert client.queue_prompt({}, "client") == "server-id"
    assert client.session.post.call_args.kwargs["json"] == {"prompt": {}, "client_id": "client"}
    assert client.session.post.call_args.kwargs["timeout"] == 30
