"""A user generation shares one deadline through validation and final saving."""

from types import SimpleNamespace

import pytest

from app import pipeline as pipeline_module
from app import states
from app.errors import ConfigError, TimeoutError_
from app.storage import Job
from tests.fixtures import make_app_config, marker_drawing_png, solid_png


@pytest.mark.parametrize("budget", [0, -1, True, "10", [], float("nan"), float("inf")])
def test_generation_time_budget_requires_positive_finite_seconds(tmp_path, budget):
    with pytest.raises(ConfigError, match="generation.time_budget_sec"):
        make_app_config(tmp_path, generation={"time_budget_sec": budget})


@pytest.mark.parametrize("budget", [None, 10, 0.5])
def test_generation_time_budget_is_optional(tmp_path, budget):
    config = make_app_config(tmp_path, generation={"time_budget_sec": budget})
    assert config.generation_time_budget_sec == budget
    assert make_app_config(tmp_path).generation_time_budget_sec is None


@pytest.fixture
def deadline_env(pipeline_env, monkeypatch):
    pipeline, store, generator, checker, registry = pipeline_env(
        overrides={"generation": {"time_budget_sec": 10}, "checker": {"enabled": False}}
    )
    clock = SimpleNamespace(value=100.0)
    monkeypatch.setattr(pipeline_module.time, "monotonic", lambda: clock.value)
    guards = []

    class ImmediateGuard:
        def __init__(self, delegate, root, *, deadline, finalize_margin_sec):
            self.delegate = delegate
            self.root = root
            self.deadline = deadline
            self.margin = finalize_margin_sec
            guards.append(self)

        def __getattr__(self, name):
            return getattr(self.delegate, name)

    monkeypatch.setattr(pipeline_module, "GuardedGenerator", ImmediateGuard)
    return SimpleNamespace(
        pipeline=pipeline, store=store, generator=generator, checker=checker,
        registry=registry, clock=clock, guards=guards,
    )


def assert_timeout(env, summary):
    job = env.store.load(summary["job_id"])
    assert summary["state"] == states.FAILED
    assert summary["has_output"] is False
    assert not job.output_path.exists()
    assert job.meta["errors"][-1]["category"] == "timeout"
    assert env.store.find_active() is None
    assert states.COMPLETED not in [item["state"] for item in job.meta["state_history"]]
    assert env.pipeline.generator is env.generator
    assert env.pipeline._run_deadline is None
    return job


def test_success_uses_shared_start_and_reports_budget(deadline_env, monkeypatch):
    env = deadline_env
    generate = env.generator.generate

    def finish(**kwargs):
        result = generate(**kwargs)
        env.clock.value = 104.5
        return result

    monkeypatch.setattr(env.generator, "generate", finish)
    summary = env.pipeline.run(marker_drawing_png(), started_at=95)
    job = env.store.load(summary["job_id"])
    assert summary["state"] == states.COMPLETED
    assert summary["generation_deadline"] == {
        "budget_sec": 10.0, "elapsed_sec": 9.5, "met": True,
    }
    assert job.meta["generation_deadline"] == summary["generation_deadline"]
    assert job.meta["config"]["time_budget_sec"] == 10.0
    assert env.guards[0].deadline == 105
    assert env.guards[0].margin == 0.2
    assert env.guards[0].root == env.pipeline.config.project_root
    assert env.pipeline.generator is env.generator


def test_late_generation_fails_once_and_next_job_can_run(deadline_env, monkeypatch):
    env = deadline_env
    generate = env.generator.generate

    def late_once(**kwargs):
        result = generate(**kwargs)
        if len(env.generator.calls) == 1:
            env.clock.value = 111
        return result

    monkeypatch.setattr(env.generator, "generate", late_once)
    summary = env.pipeline.run(marker_drawing_png())
    job = assert_timeout(env, summary)
    assert job.meta["attempt_count"] == 1
    assert not job.attempt_path(2).exists()
    assert not job.attempt_path(1).exists()
    assert summary["generation_deadline"]["met"] is False
    second = env.pipeline.run(marker_drawing_png())
    assert second["state"] == states.COMPLETED
    assert second["job_id"] != summary["job_id"]
    assert len(env.generator.calls) == 2
    assert env.guards[1].deadline == 121


def test_supervisor_timeout_remains_technical_failure_without_retry(deadline_env, monkeypatch):
    env = deadline_env

    def timeout(**kwargs):
        env.generator.calls.append(kwargs)
        env.clock.value = 109.8
        raise TimeoutError_("worker reached reserved deadline", stage=states.GENERATING)

    monkeypatch.setattr(env.generator, "generate", timeout)
    job = assert_timeout(env, env.pipeline.run(marker_drawing_png()))
    assert job.meta["attempt_count"] == len(env.generator.calls) == 1
    assert job.meta["attempts"][0]["error"]["category"] == "timeout"


def test_expired_external_start_never_submits_generation(deadline_env):
    env = deadline_env
    summary = env.pipeline.run(marker_drawing_png(), started_at=89.9)
    assert_timeout(env, summary)
    assert not env.generator.calls
    assert summary["generation_deadline"]["elapsed_sec"] == 10.1


@pytest.mark.parametrize("expired_decode", [1, 2])
def test_output_decoding_uses_remaining_budget(deadline_env, monkeypatch, expired_decode):
    env = deadline_env
    decode = pipeline_module.decode_image_bytes
    candidate_decodes = 0

    def late_decode(data, name):
        nonlocal candidate_decodes
        image = decode(data, name)
        if name.startswith("attempt_"):
            candidate_decodes += 1
            if candidate_decodes == expired_decode:
                env.clock.value = 110
        return image

    monkeypatch.setattr(pipeline_module, "decode_image_bytes", late_decode)
    job = assert_timeout(env, env.pipeline.run(marker_drawing_png()))
    assert job.meta["attempt_count"] == 1


def test_output_downsampling_cannot_publish_after_budget(deadline_env, monkeypatch):
    env = deadline_env
    env.pipeline.config.pre_paper_format = "a4"
    env.pipeline.config.pre_output_short_side = 720
    env.pipeline.config.pre_max_input_side = 1088
    generate = env.generator.generate
    downsample = pipeline_module.downsample_output

    def native_generate(**kwargs):
        _, prompt_id = generate(**kwargs)
        return solid_png(width=kwargs["width"], height=kwargs["height"]), prompt_id

    def late_resize(data, size):
        result = downsample(data, size)
        env.clock.value = 111
        return result

    monkeypatch.setattr(env.generator, "generate", native_generate)
    monkeypatch.setattr(pipeline_module, "downsample_output", late_resize)
    job = assert_timeout(env, env.pipeline.run(marker_drawing_png()))
    assert job.meta["attempt_count"] == 1
    assert not job.meta.get("output")


def test_final_file_written_after_budget_is_removed_before_failure(deadline_env, monkeypatch):
    env = deadline_env
    save_output = Job.save_output

    def late_save(job, data):
        save_output(job, data)
        env.clock.value = 111

    unrelated = env.pipeline.config.runs_dir / "unrelated.png"
    unrelated.write_bytes(b"preserve")
    monkeypatch.setattr(Job, "save_output", late_save)
    job = assert_timeout(env, env.pipeline.run(marker_drawing_png()))
    assert job.meta["attempt_count"] == 1
    assert not job.meta.get("output")
    assert unrelated.read_bytes() == b"preserve"


def test_legacy_snapshot_does_not_adopt_current_budget(pipeline_env, monkeypatch):
    pipeline, store, generator, _, _ = pipeline_env(overrides={"checker": {"enabled": False}})
    first = pipeline.run(marker_drawing_png())
    job = store.load(first["job_id"])
    job.meta["config"].pop("time_budget_sec")
    job.save()
    pipeline.config.generation_time_budget_sec = 10

    def forbidden_guard(*args, **kwargs):
        raise AssertionError("historical job must retain no deadline")

    monkeypatch.setattr(pipeline_module, "GuardedGenerator", forbidden_guard)
    second = pipeline.run(marker_drawing_png(), job_id=job.job_id, force=True)
    assert second["state"] == states.COMPLETED
    assert "generation_deadline" not in second
    assert len(generator.calls) == 2


def test_existing_budget_snapshot_wins_over_changed_runtime(deadline_env):
    env = deadline_env
    first = env.pipeline.run(marker_drawing_png())
    env.pipeline.config.generation_time_budget_sec = 2
    second = env.pipeline.run(marker_drawing_png(), job_id=first["job_id"], force=True)
    assert second["state"] == states.COMPLETED
    assert second["generation_deadline"]["budget_sec"] == 10
    assert env.guards[-1].deadline == 110


@pytest.mark.parametrize("started_at", [True, "100", float("nan"), float("inf"), 101])
def test_invalid_external_start_is_rejected_before_creating_job(deadline_env, started_at):
    env = deadline_env
    with pytest.raises(ConfigError, match="started_at"):
        env.pipeline.run(marker_drawing_png(), started_at=started_at)
    assert env.store.find_active() is None
    assert not env.generator.calls
