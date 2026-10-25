"""No-review generation is distinct from a model-approved delivery."""

from unittest.mock import Mock

import pytest

from app import main as entrypoint
from app import states
from app.errors import ConfigError
from tests.fixtures import make_app_config, make_registry, marker_drawing_png, solid_png


def test_checker_enabled_defaults_true_for_existing_configurations(tmp_path):
    assert make_app_config(tmp_path).checker_enabled is True


@pytest.mark.parametrize("enabled", ["false", "true", 0, 1, None, []])
def test_checker_enabled_requires_a_json_boolean(tmp_path, enabled):
    with pytest.raises(ConfigError, match="checker.enabled"):
        make_app_config(tmp_path, checker={"enabled": enabled})


def test_disabled_pipeline_generates_once_and_records_no_review_without_pass_claim(
    pipeline_env, monkeypatch
):
    pipeline, store, generator, checker, _ = pipeline_env(overrides={"checker": {"enabled": False}})
    checker.technical_error = AssertionError("checker must not be called")
    pipeline.check_prompt_template = None
    monkeypatch.setattr(
        pipeline,
        "build_check_prompt",
        Mock(side_effect=AssertionError("check prompt must not be built")),
    )
    summary = pipeline.run(marker_drawing_png(), "paper.png")
    job = store.load(summary["job_id"])
    assert summary["state"] == states.COMPLETED
    assert len(generator.calls) == 1 and checker.calls == []
    assert job.output_path.read_bytes() == job.attempt_path(1).read_bytes()
    assert not job.check_path(1).exists() and not job.attempt_path(2).exists()
    snapshot = job.meta["config"]
    assert snapshot["checker_enabled"] is False
    assert snapshot["style_check_source"] == "not_performed"
    assert snapshot["check_prompt_file"] is snapshot["checker_prompt_version"] is None
    assert set(job.meta["prompts"]) == {"generate"}
    assert job.meta["review"] == {
        "enabled": False,
        "status": "not_performed",
        "validation_scope": "file_integrity_only",
        "reason": "checker_disabled_by_configuration",
    }
    assert summary["review_status"] == "not_performed"
    assert summary["validation_scope"] == "file_integrity_only"
    attempt = job.meta["attempts"][0]
    assert attempt["file_integrity"] == {"decodable": True, "width": 96, "height": 96}
    assert attempt["check_result"] is attempt["check_passed"] is attempt["check_sec"] is None
    for field in ("style_match", "content_preserved"):
        assert field not in attempt
    assert "confirmed_at" not in job.meta["output"]
    assert job.meta["output"]["review_status"] == "not_performed"
    path = [entry["state"] for entry in job.meta["state_history"]]
    assert states.CHECKING not in path and states.CONFIRMED not in path
    assert path[-2:] == [states.GENERATING, states.COMPLETED]


def test_disabled_snapshot_does_not_attribute_style_to_a_dormant_override(pipeline_env):
    pipeline, _, _, _, _ = pipeline_env(
        overrides={
            "checker": {
                "enabled": False,
                "mode": "comparison",
                "style_validation": "deferred",
            }
        }
    )
    snapshot = pipeline.config_snapshot()
    assert snapshot["style_check_source"] == "not_performed"
    assert snapshot["style_calibration_pending"] is None


def test_disabled_pipeline_accepts_absent_checker_adapter(pipeline_env):
    pipeline, store, generator, _, _ = pipeline_env(overrides={"checker": {"enabled": False}})
    pipeline.checker = None
    summary = pipeline.run(marker_drawing_png(), "paper.png")
    assert summary["state"] == states.COMPLETED and len(generator.calls) == 1
    assert store.load(summary["job_id"]).output_path.exists()


@pytest.mark.parametrize("bad_image", [b"", b"not an image", solid_png(width=8, height=8)])
def test_bad_generation_file_fails_once_without_output_or_review(
    pipeline_env, monkeypatch, bad_image
):
    pipeline, store, generator, checker, _ = pipeline_env(overrides={"checker": {"enabled": False}})
    generate = generator.generate

    def broken(**kwargs):
        _, prompt_id = generate(**kwargs)
        return bad_image, prompt_id

    monkeypatch.setattr(generator, "generate", broken)
    summary = pipeline.run(marker_drawing_png(), "paper.png")
    job = store.load(summary["job_id"])
    assert summary["state"] == states.FAILED
    assert len(generator.calls) == 1 and checker.calls == []
    assert not job.output_path.exists() and not job.attempt_path(2).exists()
    assert not job.check_path(1).exists()
    assert job.meta["errors"][-1]["category"] == "generation_error"


def test_generator_service_failure_is_not_retried_when_review_disabled(pipeline_env):
    pipeline, store, generator, checker, _ = pipeline_env(
        generator_fail_on={1}, overrides={"checker": {"enabled": False}}
    )
    summary = pipeline.run(marker_drawing_png(), "paper.png")
    assert summary["state"] == states.FAILED and len(generator.calls) == 1
    assert checker.calls == [] and not store.load(summary["job_id"]).output_path.exists()


def test_disabled_configuration_and_app_have_no_check_prompt_or_checker_constructor_dependency(
    tmp_path, monkeypatch
):
    config = make_app_config(tmp_path, checker={"enabled": False})
    config.check_prompt_path.unlink()
    config.require_prompt_files()
    generation, review = config.load_prompts(include_check=False)
    assert generation and review is None
    monkeypatch.setattr(config, "load_styles", lambda: make_registry())
    constructor = Mock(side_effect=AssertionError("Ollama adapter must not be constructed"))
    monkeypatch.setattr(entrypoint, "VisionChecker", constructor)
    app = entrypoint.App(config=config)
    assert app.checker is None
    constructor.assert_not_called()
    assert app.pipeline.check_prompt_template is None


def test_enabled_configuration_still_requires_check_prompt(tmp_path):
    config = make_app_config(tmp_path)
    config.check_prompt_path.unlink()
    with pytest.raises(ConfigError, match="Prompt"):
        config.load_prompts()


def test_disabled_doctor_skips_ollama_and_cli_readiness_succeeds(tmp_path, monkeypatch):
    config = make_app_config(tmp_path, checker={"enabled": False})
    monkeypatch.setattr(config, "load_styles", lambda: make_registry())
    app = entrypoint.App(config=config)
    monkeypatch.setattr(app.comfy_client, "system_stats", lambda: {"devices": []})
    monkeypatch.setattr(app.generator, "verify_checkpoint", lambda: None)
    report = app.doctor()
    assert report["vision"] == {
        "probe_status": "skipped",
        "reachable": None,
        "model_ok": None,
        "reason": "checker_disabled",
    }
    assert report["config"]["checker_enabled"] is False
    assert report["config"]["checker_model"] is report["config"]["checker_base_url"] is None
    monkeypatch.setattr(entrypoint, "App", lambda: app)
    assert entrypoint.main(["doctor"]) == 0


@pytest.mark.parametrize("legacy_missing_enabled", [True, False])
def test_disabled_runtime_cannot_force_rerun_an_old_reviewed_job(
    pipeline_env, legacy_missing_enabled
):
    mismatch = {"style_match": "mismatch", "content_preserved": True, "reason": "style mismatch"}
    pipeline, store, generator, _, _ = pipeline_env([mismatch, mismatch])
    original = pipeline.run(marker_drawing_png(), "paper.png")
    job = store.load(original["job_id"])
    if legacy_missing_enabled:
        job.meta["config"].pop("checker_enabled")
        job.save()
    original_meta = job.meta_path.read_bytes()
    calls = len(generator.calls)
    pipeline.config.checker_enabled = False
    pipeline.checker = None
    with pytest.raises(ConfigError, match="原任务启用了审查"):
        pipeline.run(marker_drawing_png(), "paper.png", job_id=job.job_id, force=True)
    assert len(generator.calls) == calls
    assert job.meta_path.read_bytes() == original_meta
    assert not job.output_path.exists()


def test_no_review_job_keeps_its_snapshot_when_app_is_later_enabled(pipeline_env):
    pipeline, store, generator, checker, _ = pipeline_env(overrides={"checker": {"enabled": False}})
    original = pipeline.run(marker_drawing_png(), "paper.png")
    pipeline.config.checker_enabled = True
    checker.technical_error = AssertionError("historical no-review policy must remain unchanged")
    rerun = pipeline.run(marker_drawing_png(), "paper.png", job_id=original["job_id"], force=True)
    assert rerun["state"] == states.COMPLETED
    assert len(generator.calls) == 2 and checker.calls == []
    assert store.load(original["job_id"]).meta["config"]["checker_enabled"] is False


def test_enabled_path_still_requires_an_actual_checker(pipeline_env):
    pipeline, store, generator, _, _ = pipeline_env()
    pipeline.checker = None
    summary = pipeline.run(marker_drawing_png(), "paper.png")
    assert summary["state"] == states.FAILED and generator.calls == []
    assert not store.load(summary["job_id"]).output_path.exists()


def test_direct_completion_is_an_explicit_state_transition():
    states.assert_transition(states.GENERATING, states.COMPLETED)
    assert not states.can_transition(states.GENERATING, states.CONFIRMED)


def test_offline_calibration_can_explicitly_read_check_prompt_without_enabling_production(tmp_path):
    config = make_app_config(tmp_path, checker={"enabled": False})
    generation, review = config.load_prompts()
    assert generation and review
    assert config.checker_enabled is False
