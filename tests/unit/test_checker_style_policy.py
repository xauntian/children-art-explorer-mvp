"""Temporary user style override keeps theme and technical checks independent."""

import json

import pytest

from app import states
from app.checker import COMPARISON_CONTENT_PROMPT, COMPARISON_OBSERVATION_PROMPT, VisionChecker
from app.errors import CheckTechnicalError, ConfigError, ServiceUnavailableError
from app.main import App
from tests.fixtures import make_app_config, make_registry, marker_drawing_png, solid_png

CONTENT = {
    "original_subjects": ["house"],
    "candidate_subjects": ["house"],
    "missing_subjects": [],
    "added_subjects": [],
    "layout_changed": False,
    "content_preserved": True,
    "reason": "The house retains the original home theme.",
}
OBSERVATION = {
    "original_treatment": "marker",
    "candidate_treatment": "paint",
    "transformation": "technique",
    "reason": "Visible painted marks differ.",
}
STYLE_FAIL = {"style_match": "mismatch", "reason": "Requested style is not visible."}


def checker(monkeypatch, *, policy="deferred", content=None, technical_error=None):
    instance = VisionChecker(
        "http://unused", "local-model", mode="comparison", style_validation=policy
    )
    calls = []

    def respond(prompt, images):
        calls.append((prompt, images))
        if technical_error is not None:
            raise technical_error
        if prompt == COMPARISON_CONTENT_PROMPT:
            return json.dumps(CONTENT if content is None else content)
        if prompt == COMPARISON_OBSERVATION_PROMPT:
            return json.dumps(OBSERVATION)
        return json.dumps(STYLE_FAIL)

    monkeypatch.setattr(instance, "_call", respond)
    return instance, calls


def check(instance, *, style="Fauvism", **kwargs):
    return instance.check_pair(
        marker_drawing_png(),
        solid_png(),
        "check prompt",
        style,
        "style features",
        "theme association",
        attempt=2,
        **kwargs,
    )


def test_missing_configuration_defaults_to_enforced(tmp_path):
    assert make_app_config(tmp_path).checker_style_validation == "enforced"


@pytest.mark.parametrize("value", ["disabled", "DEFERRED", True, None])
def test_invalid_policy_is_config_error(tmp_path, value):
    with pytest.raises(ConfigError, match="style_validation"):
        make_app_config(tmp_path, checker={"mode": "comparison", "style_validation": value})


@pytest.mark.parametrize("mode", ["two_stage", "pairwise"])
def test_deferred_only_supports_comparison_and_rejects_other_modes(tmp_path, mode):
    with pytest.raises(ConfigError, match="comparison"):
        make_app_config(tmp_path, checker={"mode": mode, "style_validation": "deferred"})
    with pytest.raises(CheckTechnicalError, match="comparison"):
        VisionChecker("http://unused", "model", mode=mode, style_validation="deferred")


@pytest.mark.parametrize(
    "style", ["Fauvism", "Cubism", "Impressionism", "Future experimental style"]
)
def test_deferred_applies_to_every_style_without_style_or_rendering_model_calls(monkeypatch, style):
    instance, calls = checker(monkeypatch)
    result = check(instance, style=style)
    evidence = result.to_dict()
    assert result.passed and result.style_match == "match"
    assert len(calls) == 1 and calls[0][0] == COMPARISON_CONTENT_PROMPT
    assert len(calls[0][1]) == 1
    assert evidence["style_validation"] == "deferred"
    assert evidence["style_check_source"] == "user_override"
    assert evidence["style_calibration_pending"] is True
    assert evidence["model_style_match"] is None and evidence["style_judgement"] is None
    assert evidence["comparison_observation"] is None and evidence["style_model_called"] is False
    assert set(evidence["comparison_raw"]) == set(evidence["comparison_prompts"]) == {"content"}
    assert json.loads(result.raw_text) == evidence["comparison_raw"]
    assert "not performed" in result.reason


@pytest.mark.parametrize("preserved", [False, None])
def test_user_style_override_does_not_promote_false_or_unknown_content(monkeypatch, preserved):
    instance, calls = checker(monkeypatch, content={**CONTENT, "content_preserved": preserved})
    result = check(instance)
    assert result.style_match == "match" and result.content_preserved is preserved
    assert not result.passed and len(calls) == 1


@pytest.mark.parametrize("preserved", ["true", 1])
def test_malformed_content_is_still_technical_error(monkeypatch, preserved):
    instance, calls = checker(monkeypatch, content={**CONTENT, "content_preserved": preserved})
    with pytest.raises(CheckTechnicalError) as error:
        check(instance)
    assert len(calls) == 1
    assert error.value.details["style_validation"] == "deferred"
    assert error.value.details["comparison_component"] == "content"


def test_transport_failure_is_not_a_successful_override(monkeypatch):
    instance, calls = checker(monkeypatch, technical_error=ServiceUnavailableError("offline"))
    with pytest.raises(ServiceUnavailableError) as error:
        check(instance)
    assert len(calls) == 1
    assert error.value.details["style_validation"] == "deferred"
    assert error.value.details["comparison_raw"] == {}


def test_enforced_per_job_policy_overrides_new_deferred_constructor_default(monkeypatch):
    instance, calls = checker(monkeypatch)
    result = check(instance, style_validation="enforced")
    assert len(calls) == 3
    assert result.style_match == "mismatch" and not result.passed
    assert result.extra["style_validation"] == "enforced"
    assert result.extra["style_check_source"] == "model"
    assert result.extra["style_calibration_pending"] is False


def test_legacy_pairwise_entry_cannot_silently_use_deferred_policy(monkeypatch):
    instance, calls = checker(monkeypatch)
    with pytest.raises(CheckTechnicalError, match="check_pair"):
        instance.check(marker_drawing_png(), solid_png(), "legacy prompt")
    assert calls == []


def test_application_constructor_explicitly_wires_policy(tmp_path, monkeypatch):
    config = make_app_config(
        tmp_path, checker={"mode": "comparison", "style_validation": "deferred"}
    )
    monkeypatch.setattr(config, "load_styles", lambda: make_registry())
    app = App(config=config)
    assert app.checker.style_validation == "deferred"
    assert app.pipeline.config_snapshot()["checker_style_validation"] == "deferred"


def test_new_job_saves_policy_and_user_attribution_beside_real_content(pipeline_env, monkeypatch):
    pipeline, store, generator, _, _ = pipeline_env(
        overrides={
            "checker": {
                "mode": "comparison",
                "style_validation": "deferred",
            }
        }
    )
    # Per-job snapshot is authoritative even if a test adapter was constructed
    # with a different default. Runtime still wires its constructor explicitly.
    pipeline.checker, calls = checker(monkeypatch, policy="enforced")
    summary = pipeline.run(marker_drawing_png(), "paper.png")
    job = store.load(summary["job_id"])
    assert summary["state"] == states.COMPLETED
    assert len(generator.calls) == len(calls) == 1
    snapshot = job.meta["config"]
    assert snapshot["checker_style_validation"] == "deferred"
    assert snapshot["style_check_source"] == "user_override"
    assert snapshot["style_calibration_pending"] is True
    evidence = job.read_check_result(1)
    assert evidence["style_validation"] == "deferred"
    assert evidence["style_check_source"] == "user_override"
    assert evidence["comparison_content"]["content_preserved"] is True
    assert job.meta["attempts"][0]["check_result"] == evidence


def test_unrelated_content_still_retries_and_fails_with_policy_provenance(
    pipeline_env, monkeypatch
):
    pipeline, store, generator, _, _ = pipeline_env(
        overrides={
            "checker": {
                "mode": "comparison",
                "style_validation": "deferred",
            }
        }
    )
    pipeline.checker, calls = checker(monkeypatch, content={**CONTENT, "content_preserved": False})
    summary = pipeline.run(marker_drawing_png(), "paper.png")
    job = store.load(summary["job_id"])
    assert summary["state"] == states.FAILED and not job.output_path.exists()
    assert len(generator.calls) == len(calls) == 2
    assert job.meta["final_quality_result"]["content_preserved"] is False
    assert job.meta["final_quality_result"]["style_check_source"] == "user_override"
    assert job.meta["final_quality_result"]["style_calibration_pending"] is True


def test_deferred_technical_failure_does_not_retry_or_create_output(pipeline_env, monkeypatch):
    pipeline, store, generator, _, _ = pipeline_env(
        overrides={
            "checker": {
                "mode": "comparison",
                "style_validation": "deferred",
            }
        }
    )
    pipeline.checker, calls = checker(
        monkeypatch, technical_error=ServiceUnavailableError("offline")
    )
    summary = pipeline.run(marker_drawing_png(), "paper.png")
    job = store.load(summary["job_id"])
    assert summary["state"] == states.FAILED and not job.output_path.exists()
    assert len(generator.calls) == len(calls) == 1
    evidence = job.read_check_result(1)
    assert evidence.get("style_match") is None
    assert evidence["technical_error"]["details"]["style_validation"] == "deferred"


def test_force_rerun_of_legacy_job_retains_enforced_policy(pipeline_env, monkeypatch):
    mismatch = {"style_match": "mismatch", "content_preserved": True, "reason": "style mismatch"}
    pipeline, store, _, _, _ = pipeline_env(
        [mismatch, mismatch], overrides={"checker": {"mode": "comparison"}}
    )
    first = pipeline.run(marker_drawing_png(), "paper.png")
    legacy = store.load(first["job_id"])
    legacy.meta["config"].pop("checker_style_validation")
    legacy.save()
    pipeline.config.checker_style_validation = "deferred"
    pipeline.checker, calls = checker(monkeypatch)
    summary = pipeline.run(marker_drawing_png(), "paper.png", job_id=legacy.job_id, force=True)
    rerun = store.load(legacy.job_id)
    assert summary["state"] == states.FAILED and not rerun.output_path.exists()
    assert len(calls) == 6
    assert "checker_style_validation" not in rerun.meta["config"]
    assert rerun.read_check_result(1)["style_validation"] == "enforced"
    assert rerun.read_check_result(2)["style_validation"] == "enforced"


def test_direct_two_stage_entry_cannot_bypass_deferred_comparison_contract(monkeypatch):
    instance, calls = checker(monkeypatch)
    with pytest.raises(CheckTechnicalError, match="check_pair"):
        instance.check_two_stage(marker_drawing_png(), solid_png(), "Fauvism", "features", "theme")
    assert calls == []
