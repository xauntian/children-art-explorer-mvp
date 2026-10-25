"""Doctor reports the active model route and distinguishes unprobed services."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from app import main as entrypoint
from app.errors import ServiceUnavailableError
from app.generator import ComfyClient, ComfyGenerator, load_workflow
from tests.fixtures import make_app_config, make_registry

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def doctor_app(tmp_path):
    def build(*, uses_unet=True, missing_lora=False):
        config = make_app_config(
            tmp_path,
            comfy={
                "unet_name": "edit.gguf",
                "text_encoder": "encoder.gguf",
                "vae_name": "vae.safetensors",
                "lora_name": "lightning.safetensors",
            },
        )
        workflow = (
            ROOT / "workflows" / "image_edit_qwen.json"
            if uses_unet
            else config.comfy_workflow_path
        )
        config.comfy_workflow_path = workflow
        client = Mock(spec=ComfyClient)
        client.base_url = config.comfy_base_url
        client.request_timeout = config.comfy_request_timeout
        client.system_stats.return_value = {"devices": [], "system": {"comfyui_version": "test"}}
        client.available_checkpoints.return_value = [config.comfy_checkpoint]
        options = {
            "UnetLoaderGGUF": ("unet_name", [config.comfy_unet_name]),
            "CLIPLoaderGGUF": ("clip_name", [config.comfy_text_encoder]),
            "VAELoader": ("vae_name", [config.comfy_vae_name]),
            "LoraLoaderModelOnly": (
                "lora_name", [] if missing_lora else [config.comfy_lora_name]
            ),
        }

        def object_info(url, **_kwargs):
            class_type = url.rsplit("/", 1)[1]
            field, values = options[class_type]
            response = Mock()
            response.json.return_value = {
                class_type: {"input": {"required": {field: [values]}}}
            }
            return response

        client.session = Mock()
        client.session.get.side_effect = object_info
        app = entrypoint.App.__new__(entrypoint.App)
        app.config = config
        app.registry = make_registry()
        app.comfy_client = client
        app.generator = ComfyGenerator(
            client=client,
            workflow_nodes=load_workflow(workflow),
            checkpoint=config.comfy_checkpoint,
            negative_prompt=config.negative_prompt,
            unet_name=config.comfy_unet_name,
            text_encoder=config.comfy_text_encoder,
            vae_name=config.comfy_vae_name,
            lora_name=config.comfy_lora_name,
        )
        app.checker = Mock()
        app.checker.list_models.return_value = [config.checker_model]
        app.store = Mock()
        app.store.find_active.return_value = None
        app.store.find_incomplete.return_value = []
        return app

    return build


def test_unet_doctor_checks_actual_model_and_dependencies(doctor_app, monkeypatch):
    app = doctor_app()
    monkeypatch.setattr(entrypoint, "App", lambda: app)

    report = app.doctor()

    assert report["config"]["generation_model"] == "edit.gguf"
    assert report["config"]["model_route"] == "unet"
    assert report["config"]["checkpoint"] is None
    assert report["config"]["text_encoder"] == "encoder.gguf"
    assert report["config"]["vae_name"] == "vae.safetensors"
    assert report["config"]["lora_name"] == "lightning.safetensors"
    assert report["comfyui"]["generation_model_ok"] is True
    assert report["comfyui"]["reachable"] is True
    assert report["comfyui"]["probe_status"] == "completed"
    assert {
        call.args[0].rsplit("/", 1)[1] for call in app.comfy_client.session.get.call_args_list
    } == {"UnetLoaderGGUF", "CLIPLoaderGGUF", "VAELoader", "LoraLoaderModelOnly"}
    app.comfy_client.available_checkpoints.assert_not_called()
    assert entrypoint.main(["doctor"]) == 0


def test_missing_lora_fails_readiness_while_service_is_reachable(doctor_app, monkeypatch):
    app = doctor_app(missing_lora=True)
    monkeypatch.setattr(entrypoint, "App", lambda: app)

    report = app.doctor()

    assert report["comfyui"]["reachable"] is True
    assert report["comfyui"]["generation_model_ok"] is False
    assert "lightning.safetensors" in report["comfyui"]["error"]
    assert entrypoint.main(["doctor"]) == 1


def test_checkpoint_workflow_checks_checkpoint_route(doctor_app):
    app = doctor_app(uses_unet=False)

    report = app.doctor()

    assert report["config"]["generation_model"] == app.config.comfy_checkpoint
    assert report["config"]["model_route"] == "checkpoint"
    assert report["config"]["checkpoint"] == app.config.comfy_checkpoint
    assert report["config"]["unet_name"] is None
    assert report["config"]["lora_name"] is None
    assert report["comfyui"]["generation_model_ok"] is True
    app.comfy_client.available_checkpoints.assert_called_once_with()
    app.comfy_client.session.get.assert_not_called()


@pytest.mark.parametrize("missing_lora", [False, True])
def test_checkpoint_lora_route_is_reported_and_checked(doctor_app, missing_lora):
    app = doctor_app(uses_unet=False, missing_lora=missing_lora)
    app.generator.workflow_nodes["14"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"model": ["2", 0], "lora_name": "__lora_name__"},
    }
    report = app.doctor()
    assert report["config"]["lora_name"] == "lightning.safetensors"
    assert report["comfyui"]["generation_model_ok"] is (not missing_lora)
    if missing_lora:
        assert "lightning.safetensors" in report["comfyui"]["error"]
    app.comfy_client.available_checkpoints.assert_called_once_with()
    assert app.comfy_client.session.get.call_args.args[0].endswith("/LoraLoaderModelOnly")


def test_offline_doctor_does_not_claim_services_are_disconnected(doctor_app, monkeypatch):
    app = doctor_app()
    monkeypatch.setattr(entrypoint, "App", lambda: app)

    report = app.doctor(probe_models=False)

    assert report["comfyui"] == {
        "probe_status": "not_run", "reachable": None, "generation_model_ok": None
    }
    assert report["vision"] == {
        "probe_status": "not_run", "reachable": None, "model_ok": None
    }
    assert report["config"]["generation_model"] == "edit.gguf"
    assert entrypoint.main(["doctor", "--offline"]) == 0
    app.comfy_client.system_stats.assert_not_called()
    app.comfy_client.session.get.assert_not_called()
    app.comfy_client.available_checkpoints.assert_not_called()
    app.checker.list_models.assert_not_called()


def test_service_failures_are_distinct_from_skipped_probes(doctor_app):
    app = doctor_app()
    app.comfy_client.system_stats.side_effect = ServiceUnavailableError("ComfyUI stopped")
    app.checker.list_models.side_effect = ServiceUnavailableError("VLM stopped")

    report = app.doctor()

    for service in ("comfyui", "vision"):
        assert report[service]["probe_status"] == "completed"
        assert report[service]["reachable"] is False
        assert "stopped" in report[service]["error"]
    assert report["comfyui"]["generation_model_ok"] is False
    assert report["vision"]["model_ok"] is False


def test_offline_cli_requires_model_for_active_route(doctor_app, monkeypatch):
    app = doctor_app()
    app.config.comfy_unet_name = None
    monkeypatch.setattr(entrypoint, "App", lambda: app)

    assert app.config.comfy_checkpoint
    assert entrypoint.main(["doctor", "--offline"]) == 1


@pytest.mark.parametrize("requested,installed,available", [
    ("qwen3-vl:8b-instruct", ["qwen3-vl:8b"], False),
    ("qwen3-vl:8b-instruct", ["qwen3-vl:8b-instruct"], True),
    ("qwen3-vl", ["qwen3-vl:8b"], False),
    ("qwen3-vl", ["qwen3-vl:latest"], True),
    ("qwen3-vl:latest", ["qwen3-vl"], True),
])
def test_doctor_checks_exact_checker_tag(doctor_app, requested, installed, available):
    app = doctor_app()
    app.config.checker_model = requested
    app.checker.list_models.return_value = installed
    vision = app.doctor()["vision"]
    assert vision["reachable"] is True
    assert vision["model_ok"] is available
    app.checker.list_models.assert_called_once_with()
