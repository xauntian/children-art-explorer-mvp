"""Real workflow connectivity and checker-evidence regressions (no model required)."""

from pathlib import Path

import pytest

from app.checker import VisionChecker
from app.errors import CheckTechnicalError
from app.generator import ComfyClient, ComfyGenerator, load_workflow


@pytest.mark.parametrize("lora", [None, "lightning.safetensors"])
def test_qwen_sampling_chain_uses_shift_and_norm_with_or_without_lora(lora):
    path = Path(__file__).resolve().parents[2] / "workflows/image_edit_qwen.json"
    generator = ComfyGenerator(
        ComfyClient("http://127.0.0.1:1"), load_workflow(path), "unused", "",
        unet_name="qwen.gguf", text_encoder="clip.gguf", vae_name="vae.safetensors",
        lora_name=lora, shift=2.75,
    )
    nodes = generator.build_workflow(
        "input.png", "preserve shapes", 512, 512, 123,
        {"steps": 4, "cfg": 1, "denoise": 0.85, "sampler_name": "euler", "scheduler": "simple"},
    )
    assert nodes[generator.binding["ksampler"][0]]["inputs"]["denoise"] == 0.85
    chain = []
    node_id = nodes[generator.binding["ksampler"][0]]["inputs"]["model"][0]
    while node_id:
        node = nodes[node_id]
        chain.append(node["class_type"])
        if node["class_type"] == "ModelSamplingAuraFlow":
            assert node["inputs"]["shift"] == 2.75
        model = node["inputs"].get("model")
        node_id = model[0] if model else None
    expected = ["CFGNorm", "ModelSamplingAuraFlow"]
    if lora:
        expected.append("LoraLoaderModelOnly")
    assert chain == expected + ["UnetLoaderGGUF"]


@pytest.mark.parametrize("api", ["chat", "generate"])
def test_length_limited_model_response_cannot_be_used_as_evidence(monkeypatch, api):
    checker = VisionChecker("http://127.0.0.1:1", "model", api=api)
    monkeypatch.setattr(checker, "_post", lambda *args: {
        "done_reason": "length", "message": {"content": "partial facts"},
        "response": "partial facts",
    })
    with pytest.raises(CheckTechnicalError, match="长度上限"):
        checker._call("describe", [b"image"])


def test_control_judges_original_instead_of_reusing_candidate(monkeypatch):
    checker = VisionChecker("http://127.0.0.1:1", "model")
    calls = []
    original = (
        "STYLE: ORIGINAL MARKER\nSUBJECTS: one circle\n"
        "LAYOUT: centre\nCOLOURS: red"
    )
    candidate = (
        "STYLE: CANDIDATE PAINT\nSUBJECTS: one circle\n"
        "LAYOUT: centre\nCOLOURS: blue"
    )
    replies = iter([
        original, candidate,
        '{"content_preserved":true,"reason":"same circle"}',
        '{"style_match":"match","reason":"painted"}',
        '{"content_preserved":true,"reason":"same original"}',
        '{"style_match":"mismatch","reason":"unchanged marker"}',
    ])

    def respond(prompt, images):
        calls.append((prompt, images))
        return next(replies)

    monkeypatch.setattr(checker, "_call", respond)
    result = checker.check_two_stage(b"original", b"candidate", "Fauvism", "colour", "layout", control=True)
    assert calls[0][1] == [b"original"]
    assert calls[1][1] == [b"candidate"]
    assert "CANDIDATE style evidence:\nSTYLE: CANDIDATE PAINT" in calls[3][0]
    assert "CANDIDATE style evidence:\nSTYLE: ORIGINAL MARKER" in calls[5][0]
    assert "CANDIDATE PAINT" not in calls[5][0]
    assert all(images == [] for _, images in calls[2:])
    assert result.passed
    assert result.extra["control"]["passed"] is False
    assert result.extra["control"]["content_preserved"] is True
