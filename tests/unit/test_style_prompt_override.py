"""Keep selected whole-style prompts intact while preserving template defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import states
from app.config import StyleConfig, load_app_config
from app.errors import ConfigError
from app.pipeline import Pipeline
from tests.fixtures import marker_drawing_png

ROOT = Path(__file__).resolve().parents[2]
LEGACY_STYLES = ("fauvism", "cubism", "impressionism")
EXTENDED_STYLES = (
    "post_impressionism", "expressionism", "surrealism", "neoplasticism", "pop_art",
)


def style_raw() -> dict:
    return {
        "id": "surrealism",
        "name": "Surrealism",
        "name_zh": "超现实主义",
        "enabled": True,
        "style_features": "dream space",
        "preservation_rules": "retain source motifs",
    }


@pytest.mark.parametrize("value", [None, "", " \n\t", False, 1, [], {}])
def test_provided_prompt_requires_nonempty_string(value):
    with pytest.raises(ConfigError, match=r"styles\[0\]\.generate_prompt"):
        StyleConfig({**style_raw(), "generate_prompt": value}, 0)


def test_absent_override_keeps_legacy_snapshot_schema():
    style = StyleConfig(style_raw(), 0)
    assert style.generate_prompt is None
    assert "generate_prompt" not in style.to_dict()


def test_override_is_returned_verbatim_without_template_interpolation(pipeline_env):
    pipeline, *_ = pipeline_env()
    prompt = "  Entire selected prompt.\nKeep literal {style_name} and {unknown}.\n"
    style = StyleConfig({**style_raw(), "generate_prompt": prompt}, 0)
    pipeline.generate_prompt_template = "invalid template without placeholders"
    assert pipeline.build_generate_prompt(style) == prompt


def test_style_snapshot_roundtrip_preserves_exact_override():
    prompt = "Selected prompt.\nPreserve whitespace and 字符.\n"
    original = StyleConfig({**style_raw(), "generate_prompt": prompt}, 0)
    snapshot = json.loads(json.dumps(original.to_dict(), ensure_ascii=False))
    restored = StyleConfig(snapshot, 0)
    assert snapshot["generate_prompt"] == prompt
    assert restored.to_dict() == original.to_dict()
    assert restored.generate_prompt == prompt


def test_new_job_records_and_uses_its_whole_prompt(pipeline_env):
    pipeline, store, generator, _, registry = pipeline_env(
        overrides={"checker": {"enabled": False}}
    )
    prompt = "The complete selected painting prompt.\n"
    registry.get("fauvism").generate_prompt = prompt
    summary = pipeline.run(marker_drawing_png(), style_id="fauvism")
    assert summary["state"] == states.COMPLETED
    job = store.load(summary["job_id"])
    assert generator.calls[0]["positive_prompt"] == prompt
    assert job.meta["prompts"]["generate"] == prompt
    assert job.meta["style"]["generate_prompt"] == prompt
    restored = StyleConfig(job.meta["style"], 0)
    assert pipeline.build_generate_prompt(restored) == prompt


@pytest.mark.parametrize("style_id", LEGACY_STYLES)
def test_production_legacy_styles_keep_existing_template(style_id):
    config = load_app_config(ROOT)
    style = config.load_styles().get(style_id)
    template = config.generate_prompt_path.read_text(encoding="utf-8")
    pipeline = object.__new__(Pipeline)
    pipeline.generate_prompt_template = template
    assert style.generate_prompt is None
    assert pipeline.build_generate_prompt(style) == template.format(
        style_name=style.name,
        style_features=style.style_features,
        preservation_rules=style.preservation_rules,
    )


@pytest.mark.parametrize("style_id", EXTENDED_STYLES)
def test_production_extended_styles_use_the_selected_experiment_prompt(style_id):
    config = load_app_config(ROOT)
    style = config.load_styles().get(style_id)
    source = (
        ROOT / "docs" / "experiments" / "2026-10-19" / "extended-styles-v1"
        / "prompts" / f"{style_id}.txt"
    ).read_text(encoding="utf-8").strip()
    pipeline = object.__new__(Pipeline)
    pipeline.generate_prompt_template = config.generate_prompt_path.read_text(encoding="utf-8")
    assert style.generate_prompt == source
    assert pipeline.build_generate_prompt(style) == source
