"""A4 720p 保留构图、原生模型尺寸与历史任务尺寸策略。"""

import numpy as np
import pytest

from app.errors import ConfigError, PreprocessError
from app.preprocess import (
    a4_dimensions,
    contain_a4,
    decode_image_bytes,
    downsample_output,
    encode_png,
    preprocess,
)
from tests.fixtures import make_app_config, marker_drawing_png, solid_png

A4 = {"paper_format": "a4", "output_short_side": 720, "max_input_side": 1088}


@pytest.mark.parametrize(
    "landscape,native,output",
    [(True, (1088, 768), (1018, 720)), (False, (768, 1088), (720, 1018))],
)
def test_a4_dimensions_are_native_32_aligned_and_final_short_side_720(landscape, native, output):
    assert a4_dimensions(720, landscape=landscape) == (native, output)
    assert all(side % 32 == 0 for side in native)
    assert max(output) / min(output) == pytest.approx(297 / 210, rel=0.001)
    assert all(source >= target for source, target in zip(native, output, strict=True))


@pytest.mark.parametrize("portrait", [False, True])
def test_a4_contains_full_image_with_undistorted_content_and_all_edges(portrait):
    image = np.full((1000, 1600, 3), 128, dtype=np.uint8)
    image[:12] = (0, 0, 255)
    image[-12:] = (0, 255, 0)
    image[:, :12] = (255, 0, 0)
    image[:, -12:] = (0, 255, 255)
    if portrait:
        image = np.rot90(image).copy()
    result, layout = contain_a4(image, 720)
    left, top, right, bottom = layout["content_box"]
    content = result[top:bottom, left:right]
    assert (right - left) / (bottom - top) == pytest.approx(
        image.shape[1] / image.shape[0], rel=0.002
    )
    # 原画四条边的中点仍然位于内容框的四条边上。
    for source, actual in [
        (image[0, image.shape[1] // 2], content[0, content.shape[1] // 2]),
        (image[-1, image.shape[1] // 2], content[-1, content.shape[1] // 2]),
        (image[image.shape[0] // 2, 0], content[content.shape[0] // 2, 0]),
        (image[image.shape[0] // 2, -1], content[content.shape[0] // 2, -1]),
    ]:
        np.testing.assert_array_equal(source, actual)
    outside = np.ones(result.shape[:2], dtype=bool)
    outside[top:bottom, left:right] = False
    assert np.all(result[outside] == 255)
    assert layout["orientation"] == ("portrait" if portrait else "landscape")


def test_a4_preprocess_selects_native_dimensions_and_metadata():
    result = preprocess(
        np.full((1400, 1000, 3), 128, dtype=np.uint8), max_input_side=1088,
        frame_detect=False, border_trim_ratio=0, paper_format="a4", output_short_side=720,
    )
    assert result.output_size == (768, 1088)
    assert result.paper_layout["output_size"] == [720, 1018]


def test_a4_final_output_is_downsampled_from_native_not_upscaled():
    output, meta = downsample_output(solid_png(1088, 768), (1018, 720))
    assert decode_image_bytes(output).shape[:2] == (720, 1018)
    assert meta == {
        "method": "lanczos4", "operation": "downsample", "upscaled": False,
        "native_size": [1088, 768], "output_size": [1018, 720],
    }
    with pytest.raises(PreprocessError, match="不能通过放大"):
        downsample_output(solid_png(768, 512), (1018, 720))


@pytest.mark.parametrize(
    "fields", [
        {"paper_format": "letter"},
        {"paper_format": "a4", "output_short_side": True, "max_input_side": 1088},
        {"paper_format": "a4", "output_short_side": 720.0, "max_input_side": 1088},
        {"paper_format": "a4", "output_short_side": 0, "max_input_side": 1088},
        {"paper_format": "a4", "output_short_side": 720, "max_input_side": 768},
        {"output_short_side": 720},
    ],
)
def test_a4_config_rejects_invalid_or_insufficient_sizes(tmp_path, fields):
    with pytest.raises(ConfigError):
        make_app_config(tmp_path, preprocess=fields)


def test_a4_default_short_side_and_old_config_compatibility(tmp_path):
    old = make_app_config(tmp_path)
    assert old.pre_paper_format is old.pre_output_short_side is None
    a4 = make_app_config(tmp_path, preprocess={"paper_format": "a4", "max_input_side": 1088})
    assert a4.pre_output_short_side == 720


def _native_generator(generator, monkeypatch):
    original = generator.generate

    def generate(**kwargs):
        _, prompt_id = original(**kwargs)
        return kwargs["input_bytes"], prompt_id

    monkeypatch.setattr(generator, "generate", generate)


@pytest.mark.parametrize("portrait", [False, True])
@pytest.mark.parametrize("review", [False, True])
def test_a4_pipeline_preserves_native_attempt_and_delivers_final_size(
    pipeline_env, monkeypatch, portrait, review
):
    pipeline, store, generator, _, _ = pipeline_env(
        overrides={"preprocess": A4, "checker": {"enabled": review}}
    )
    _native_generator(generator, monkeypatch)
    image = np.full((1400, 1000, 3) if portrait else (1000, 1400, 3), 200, dtype=np.uint8)
    summary = pipeline.run(encode_png(image), "paper.png")
    assert summary["state"] == "COMPLETED"
    job = store.load(summary["job_id"])
    native, output = a4_dimensions(720, landscape=not portrait)
    assert decode_image_bytes(job.attempt_path(1).read_bytes()).shape[:2] == native[::-1]
    assert decode_image_bytes(job.output_path.read_bytes()).shape[:2] == output[::-1]
    assert job.meta["output"]["resize"]["native_size"] == list(native)
    assert job.meta["output"]["width"] == output[0]
    assert job.meta["output"]["height"] == output[1]
    assert job.meta["config"]["preprocess"]["paper_format"] == "a4"
    assert job.meta["config"]["preprocess"]["output_short_side"] == 720


def test_a4_job_rerun_retains_saved_paper_size_when_current_default_changes(pipeline_env, monkeypatch):
    pipeline, store, generator, _, _ = pipeline_env(
        overrides={"preprocess": A4, "checker": {"enabled": False}}
    )
    _native_generator(generator, monkeypatch)
    source = marker_drawing_png(1400, 1000)
    original = pipeline.run(source, "paper.png")
    pipeline.config.pre_paper_format = None
    pipeline.config.pre_output_short_side = None
    pipeline.config.pre_max_input_side = 128
    summary = pipeline.run(source, "paper.png", job_id=original["job_id"], force=True)
    assert summary["state"] == "COMPLETED"
    job = store.load(original["job_id"])
    assert job.meta["preprocess"]["input_size"] == [1088, 768]
    assert decode_image_bytes(job.output_path.read_bytes()).shape[:2] == (720, 1018)


def test_legacy_job_does_not_inherit_new_a4_default_on_rerun(pipeline_env):
    pipeline, store, generator, _, _ = pipeline_env(overrides={"checker": {"enabled": False}})
    source = marker_drawing_png()
    original = pipeline.run(source, "paper.png")
    original_input_size = store.load(original["job_id"]).meta["preprocess"]["input_size"]
    pipeline.config.pre_paper_format = "a4"
    pipeline.config.pre_output_short_side = 720
    pipeline.config.pre_max_input_side = 1088
    summary = pipeline.run(source, "paper.png", job_id=original["job_id"], force=True)
    assert summary["state"] == "COMPLETED"
    job = store.load(original["job_id"])
    assert job.meta["preprocess"]["input_size"] == original_input_size
    assert "paper_layout" not in job.meta["preprocess"]
    assert "resize" not in job.meta["output"]
    assert len(generator.calls) == 2
