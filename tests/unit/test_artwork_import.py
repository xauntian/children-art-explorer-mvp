"""完整画作导入保留画面边缘，且不改变相机与默认导入配置。"""

from copy import deepcopy

import numpy as np
import pytest

from app import main as entrypoint
from app.config import AppConfig
from app.preprocess import decode_image_bytes, encode_png
from tests.fixtures import make_app_config, make_registry
from tools import bench_generate, evaluate_styles

PHOTO_PREPROCESS = {
    "frame_detect": True,
    "border_trim_ratio": 0.06,
    "crop_ratios": [0.1, 0.1, 0.1, 0.1],
}
ARTWORK_PREPROCESS = {
    "frame_detect": False,
    "border_trim_ratio": 0.0,
    "crop_ratios": [0.0, 0.0, 0.0, 0.0],
}


def test_artwork_config_is_isolated_from_original(tmp_path):
    config = make_app_config(tmp_path, preprocess=PHOTO_PREPROCESS)
    before = deepcopy(config.raw)
    artwork = config.for_artwork()
    assert config.raw == before
    assert config.pre_frame_detect is True
    assert config.pre_border_trim_ratio == 0.06
    assert config.pre_crop_ratios == (0.1, 0.1, 0.1, 0.1)
    expected = deepcopy(before)
    expected["preprocess"].update(ARTWORK_PREPROCESS)
    assert artwork.raw == expected
    assert artwork.project_root == config.project_root
    # 深层配置也必须隔离，避免后续实验改动污染默认配置。
    artwork.raw["generation"]["samplers"]["fauvism"]["steps"] = 777
    artwork.raw["preprocess"]["crop_ratios"][0] = 0.25
    assert config.raw == before


def test_artwork_pipeline_keeps_all_four_edges_and_records_settings(pipeline_env):
    pipeline, store, generator, checker, _ = pipeline_env(
        overrides={"preprocess": PHOTO_PREPROCESS}
    )
    pipeline.config = pipeline.config.for_artwork()
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[0, :] = (0, 0, 255)
    image[-1, :] = (0, 255, 0)
    image[:, 0] = (255, 0, 0)
    image[:, -1] = (0, 255, 255)
    summary = pipeline.run(encode_png(image), "full-artwork.png")
    job = store.load(summary["job_id"])
    actual = decode_image_bytes(job.input_path.read_bytes())
    np.testing.assert_array_equal(actual, image)
    assert generator.calls[0]["input_bytes"] == checker.calls[0]["original_bytes"]
    assert job.meta["config"]["preprocess"] == ARTWORK_PREPROCESS
    for key, value in ARTWORK_PREPROCESS.items():
        assert job.meta["preprocess"][key] == value
    assert job.meta["preprocess"]["corners"] is None
    assert job.meta["preprocess"]["source_size"] == [64, 64]
    assert job.meta["preprocess"]["input_size"] == [64, 64]


class StopBeforeModels(Exception):
    """在装配模型前结束入口测试。"""


@pytest.mark.parametrize("artwork", [False, True])
@pytest.mark.parametrize("module", [entrypoint, evaluate_styles])
def test_import_and_evaluation_pass_artwork_config_only_when_requested(
    monkeypatch, tmp_path, module, artwork,
):
    config = make_app_config(tmp_path, preprocess=PHOTO_PREPROCESS)
    captured = {}

    def capture_app(**kwargs):
        captured.update(kwargs)
        raise StopBeforeModels

    monkeypatch.setattr(module, "load_app_config", lambda *args: config)
    monkeypatch.setattr(module, "App", capture_app)
    argv = (
        ["import", "--photo", "artwork.png"] if module is entrypoint
        else ["--inputs", "artworks"]
    )
    if artwork:
        argv.append("--artwork")
    with pytest.raises(StopBeforeModels):
        module.main(argv)
    if artwork:
        actual = captured["config"]
        assert actual.pre_frame_detect is False
        assert actual.pre_border_trim_ratio == 0.0
        assert actual.pre_crop_ratios == (0.0, 0.0, 0.0, 0.0)
    else:
        assert captured == {}
    assert config.pre_frame_detect is True


@pytest.mark.parametrize("artwork", [False, True])
def test_benchmark_passes_actual_artwork_settings_to_preprocess(
    monkeypatch, tmp_path, artwork,
):
    config = make_app_config(tmp_path, preprocess=PHOTO_PREPROCESS)
    captured = {}

    def capture_prepare(*args, **kwargs):
        captured.update(kwargs)
        raise StopBeforeModels

    monkeypatch.setattr(bench_generate, "load_app_config", lambda _: config)
    monkeypatch.setattr(AppConfig, "load_styles", lambda _: make_registry())
    monkeypatch.setattr(bench_generate, "import_photo", lambda _: None)
    monkeypatch.setattr(bench_generate, "prepare_model_input", capture_prepare)
    argv = ["--photo", "artwork.png"]
    if artwork:
        argv.append("--artwork")
    with pytest.raises(StopBeforeModels):
        bench_generate.main(argv)
    expected = ARTWORK_PREPROCESS if artwork else PHOTO_PREPROCESS
    assert captured["frame_detect"] is expected["frame_detect"]
    assert captured["border_trim_ratio"] == expected["border_trim_ratio"]
    assert captured["crop_ratios"] == tuple(expected["crop_ratios"])


def test_camera_does_not_accept_artwork_flag():
    with pytest.raises(SystemExit) as exc:
        entrypoint.build_parser().parse_args(["camera", "--artwork"])
    assert exc.value.code == 2
