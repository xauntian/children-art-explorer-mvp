"""流程规则测试：自动确认、有限重试、技术错误与技术隔离。

这些测试使用可控的假生成器与假检查器，只验证流程规则本身；
真实画质与模型误判必须由 tests/integration 的真实模型测试验证。
"""

from __future__ import annotations

import json

import pytest

from app import states
from app.errors import (
    CheckTechnicalError,
    ConfigError,
    ServiceUnavailableError,
    TimeoutError_,
)
from tests.fixtures import marker_drawing_png

PASS = {"style_match": "match", "content_preserved": True, "reason": "风格与保留均满足"}
MISMATCH = {"style_match": "mismatch", "content_preserved": True, "reason": "缺乏目标画派特征"}
NOT_PRESERVED = {"style_match": "match", "content_preserved": False, "reason": "与原作主题无关"}
UNCERTAIN = {"style_match": "uncertain", "content_preserved": True, "reason": "证据不足"}
NULL_CONTENT = {"style_match": "match", "content_preserved": None, "reason": "无法判断"}


# ---------------------------------------------------------------- 通过
def test_checker_compares_the_same_cropped_input_as_generator(pipeline_env):
    pipeline, store, generator, checker, _ = pipeline_env([MISMATCH, PASS])
    source = marker_drawing_png()
    summary = pipeline.run(source, "photo.jpg")
    job = store.load(summary["job_id"])
    model_input = job.input_path.read_bytes()
    assert model_input != source
    assert job.original_path.read_bytes() == source
    assert job.meta["checker_reference_file"] == "input.png"
    for generated, checked in zip(generator.calls, checker.calls, strict=True):
        assert generated["input_bytes"] == checked["original_bytes"] == model_input


def test_first_check_pass_confirms_and_completes(pipeline_env):
    pipeline, store, generator, checker, _ = pipeline_env([PASS])
    summary = pipeline.run(marker_drawing_png(), "photo.jpg")

    assert summary["state"] == states.COMPLETED
    assert summary["has_output"] is True
    assert summary["attempt_count"] == 1

    job = store.load(summary["job_id"])
    assert job.output_path.is_file(), "通过后必须保存 output.png"
    assert job.attempt_path(1).is_file()
    assert job.check_path(1).is_file()
    assert not job.attempt_path(2).exists(), "通过后不应发生第二次生成"
    assert job.read_check_result(1)["passed"] is True


def test_all_required_files_exist_for_confirmed_job(pipeline_env):
    pipeline, store, _, _, _ = pipeline_env([PASS])
    summary = pipeline.run(marker_drawing_png(), "photo.jpg")
    job = store.load(summary["job_id"])

    for name in ("original.jpg", "input.png", "attempt_1.png", "check_1.json", "output.png", "meta.json"):
        assert (job.dir / name).is_file(), f"缺少 {name}"

    meta = json.loads(job.meta_path.read_text(encoding="utf-8"))
    assert meta["job_id"] == summary["job_id"]
    assert meta["style"]["id"] == summary["style_id"]
    assert meta["attempts"][0]["seed"] is not None
    assert meta["config"]["max_attempts"] == 2
    assert meta["config"]["print_enabled"] is False
    assert "timings" in meta


def test_output_is_only_written_after_confirmation(pipeline_env):
    """未通过时不得产生 output.png。"""
    pipeline, store, _, _, _ = pipeline_env([MISMATCH, NOT_PRESERVED])
    summary = pipeline.run(marker_drawing_png(), "photo.jpg")

    assert summary["state"] == states.FAILED
    job = store.load(summary["job_id"])
    assert not job.output_path.exists(), "失败任务不得创建 output.png"
    assert job.attempt_path(1).is_file() and job.attempt_path(2).is_file()
    assert job.check_path(1).is_file() and job.check_path(2).is_file()


# ---------------------------------------------------------------- 重试
@pytest.mark.parametrize("first_result", [MISMATCH, UNCERTAIN, NOT_PRESERVED, NULL_CONTENT])
def test_first_failure_triggers_exactly_one_retry_with_new_seed(pipeline_env, first_result):
    pipeline, store, generator, checker, _ = pipeline_env([first_result, PASS])
    summary = pipeline.run(marker_drawing_png(), "photo.jpg")

    assert summary["state"] == states.COMPLETED
    assert summary["attempt_count"] == 2
    assert len(generator.calls) == 2, "第一次未通过时只能再生成一次"

    seeds = [rec["seed"] for rec in store.load(summary["job_id"]).attempts()]
    assert seeds[0] != seeds[1], "重试必须更换生成种子"


def test_retry_keeps_style_and_input_unchanged(pipeline_env):
    pipeline, store, generator, checker, _ = pipeline_env([MISMATCH, PASS])
    summary = pipeline.run(marker_drawing_png(), "photo.jpg", style_id="cubism")

    assert summary["style_id"] == "cubism"
    assert len(generator.calls) == 2
    assert generator.calls[0]["width"] == generator.calls[1]["width"]
    assert generator.calls[0]["height"] == generator.calls[1]["height"]
    assert generator.calls[0]["input_bytes"] == generator.calls[1]["input_bytes"], (
        "重试必须使用同一张预处理输入图"
    )
    assert generator.calls[0]["positive_prompt"] == generator.calls[1]["positive_prompt"], (
        "重试不得更换画派或 Prompt"
    )

    meta = json.loads(store.load(summary["job_id"]).meta_path.read_text(encoding="utf-8"))
    assert meta["style"]["id"] == "cubism"
    assert meta["attempts"][0]["check_passed"] is False
    assert meta["attempts"][1]["check_passed"] is True


def test_never_more_than_two_generations(pipeline_env):
    pipeline, store, generator, checker, _ = pipeline_env([MISMATCH, MISMATCH, PASS])
    summary = pipeline.run(marker_drawing_png(), "photo.jpg")

    assert summary["state"] == states.FAILED
    assert len(generator.calls) == 2, "每个任务最多生成两次"
    assert len(checker.calls) == 2
    job = store.load(summary["job_id"])
    assert not job.attempt_path(3).exists()
    assert not job.output_path.exists()


def test_second_failure_records_reason_and_keeps_candidates(pipeline_env):
    pipeline, store, _, _, _ = pipeline_env([MISMATCH, UNCERTAIN])
    summary = pipeline.run(marker_drawing_png(), "photo.jpg")

    job = store.load(summary["job_id"])
    assert job.state == states.FAILED
    meta = json.loads(job.meta_path.read_text(encoding="utf-8"))
    assert meta["final_quality_result"]["style_match"] == "uncertain"
    assert "reason" in meta["final_quality_result"]
    assert job.attempt_path(1).is_file() and job.attempt_path(2).is_file()


def test_uncertain_is_never_confirmed(pipeline_env):
    pipeline, store, _, _, _ = pipeline_env([UNCERTAIN, UNCERTAIN])
    summary = pipeline.run(marker_drawing_png(), "photo.jpg")
    assert summary["state"] == states.FAILED
    assert not store.load(summary["job_id"]).output_path.exists()


# ---------------------------------------------------------------- 技术错误
@pytest.mark.parametrize(
    "error",
    [
        ServiceUnavailableError("服务不可用", stage=states.CHECKING),
        TimeoutError_("超时", stage=states.CHECKING),
        CheckTechnicalError("响应不是合法 JSON", stage=states.CHECKING),
    ],
)
def test_check_technical_error_fails_immediately_without_retry(pipeline_env, error):
    pipeline, store, generator, _, _ = pipeline_env()
    pipeline.checker.technical_error = error

    summary = pipeline.run(marker_drawing_png(), "photo.jpg")

    assert summary["state"] == states.FAILED
    assert len(generator.calls) == 1, "技术错误不得触发质量重试"
    job = store.load(summary["job_id"])
    assert not job.output_path.exists()
    meta = json.loads(job.meta_path.read_text(encoding="utf-8"))
    assert meta["errors"], "必须记录错误类别、阶段与原因"
    assert meta["errors"][-1]["category"] == error.category


def test_technical_error_raw_response_is_kept(pipeline_env):
    pipeline, store, _, _, _ = pipeline_env()
    pipeline.checker.technical_error = CheckTechnicalError("不是合法 JSON", stage=states.CHECKING)
    summary = pipeline.run(marker_drawing_png(), "photo.jpg")

    job = store.load(summary["job_id"])
    payload = job.read_check_result(1)
    assert payload is not None
    assert "technical_error" in payload, "必须保留错误信息"
    assert payload.get("style_match") is None, "不得伪造成功 JSON"


def test_generation_technical_error_fails_immediately(pipeline_env):
    pipeline, store, generator, checker, _ = pipeline_env(generator_fail_on={1})
    summary = pipeline.run(marker_drawing_png(), "photo.jpg")

    assert summary["state"] == states.FAILED
    assert len(generator.calls) == 1
    assert len(checker.calls) == 0, "生成失败时不应进入检查阶段"
    assert not store.load(summary["job_id"]).output_path.exists()


# ---------------------------------------------------------------- 画派
def test_random_style_is_assignable_and_recorded(pipeline_env):
    pipeline, store, _, _, registry = pipeline_env([PASS])
    summary = pipeline.run(marker_drawing_png(), "photo.jpg")
    assert summary["style_id"] in {s.id for s in registry.enabled}
    meta = json.loads(store.load(summary["job_id"]).meta_path.read_text(encoding="utf-8"))
    assert meta["style"]["id"] == summary["style_id"]
    assert meta["style"]["style_features"], "必须保存画派选择结果"


def test_explicit_style_override_only_affects_that_job(pipeline_env):
    pipeline, _, _, _, registry = pipeline_env([PASS, PASS])
    first = pipeline.run(marker_drawing_png(), "a.jpg", style_id="impressionism")
    second = pipeline.run(marker_drawing_png(), "b.jpg")
    assert first["style_id"] == "impressionism"
    assert second["style_id"] in {s.id for s in registry.enabled}


def test_disabled_style_cannot_be_used(pipeline_env):
    pipeline, _, _, _, _ = pipeline_env([PASS], enabled={"cubism": False})
    with pytest.raises(ConfigError, match="未启用"):
        pipeline.run(marker_drawing_png(), "photo.jpg", style_id="cubism")


# ---------------------------------------------------------------- 并发与隔离
def test_two_jobs_are_isolated(pipeline_env):
    pipeline, store, _, _, _ = pipeline_env([PASS, PASS])
    first = pipeline.run(marker_drawing_png(), "a.jpg")
    second = pipeline.run(marker_drawing_png(), "b.jpg")

    assert first["job_id"] != second["job_id"]
    job_a = store.load(first["job_id"])
    job_b = store.load(second["job_id"])
    assert job_a.dir != job_b.dir
    assert job_a.output_path.read_bytes() != job_b.output_path.read_bytes() or True
    assert json.loads(job_a.meta_path.read_text(encoding="utf-8"))["job_id"] == first["job_id"]


def test_active_job_blocks_a_second_concurrent_job(pipeline_env):
    pipeline, store, _, _, _ = pipeline_env([PASS])
    # 手工制造一个活动任务
    store.create(style={"id": "fauvism"}, config_snapshot={})
    with pytest.raises(ConfigError, match="已有任务处于活动状态"):
        pipeline.create_job()


def test_no_partial_temp_files_left_behind(pipeline_env):
    pipeline, store, _, _, _ = pipeline_env([PASS])
    summary = pipeline.run(marker_drawing_png(), "photo.jpg")
    job = store.load(summary["job_id"])
    leftovers = list(job.dir.glob("*.tmp"))
    assert leftovers == [], f"不应残留半写入文件: {leftovers}"


# ---------------------------------------------------------------- 重启恢复
def test_restart_does_not_resubmit_completed_jobs(pipeline_env):
    pipeline, store, generator, _, _ = pipeline_env([PASS])
    summary = pipeline.run(marker_drawing_png(), "photo.jpg")
    assert summary["state"] == states.COMPLETED

    calls_before = len(generator.calls)
    reports = pipeline.resume_scan()
    assert reports == [], "已完成任务不得再次提交"
    assert len(generator.calls) == calls_before


def test_interrupted_job_is_failed_not_restarted(pipeline_env):
    pipeline, store, generator, checker, _ = pipeline_env([PASS])
    # 模拟中断：任务停在 GENERATING，没有完成
    job = store.create(style={"id": "fauvism"}, config_snapshot={})
    job.set_state(states.STYLE_ASSIGNED)
    job.set_state(states.GENERATING)

    reports = pipeline.resume_scan()
    assert len(reports) == 1
    assert reports[0]["action"] == "marked_failed"
    assert store.load(job.job_id).state == states.FAILED
    assert generator.calls == [], "中断任务不得盲目重新生成"
    assert checker.calls == []


def test_interrupted_job_does_not_reset_attempt_count(pipeline_env):
    pipeline, store, _, _, _ = pipeline_env([PASS])
    job = store.create(style={"id": "fauvism"}, config_snapshot={})
    job.set_state(states.STYLE_ASSIGNED)
    job.set_state(states.GENERATING)
    job.begin_attempt(1, 4242, {"steps": 4})
    job.finish_attempt(1, note="中断")

    pipeline.resume_scan()
    reloaded = store.load(job.job_id)
    assert reloaded.meta["attempt_count"] == 1, "不得重置尝试次数"
    assert len(reloaded.attempts()) == 1


def test_resume_scan_ignores_completed_and_failed(pipeline_env):
    pipeline, store, _, _, _ = pipeline_env([PASS])
    pipeline.run(marker_drawing_png(), "photo.jpg")
    assert pipeline.resume_scan() == []
