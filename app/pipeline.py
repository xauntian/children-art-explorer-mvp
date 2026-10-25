"""状态流转、生成输出与可选审查。

规则要点：
- 关闭审查时只生成一次；启用审查时最多两次，重试保持原图与原画派，只更换种子。
- 技术错误立即失败，不触发质量重试，不转人工审批，不调用云端替代服务。
- 关闭审查的新任务只校验生成文件完整性，直接保存输出，不伪造风格或主题判定。
- 开启审查时，生效的 style_match == "match" 且 content_preserved is True 才保存输出。
  风格可来自明确记录的 user_override；这不表示模型独立验收了画派。
- 同一时刻只处理一件作品。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from math import isfinite

from . import states
from .checker import COMPARISON_VERSION, CheckResult, VisionChecker
from .config import AppConfig, StyleConfig, StyleRegistry
from .errors import (
    ConfigError,
    GenerationError,
    PipelineError,
    PreprocessError,
    TimeoutError_,
)
from .generation_guard import GuardedGenerator
from .generator import ComfyGenerator
from .preprocess import decode_image_bytes, downsample_output, encode_png, preprocess
from .storage import Job, JobStore, utc_now_iso
from .styles import StyleAssigner


@contextmanager
def timed(meta: dict, key: str):
    """把一段耗时记入 meta['timings'][key]。"""
    started = time.monotonic()
    try:
        yield
    finally:
        meta.setdefault("timings", {})[key] = round(time.monotonic() - started, 3)


class Pipeline:
    """串联采集、预处理、随机画派、本地生成、可选检查与保存。"""

    def __init__(
        self,
        config: AppConfig,
        registry: StyleRegistry,
        store: JobStore,
        generator: ComfyGenerator,
        checker: VisionChecker | None,
        generate_prompt_template: str,
        check_prompt_template: str | None,
        style_assigner: StyleAssigner | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.store = store
        self.generator = generator
        self.checker = checker
        self.generate_prompt_template = generate_prompt_template
        self.check_prompt_template = check_prompt_template
        self.style_assigner = style_assigner or StyleAssigner(registry)
        self._run_deadline: float | None = None
        self._run_output_written = False

    # ---------- Prompt ----------
    @staticmethod
    def _fill(template: str, mapping: dict[str, str], label: str) -> str:
        text = template
        for key, value in mapping.items():
            placeholder = "{" + key + "}"
            if placeholder not in text:
                raise ConfigError(f"{label} 缺少占位符 {placeholder}")
            text = text.replace(placeholder, value)
        return text

    def build_generate_prompt(self, style: StyleConfig) -> str:
        if style.generate_prompt is not None:
            return style.generate_prompt
        return self._fill(
            self.generate_prompt_template,
            {
                "style_name": style.name,
                "style_features": style.style_features,
                "preservation_rules": style.preservation_rules,
            },
            "生成 Prompt (prompts/generate.txt)",
        )

    def build_check_prompt(self, style: StyleConfig) -> str:
        if self.check_prompt_template is None:
            raise ConfigError("审查 Prompt 未加载；要审查历史任务，请先启用 checker")
        return self._fill(
            self.check_prompt_template,
            {
                "style_name": style.name,
                "style_features": style.style_features,
                "preservation_rules": style.preservation_rules,
            },
            "检查 Prompt (prompts/check.txt)",
        )

    # ---------- 种子 ----------
    def seed_for_attempt(self, attempt: int) -> int:
        base = self.config.base_seed
        if base is None:
            base = int.from_bytes(__import__("os").urandom(4), "big") % (2**31)
        return int(base + (attempt - 1) * self.config.seed_stride) % (2**31)

    # ---------- 任务创建 ----------
    def config_snapshot(self, style: StyleConfig | None = None) -> dict:
        workflow_path = self.config.comfy_workflow_path
        return {
            "app_config_version": self.config.config_version,
            "style_config_version": self.registry.prompt_version,
            "prompt_version": self.registry.prompt_version,
            "generate_prompt_file": str(self.config.generate_prompt_path.relative_to(self.config.project_root))
            if self.config.generate_prompt_path.is_relative_to(self.config.project_root)
            else str(self.config.generate_prompt_path),
            "check_prompt_file": (
                str(self.config.check_prompt_path.relative_to(self.config.project_root))
                if self.config.check_prompt_path.is_relative_to(self.config.project_root)
                else str(self.config.check_prompt_path)
            ) if self.config.checker_enabled else None,
            "workflow_file": str(workflow_path.relative_to(self.config.project_root))
            if workflow_path.is_relative_to(self.config.project_root)
            else str(workflow_path),
            "workflow_nodes": sorted(
                n["class_type"] for n in self.generator.workflow_nodes.values()
            ),
            "checkpoint": self.config.comfy_checkpoint,
            "controlnet": self.config.comfy_controlnet,
            # 编辑模型路径下的模型与加速配置，保证任务可复现
            "unet_name": self.config.comfy_unet_name,
            "text_encoder": self.config.comfy_text_encoder,
            "vae_name": self.config.comfy_vae_name,
            "lora_name": self.config.comfy_lora_name,
            "lora_strength": self.config.comfy_lora_strength,
            "shift": self.config.comfy_shift,
            "samplers": self.config.samplers,
            "checker_enabled": self.config.checker_enabled,
            "checker_model": self.config.checker_model,
            "checker_mode": self.config.checker_mode,
            "checker_style_validation": self.config.checker_style_validation,
            "style_check_source": (
                "not_performed" if not self.config.checker_enabled else
                "user_override" if self.config.checker_style_validation == "deferred" else "model"
            ),
            "style_calibration_pending": (
                self.config.checker_style_validation == "deferred"
                if self.config.checker_enabled else None
            ),
            "checker_control": self.config.checker_control,
            "checker_api": self.config.checker_api,
            "checker_options": dict(self.config.checker_options),
            "checker_request_timeout": self.config.checker_request_timeout,
            "checker_keep_alive": self.config.checker_keep_alive,
            "checker_prompt_version": (
                COMPARISON_VERSION if self.config.checker_mode == "comparison"
                else self.registry.prompt_version
            ) if self.config.checker_enabled else None,
            "max_attempts": self.config.max_attempts,
            "time_budget_sec": self.config.generation_time_budget_sec,
            "print_enabled": self.config.print_enabled,
            "max_input_side": self.config.pre_max_input_side,
            "preprocess": {
                "frame_detect": self.config.pre_frame_detect,
                "border_trim_ratio": self.config.pre_border_trim_ratio,
                "crop_ratios": list(self.config.pre_crop_ratios),
                **({
                    "paper_format": self.config.pre_paper_format,
                    "output_short_side": self.config.pre_output_short_side,
                } if self.config.pre_paper_format is not None else {}),
            },
        }

    def ensure_idle(self, ignore_job_id: str | None = None) -> None:
        """同一时刻仅处理一个作品任务。

        ignore_job_id 用于「重跑同一个已失败任务」的场景：该任务本身允许继续，
        但仍不允许同时存在另一件活动任务。
        """
        active = self.store.find_active()
        if active is not None and active.job_id != ignore_job_id:
            raise ConfigError(
                f"已有任务处于活动状态，不能同时处理多件作品: {active.job_id} ({active.state})"
            )

    def create_job(self, style_id: str | None = None, job_id: str | None = None) -> Job:
        self.ensure_idle()
        style = (
            self.style_assigner.assign_explicit(style_id)
            if style_id
            else self.style_assigner.assign()
        )
        return self.store.create(
            style=style.to_dict(), config_snapshot=self.config_snapshot(), job_id=job_id
        )

    # ---------- 主流程 ----------
    def run(
        self,
        source_bytes: bytes,
        source_name: str = "photo",
        style_id: str | None = None,
        job_id: str | None = None,
        force: bool = False,
        started_at: float | None = None,
    ) -> dict:
        """执行一件作品的完整流程（阻塞直到 COMPLETED 或 FAILED）。

        force=True 用于重跑一个已失败（终态）的任务：沿用其已保存的画派，
        不重新随机选择画派，不覆盖其他任务的文件。
        """
        started = time.monotonic() if started_at is None else started_at
        if (
            isinstance(started, bool) or not isinstance(started, (int, float))
            or not isfinite(started) or started > time.monotonic()
        ):
            raise ConfigError("started_at 必须是不晚于当前时间的有限 monotonic 时间")
        job = self._prepare_job(style_id=style_id, job_id=job_id, force=force)
        # A historical job without this snapshot key keeps its original policy.
        budget = (job.meta.get("config") or {}).get("time_budget_sec")
        original_generator = self.generator
        previous_deadline = self._run_deadline
        previous_output_written = self._run_output_written
        self._run_deadline = None
        self._run_output_written = False
        run_budget = None
        try:
            if budget is not None:
                if (
                    isinstance(budget, bool) or not isinstance(budget, (int, float))
                    or not isfinite(budget) or budget <= 0
                ):
                    raise ConfigError("任务快照 time_budget_sec 必须是有限的正秒数或 null")
                run_budget = float(budget)
                self._run_deadline = started + run_budget
                job.meta["generation_deadline"] = {"budget_sec": run_budget}
                self.generator = GuardedGenerator(
                    original_generator, self.config.project_root,
                    deadline=self._run_deadline, finalize_margin_sec=0.2,
                )
                self._check_deadline(job)
            self._run_job(job, source_bytes, source_name)
        except PipelineError as exc:
            if isinstance(exc, TimeoutError_) and self._run_output_written:
                # The parent alone writes this exact Job output; a timed-out
                # child cannot recreate it after we discard a late result.
                try:
                    job.output_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    exc.details = {
                        **(exc.details or {}), "output_cleanup_error": str(cleanup_error),
                    }
                job.meta.pop("output", None)
            if not states.is_terminal(job.state):
                job.fail(exc)
        except Exception as exc:  # noqa: BLE001 - 未预期异常也进入失败状态
            if not states.is_terminal(job.state):
                job.fail(PipelineError(f"未预期异常: {exc}", stage=job.state))
        finally:
            self.generator = original_generator
            self._run_deadline = previous_deadline
            self._run_output_written = previous_output_written
            elapsed = time.monotonic() - started
            if run_budget is not None:
                job.meta["generation_deadline"].update(
                    elapsed_sec=round(elapsed, 3), met=elapsed < run_budget,
                )
        job.meta.setdefault("timings", {})["total_sec"] = round(time.monotonic() - started, 3)
        job.save()
        return self.summary(job)

    def _check_deadline(self, job: Job) -> None:
        if self._run_deadline is not None and time.monotonic() >= self._run_deadline:
            budget = (job.meta.get("config") or {}).get("time_budget_sec")
            raise TimeoutError_(
                f"AI 生成超过 {budget:g} 秒，本轮结果未输出",
                stage=states.GENERATING, details={"time_budget_sec": budget},
            )

    def _save_output_before_deadline(self, job: Job, data: bytes) -> None:
        self._check_deadline(job)
        self._run_output_written = True
        try:
            job.save_output(data)
        finally:
            self._check_deadline(job)

    def _prepare_job(
        self, style_id: str | None = None, job_id: str | None = None, force: bool = False
    ) -> Job:
        if job_id and self.store.exists(job_id) and force:
            job = self.store.load(job_id)
            if not states.is_terminal(job.state):
                raise ConfigError(
                    f"任务 {job_id} 仍在活动状态（{job.state}），不能重跑；"
                    "请先运行 resume 处理中断任务"
                )
            if style_id and style_id != (job.meta.get("style") or {}).get("id"):
                raise ConfigError(
                    f"任务 {job_id} 已固定画派 {(job.meta.get('style') or {}).get('id')}，"
                    "重跑不得更换画派"
                )
            self.ensure_idle(ignore_job_id=job_id)
            if (job.meta.get("config") or {}).get("checker_enabled", True) and (
                not self.config.checker_enabled or self.checker is None
            ):
                raise ConfigError("原任务启用了审查；请先启用 checker 后再重跑，不能改为免审查")
            previous = job.state
            job.meta["state"] = states.CAPTURED
            job.meta.setdefault("reruns", []).append(
                {"from_state": previous, "at": utc_now_iso()}
            )
            job.meta.setdefault("state_history", []).append(
                {"state": states.CAPTURED, "at": utc_now_iso(), "note": "手动重跑"}
            )
            job.save()
            return job
        return self.create_job(style_id=style_id, job_id=job_id)


    def _run_job(self, job: Job, source_bytes: bytes, source_name: str) -> None:
        review_enabled = (job.meta.get("config") or {}).get("checker_enabled", True)
        if type(review_enabled) is not bool:
            raise ConfigError("任务快照 checker_enabled 必须是布尔值")
        if review_enabled and self.checker is None:
            raise ConfigError("该任务要求审查；请启用 checker 后再执行")
        if not review_enabled:
            job.meta["review"] = {
                "enabled": False, "status": "not_performed",
                "validation_scope": "file_integrity_only",
                "reason": "checker_disabled_by_configuration",
            }
            job.meta["review_status"] = "not_performed"
            job.meta["validation_scope"] = "file_integrity_only"
        # 1) 采集结果落盘（原始照片）
        with timed(job.meta, "capture_sec"):
            if not source_bytes:
                raise PreprocessError(f"原始照片为空: {source_name}", stage=states.CAPTURED)
            job.save_original(source_bytes)
        self._check_deadline(job)
        job.meta["source_name"] = source_name

        # 2) 读回原图并预处理，保存模型输入
        snapshot = job.meta.get("config") or {}
        saved_preprocess = snapshot.get("preprocess") or {}
        paper_format = saved_preprocess.get("paper_format")
        output_short_side = saved_preprocess.get("output_short_side")
        # 缺少纸张字段的历史任务继续采用原尺寸策略，不继承新 A4 默认值。
        frame_detect = saved_preprocess.get("frame_detect", self.config.pre_frame_detect)
        border_trim = saved_preprocess.get("border_trim_ratio", self.config.pre_border_trim_ratio)
        crop_ratios = tuple(saved_preprocess.get("crop_ratios", self.config.pre_crop_ratios))
        with timed(job.meta, "preprocess_sec"):
            original = decode_image_bytes(source_bytes, source_name)
            result = preprocess(
                original,
                max_input_side=snapshot.get("max_input_side", self.config.pre_max_input_side),
                frame_detect=frame_detect,
                border_trim_ratio=border_trim,
                crop_ratios=crop_ratios,
                min_area_ratio=self.config.pre_min_area_ratio,
                max_area_ratio=self.config.pre_max_area_ratio,
                approx_epsilon_ratio=self.config.pre_approx_epsilon_ratio,
                paper_format=paper_format,
                output_short_side=output_short_side,
            )
            input_bytes = encode_png(result.image)
            job.save_input(input_bytes)
        self._check_deadline(job)
        job.meta["preprocess"] = {
            "frame_detect": frame_detect,
            "border_trim_ratio": border_trim,
            "crop_ratios": list(crop_ratios),
            "crop_used": result.crop_used,
            "corners": result.corners,
            "source_size": list(result.source_size),
            "input_size": list(result.output_size),
        }
        if result.paper_layout is not None:
            job.meta["preprocess"]["paper_layout"] = result.paper_layout
        if review_enabled:
            job.meta["checker_reference_file"] = "input.png"
        job.save()

        style = self.registry.get((job.meta.get("style") or {}).get("id", ""))
        job.set_state(states.STYLE_ASSIGNED)

        generate_prompt = self.build_generate_prompt(style)
        check_prompt = self.build_check_prompt(style) if review_enabled else None
        job.meta["prompts"] = {"generate": generate_prompt}
        if review_enabled:
            job.meta["prompts"]["check"] = check_prompt
            job.meta["check_prompt_version"] = (
                COMPARISON_VERSION if self.config.checker_mode == "comparison"
                else self.registry.prompt_version
            )
        job.meta["generate_prompt_version"] = self.registry.prompt_version
        job.save()

        width, height = result.output_size
        params = dict(self.config.sampler_for(style.id))

        attempt = 1
        while attempt <= self.config.max_attempts:
            self._check_deadline(job)
            seed = self.seed_for_attempt(attempt)
            # 进入本地模型调用前先保存任务与尝试信息
            job.begin_attempt(attempt, seed, {**params, "width": width, "height": height})
            if job.state != states.GENERATING:
                job.set_state(states.GENERATING)

            candidate_bytes, prompt_id = self._generate_once(
                job, attempt, seed, input_bytes, width, height, generate_prompt, params
            )
            self._check_deadline(job)
            if not review_enabled:
                image = decode_image_bytes(candidate_bytes, f"attempt_{attempt}.png")
                self._check_deadline(job)
                job.finish_attempt(
                    attempt, review_status="not_performed",
                    validation_scope="file_integrity_only",
                    file_integrity={"decodable": True, "width": image.shape[1], "height": image.shape[0]},
                )
                with timed(job.meta, "save_output_sec"):
                    output_bytes = candidate_bytes
                    resize_meta = None
                    if result.paper_layout is not None:
                        output_bytes, resize_meta = downsample_output(
                            candidate_bytes, tuple(result.paper_layout["output_size"])
                        )
                    self._check_deadline(job)
                    self._save_output_before_deadline(job, output_bytes)
                output_size = (
                    resize_meta["output_size"] if resize_meta is not None
                    else [image.shape[1], image.shape[0]]
                )
                job.meta["output"] = {
                    "file": "output.png", "bytes": len(output_bytes),
                    "attempt": attempt, "saved_at": utc_now_iso(),
                    "review_status": "not_performed", "validation_scope": "file_integrity_only",
                    "width": output_size[0], "height": output_size[1],
                }
                if resize_meta is not None:
                    job.meta["output"]["resize"] = resize_meta
                job.save()
                self._check_deadline(job)
                job.set_state(states.COMPLETED)
                return
            check_result = self._check_once(
                job, attempt, input_bytes, candidate_bytes, check_prompt, style=style
            )
            self._check_deadline(job)

            if check_result.passed:
                with timed(job.meta, "confirm_sec"):
                    if job.state != states.CONFIRMED:
                        job.set_state(states.CONFIRMED)
                    output_bytes = candidate_bytes
                    resize_meta = None
                    if result.paper_layout is not None:
                        output_bytes, resize_meta = downsample_output(
                            candidate_bytes, tuple(result.paper_layout["output_size"])
                        )
                    self._check_deadline(job)
                    self._save_output_before_deadline(job, output_bytes)
                job.meta["output"] = {
                    "file": "output.png",
                    "bytes": len(output_bytes),
                    "attempt": attempt,
                    "confirmed_at": utc_now_iso(),
                }
                if resize_meta is not None:
                    job.meta["output"].update(
                        width=resize_meta["output_size"][0], height=resize_meta["output_size"][1],
                        resize=resize_meta,
                    )
                job.save()
                self._check_deadline(job)
                job.set_state(states.COMPLETED)
                return

            # 本次质量检查未通过：合法结果但未达标
            if attempt < self.config.max_attempts:
                job.set_state(states.RETRYING)
                attempt += 1
                continue

            job.meta["final_quality_result"] = {**check_result.to_dict(), "attempt": attempt}
            raise PipelineError(
                "第二次质量检查仍未通过，任务失败（保留候选图与检查原因，不产生 output.png）",
                stage=states.CHECKING,
            )

    def _generate_once(
        self,
        job: Job,
        attempt: int,
        seed: int,
        input_bytes: bytes,
        width: int,
        height: int,
        generate_prompt: str,
        params: dict,
    ) -> tuple[bytes, str]:
        started = time.monotonic()
        try:
            candidate_bytes, prompt_id = self.generator.generate(
                input_bytes=input_bytes,
                input_filename=f"{job.job_id}-input.png",
                positive_prompt=generate_prompt,
                width=width,
                height=height,
                seed=seed,
                params=params,
                filename_prefix=f"amcc/{job.job_id}/attempt_{attempt}",
            )
            self._check_deadline(job)
        except PipelineError as exc:
            job.finish_attempt(
                attempt,
                generate_sec=round(time.monotonic() - started, 3),
                error=exc.to_dict(),
            )
            if exc.stage is None:
                exc.stage = states.GENERATING
            raise

        job.save_attempt(attempt, candidate_bytes)
        self._check_deadline(job)
        job.finish_attempt(
            attempt,
            generate_sec=round(time.monotonic() - started, 3),
            comfy_prompt_id=prompt_id,
            candidate_file=f"attempt_{attempt}.png",
            candidate_bytes=len(candidate_bytes),
        )
        # 生成文件必须可解码、尺寸有效
        try:
            decode_image_bytes(candidate_bytes, f"attempt_{attempt}.png")
        except PipelineError as exc:
            job.finish_attempt(attempt, error=exc.to_dict())
            raise GenerationError(
                f"第 {attempt} 次生成结果无法解码: {exc.message}", stage=states.GENERATING
            ) from exc
        self._check_deadline(job)
        return candidate_bytes, prompt_id

    def _check_once(
        self,
        job: Job,
        attempt: int,
        original_bytes: bytes,
        candidate_bytes: bytes,
        check_prompt: str,
        style: StyleConfig | None = None,
    ) -> CheckResult:
        if job.state != states.CHECKING:
            job.set_state(states.CHECKING)
        started = time.monotonic()
        try:
            if style is not None and hasattr(self.checker, "check_pair"):
                result = self.checker.check_pair(
                    original_bytes=original_bytes,
                    candidate_bytes=candidate_bytes,
                    prompt=check_prompt,
                    style_name=style.name,
                    style_features=style.style_features,
                    preservation_rules=style.preservation_rules,
                    attempt=attempt,
                    duration_sec=None,
                    # A rerun retains its original policy. Missing legacy keys
                    # mean enforced, regardless of today's application default.
                    style_validation=(job.meta.get("config") or {}).get(
                        "checker_style_validation", "enforced"
                    ),
                )
            else:
                result = self.checker.check(
                    original_bytes=original_bytes,
                    candidate_bytes=candidate_bytes,
                    prompt=check_prompt,
                    attempt=attempt,
                    duration_sec=None,
                )
        except PipelineError as exc:
            duration = round(time.monotonic() - started, 3)
            # 技术错误：保留可用的原始响应，不伪造成功 JSON
            raw = getattr(exc, "raw_text", None)
            payload = {
                "technical_error": exc.to_dict(),
                "attempt": attempt,
                "timestamp": utc_now_iso(),
            }
            if raw:
                payload["raw_response"] = raw[:4000]
            job.save_check_result(attempt, payload)
            job.finish_attempt(attempt, check_sec=duration, error=exc.to_dict())
            if exc.stage is None:
                exc.stage = states.CHECKING
            raise

        duration = round(time.monotonic() - started, 3)
        result.duration_sec = duration
        job.save_check_result(attempt, result.to_dict())
        job.finish_attempt(
            attempt,
            check_sec=duration,
            check_result=result.to_dict(),
            check_passed=result.passed,
        )
        return result

    # ---------- 重启恢复 ----------
    def resume_scan(self) -> list[dict]:
        """程序启动时读取已有记录：已完成任务不再提交，中断任务不盲目重跑。"""
        reports: list[dict] = []
        for job in self.store.find_incomplete():
            entry = self._resume_one(job)
            reports.append(entry)
        return reports

    def _resume_one(self, job: Job) -> dict:
        state = job.state
        if state in (states.CAPTURED, states.STYLE_ASSIGNED):
            # 尚未提交本地模型调用，可安全标记失败并要求重新发起
            job.fail(
                PipelineError(
                    f"任务在 {state} 阶段中断（未提交生成），已标记失败；"
                    "如需继续请重新发起新任务",
                    stage=state,
                )
            )
            return {"job_id": job.job_id, "previous_state": state, "action": "marked_failed"}

        # 已进入本地模型调用：优先通过 ComfyUI 返回的任务 ID 查询进度
        prompt_id = None
        for record in reversed(job.attempts()):
            if record.get("comfy_prompt_id"):
                prompt_id = record["comfy_prompt_id"]
                break

        if prompt_id:
            try:
                record = self.generator.client.history(prompt_id)
            except PipelineError as exc:
                job.fail(
                    PipelineError(
                        f"无法确认中断任务在本地服务中的执行状态: {exc.message}", stage=state
                    )
                )
                return {
                    "job_id": job.job_id,
                    "previous_state": state,
                    "action": "marked_failed",
                    "reason": "service_unreachable",
                }
            entry = record.get(prompt_id) if isinstance(record, dict) else None
            if entry:
                status = entry.get("status") or {}
                if status.get("completed") is True:
                    job.fail(
                        PipelineError(
                            "任务在中断前已完成生成，但未完成自动确认；"
                            "为避免重复生成与重复确认，标记失败并保留候选图",
                            stage=state,
                        )
                    )
                    return {
                        "job_id": job.job_id,
                        "previous_state": state,
                        "action": "marked_failed",
                        "reason": "generation_already_completed",
                    }
            job.fail(
                PipelineError(
                    f"无法确认本地服务中任务 {prompt_id} 是否仍在执行，标记失败；"
                    "不盲目重新生成，也不重置尝试次数",
                    stage=state,
                )
            )
            return {
                "job_id": job.job_id,
                "previous_state": state,
                "action": "marked_failed",
                "reason": "unknown_execution_state",
            }

        job.fail(
            PipelineError(f"任务在 {state} 阶段中断且无法确认执行状态", stage=state)
        )
        return {
            "job_id": job.job_id,
            "previous_state": state,
            "action": "marked_failed",
            "reason": "no_prompt_id",
        }

    # ---------- 视图 ----------
    def summary(self, job: Job) -> dict:
        summary = job.to_summary()
        summary["error"] = (job.meta.get("errors") or [None])[-1]
        summary["timings"] = job.meta.get("timings", {})
        if "generation_deadline" in job.meta:
            summary["generation_deadline"] = dict(job.meta["generation_deadline"])
        if "review" in job.meta:
            summary["review"] = dict(job.meta["review"])
            summary["review_status"] = job.meta["review_status"]
            summary["validation_scope"] = job.meta["validation_scope"]
        if summary.get("has_output"):
            summary["output_path"] = str(job.output_path)
        return summary
