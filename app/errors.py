"""错误分类与异常定义。

技术错误（TechnicalError）直接导致任务失败，不触发质量重试。
"""

from __future__ import annotations


class PipelineError(Exception):
    """流程基础异常。"""

    category = "pipeline_error"

    def __init__(
        self, message: str, stage: str | None = None, *, details: dict | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.details = dict(details) if details is not None else None

    def to_dict(self) -> dict:
        payload = {
            "category": self.category,
            "stage": self.stage,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload


class ConfigError(PipelineError):
    """配置缺失、字段非法或没有启用画派。"""

    category = "config_error"


class CaptureError(PipelineError):
    """摄像头或照片导入失败。"""

    category = "capture_error"


class PreprocessError(PipelineError):
    """裁切、透视校正或图像解码失败。"""

    category = "preprocess_error"


class ServiceUnavailableError(PipelineError):
    """本地服务不可用或连接失败。"""

    category = "service_unavailable"


class TimeoutError_(PipelineError):
    """本地服务超时。"""

    category = "timeout"


class GenerationError(PipelineError):
    """生成阶段技术错误（工作流提交失败、输出文件缺失或损坏）。"""

    category = "generation_error"


class CheckTechnicalError(PipelineError):
    """检查阶段技术错误（响应非 JSON、字段缺失、类型或枚举非法、文件损坏）。"""

    category = "check_technical_error"
