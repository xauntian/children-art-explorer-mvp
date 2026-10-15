"""打印占位接口。

本阶段 print_enabled=false，正常流程保存数字结果后结束，不调用本接口。
即使配置被误设为启用，也不得转而执行真实打印：
不导入打印驱动、不访问打印机、不提交系统打印任务。
"""

from __future__ import annotations

from pathlib import Path

STATUS_NOT_CONNECTED = "not_connected"


def export_to_print(job_id: str, image_path: str | Path, profile: dict | None = None) -> dict:
    """已确认作品的打印占位接口。

    参数:
        job_id: 已确认作品的编号。
        image_path: 已确认图像路径。
        profile: 可选打印配置（本阶段不使用）。

    返回:
        包含 job_id、status、message 的对象。本阶段固定返回 not_connected。

    说明:
        本函数不产生任何打印副作用，也不改变图像的已确认状态。
    """
    return {
        "job_id": job_id,
        "status": STATUS_NOT_CONNECTED,
        "message": (
            "打印模块未接入：本阶段 print_enabled=false，仅提供占位接口，"
            "未导入打印驱动、未访问打印机、未提交系统打印任务。"
        ),
        "image_path": str(image_path),
        "profile": profile,
        "print_enabled": False,
    }


def is_print_enabled(config) -> bool:
    """读取配置中的打印开关（仅用于记录，不影响本阶段的占位行为）。"""
    return bool(getattr(config, "print_enabled", False))
