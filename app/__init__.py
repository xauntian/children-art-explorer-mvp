"""儿童艺术风格探索装置 MVP —— 后台流程包。

流程：粗马克笔创作 → 拍摄或导入 → 裁切预处理 → 随机画派
      → ComfyUI 本地图像生成 → 本地视觉模型粗检 → 自动确认 → 保存数字结果
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "capture",
    "checker",
    "config",
    "errors",
    "generator",
    "main",
    "pipeline",
    "preprocess",
    "print_adapter",
    "states",
    "storage",
    "styles",
]
