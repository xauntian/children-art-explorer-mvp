"""画派配置校验与随机分配。

随机选择只在新任务中执行一次；重试、检查失败与保存失败都不得重新选择画派。
"""

from __future__ import annotations

import random

from .config import StyleConfig, StyleRegistry
from .errors import ConfigError


class StyleAssigner:
    """等概率随机分配启用画派。

    rng 可由测试注入，以便用可控随机输入验证选择逻辑，
    而不是依赖偶然抽样次数。
    """

    def __init__(self, registry: StyleRegistry, rng: random.Random | None = None) -> None:
        self.registry = registry
        self.rng = rng or random.Random()

    def enabled_ids(self) -> list[str]:
        return [s.id for s in self.registry.require_enabled()]

    def assign(self) -> StyleConfig:
        """新任务：从启用列表等概率选择一种画派。"""
        candidates = self.registry.require_enabled()
        return self.rng.choice(candidates)

    def assign_explicit(self, style_id: str) -> StyleConfig:
        """测试入口：显式指定画派，覆盖随机行为；不改变正常入口的随机逻辑。"""
        return self.registry.resolve(style_id)


def validate_registry(registry: StyleRegistry) -> list[str]:
    """启动前校验画派列表、字段与启用状态；返回启用画派 id。"""
    if not registry.styles:
        raise ConfigError("画派列表为空，请检查 configs/styles.json")
    for style in registry.styles:
        for field in ("id", "name", "style_features", "preservation_rules"):
            value = getattr(style, field, None)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"画派 {style.id or '?'} 的字段 {field} 不能为空")
    return [s.id for s in registry.require_enabled()]
