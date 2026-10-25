"""画派配置与随机分配测试。

要求：画派只能来自启用列表，新任务等概率选择；
用可控制的随机输入验证选择逻辑，避免用偶然抽样次数作为测试通过条件。
"""

from __future__ import annotations

import random

import pytest

from app.config import StyleConfig
from app.errors import ConfigError
from app.styles import StyleAssigner, validate_registry
from tests.fixtures import make_registry


def test_registry_requires_enabled_styles():
    registry = make_registry(enabled={"fauvism": False, "cubism": False, "impressionism": False})
    with pytest.raises(ConfigError, match="没有启用任何画派"):
        registry.require_enabled()
    with pytest.raises(ConfigError, match="没有启用任何画派"):
        StyleAssigner(registry).assign()


def test_enabled_list_excludes_disabled_styles():
    registry = make_registry(enabled={"cubism": False})
    assert [s.id for s in registry.enabled] == ["fauvism", "impressionism"]
    assert [s.id for s in registry.require_enabled()] == ["fauvism", "impressionism"]


def test_assign_uses_exactly_the_enabled_list():
    """可控随机输入：choice 收到的候选集必须严格等于启用列表。"""
    seen: dict = {}

    class SpyRandom(random.Random):
        def choice(self, seq):  # type: ignore[override]
            seen["seq"] = list(seq)
            return seq[0]

    registry = make_registry(enabled={"impressionism": False})
    assigner = StyleAssigner(registry, SpyRandom(0))
    chosen = assigner.assign()

    assert [s.id for s in seen["seq"]] == ["fauvism", "cubism"]
    assert chosen.id == "fauvism"


def test_assign_is_deterministic_for_a_seeded_rng_and_covers_the_list():
    """同一随机种子的选择序列可复现，并且只在启用列表内取值。"""
    registry = make_registry()
    assigner_a = StyleAssigner(registry, random.Random(7))
    assigner_b = StyleAssigner(registry, random.Random(7))

    seq_a = [assigner_a.assign().id for _ in range(12)]
    seq_b = [assigner_b.assign().id for _ in range(12)]

    assert seq_a == seq_b, "同一随机种子的选择序列必须可复现"
    assert set(seq_a) <= {"fauvism", "cubism", "impressionism"}


def test_assign_can_reach_every_enabled_style():
    """遍历 RNG 的取值，验证每种启用画派都可能被选中（不依赖偶然抽样）。"""
    registry = make_registry()
    enabled_ids = {s.id for s in registry.enabled}

    class SweepRandom(random.Random):
        """每次 choice 返回候选列表中的下一个，循环覆盖全部候选。"""

        def __init__(self):
            super().__init__(0)
            self.calls = 0

        def choice(self, seq):  # type: ignore[override]
            value = seq[self.calls % len(seq)]
            self.calls += 1
            return value

    assigner = StyleAssigner(registry, SweepRandom())
    picked = {assigner.assign().id for _ in range(len(enabled_ids))}
    assert picked == enabled_ids


def test_assign_explicit_accepts_enabled_style():
    registry = make_registry()
    style = StyleAssigner(registry).assign_explicit("cubism")
    assert style.id == "cubism"
    assert style.name == "Cubism"


def test_assign_explicit_rejects_disabled_style():
    registry = make_registry(enabled={"cubism": False})
    with pytest.raises(ConfigError, match="未启用"):
        StyleAssigner(registry).assign_explicit("cubism")


def test_assign_explicit_rejects_unknown_style():
    registry = make_registry()
    with pytest.raises(ConfigError, match="未知画派"):
        StyleAssigner(registry).assign_explicit("surrealism")


def test_duplicate_style_ids_rejected():
    from app.config import StyleRegistry

    raw = {
        "id": "fauvism",
        "name": "Fauvism",
        "enabled": True,
        "style_features": "x",
        "preservation_rules": "y",
    }
    with pytest.raises(ConfigError, match="画派 id 重复"):
        StyleRegistry([StyleConfig(raw, 0), StyleConfig(raw, 1)], "v1")


def test_style_requires_all_fields():
    for missing in ("id", "name", "enabled", "style_features", "preservation_rules"):
        raw = {
            "id": "fauvism",
            "name": "Fauvism",
            "enabled": True,
            "style_features": "x",
            "preservation_rules": "y",
        }
        raw.pop(missing)
        with pytest.raises(ConfigError, match=missing):
            StyleConfig(raw, 0)


def test_validate_registry_returns_enabled_ids_and_rejects_empty():
    registry = make_registry(enabled={"impressionism": False})
    assert validate_registry(registry) == ["fauvism", "cubism"]

    from app.config import StyleRegistry

    with pytest.raises(ConfigError, match="画派列表为空"):
        validate_registry(StyleRegistry([], "v1"))


def test_real_project_style_config_is_valid():
    """正式八画派全部可随机选择，并各自具备当前模型的生成参数。"""
    from pathlib import Path

    from app.config import load_app_config

    root = Path(__file__).resolve().parents[2]
    config = load_app_config(root)
    registry = config.load_styles()
    enabled = validate_registry(registry)
    assert set(enabled) == {
        "fauvism", "cubism", "impressionism", "post_impressionism",
        "expressionism", "surrealism", "neoplasticism", "pop_art",
    }
    for style in registry.enabled:
        assert style.style_features.strip()
        assert style.preservation_rules.strip()
        assert config.sampler_for(style.id) == {
            "steps": 4, "cfg": 1.0, "denoise": 1.0,
            "sampler_name": "euler", "scheduler": "flux2",
        }
