"""pytest 根配置：项目根目录入 sys.path，并暴露共享夹具。"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline import Pipeline  # noqa: E402
from app.storage import JobStore  # noqa: E402
from app.styles import StyleAssigner  # noqa: E402
from tests.fixtures import (  # noqa: E402
    FakeChecker,
    FakeGenerator,
    make_app_config,
    make_registry,
)

__all__ = ["pipeline_env"]


@pytest.fixture
def pipeline_env(tmp_path: Path):
    """返回构造函数 build(checker_payloads, generator_fail_on, enabled, overrides)。

    返回 (pipeline, store, generator, checker, registry)。
    """

    def build(
        checker_payloads=None,
        generator_fail_on=None,
        enabled=None,
        overrides=None,
    ):
        config = make_app_config(tmp_path, **(overrides or {}))
        registry = make_registry(enabled=enabled)
        store = JobStore(config.runs_dir, config.temp_suffix)
        generator = FakeGenerator(fail_on=generator_fail_on)
        checker = FakeChecker(checker_payloads or [])
        pipeline = Pipeline(
            config=config,
            registry=registry,
            store=store,
            generator=generator,
            checker=checker,
            generate_prompt_template=(tmp_path / "prompts" / "generate.txt").read_text(
                encoding="utf-8"
            ),
            check_prompt_template=(tmp_path / "prompts" / "check.txt").read_text(
                encoding="utf-8"
            ),
            style_assigner=StyleAssigner(registry, random.Random(1234)),
        )
        return pipeline, store, generator, checker, registry

    return build
