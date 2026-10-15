"""pytest 标记与默认收集规则。

集成测试默认不运行，避免在模型或服务缺失时误报通过：
    python -m pytest tests/unit -v
    python -m pytest tests/integration -v -m integration
"""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: 需要真实本地服务与模型权重的测试")
