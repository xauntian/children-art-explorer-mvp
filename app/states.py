"""状态定义与状态机。

启用审查的状态路径：
    CAPTURED → STYLE_ASSIGNED → GENERATING → CHECKING → CONFIRMED → COMPLETED

关闭审查：GENERATING → COMPLETED（只校验生成文件完整性）
首次质量检查未通过：CHECKING → RETRYING → GENERATING
达到重试上限或技术错误：→ FAILED
"""

from __future__ import annotations

from .errors import ConfigError

CAPTURED = "CAPTURED"
STYLE_ASSIGNED = "STYLE_ASSIGNED"
GENERATING = "GENERATING"
CHECKING = "CHECKING"
RETRYING = "RETRYING"
CONFIRMED = "CONFIRMED"
COMPLETED = "COMPLETED"
FAILED = "FAILED"

ALL_STATES = (
    CAPTURED,
    STYLE_ASSIGNED,
    GENERATING,
    CHECKING,
    RETRYING,
    CONFIRMED,
    COMPLETED,
    FAILED,
)

#: 终态：不再发生状态变化
TERMINAL_STATES = (COMPLETED, FAILED)

#: 活动状态：同一时刻只允许一件作品处于这些状态
ACTIVE_STATES = (CAPTURED, STYLE_ASSIGNED, GENERATING, CHECKING, RETRYING, CONFIRMED)

TRANSITIONS: dict[str, tuple[str, ...]] = {
    CAPTURED: (STYLE_ASSIGNED, FAILED),
    STYLE_ASSIGNED: (GENERATING, FAILED),
    GENERATING: (CHECKING, COMPLETED, FAILED),
    CHECKING: (CONFIRMED, RETRYING, FAILED),
    RETRYING: (GENERATING, FAILED),
    CONFIRMED: (COMPLETED, FAILED),
    COMPLETED: (),
    FAILED: (),
}


def is_terminal(state: str) -> bool:
    return state in TERMINAL_STATES


def is_active(state: str) -> bool:
    return state in ACTIVE_STATES


def can_transition(current: str, target: str) -> bool:
    if current not in TRANSITIONS:
        return False
    return target in TRANSITIONS[current]


def assert_transition(current: str, target: str) -> None:
    if current not in TRANSITIONS:
        raise ConfigError(f"未知状态: {current}")
    if target not in TRANSITIONS[current]:
        raise ConfigError(f"非法状态流转: {current} → {target}")
