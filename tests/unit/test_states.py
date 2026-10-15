"""状态机与状态流转测试。"""

from __future__ import annotations

import pytest

from app import states
from app.errors import ConfigError


def test_main_path_is_legal():
    path = [
        states.CAPTURED,
        states.STYLE_ASSIGNED,
        states.GENERATING,
        states.CHECKING,
        states.CONFIRMED,
        states.COMPLETED,
    ]
    for current, target in zip(path, path[1:], strict=False):
        states.assert_transition(current, target)


def test_retry_path_is_legal():
    states.assert_transition(states.CHECKING, states.RETRYING)
    states.assert_transition(states.RETRYING, states.GENERATING)


def test_cannot_skip_or_reverse_states():
    illegal = [
        (states.CAPTURED, states.GENERATING),
        (states.CAPTURED, states.COMPLETED),
        (states.STYLE_ASSIGNED, states.CHECKING),
        (states.GENERATING, states.CONFIRMED),
        (states.COMPLETED, states.FAILED),
        (states.FAILED, states.GENERATING),
        (states.CONFIRMED, states.RETRYING),
    ]
    for current, target in illegal:
        with pytest.raises(ConfigError, match="非法状态流转"):
            states.assert_transition(current, target)


def test_any_active_state_can_fail():
    for state in states.ACTIVE_STATES:
        if state == states.FAILED:
            continue
        assert states.can_transition(state, states.FAILED)
    assert not states.can_transition(states.COMPLETED, states.FAILED)


def test_terminal_states_have_no_outgoing_transitions():
    for state in states.TERMINAL_STATES:
        assert states.TRANSITIONS[state] == ()
        assert states.is_terminal(state)
        assert not states.is_active(state)


def test_active_states_exclude_terminal():
    for state in states.ACTIVE_STATES:
        assert not states.is_terminal(state)


def test_unknown_state_rejected():
    with pytest.raises(ConfigError, match="未知状态"):
        states.assert_transition("SOMETHING_ELSE", states.FAILED)
