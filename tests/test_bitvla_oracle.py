from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_policy_bitwam.bitvla_oracle import goal_progress, make_action_candidates


def test_candidates_are_nested_and_preserve_base_and_gripper() -> None:
    actions = np.zeros((8, 7), dtype=np.float32)
    actions[:, 6] = np.linspace(0, 1, 8)
    kwargs = {
        "actions": actions,
        "rng": np.random.default_rng(5),
        "action_std": np.ones(7, dtype=np.float32),
        "action_low": np.full(7, -0.25, dtype=np.float32),
        "action_high": np.full(7, 0.25, dtype=np.float32),
        "noise_scale": 0.1,
    }
    four = make_action_candidates(count=4, **kwargs)
    kwargs["rng"] = np.random.default_rng(5)
    eight = make_action_candidates(count=8, **kwargs)

    np.testing.assert_array_equal(four, eight[:4])
    np.testing.assert_array_equal(four[0], actions)
    np.testing.assert_array_equal(four[:, :, 6], np.repeat(actions[None, :, 6], 4, axis=0))
    assert np.all(four[:, :, :6] <= 0.25)
    assert np.all(four[:, :, :6] >= -0.25)


def test_candidates_reject_invalid_count_and_shape() -> None:
    kwargs = {
        "rng": np.random.default_rng(0),
        "action_std": np.ones(7),
        "action_low": np.full(7, -1),
        "action_high": np.full(7, 1),
        "noise_scale": 0.1,
    }
    with pytest.raises(ValueError, match="positive"):
        make_action_candidates(np.zeros((8, 7)), count=0, **kwargs)
    with pytest.raises(ValueError, match=r"\[H, 7\]"):
        make_action_candidates(np.zeros((8, 6)), count=2, **kwargs)


def test_goal_progress_uses_libero_predicates() -> None:
    raw = SimpleNamespace(
        parsed_problem={"goal_state": ["a", "b", "c"]},
        _eval_predicate=lambda goal: goal != "b",
    )
    assert goal_progress(SimpleNamespace(env=raw)) == (2, 3)
