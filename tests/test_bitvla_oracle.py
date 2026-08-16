from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_policy_bitwam.bitvla_oracle import (
    _build_nominal_plan,
    goal_progress,
    make_action_candidates,
)


def test_candidates_are_nested_and_preserve_base_and_gripper() -> None:
    actions = np.zeros((32, 7), dtype=np.float32)
    actions[:, 6] = np.linspace(0, 1, 32)
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


def test_nominal_plan_queries_only_the_shared_policy_tail() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.env = SimpleNamespace(timestep=0, done=False)
            self.steps = 0

        def regenerate_obs_from_state(self, state: np.ndarray) -> dict[str, int]:
            return {"step": self.steps}

        def step(self, action: list[float]) -> tuple[dict[str, int], float, bool, dict]:
            self.steps += 1
            return {"step": self.steps}, 0.0, False, {}

    class FakeConnection:
        def __init__(self) -> None:
            self.sent: list[dict] = []
            self.responses = [
                {"type": "action_response", "actions": np.full((8, 7), 2.0)},
                {"type": "action_response", "actions": np.full((8, 7), 3.0)},
            ]

        def send(self, message: dict) -> None:
            self.sent.append(message)

        def recv(self) -> dict:
            return self.responses.pop(0)

    env = FakeEnv()
    connection = FakeConnection()
    nominal, policy_calls = _build_nominal_plan(
        env=env,
        evaluator=SimpleNamespace(process_action=lambda action, family: action),
        cfg=SimpleNamespace(model_family="bitnet"),
        state=np.zeros(4),
        timestep=9,
        base_actions=np.ones((8, 7), dtype=np.float32),
        planning_horizon=20,
        connection=connection,
    )

    assert nominal.shape == (20, 7)
    np.testing.assert_array_equal(nominal[:8], 1.0)
    np.testing.assert_array_equal(nominal[8:16], 2.0)
    np.testing.assert_array_equal(nominal[16:], 3.0)
    assert policy_calls == 2
    assert env.steps == 16
    assert [message["type"] for message in connection.sent] == [
        "action_request",
        "action_request",
    ]
