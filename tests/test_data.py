"""The relative representation and the training windows."""

from __future__ import annotations

import numpy as np
import pytest

from data import (
    GOAL_POSE,
    Normalizer,
    absolute_actions,
    build_windows,
    previous_set_points,
    relative_actions,
    relative_observation,
)

rng = np.random.default_rng(0)
STATES = np.concatenate(
    (rng.uniform(0, 512, (32, 3, 4)), rng.uniform(0, 2 * np.pi, (32, 3, 1))), axis=-1
)


@pytest.mark.parametrize("frame", ["block", "agent"])
def test_actions_round_trip(frame):
    """
    Relative actions map back to the world targets they came from, in both frames.
    """
    actions = rng.uniform(0, 512, (32, 8, 2))
    relative = relative_actions(actions, STATES[:, -1], frame)
    assert np.allclose(absolute_actions(relative, STATES[:, -1], frame), actions)
    # The current agent position is the origin of every action.
    here = relative_actions(STATES[:, -1:, 0:2], STATES[:, -1], frame)
    assert np.allclose(here, 0.0)


def test_observation_of_the_current_frame():
    """
    The current block is the frame's origin, the goal token recovers the goal, set-points add two features.
    """
    frames, goal = relative_observation(STATES, "block")
    # The block frame at the current state: the block sits at the origin with angle 0, exactly.
    assert np.array_equal(frames[:, -1, 2:6], np.tile([0.0, 0.0, 1.0, 0.0], (32, 1)))
    # The goal token recovers the frame's world pose: rotate back and translate.
    heading = STATES[:, -1, 4]
    goal_angle = np.arctan2(goal[:, 3], goal[:, 2]) + heading
    assert np.allclose(np.cos(goal_angle), np.cos(GOAL_POSE[2]))
    with_set_points, _ = relative_observation(STATES, "agent", STATES[..., 0:2])
    assert with_set_points.shape == (32, 3, 8)
    assert np.allclose(with_set_points[:, -1, 6:8], 0.0)


@pytest.mark.parametrize("padding", ["hold", "mask", "drop"])
def test_windows(padding):
    """
    Windows, validity and history padding for each padding mode.
    """
    episode_ends = np.array([5, 9])
    t, start, history, future, valid = build_windows(
        episode_ends, obs_horizon=3, pred_horizon=4, padding=padding
    )
    if padding == "drop":
        assert t.tolist() == [0, 1, 5]
        assert valid.all()
    else:
        assert t.tolist() == list(range(9))
        assert valid[:, 0].all() and valid.sum() == 9 * 4 - 2 * (1 + 2 + 3)
    # History repeats the episode's first state, futures repeat its last step.
    assert history[0].tolist() == [0, 0, 0] and history[
        t.tolist().index(5)
    ].tolist() == [5, 5, 5]
    assert future.max() <= 8 and np.all(future[start == 0] <= 4)


def test_previous_set_points():
    """
    Frame t pairs with a_{t-1}, and the initial position stands in before the first action.
    """
    states = np.arange(10 * 5, dtype=float).reshape(10, 5)
    actions = 1000 + np.arange(10 * 2, dtype=float).reshape(10, 2)
    t, start = np.array([0, 2, 6]), np.array([0, 0, 5])
    points = previous_set_points(states, actions, t, start, obs_horizon=2)
    # Frame t pairs with a_{t-1}; before the first action the agent holds its initial position.
    assert points[0].tolist() == [states[0, 0:2].tolist()] * 2
    assert points[1].tolist() == [actions[0].tolist(), actions[1].tolist()]
    assert points[2].tolist() == [states[5, 0:2].tolist(), actions[5].tolist()]


def test_normalizer_maps_constant_features_to_zero():
    """
    Constant features normalize to 0 and masked targets stay out of the statistics.
    """
    obs = np.concatenate((rng.normal(size=(100, 2, 1)), np.full((100, 2, 1), 0.3)), -1)
    actions = rng.normal(size=(100, 4, 2))
    weight = np.ones((100, 4))
    weight[:50, -1] = 0
    actions[:50, -1] = 1e6  # masked targets do not enter the statistics
    normalizer = Normalizer.from_data(obs, obs[:, 0], actions, weight)
    assert np.all(normalizer.normalize_obs(obs)[..., 1] == 0.0)
    assert np.all(np.abs(normalizer.action_mean[-1]) < 1.0)
