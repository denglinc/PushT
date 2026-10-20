"""
Dataset utilities for Push-T: download, relative representation, normalizer and on-device chunk windows.

Shapes: bs is the batch size, n a number of points, K the observation history and H the chunk length.
"""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import zarr

PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
ZARR_RELATIVE_PATH = Path("pusht") / "pusht_cchi_v7_replay.zarr"

# The env's fixed goal pose (x, y, theta) in world coordinates, the same body-origin convention as the block.
GOAL_POSE = np.array([256.0, 256.0, np.pi / 4])

Frame = Literal["block", "agent"]
Padding = Literal["hold", "mask", "drop"]


def reference_frame(state: np.ndarray, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the reference frame of world states.
    Note: block is the block's pose; agent is the agent's position with the world's axes (the agent is a disk
    without orientation).
    Args:
        - state: world states [agent xy, block xy, block angle], shape (..., 5)
        - frame: block | agent
    Returns:
        - origin: shape (..., 2)
        - heading: shape (...)
    """
    if frame == "agent":
        return state[..., 0:2], np.zeros(state.shape[:-1])
    return state[..., 2:4], state[..., 4]


def to_frame(points: np.ndarray, origin: np.ndarray, heading: np.ndarray) -> np.ndarray:
    r"""
    Expresses world points in a frame: $R(\phi)^\top (r - o)$.
    Args:
        - points: world points, shape (bs, n, 2)
        - origin: frame origins, shape (bs, 2)
        - heading: frame headings, shape (bs,)
    Returns:
        - points in the frame, shape (bs, n, 2)
    """
    # Subtract first, then rotate, so a frame's own origin maps to exactly (0, 0).
    d = points - origin[:, None, :]
    cos, sin = np.cos(heading)[:, None], np.sin(heading)[:, None]
    return np.stack(
        (cos * d[..., 0] + sin * d[..., 1], -sin * d[..., 0] + cos * d[..., 1]), axis=-1
    )


def from_frame(
    points: np.ndarray, origin: np.ndarray, heading: np.ndarray
) -> np.ndarray:
    r"""
    Maps frame coordinates back to world points: $o + R(\phi)\,r'$, the inverse of to_frame.
    Args:
        - points: points in the frame, shape (bs, n, 2)
        - origin: frame origins, shape (bs, 2)
        - heading: frame headings, shape (bs,)
    Returns:
        - world points, shape (bs, n, 2)
    """
    cos, sin = np.cos(heading)[:, None], np.sin(heading)[:, None]
    rotated = np.stack(
        (
            cos * points[..., 0] - sin * points[..., 1],
            sin * points[..., 0] + cos * points[..., 1],
        ),
        axis=-1,
    )
    return rotated + origin[:, None, :]


def relative_observation(
    states: np.ndarray, frame: Frame, set_points: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expresses a state history and the goal in the reference frame of the current (last) state.
    Note: a frame is [agent xy, block xy, cos, sin of the block angle] and the goal [xy, cos, sin of its angle].
    The goal is fixed in the world, so its pose in the frame recovers the frame's world pose and nothing is lost.
    Args:
        - states: world state history, shape (bs, K, 5)
        - frame: block | agent
        - set_points: the targets that produced the K states, shape (bs, K, 2), or None
    Returns:
        - frames: shape (bs, K, 6), or (bs, K, 8) with the set-points as displacements from the current agent
        - goal: shape (bs, 4)
    """
    origin, heading = reference_frame(states[:, -1], frame)
    agent = to_frame(states[..., 0:2], origin, heading)
    block = to_frame(states[..., 2:4], origin, heading)
    # Angles enter as (cos, sin) of the angle relative to the frame, which removes the 0 / 2 pi wrap.
    angle = states[..., 4] - heading[:, None]
    frames = np.concatenate(
        (agent, block, np.cos(angle)[..., None], np.sin(angle)[..., None]), axis=-1
    )
    if set_points is not None:
        frames = np.concatenate(
            (frames, to_frame(set_points, states[:, -1, 0:2], heading)), axis=-1
        )
    goal_xy = to_frame(
        np.broadcast_to(GOAL_POSE[:2], (len(states), 1, 2)), origin, heading
    )[:, 0]
    goal_angle = GOAL_POSE[2] - heading
    goal = np.concatenate(
        (goal_xy, np.cos(goal_angle)[:, None], np.sin(goal_angle)[:, None]), axis=-1
    )
    return frames, goal


def relative_actions(
    actions: np.ndarray, state: np.ndarray, frame: Frame
) -> np.ndarray:
    """
    Expresses world targets as displacements from the current agent position, in the frame's axes.
    Args:
        - actions: world targets, shape (bs, H, 2)
        - state: the current world state, shape (bs, 5)
        - frame: block | agent
    Returns:
        - relative actions, shape (bs, H, 2)
    """
    _, heading = reference_frame(state, frame)
    return to_frame(actions, state[:, 0:2], heading)


def absolute_actions(
    actions: np.ndarray, state: np.ndarray, frame: Frame
) -> np.ndarray:
    """
    Maps relative actions back to world targets with the pose of the query state.
    Args:
        - actions: displacements from the agent position, shape (bs, H, 2)
        - state: the query's world state, shape (bs, 5)
        - frame: block | agent
    Returns:
        - world targets, shape (bs, H, 2)
    """
    _, heading = reference_frame(state, frame)
    return from_frame(actions, state[:, 0:2], heading)


@dataclass(frozen=True)
class Normalizer:
    """
    Per-index z-score statistics: one (mean, std) per history lag, chunk step and feature.
    Note: every timestep starts a window and history padding repeats edge frames, so the statistics of a given
    lag or step are the same for every K and H (with hold or mask; drop removes windows by H). Features whose std is
    below eps ($10^{-4}$) are constant: the reference object's own entries of the current frame and, in the agent
    frame, the goal's angle pair. They get std 1 and their constant as mean, so they normalize to exactly 0.
    """

    obs_mean: np.ndarray
    obs_std: np.ndarray
    goal_mean: np.ndarray
    goal_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def _stats(
        x: np.ndarray, weight: np.ndarray | None = None, eps: float = 1e-4
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes weighted statistics over the window axis.
        Args:
            - x: features, shape (N, ..., F)
            - weight: 0 on entries to leave out (padded targets), shape (N, ...), or None
            - eps: standard deviations below this mark a constant feature
        Returns:
            - mean: shape (..., F)
            - std: shape (..., F)
        """
        if weight is None:
            weight = np.ones(x.shape[:-1])
        w = weight[..., None] / weight.sum(axis=0)[..., None]
        mean = (w * x).sum(axis=0)
        std = np.sqrt((w * (x - mean) ** 2).sum(axis=0))
        constant = std < eps
        # Use the constant itself as the mean, since the weighted sum can be off by rounding.
        return np.where(constant, x[0], mean), np.where(constant, 1.0, std)

    @classmethod
    def from_data(
        cls,
        obs: np.ndarray,
        goal: np.ndarray,
        actions: np.ndarray,
        action_weight: np.ndarray | None = None,
    ) -> Normalizer:
        """
        Fits the statistics to relative windows.
        Args:
            - obs: shape (N, K, F)
            - goal: shape (N, 4)
            - actions: shape (N, H, 2)
            - action_weight: 0 on padded targets, shape (N, H), or None
        Returns:
            - the normalizer
        """
        return cls(
            *cls._stats(obs), *cls._stats(goal), *cls._stats(actions, action_weight)
        )

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalizes history frames.
        Args:
            - obs: shape (bs, K, F)
        Returns:
            - normalized frames, shape (bs, K, F)
        """
        return (obs - self.obs_mean) / self.obs_std

    def normalize_goal(self, goal: np.ndarray) -> np.ndarray:
        """
        Normalizes goal tokens.
        Args:
            - goal: shape (bs, 4)
        Returns:
            - normalized goals, shape (bs, 4)
        """
        return (goal - self.goal_mean) / self.goal_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalizes relative action chunks.
        Args:
            - action: shape (bs, H, 2)
        Returns:
            - normalized chunks, shape (bs, H, 2)
        """
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Maps normalized action chunks back to relative actions.
        Args:
            - action: shape (bs, H, 2)
        Returns:
            - relative chunks, shape (bs, H, 2)
        """
        return action * self.action_std + self.action_mean


def download_pusht(dataset_dir: Path) -> Path:
    """
    Downloads and extracts the Push-T dataset if needed.
    Args:
        - dataset_dir: where to keep the dataset
    Returns:
        - the path to the extracted Zarr dataset
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = dataset_dir / ZARR_RELATIVE_PATH
    if zarr_path.exists():
        return zarr_path

    zip_path = dataset_dir / "pusht.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(PUSHT_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    return zarr_path


def load_pusht_zarr(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the flat step arrays of a Push-T Zarr dataset.
    Args:
        - zarr_path: the dataset
    Returns:
        - states: shape (T, 5)
        - actions: the target applied after each state, shape (T, 2)
        - episode_ends: one past each episode's last step, shape (episodes,)
    """
    # float64 so that the relative conversion matches evaluation, whose env observations are float64.
    root = zarr.open(zarr_path, mode="r")
    states = np.asarray(root["data"]["state"][:], dtype=np.float64)
    actions = np.asarray(root["data"]["action"][:], dtype=np.float64)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    return states, actions, episode_ends


def build_windows(
    episode_ends: np.ndarray, *, obs_horizon: int, pred_horizon: int, padding: Padding
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds window indices into the flat step arrays: every step of every episode starts one window.
    Note: history before an episode starts repeats its first state, which is also what the env looks like right
    after reset (velocity 0). Future steps after an episode ends repeat its last action: hold trains on them, mask
    keeps the windows but marks those steps invalid, drop discards the windows.
    Args:
        - episode_ends: shape (episodes,)
        - obs_horizon: K
        - pred_horizon: H
        - padding: hold | mask | drop
    Returns:
        - t: the current step, shape (N,)
        - start: the first step of its episode, shape (N,)
        - history: history steps, shape (N, K)
        - future: future steps, shape (N, H)
        - valid: whether a future step lies inside the episode, shape (N, H)
    """
    starts = np.concatenate(([0], episode_ends[:-1]))
    lengths = episode_ends - starts
    t = np.arange(episode_ends[-1])
    start, end = np.repeat(starts, lengths), np.repeat(episode_ends, lengths)
    if padding == "drop":
        keep = t + pred_horizon <= end
        t, start, end = t[keep], start[keep], end[keep]
    history = np.maximum(t[:, None] + np.arange(-obs_horizon + 1, 1), start[:, None])
    future = t[:, None] + np.arange(pred_horizon)
    valid = future < end[:, None]
    return t, start, history, np.minimum(future, end[:, None] - 1), valid


def previous_set_points(
    states: np.ndarray,
    actions: np.ndarray,
    t: np.ndarray,
    start: np.ndarray,
    obs_horizon: int,
) -> np.ndarray:
    """
    Returns the set-points that produced the K history frames: frame $t-k$ pairs with $a_{t-k-1}$.
    Note: before an episode's first action the agent holds its initial position, which is also all an evaluation
    knows right after reset.
    Args:
        - states: shape (T, 5)
        - actions: shape (T, 2)
        - t: current steps, shape (N,)
        - start: first steps of their episodes, shape (N,)
        - obs_horizon: K
    Returns:
        - set-points in world units, shape (N, K, 2)
    """
    previous = t[:, None] + np.arange(-obs_horizon, 0)
    before = (previous < start[:, None])[..., None]
    return np.where(
        before,
        states[start, None, 0:2],
        actions[np.maximum(previous, start[:, None])],
    )


class PushtChunkDataset:
    """
    Normalized relative windows (obs history, goal, action chunk), precomputed and kept on one device.
    Note: one item is obs (K, 6), or (K, 8) with the action history, goal (4,), actions (H, 2) and a loss weight
    (H,), 0 on masked padding; batch gathers bs items at once.
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        *,
        frame: Frame,
        obs_horizon: int,
        pred_horizon: int,
        action_history: bool,
        padding: Padding,
        device: torch.device,
    ) -> None:
        """
        Args:
            - states, actions, episode_ends: flat step arrays (see load_pusht_zarr)
            - frame: block | agent
            - obs_horizon: K
            - pred_horizon: H
            - action_history: whether each history frame carries the set-point that produced it
            - padding: hold | mask | drop
            - device: where the windows live
        """
        if frame not in ("block", "agent") or padding not in ("hold", "mask", "drop"):
            raise ValueError(f"Unknown frame {frame!r} or padding {padding!r}.")
        t, start, history, future, valid = build_windows(
            episode_ends,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            padding=padding,
        )
        set_points = None
        if action_history:
            set_points = previous_set_points(states, actions, t, start, obs_horizon)
        # Raw environment units become relative features of the window's current state here.
        obs, goal = relative_observation(states[history], frame, set_points)
        chunk = relative_actions(actions[future], states[t], frame)
        weight = valid.astype(np.float64) if padding == "mask" else np.ones(valid.shape)
        # Step 0 of a window is always inside its episode, so this holds by construction; the loss relies on it.
        if not weight.any(axis=1).all():
            raise ValueError("A window has no valid target step.")
        self.normalizer = Normalizer.from_data(obs, goal, chunk, weight)
        self.num_padded = int((~valid).sum())

        def to_device(x: np.ndarray) -> torch.Tensor:
            """
            Moves a float32 copy of an array to the dataset's device.
            """
            return torch.from_numpy(x).float().to(device)

        self.obs = to_device(self.normalizer.normalize_obs(obs))
        self.goal = to_device(self.normalizer.normalize_goal(goal))
        self.actions = to_device(self.normalizer.normalize_action(chunk))
        self.weight = to_device(weight)

    def __len__(self) -> int:
        """
        Returns the number of windows.
        """
        return len(self.obs)

    def batch(
        self, index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gathers windows by index.
        Args:
            - index: window indices, shape (bs,)
        Returns:
            - obs: shape (bs, K, F)
            - goal: shape (bs, 4)
            - actions: normalized relative chunks, shape (bs, H, 2)
            - weight: shape (bs, H)
        """
        return (
            self.obs[index],
            self.goal[index],
            self.actions[index],
            self.weight[index],
        )
