from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ReachEpisode:
    observations: np.ndarray
    actions: np.ndarray


def generate_episode(
    rng: np.random.Generator,
    episode_length: int,
    dt: float,
    expert_gain: float,
    action_limit: float,
) -> ReachEpisode:
    """Generate a 3-D point-mass reach trajectory from a proportional expert."""
    position = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
    goal = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for _ in range(episode_length):
        observations.append(np.concatenate([position, goal]).astype(np.float32))
        action = np.clip(expert_gain * (goal - position), -action_limit, action_limit)
        actions.append(action.astype(np.float32))
        position = position + dt * action

    return ReachEpisode(np.stack(observations), np.stack(actions))


class SyntheticReachDataset(Dataset[dict[str, torch.Tensor]]):
    """Window expert trajectories into observation histories and future action chunks."""

    obs_dim = 6
    action_dim = 3

    def __init__(
        self,
        num_episodes: int,
        episode_length: int,
        obs_horizon: int,
        action_horizon: int,
        dt: float,
        expert_gain: float,
        action_limit: float,
        seed: int,
    ) -> None:
        if min(num_episodes, episode_length, obs_horizon, action_horizon) <= 0:
            raise ValueError("Episode counts and horizons must be positive")

        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        rng = np.random.default_rng(seed)
        self.episodes = [
            generate_episode(rng, episode_length, dt, expert_gain, action_limit)
            for _ in range(num_episodes)
        ]
        self.index = [
            (episode_index, timestep)
            for episode_index in range(num_episodes)
            for timestep in range(episode_length)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, timestep = self.index[index]
        episode = self.episodes[episode_index]

        obs_start = max(0, timestep - self.obs_horizon + 1)
        obs = episode.observations[obs_start : timestep + 1]
        if len(obs) < self.obs_horizon:
            padding = np.repeat(obs[:1], self.obs_horizon - len(obs), axis=0)
            obs = np.concatenate([padding, obs], axis=0)

        action_end = min(len(episode.actions), timestep + self.action_horizon)
        actions = episode.actions[timestep:action_end]
        valid_actions = len(actions)
        if valid_actions < self.action_horizon:
            padding = np.zeros(
                (self.action_horizon - valid_actions, self.action_dim), dtype=np.float32
            )
            actions = np.concatenate([actions, padding], axis=0)

        mask = np.zeros(self.action_horizon, dtype=np.bool_)
        mask[:valid_actions] = True
        return {
            "observations": torch.from_numpy(obs.copy()),
            "actions": torch.from_numpy(actions.copy()),
            "action_mask": torch.from_numpy(mask),
        }

