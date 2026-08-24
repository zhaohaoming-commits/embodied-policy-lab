from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def sorted_trajectory_names(handle: h5py.File) -> list[str]:
    def trajectory_id(name: str) -> int:
        try:
            return int(name.rsplit("_", maxsplit=1)[-1])
        except ValueError:
            return 0

    return sorted((name for name in handle if name.startswith("traj_")), key=trajectory_id)


class ManiSkillTrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    """Lazily window flattened ManiSkill HDF5 trajectories.

    Episodes are split before this class is constructed so no trajectory leaks
    between training and validation datasets.
    """

    def __init__(
        self,
        path: str | Path,
        episode_names: Sequence[str],
        obs_horizon: int,
        action_horizon: int,
    ) -> None:
        self.path = str(Path(path))
        self.episode_names = list(episode_names)
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self._handle: h5py.File | None = None
        self.obs_mean: np.ndarray | None = None
        self.obs_std: np.ndarray | None = None
        self.action_mean: np.ndarray | None = None
        self.action_std: np.ndarray | None = None

        if not self.episode_names:
            raise ValueError("At least one trajectory is required")
        if min(obs_horizon, action_horizon) <= 0:
            raise ValueError("Observation and action horizons must be positive")

        with h5py.File(self.path, "r") as handle:
            first = handle[self.episode_names[0]]
            if not isinstance(first["obs"], h5py.Dataset):
                raise ValueError("Expected flattened observations. Replay with --obs-mode state.")
            self.obs_dim = int(first["obs"].shape[-1])
            self.action_dim = int(first["actions"].shape[-1])
            self.index: list[tuple[str, int]] = []
            for episode_name in self.episode_names:
                action_count = int(handle[episode_name]["actions"].shape[0])
                self.index.extend((episode_name, timestep) for timestep in range(action_count))

    @property
    def handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.index)

    def compute_normalization(self) -> dict[str, np.ndarray]:
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        with h5py.File(self.path, "r") as handle:
            for episode_name in self.episode_names:
                observations.append(np.asarray(handle[episode_name]["obs"], dtype=np.float32))
                actions.append(np.asarray(handle[episode_name]["actions"], dtype=np.float32))
        all_observations = np.concatenate(observations, axis=0)
        all_actions = np.concatenate(actions, axis=0)
        return {
            "obs_mean": all_observations.mean(axis=0),
            "obs_std": np.maximum(all_observations.std(axis=0), 1e-4),
            "action_mean": all_actions.mean(axis=0),
            "action_std": np.maximum(all_actions.std(axis=0), 1e-4),
        }

    def set_normalization(self, stats: dict[str, np.ndarray]) -> None:
        self.obs_mean = np.asarray(stats["obs_mean"], dtype=np.float32)
        self.obs_std = np.asarray(stats["obs_std"], dtype=np.float32)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action_std"], dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_name, timestep = self.index[index]
        episode = self.handle[episode_name]
        observations_dataset = episode["obs"]
        actions_dataset = episode["actions"]

        obs_start = max(0, timestep - self.obs_horizon + 1)
        observations = np.asarray(
            observations_dataset[obs_start : timestep + 1], dtype=np.float32
        )
        if len(observations) < self.obs_horizon:
            padding = np.repeat(observations[:1], self.obs_horizon - len(observations), axis=0)
            observations = np.concatenate([padding, observations], axis=0)

        if self.obs_mean is not None and self.obs_std is not None:
            observations = (observations - self.obs_mean) / self.obs_std

        action_end = min(len(actions_dataset), timestep + self.action_horizon)
        actions = np.asarray(actions_dataset[timestep:action_end], dtype=np.float32)
        if self.action_mean is not None and self.action_std is not None:
            actions = (actions - self.action_mean) / self.action_std
        valid_actions = len(actions)
        if valid_actions < self.action_horizon:
            padding = np.zeros(
                (self.action_horizon - valid_actions, self.action_dim), dtype=np.float32
            )
            actions = np.concatenate([actions, padding], axis=0)

        action_mask = np.zeros(self.action_horizon, dtype=np.bool_)
        action_mask[:valid_actions] = True
        return {
            "observations": torch.from_numpy(observations.copy()),
            "actions": torch.from_numpy(actions.copy()),
            "action_mask": torch.from_numpy(action_mask),
        }
