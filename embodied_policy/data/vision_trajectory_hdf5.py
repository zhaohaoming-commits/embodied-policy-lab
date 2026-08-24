from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class VisionStateTrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    """Window aligned RGB, state and action data from a ManiSkill trajectory HDF5.

    For each action index ``t`` the returned input contains the camera image and
    state at ``t``.  ManiSkill stores one additional terminal observation, so
    RGB/state have length ``T + 1`` while actions have length ``T``.
    """

    def __init__(
        self,
        path: str | Path,
        episode_names: Sequence[str],
        obs_horizon: int,
        action_horizon: int,
        camera_name: str = "base_camera",
        state_indices: Sequence[int] | None = None,
    ) -> None:
        self.path = str(Path(path))
        self.episode_names = list(episode_names)
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.camera_name = camera_name
        self._handle: h5py.File | None = None
        self.normalization: dict[str, np.ndarray] | None = None
        if not self.episode_names or min(obs_horizon, action_horizon) <= 0:
            raise ValueError("At least one episode and positive horizons are required")

        with h5py.File(self.path, "r") as handle:
            first = handle[self.episode_names[0]]
            states, images, actions = self._datasets(first)
            raw_obs_dim = int(states.shape[-1])
            self.state_indices = np.asarray(
                list(state_indices) if state_indices is not None else list(range(raw_obs_dim)),
                dtype=np.int64,
            )
            if not len(self.state_indices) or self.state_indices.min() < 0 or self.state_indices.max() >= raw_obs_dim:
                raise ValueError(f"state_indices must be within [0, {raw_obs_dim})")
            self.obs_dim = int(len(self.state_indices))
            self.action_dim = int(actions.shape[-1])
            if images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError("Expected RGB images shaped [time, height, width, 3]")
            self.image_shape = (int(images.shape[-1]), int(images.shape[1]), int(images.shape[2]))
            self.index: list[tuple[str, int]] = []
            for name in self.episode_names:
                state_data, image_data, action_data = self._datasets(handle[name])
                if len(state_data) != len(action_data) + 1 or len(image_data) != len(action_data) + 1:
                    raise ValueError(f"Episode {name} has misaligned state/image/action lengths")
                self.index.extend((name, timestep) for timestep in range(len(action_data)))

    def _datasets(self, episode: h5py.Group) -> tuple[h5py.Dataset, h5py.Dataset, h5py.Dataset]:
        try:
            return (
                episode["obs/state"],
                episode[f"obs/sensor_data/{self.camera_name}/rgb"],
                episode["actions"],
            )
        except KeyError as error:
            raise ValueError(
                f"Missing rgb+state fields for camera '{self.camera_name}' in {episode.name}"
            ) from error

    @property
    def handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.index)

    def compute_normalization(self) -> dict[str, np.ndarray]:
        state_values: list[np.ndarray] = []
        action_values: list[np.ndarray] = []
        image_sum = np.zeros(3, dtype=np.float64)
        image_square_sum = np.zeros(3, dtype=np.float64)
        pixel_count = 0
        with h5py.File(self.path, "r") as handle:
            for name in self.episode_names:
                states, images, actions = self._datasets(handle[name])
                state_values.append(np.asarray(states[:-1, self.state_indices], dtype=np.float32))
                action_values.append(np.asarray(actions, dtype=np.float32))
                rgb = np.asarray(images[:-1], dtype=np.float64) / 255.0
                image_sum += rgb.sum(axis=(0, 1, 2))
                image_square_sum += np.square(rgb).sum(axis=(0, 1, 2))
                pixel_count += int(np.prod(rgb.shape[:3]))
        image_mean = image_sum / pixel_count
        image_variance = np.maximum(image_square_sum / pixel_count - np.square(image_mean), 1e-8)
        return {
            "obs_mean": np.concatenate(state_values).mean(axis=0),
            "obs_std": np.maximum(np.concatenate(state_values).std(axis=0), 1e-4),
            "action_mean": np.concatenate(action_values).mean(axis=0),
            "action_std": np.maximum(np.concatenate(action_values).std(axis=0), 1e-4),
            "image_mean": image_mean.astype(np.float32),
            "image_std": np.sqrt(image_variance).astype(np.float32),
        }

    def set_normalization(self, stats: dict[str, np.ndarray]) -> None:
        self.normalization = {key: np.asarray(value, dtype=np.float32) for key, value in stats.items()}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        name, timestep = self.index[index]
        states, image_data, action_data = self._datasets(self.handle[name])
        obs_start = max(0, timestep - self.obs_horizon + 1)
        observations = np.asarray(states[obs_start : timestep + 1, self.state_indices], dtype=np.float32)
        images = np.asarray(image_data[obs_start : timestep + 1], dtype=np.float32) / 255.0
        if len(observations) < self.obs_horizon:
            pad_count = self.obs_horizon - len(observations)
            observations = np.concatenate([np.repeat(observations[:1], pad_count, axis=0), observations])
            images = np.concatenate([np.repeat(images[:1], pad_count, axis=0), images])
        images = np.moveaxis(images, -1, -3)

        action_end = min(len(action_data), timestep + self.action_horizon)
        actions = np.asarray(action_data[timestep:action_end], dtype=np.float32)
        valid_actions = len(actions)
        if valid_actions < self.action_horizon:
            actions = np.concatenate(
                [actions, np.zeros((self.action_horizon - valid_actions, self.action_dim), dtype=np.float32)]
            )
        if self.normalization is not None:
            observations = (observations - self.normalization["obs_mean"]) / self.normalization["obs_std"]
            actions[:valid_actions] = (
                actions[:valid_actions] - self.normalization["action_mean"]
            ) / self.normalization["action_std"]
            images = (images - self.normalization["image_mean"][None, :, None, None]) / self.normalization["image_std"][None, :, None, None]
        action_mask = np.zeros(self.action_horizon, dtype=np.bool_)
        action_mask[:valid_actions] = True
        return {
            "observations": torch.from_numpy(observations.copy()),
            "images": torch.from_numpy(images.copy()),
            "actions": torch.from_numpy(actions.copy()),
            "action_mask": torch.from_numpy(action_mask),
        }
