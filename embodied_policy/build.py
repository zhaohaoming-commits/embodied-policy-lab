from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from embodied_policy.data import (
    ManiSkillTrajectoryDataset,
    SyntheticReachDataset,
    VisionStateTrajectoryDataset,
)
from embodied_policy.data.trajectory_hdf5 import sorted_trajectory_names
from embodied_policy.models import (
    ActionChunkingTransformer,
    SingleStepMLP,
    VisionStateActionChunkingTransformer,
)


def build_dataset(
    config: dict[str, Any], split: str
) -> SyntheticReachDataset | ManiSkillTrajectoryDataset | VisionStateTrajectoryDataset:
    data = config["data"]
    kind = data.get("kind", "synthetic_reach")
    if kind in {"maniskill_hdf5", "maniskill_rgb_state_hdf5"}:
        with h5py.File(data["path"], "r") as handle:
            episode_names = sorted_trajectory_names(handle)
        # Keep the episode split fixed while varying the training seed.  This
        # lets a seed sweep measure optimization/model variance instead of
        # silently changing the training data for every run.
        split_seed = data.get("split_seed", config["seed"])
        rng = np.random.default_rng(split_seed)
        rng.shuffle(episode_names)
        train_count = data["train_episodes"]
        val_count = data["val_episodes"]
        if train_count + val_count > len(episode_names):
            raise ValueError("Requested more episodes than the trajectory file contains")
        selected = (
            episode_names[:train_count]
            if split == "train"
            else episode_names[train_count : train_count + val_count]
        )
        if kind == "maniskill_hdf5":
            return ManiSkillTrajectoryDataset(
                path=data["path"],
                episode_names=selected,
                obs_horizon=data["obs_horizon"],
                action_horizon=data["action_horizon"],
            )
        return VisionStateTrajectoryDataset(
            path=data["path"],
            episode_names=selected,
            obs_horizon=data["obs_horizon"],
            action_horizon=data["action_horizon"],
            camera_name=data.get("camera_name", "base_camera"),
        )
    if kind != "synthetic_reach":
        raise ValueError(f"Unsupported dataset kind: {kind}")
    seed_offset = 0 if split == "train" else 10_000
    return SyntheticReachDataset(
        num_episodes=data[f"{split}_episodes"],
        episode_length=data["episode_length"],
        obs_horizon=data["obs_horizon"],
        action_horizon=data["action_horizon"],
        dt=data["dt"],
        expert_gain=data["expert_gain"],
        action_limit=data["action_limit"],
        seed=config["seed"] + seed_offset,
    )


def build_model(
    config: dict[str, Any],
) -> ActionChunkingTransformer | SingleStepMLP | VisionStateActionChunkingTransformer:
    data = config["data"]
    model = config["model"]
    kind = data.get("kind", "synthetic_reach")
    if kind == "synthetic_reach":
        obs_dim = SyntheticReachDataset.obs_dim
        action_dim = SyntheticReachDataset.action_dim
    elif kind in {"maniskill_hdf5", "maniskill_rgb_state_hdf5"}:
        with h5py.File(data["path"], "r") as handle:
            first = handle[sorted_trajectory_names(handle)[0]]
            state_data = first["obs/state"] if kind == "maniskill_rgb_state_hdf5" else first["obs"]
            obs_dim = int(state_data.shape[-1])
            action_dim = int(first["actions"].shape[-1])
            image_channels = (
                int(first[f"obs/sensor_data/{data.get('camera_name', 'base_camera')}/rgb"].shape[-1])
                if kind == "maniskill_rgb_state_hdf5"
                else None
            )
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")
    model_type = model.get("type", "action_chunking_transformer")
    if model_type == "single_step_mlp":
        if data["action_horizon"] != 1:
            raise ValueError("single_step_mlp requires action_horizon=1")
        return SingleStepMLP(
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=data["obs_horizon"],
            hidden_dims=model["hidden_dims"],
            dropout=model["dropout"],
        )
    if model_type != "action_chunking_transformer":
        if model_type != "vision_state_action_chunking_transformer":
            raise ValueError(f"Unsupported model type: {model_type}")
        if kind != "maniskill_rgb_state_hdf5":
            raise ValueError("vision_state_action_chunking_transformer requires rgb+state HDF5 data")
        if image_channels is None:
            raise AssertionError("RGB data must define image channels")
        return VisionStateActionChunkingTransformer(
            image_channels=image_channels,
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=data["obs_horizon"],
            action_horizon=data["action_horizon"],
            d_model=model["d_model"],
            nhead=model["nhead"],
            num_layers=model["num_layers"],
            dim_feedforward=model["dim_feedforward"],
            dropout=model["dropout"],
        )
    return ActionChunkingTransformer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_horizon=data["obs_horizon"],
        action_horizon=data["action_horizon"],
        d_model=model["d_model"],
        nhead=model["nhead"],
        num_layers=model["num_layers"],
        dim_feedforward=model["dim_feedforward"],
        dropout=model["dropout"],
    )
