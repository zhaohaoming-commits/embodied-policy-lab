from __future__ import annotations

import argparse
from collections.abc import Mapping

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch


def print_tree(value: object, prefix: str = "") -> int:
    if isinstance(value, Mapping):
        return sum(print_tree(child, f"{prefix}.{key}" if prefix else str(key)) for key, child in value.items())
    if isinstance(value, torch.Tensor):
        feature_count = int(np.prod(value.shape[1:]))
        print(f"{prefix:<32} shape={tuple(value.shape)!s:<16} features={feature_count}")
        return feature_count
    array = np.asarray(value)
    feature_count = int(np.prod(array.shape[1:]))
    print(f"{prefix:<32} shape={array.shape!s:<16} features={feature_count}")
    return feature_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="state_dict",
        control_mode=args.control_mode,
        render_mode=None,
        sim_backend="physx_cpu",
    )
    try:
        observation, info = env.reset(seed=args.seed)
        total = print_tree(observation)
        print(f"total_features={total}")
        print(f"action_shape={env.action_space.shape}")
        print(f"info_keys={sorted(info)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

