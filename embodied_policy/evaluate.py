from __future__ import annotations

import argparse
from collections import Counter
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embodied_policy.build import build_model
from embodied_policy.config import load_config
from embodied_policy.utils import choose_device, seed_everything


@torch.no_grad()
def evaluate(config: dict[str, Any], checkpoint_path: str | Path) -> dict[str, Any]:
    if config["data"].get("kind", "synthetic_reach") == "maniskill_hdf5":
        return evaluate_maniskill(config, checkpoint_path)
    seed_everything(config["seed"])
    device = choose_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data_cfg = config["data"]
    eval_cfg = config["eval"]
    rng = np.random.default_rng(config["seed"] + 20_000)
    successes = 0
    final_distances: list[float] = []
    steps_taken: list[int] = []

    for _ in range(eval_cfg["episodes"]):
        position = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        goal = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        initial_obs = np.concatenate([position, goal]).astype(np.float32)
        history: deque[np.ndarray] = deque(
            [initial_obs.copy() for _ in range(data_cfg["obs_horizon"])],
            maxlen=data_cfg["obs_horizon"],
        )
        action_queue: deque[np.ndarray] = deque()

        for step in range(1, eval_cfg["max_steps"] + 1):
            if not action_queue or (step - 1) % eval_cfg["replan_interval"] == 0:
                observations = torch.from_numpy(np.stack(history)).unsqueeze(0).to(device)
                action_chunk = model(observations).squeeze(0).cpu().numpy()
                action_queue = deque(action_chunk[: eval_cfg["replan_interval"]])
            action = np.clip(
                action_queue.popleft(), -data_cfg["action_limit"], data_cfg["action_limit"]
            )
            position = position + data_cfg["dt"] * action
            observation = np.concatenate([position, goal]).astype(np.float32)
            history.append(observation)
            if np.linalg.norm(goal - position) < eval_cfg["success_threshold"]:
                successes += 1
                break

        final_distances.append(float(np.linalg.norm(goal - position)))
        steps_taken.append(step)

    metrics = {
        "success_rate": successes / eval_cfg["episodes"],
        "mean_final_distance": float(np.mean(final_distances)),
        "mean_steps": float(np.mean(steps_taken)),
    }
    print(
        f"episodes={eval_cfg['episodes']} success_rate={metrics['success_rate']:.3f} "
        f"mean_final_distance={metrics['mean_final_distance']:.4f} "
        f"mean_steps={metrics['mean_steps']:.2f}"
    )
    return metrics


@torch.no_grad()
def evaluate_maniskill(
    config: dict[str, Any], checkpoint_path: str | Path
) -> dict[str, Any]:
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Install the 'sim' dependencies to evaluate ManiSkill") from error

    seed_everything(config["seed"])
    device = choose_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    normalization = checkpoint.get("normalization")
    if normalization is None:
        raise ValueError("ManiSkill checkpoint is missing training-set normalization statistics")
    obs_mean = np.asarray(normalization["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(normalization["obs_std"], dtype=np.float32)
    action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    action_std = np.asarray(normalization["action_std"], dtype=np.float32)

    data_cfg = config["data"]
    eval_cfg = config["eval"]
    env = gym.make(
        data_cfg["env_id"],
        num_envs=1,
        obs_mode=data_cfg["obs_mode"],
        control_mode=data_cfg["control_mode"],
        render_mode=None,
        sim_backend=data_cfg["sim_backend"],
    )
    successes = 0
    completed_steps: list[int] = []
    failed_seeds: list[int] = []
    failure_details: list[dict[str, Any]] = []

    try:
        for episode_index in range(eval_cfg["episodes"]):
            episode_seed = config["seed"] + 30_000 + episode_index
            observation, _ = env.reset(seed=episode_seed)
            observation_array = observation.detach().cpu().numpy()[0]
            normalized_observation = (observation_array - obs_mean) / obs_std
            history: deque[np.ndarray] = deque(
                [normalized_observation.copy() for _ in range(data_cfg["obs_horizon"])],
                maxlen=data_cfg["obs_horizon"],
            )
            action_queue: deque[np.ndarray] = deque()
            episode_succeeded = False
            ever_grasped = False
            ever_placed = False
            final_is_grasped = False
            final_is_placed = False
            final_is_static = False

            for step in range(1, eval_cfg["max_steps"] + 1):
                if not action_queue or (step - 1) % eval_cfg["replan_interval"] == 0:
                    observations = torch.from_numpy(np.stack(history)).unsqueeze(0).to(device)
                    normalized_action_chunk = model(observations).squeeze(0).cpu().numpy()
                    action_chunk = normalized_action_chunk * action_std + action_mean
                    action_queue = deque(action_chunk[: eval_cfg["replan_interval"]])
                action = action_queue.popleft()
                action = np.clip(action, env.action_space.low, env.action_space.high)[None, :]
                observation, _, terminated, truncated, info = env.step(action)
                observation_array = observation.detach().cpu().numpy()[0]
                history.append((observation_array - obs_mean) / obs_std)
                success = bool(info["success"].detach().cpu().item())
                final_is_grasped = bool(info["is_grasped"].detach().cpu().item())
                final_is_placed = bool(info["is_obj_placed"].detach().cpu().item())
                final_is_static = bool(info["is_robot_static"].detach().cpu().item())
                ever_grasped = ever_grasped or final_is_grasped
                ever_placed = ever_placed or final_is_placed
                if success:
                    successes += 1
                    episode_succeeded = True
                    break
                if bool(terminated.detach().cpu().item()) or bool(truncated.detach().cpu().item()):
                    break
            completed_steps.append(step)
            if not episode_succeeded:
                failed_seeds.append(episode_seed)
                if not ever_grasped:
                    category = "never_grasped"
                elif not ever_placed:
                    category = "grasped_but_never_placed"
                else:
                    category = "reached_goal_but_not_completed"
                failure_details.append(
                    {
                        "seed": episode_seed,
                        "category": category,
                        "steps": step,
                        "ever_grasped": ever_grasped,
                        "ever_placed": ever_placed,
                        "final_is_grasped": final_is_grasped,
                        "final_is_placed": final_is_placed,
                        "final_is_static": final_is_static,
                    }
                )
    finally:
        env.close()

    failure_counts = Counter(detail["category"] for detail in failure_details)
    metrics = {
        "episodes": eval_cfg["episodes"],
        "success_rate": successes / eval_cfg["episodes"],
        "mean_steps": float(np.mean(completed_steps)),
        "failed_seeds": failed_seeds,
        "failure_counts": dict(failure_counts),
        "failure_details": failure_details,
    }
    output_path = Path(config["output_dir"]) / "eval_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"episodes={eval_cfg['episodes']} success_rate={metrics['success_rate']:.3f} "
        f"mean_steps={metrics['mean_steps']:.2f}"
    )
    print(f"failed_seeds={failed_seeds}")
    print(f"failure_counts={dict(failure_counts)}")
    print(f"metrics={output_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(load_config(args.config), args.checkpoint)
