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


def build_pickcube_step_telemetry(
    observation: np.ndarray,
    action: np.ndarray,
    *,
    step: int,
    is_grasped: bool,
    is_obj_placed: bool,
    is_robot_static: bool,
    success: bool,
) -> dict[str, Any]:
    """Extract human-readable PickCube diagnostics from one post-step state.

    The fixed indices follow ManiSkill's state observation layout documented in
    ``docs/learning_guide.md``. Keeping this conversion separate from the
    simulator loop makes the saved JSON easy to inspect and unit test.
    """
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if observation.size != 42:
        raise ValueError(f"Expected a 42-D PickCube state observation, got {observation.size}")
    if action.size != 8:
        raise ValueError(f"Expected an 8-D PickCube action, got {action.size}")

    goal_position = observation[26:29]
    object_position = observation[29:32]
    return {
        "step": step,
        "object_goal_distance": float(np.linalg.norm(goal_position - object_position)),
        "object_height": float(object_position[2]),
        "goal_height": float(goal_position[2]),
        "object_goal_height_error": float(object_position[2] - goal_position[2]),
        "gripper_command": float(action[-1]),
        "arm_delta_l2": float(np.linalg.norm(action[:-1])),
        "is_grasped": is_grasped,
        "is_obj_placed": is_obj_placed,
        "is_robot_static": is_robot_static,
        "success": success,
    }


def summarize_pickcube_telemetry(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Create concise episode-level values from the step-by-step telemetry."""
    if not steps:
        return {}
    final = steps[-1]
    return {
        "min_object_goal_distance": min(item["object_goal_distance"] for item in steps),
        "final_object_goal_distance": final["object_goal_distance"],
        "final_object_height": final["object_height"],
        "final_goal_height": final["goal_height"],
        "final_object_goal_height_error": final["object_goal_height_error"],
        "final_gripper_command": final["gripper_command"],
        "final_arm_delta_l2": final["arm_delta_l2"],
        "final_is_grasped": final["is_grasped"],
        "final_is_obj_placed": final["is_obj_placed"],
        "final_is_robot_static": final["is_robot_static"],
    }


def classify_pickcube_failure(
    *, ever_grasped: bool, ever_placed: bool, final_is_grasped: bool
) -> str:
    """Give failures a mutually exclusive category from rollout state flags."""
    if not ever_grasped:
        return "never_grasped"
    if not final_is_grasped:
        return "lost_grasp_before_completion"
    if not ever_placed:
        return "holding_but_never_placed"
    return "reached_goal_but_not_completed"


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
    telemetry_episodes: list[dict[str, Any]] = []

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
            step_telemetry: list[dict[str, Any]] = []

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
                step_telemetry.append(
                    build_pickcube_step_telemetry(
                        observation_array,
                        action[0],
                        step=step,
                        is_grasped=final_is_grasped,
                        is_obj_placed=final_is_placed,
                        is_robot_static=final_is_static,
                        success=success,
                    )
                )
                if success:
                    successes += 1
                    episode_succeeded = True
                    break
                if bool(terminated.detach().cpu().item()) or bool(truncated.detach().cpu().item()):
                    break
            completed_steps.append(step)
            summary = summarize_pickcube_telemetry(step_telemetry)
            if not episode_succeeded:
                failed_seeds.append(episode_seed)
                category = classify_pickcube_failure(
                    ever_grasped=ever_grasped,
                    ever_placed=ever_placed,
                    final_is_grasped=final_is_grasped,
                )
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
            telemetry_episodes.append(
                {
                    "seed": episode_seed,
                    "success": episode_succeeded,
                    "steps": step,
                    "summary": summary,
                    "telemetry": step_telemetry,
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
    telemetry_path = output_path.parent / "rollout_telemetry.json"
    telemetry = {
        "schema_version": 1,
        "env_id": data_cfg["env_id"],
        "observation_layout": {
            "goal_position": [26, 29],
            "object_position": [29, 32],
            "action_gripper_command": 7,
        },
        "episodes": telemetry_episodes,
    }
    telemetry_path.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    print(
        f"episodes={eval_cfg['episodes']} success_rate={metrics['success_rate']:.3f} "
        f"mean_steps={metrics['mean_steps']:.2f}"
    )
    print(f"failed_seeds={failed_seeds}")
    print(f"failure_counts={dict(failure_counts)}")
    print(f"metrics={output_path}")
    print(f"telemetry={telemetry_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(load_config(args.config), args.checkpoint)
