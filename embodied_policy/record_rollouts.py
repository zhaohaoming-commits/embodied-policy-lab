from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch

from embodied_policy.build import build_model
from embodied_policy.config import load_config
from embodied_policy.utils import choose_device, seed_everything


@torch.no_grad()
def record_rollouts(
    config_path: str,
    checkpoint_path: str,
    seeds: list[int],
    output_dir: str,
) -> None:
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from mani_skill.utils.wrappers.record import RecordEpisode
    except ImportError as error:
        raise RuntimeError("Install the 'sim' dependencies to record ManiSkill videos") from error

    config = load_config(config_path)
    data_cfg = config["data"]
    eval_cfg = config["eval"]
    seed_everything(config["seed"])
    device = choose_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    normalization = checkpoint.get("normalization")
    if normalization is None:
        raise ValueError("Checkpoint is missing training-set normalization statistics")
    obs_mean = np.asarray(normalization["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(normalization["obs_std"], dtype=np.float32)
    action_mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    action_std = np.asarray(normalization["action_std"], dtype=np.float32)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        data_cfg["env_id"],
        num_envs=1,
        obs_mode=data_cfg["obs_mode"],
        control_mode=data_cfg["control_mode"],
        reward_mode="dense",
        render_mode="rgb_array",
        sim_backend=data_cfg["sim_backend"],
    )
    env = RecordEpisode(
        env,
        output_dir=str(output_path),
        save_trajectory=False,
        save_video=True,
        # ManiSkill 3.0.1 returns a batched reward even when num_envs=1. Its
        # info-overlay path treats that array as a scalar and raises TypeError.
        # Seed/outcome are encoded in the video filename; evaluation JSON keeps
        # the detailed metrics, so disabling only the overlay loses no evidence.
        info_on_video=False,
        save_on_reset=False,
        clean_on_close=False,
        # The script flushes once per complete episode below, after the success
        # state is known. A max-step auto-flush would save 50-step failures with
        # anonymous numeric filenames before we can attach their seed/outcome.
        max_steps_per_video=None,
        video_fps=20,
    )

    try:
        for episode_seed in seeds:
            observation, _ = env.reset(seed=episode_seed)
            observation_array = observation.detach().cpu().numpy()[0]
            normalized_observation = (observation_array - obs_mean) / obs_std
            history: deque[np.ndarray] = deque(
                [normalized_observation.copy() for _ in range(data_cfg["obs_horizon"])],
                maxlen=data_cfg["obs_horizon"],
            )
            action_queue: deque[np.ndarray] = deque()
            succeeded = False

            for step in range(1, eval_cfg["max_steps"] + 1):
                if not action_queue or (step - 1) % eval_cfg["replan_interval"] == 0:
                    observations = torch.from_numpy(np.stack(history)).unsqueeze(0).to(device)
                    normalized_chunk = model(observations).squeeze(0).cpu().numpy()
                    action_chunk = normalized_chunk * action_std + action_mean
                    action_queue = deque(action_chunk[: eval_cfg["replan_interval"]])
                action = action_queue.popleft()
                action = np.clip(action, env.action_space.low, env.action_space.high)[None, :]
                observation, _, terminated, truncated, info = env.step(action)
                observation_array = observation.detach().cpu().numpy()[0]
                history.append((observation_array - obs_mean) / obs_std)
                succeeded = bool(info["success"].detach().cpu().item())
                if succeeded:
                    break
                if bool(terminated.detach().cpu().item()) or bool(truncated.detach().cpu().item()):
                    break

            outcome = "success" if succeeded else "failure"
            env.flush_video(name=f"seed_{episode_seed}_{outcome}", verbose=True)
            print(f"seed={episode_seed} outcome={outcome} steps={step}")
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--output-dir", default="outputs/videos")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    record_rollouts(args.config, args.checkpoint, args.seeds, args.output_dir)
