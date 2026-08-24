"""Evaluate existing action-chunking checkpoints at several replan intervals."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from embodied_policy.config import load_config
from embodied_policy.evaluate import evaluate


def aggregate_by_replan_interval(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["replan_interval"]].append(result)
    rows: list[dict[str, Any]] = []
    for interval, group in sorted(grouped.items()):
        success_rates = np.asarray([row["success_rate"] for row in group], dtype=float)
        mean_steps = np.asarray([row["mean_steps"] for row in group], dtype=float)
        rows.append(
            {
                "replan_interval": interval,
                "runs": len(group),
                "success_rate_mean": float(success_rates.mean()),
                "success_rate_std": float(success_rates.std(ddof=1))
                if len(success_rates) > 1
                else 0.0,
                "mean_steps_mean": float(mean_steps.mean()),
                "mean_steps_std": float(mean_steps.std(ddof=1))
                if len(mean_steps) > 1
                else 0.0,
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_replan_sweep(
    config_path: str | Path,
    checkpoint_paths: Sequence[str | Path],
    replan_intervals: Sequence[int],
    output_root: str | Path,
) -> list[dict[str, Any]]:
    """Re-evaluate checkpoints without retraining them."""
    base_config = load_config(config_path)
    action_horizon = base_config["data"]["action_horizon"]
    if any(interval <= 0 or interval > action_horizon for interval in replan_intervals):
        raise ValueError(
            f"Each replan interval must be in [1, {action_horizon}], got {list(replan_intervals)}"
        )
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"Output root already exists: {root}. Choose a new directory.")
    root.mkdir(parents=True)

    results: list[dict[str, Any]] = []
    for checkpoint_path in map(Path, checkpoint_paths):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        train_seed = checkpoint["config"]["seed"]
        for interval in replan_intervals:
            config = copy.deepcopy(base_config)
            config["seed"] = train_seed
            config["eval"]["replan_interval"] = interval
            run_dir = root / f"train_seed_{train_seed}_replan_{interval}"
            config["output_dir"] = str(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== checkpoint_seed={train_seed} replan_interval={interval} ===")
            metrics = evaluate(config, checkpoint_path)
            results.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "train_seed": train_seed,
                    "replan_interval": interval,
                    "success_rate": metrics["success_rate"],
                    "mean_steps": metrics["mean_steps"],
                    "output_dir": str(run_dir),
                }
            )
            aggregate = aggregate_by_replan_interval(results)
            (root / "replan_summary.json").write_text(
                json.dumps({"runs": results, "aggregate": aggregate}, indent=2),
                encoding="utf-8",
            )
            write_csv(root / "replan_runs.csv", results)
            write_csv(root / "replan_aggregate.csv", aggregate)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--replan-intervals", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_replan_sweep(
        args.config, args.checkpoints, args.replan_intervals, args.output_root
    )
