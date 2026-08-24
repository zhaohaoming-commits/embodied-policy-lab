"""Run a reproducible, sequential training-seed sweep for one or more configs."""

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
import yaml

from embodied_policy.config import load_config
from embodied_policy.evaluate import evaluate
from embodied_policy.train import train


def prepare_run_config(
    base_config: dict[str, Any], config_name: str, train_seed: int, output_root: Path
) -> dict[str, Any]:
    """Clone a config and give one training seed an isolated output directory."""
    config = copy.deepcopy(base_config)
    run_id = f"{config_name}_train_seed_{train_seed}"
    config["seed"] = train_seed
    config["output_dir"] = str(output_root / run_id)
    return config


def aggregate_results(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute mean and sample standard deviation for each config."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["config_name"]].append(result)

    rows: list[dict[str, Any]] = []
    for config_name, group in sorted(grouped.items()):
        success_rates = np.asarray([item["success_rate"] for item in group], dtype=float)
        mean_steps = np.asarray([item["mean_steps"] for item in group], dtype=float)
        rows.append(
            {
                "config_name": config_name,
                "model_type": group[0]["model_type"],
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


def write_summaries(output_root: Path, results: Sequence[dict[str, Any]]) -> None:
    aggregate = aggregate_results(results)
    (output_root / "run_summary.json").write_text(
        json.dumps({"runs": list(results), "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    write_csv(output_root / "run_summary.csv", results)
    write_csv(output_root / "aggregate_summary.csv", aggregate)


def run_seed_sweep(
    config_paths: Sequence[str | Path], train_seeds: Sequence[int], output_root: str | Path
) -> list[dict[str, Any]]:
    """Train and evaluate every config/seed pair, one at a time, then summarize."""
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"Output root already exists: {root}. Choose a new directory.")
    root.mkdir(parents=True)

    results: list[dict[str, Any]] = []
    for config_path in config_paths:
        path = Path(config_path)
        base_config = load_config(path)
        for train_seed in train_seeds:
            config = prepare_run_config(base_config, path.stem, train_seed, root)
            run_dir = Path(config["output_dir"])
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            print(f"\n=== run={run_dir.name} ===")
            checkpoint_path = train(config)
            metrics = evaluate(config, checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            result = {
                "run_id": run_dir.name,
                "config_name": path.stem,
                "model_type": config["model"]["type"],
                "train_seed": train_seed,
                "data_split_seed": config["data"].get("split_seed", train_seed),
                "evaluation_seed": config["eval"].get("seed", train_seed),
                "best_epoch": checkpoint["epoch"],
                "best_val_loss": checkpoint["val_loss"],
                "success_rate": metrics["success_rate"],
                "mean_steps": metrics["mean_steps"],
                "output_dir": str(run_dir),
            }
            results.append(result)
            write_summaries(root, results)
            print(
                f"completed={run_dir.name} success_rate={result['success_rate']:.3f} "
                f"best_val_loss={result['best_val_loss']:.6f}"
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--train-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_seed_sweep(args.configs, args.train_seeds, args.output_root)
