from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from embodied_policy.build import build_dataset, build_model
from embodied_policy.config import load_config
from embodied_policy.data import ManiSkillTrajectoryDataset
from embodied_policy.utils import choose_device, seed_everything


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    amp: bool,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    autocast_enabled = amp and device.type == "cuda"

    for batch in loader:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        action_mask = batch["action_mask"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                predictions = model(observations)
                loss = model.masked_loss(predictions, actions, action_mask)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        batch_size = observations.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def train(config: dict[str, Any]) -> Path:
    seed_everything(config["seed"])
    device = choose_device()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val")
    normalization = None
    if isinstance(train_dataset, ManiSkillTrajectoryDataset):
        normalization = train_dataset.compute_normalization()
        train_dataset.set_normalization(normalization)
        if not isinstance(val_dataset, ManiSkillTrajectoryDataset):
            raise TypeError("Training and validation dataset types must match")
        val_dataset.set_normalization(normalization)
    train_cfg = config["train"]
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )
    best_val = math.inf
    best_path = output_dir / "best.pt"
    metrics_path = output_dir / "metrics.jsonl"

    print(f"device={device} train_samples={len(train_dataset)} val_samples={len(val_dataset)}")
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, train_cfg["epochs"] + 1):
            train_loss = run_epoch(
                model, train_loader, device, optimizer, amp=train_cfg["amp"]
            )
            with torch.no_grad():
                val_loss = run_epoch(model, val_loader, device, optimizer=None, amp=False)
            scheduler.step()
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "normalization": normalization,
                    },
                    best_path,
                )
    print(f"best_checkpoint={best_path} best_val_loss={best_val:.6f}")
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(load_config(args.config))
