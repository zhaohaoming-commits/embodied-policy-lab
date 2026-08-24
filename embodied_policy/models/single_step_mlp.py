from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class SingleStepMLP(nn.Module):
    """Behavior-cloning baseline that predicts only the next action."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.obs_horizon = obs_horizon
        layers: list[nn.Module] = []
        input_dim = obs_dim * obs_horizon
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 3 or observations.shape[1] != self.obs_horizon:
            raise ValueError(
                f"Expected observations [batch, {self.obs_horizon}, obs_dim], "
                f"received {tuple(observations.shape)}"
            )
        flattened = observations.flatten(start_dim=1)
        return self.network(flattened).unsqueeze(1)

    @staticmethod
    def masked_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        if predictions.shape != targets.shape:
            raise ValueError("Prediction and target shapes must match")
        per_value_loss = F.smooth_l1_loss(predictions, targets, reduction="none")
        mask = action_mask.unsqueeze(-1).expand_as(per_value_loss)
        return per_value_loss[mask].mean()

