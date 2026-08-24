from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ActionChunkingTransformer(nn.Module):
    """Predict a fixed-size future action chunk from recent observations."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int,
        action_horizon: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        self.obs_projection = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.action_queries = nn.Parameter(torch.empty(action_horizon, d_model))
        self.position_embedding = nn.Parameter(
            torch.empty(obs_horizon + action_horizon, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.action_head = nn.Linear(d_model, action_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.action_queries, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 3 or observations.shape[1] != self.obs_horizon:
            raise ValueError(
                f"Expected observations [batch, {self.obs_horizon}, obs_dim], "
                f"received {tuple(observations.shape)}"
            )
        batch_size = observations.shape[0]
        obs_tokens = self.obs_projection(observations)
        query_tokens = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = torch.cat([obs_tokens, query_tokens], dim=1)
        tokens = tokens + self.position_embedding.unsqueeze(0)
        encoded = self.transformer(tokens)
        return self.action_head(encoded[:, -self.action_horizon :])

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

