from __future__ import annotations

import torch
from torch import nn

from embodied_policy.models.action_chunker import ActionChunkingTransformer


class VisionStateActionChunkingTransformer(ActionChunkingTransformer):
    """Fuse a lightweight RGB encoder with state tokens before action queries."""

    requires_images = True

    def __init__(
        self, image_channels: int, *args: object, use_state: bool = True, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_state = use_state
        d_model = self.obs_projection[0].out_features
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, observations: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1] != self.obs_horizon:
            raise ValueError(f"Expected images [batch, {self.obs_horizon}, channels, height, width]")
        if observations.ndim != 3 or observations.shape[:2] != images.shape[:2]:
            raise ValueError("State and image batch/time dimensions must match")
        batch_size, horizon, channels, height, width = images.shape
        image_tokens = self.image_encoder(images.reshape(batch_size * horizon, channels, height, width))
        image_tokens = image_tokens.reshape(batch_size, horizon, -1)
        state_tokens = self.obs_projection(observations) if self.use_state else torch.zeros_like(image_tokens)
        query_tokens = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = torch.cat([state_tokens + image_tokens, query_tokens], dim=1)
        encoded = self.transformer(tokens + self.position_embedding.unsqueeze(0))
        return self.action_head(encoded[:, -self.action_horizon :])
