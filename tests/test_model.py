import unittest

import torch

from embodied_policy.models import ActionChunkingTransformer, VisionStateActionChunkingTransformer


class ActionChunkingTransformerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ActionChunkingTransformer(
            obs_dim=6,
            action_dim=3,
            obs_horizon=2,
            action_horizon=8,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            dropout=0.0,
        )

    def test_forward_and_backward(self) -> None:
        observations = torch.randn(4, 2, 6)
        targets = torch.randn(4, 8, 3)
        mask = torch.ones(4, 8, dtype=torch.bool)
        predictions = self.model(observations)
        self.assertEqual(tuple(predictions.shape), (4, 8, 3))
        loss = self.model.masked_loss(predictions, targets, mask)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))

    def test_mask_ignores_padded_targets(self) -> None:
        predictions = torch.zeros(1, 2, 3)
        targets = torch.tensor([[[1.0, 1.0, 1.0], [999.0, 999.0, 999.0]]])
        mask = torch.tensor([[True, False]])
        loss = self.model.masked_loss(predictions, targets, mask)
        self.assertAlmostEqual(loss.item(), 0.5, places=6)


class VisionStateActionChunkingTransformerTest(unittest.TestCase):
    def test_forward_and_backward(self) -> None:
        model = VisionStateActionChunkingTransformer(
            image_channels=3,
            obs_dim=6,
            action_dim=3,
            obs_horizon=2,
            action_horizon=4,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            dropout=0.0,
        )
        observations = torch.randn(2, 2, 6)
        images = torch.randn(2, 2, 3, 32, 32)
        targets = torch.randn(2, 4, 3)
        mask = torch.ones(2, 4, dtype=torch.bool)
        predictions = model(observations, images)
        self.assertEqual(tuple(predictions.shape), (2, 4, 3))
        model.masked_loss(predictions, targets, mask).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_rgb_only_mode_ignores_state_values(self) -> None:
        model = VisionStateActionChunkingTransformer(
            image_channels=3,
            use_state=False,
            obs_dim=6,
            action_dim=3,
            obs_horizon=2,
            action_horizon=4,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            dropout=0.0,
        ).eval()
        images = torch.randn(1, 2, 3, 32, 32)
        with torch.no_grad():
            first = model(torch.zeros(1, 2, 6), images)
            second = model(torch.ones(1, 2, 6), images)
        self.assertTrue(torch.equal(first, second))

    def test_resnet18_encoder_without_external_torchvision(self) -> None:
        model = VisionStateActionChunkingTransformer(
            image_channels=3, obs_dim=6, action_dim=3, obs_horizon=2, action_horizon=4,
            d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.0,
            vision_encoder="resnet18",
        ).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 2, 6), torch.randn(1, 2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 4, 3))


if __name__ == "__main__":
    unittest.main()
