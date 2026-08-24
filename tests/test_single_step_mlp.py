import unittest

import torch

from embodied_policy.models import SingleStepMLP


class SingleStepMLPTest(unittest.TestCase):
    def test_forward_and_backward(self) -> None:
        model = SingleStepMLP(
            obs_dim=42,
            action_dim=8,
            obs_horizon=2,
            hidden_dims=[32, 32],
            dropout=0.0,
        )
        observations = torch.randn(4, 2, 42)
        targets = torch.randn(4, 1, 8)
        mask = torch.ones(4, 1, dtype=torch.bool)
        predictions = model(observations)
        self.assertEqual(tuple(predictions.shape), (4, 1, 8))
        loss = model.masked_loss(predictions, targets, mask)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()

