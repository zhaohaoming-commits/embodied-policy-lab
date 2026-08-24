import unittest

from embodied_policy.data import SyntheticReachDataset


class SyntheticReachDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = SyntheticReachDataset(
            num_episodes=2,
            episode_length=5,
            obs_horizon=3,
            action_horizon=4,
            dt=0.1,
            expert_gain=2.0,
            action_limit=1.0,
            seed=0,
        )

    def test_shapes_and_left_padding(self) -> None:
        sample = self.dataset[0]
        self.assertEqual(tuple(sample["observations"].shape), (3, 6))
        self.assertEqual(tuple(sample["actions"].shape), (4, 3))
        self.assertTrue(sample["action_mask"].all())
        self.assertTrue(
            sample["observations"][0].equal(sample["observations"][1])
        )

    def test_action_padding_mask(self) -> None:
        sample = self.dataset[4]
        self.assertEqual(sample["action_mask"].tolist(), [True, False, False, False])
        self.assertEqual(float(sample["actions"][1:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

