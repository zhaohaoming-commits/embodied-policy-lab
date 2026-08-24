import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from embodied_policy.data import ManiSkillTrajectoryDataset


class ManiSkillTrajectoryDatasetTest(unittest.TestCase):
    def test_windows_and_episode_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.h5"
            with h5py.File(path, "w") as handle:
                episode = handle.create_group("traj_0")
                episode.create_dataset(
                    "obs", data=np.arange(15, dtype=np.float32).reshape(5, 3)
                )
                episode.create_dataset(
                    "actions", data=np.arange(8, dtype=np.float32).reshape(4, 2)
                )
            dataset = ManiSkillTrajectoryDataset(path, ["traj_0"], 2, 3)
            first = dataset[0]
            last = dataset[3]
            self.assertEqual(tuple(first["observations"].shape), (2, 3))
            self.assertTrue(first["observations"][0].equal(first["observations"][1]))
            self.assertEqual(last["action_mask"].tolist(), [True, False, False])
            self.assertEqual(float(last["actions"][1:].abs().sum()), 0.0)
            stats = dataset.compute_normalization()
            dataset.set_normalization(stats)
            normalized = dataset[0]
            self.assertTrue(np.isfinite(normalized["observations"].numpy()).all())
            self.assertTrue(np.isfinite(normalized["actions"].numpy()).all())
            dataset.close()


if __name__ == "__main__":
    unittest.main()
