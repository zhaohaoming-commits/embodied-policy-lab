import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from embodied_policy.data import ManiSkillTrajectoryDataset, VisionStateTrajectoryDataset


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

    def test_vision_state_alignment_and_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vision_trajectory.h5"
            with h5py.File(path, "w") as handle:
                episode = handle.create_group("traj_0")
                observation = episode.create_group("obs")
                observation.create_dataset("state", data=np.arange(20, dtype=np.float32).reshape(5, 4))
                sensor_data = observation.create_group("sensor_data").create_group("base_camera")
                rgb = np.zeros((5, 4, 6, 3), dtype=np.uint8)
                rgb[:, :, :, 0] = 255
                sensor_data.create_dataset("rgb", data=rgb)
                episode.create_dataset("actions", data=np.arange(8, dtype=np.float32).reshape(4, 2))
            dataset = VisionStateTrajectoryDataset(path, ["traj_0"], 2, 3)
            first = dataset[0]
            last = dataset[3]
            self.assertEqual(tuple(first["images"].shape), (2, 3, 4, 6))
            self.assertTrue(first["images"][0].equal(first["images"][1]))
            self.assertEqual(last["action_mask"].tolist(), [True, False, False])
            stats = dataset.compute_normalization()
            self.assertTrue(np.allclose(stats["image_mean"], [1.0, 0.0, 0.0]))
            dataset.set_normalization(stats)
            normalized = dataset[0]
            self.assertTrue(np.isfinite(normalized["images"].numpy()).all())
            self.assertTrue(np.isfinite(normalized["observations"].numpy()).all())
            dataset.close()

    def test_vision_state_can_select_proprioception_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vision_subset.h5"
            with h5py.File(path, "w") as handle:
                episode = handle.create_group("traj_0")
                obs = episode.create_group("obs")
                obs.create_dataset("state", data=np.arange(20, dtype=np.float32).reshape(5, 4))
                camera = obs.create_group("sensor_data").create_group("base_camera")
                camera.create_dataset("rgb", data=np.zeros((5, 2, 2, 3), dtype=np.uint8))
                episode.create_dataset("actions", data=np.zeros((4, 2), dtype=np.float32))
            dataset = VisionStateTrajectoryDataset(path, ["traj_0"], 1, 1, state_indices=[0, 2])
            sample = dataset[1]
            self.assertEqual(tuple(sample["observations"].shape), (1, 2))
            self.assertEqual(sample["observations"][0].tolist(), [4.0, 6.0])
            dataset.close()


if __name__ == "__main__":
    unittest.main()
