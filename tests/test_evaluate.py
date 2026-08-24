import unittest

import numpy as np
import torch

from embodied_policy.evaluate import (
    build_pickcube_step_telemetry,
    classify_pickcube_failure,
    extract_rgb_state_observation,
    summarize_pickcube_telemetry,
)


class PickCubeTelemetryTest(unittest.TestCase):
    def test_extracts_distances_actions_and_flags(self) -> None:
        observation = np.zeros(42, dtype=np.float32)
        observation[26:29] = [0.1, -0.2, 0.3]
        observation[29:32] = [0.1, -0.2, 0.1]
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

        telemetry = build_pickcube_step_telemetry(
            observation,
            action,
            step=4,
            is_grasped=True,
            is_obj_placed=False,
            is_robot_static=False,
            success=False,
        )

        self.assertEqual(telemetry["step"], 4)
        self.assertAlmostEqual(telemetry["object_goal_distance"], 0.2, places=6)
        self.assertAlmostEqual(telemetry["object_height"], 0.1, places=6)
        self.assertAlmostEqual(telemetry["goal_height"], 0.3, places=6)
        self.assertAlmostEqual(telemetry["object_goal_height_error"], -0.2, places=6)
        self.assertEqual(telemetry["gripper_command"], -1.0)
        self.assertEqual(telemetry["arm_delta_l2"], 1.0)
        self.assertTrue(telemetry["is_grasped"])
        self.assertFalse(telemetry["success"])

    def test_summary_uses_final_values_and_minimum_distance(self) -> None:
        first = {"object_goal_distance": 0.3, "object_height": 0.05, "goal_height": 0.1,
                 "object_goal_height_error": -0.05, "gripper_command": -1.0,
                 "arm_delta_l2": 0.2, "is_grasped": True, "is_obj_placed": False,
                 "is_robot_static": False}
        final = {**first, "object_goal_distance": 0.1, "is_obj_placed": True,
                 "is_robot_static": True}
        summary = summarize_pickcube_telemetry([first, final])
        self.assertEqual(summary["min_object_goal_distance"], 0.1)
        self.assertEqual(summary["final_object_goal_distance"], 0.1)
        self.assertTrue(summary["final_is_obj_placed"])
        self.assertTrue(summary["final_is_robot_static"])

    def test_rejects_wrong_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "42-D"):
            build_pickcube_step_telemetry(
                np.zeros(41), np.zeros(8), step=1, is_grasped=False,
                is_obj_placed=False, is_robot_static=False, success=False,
            )

    def test_failure_categories_distinguish_dropped_object(self) -> None:
        self.assertEqual(
            classify_pickcube_failure(
                ever_grasped=True, ever_placed=False, final_is_grasped=False
            ),
            "lost_grasp_before_completion",
        )
        self.assertEqual(
            classify_pickcube_failure(
                ever_grasped=True, ever_placed=False, final_is_grasped=True
            ),
            "holding_but_never_placed",
        )

    def test_extracts_batched_rgb_state_observation(self) -> None:
        observation = {
            "state": torch.arange(42, dtype=torch.float32).reshape(1, 42),
            "sensor_data": {"base_camera": {"rgb": torch.full((1, 4, 6, 3), 255, dtype=torch.uint8)}},
        }
        state, image = extract_rgb_state_observation(observation, "base_camera")
        self.assertEqual(tuple(state.shape), (42,))
        self.assertEqual(tuple(image.shape), (3, 4, 6))
        self.assertEqual(float(image.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
