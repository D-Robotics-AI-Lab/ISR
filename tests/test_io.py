"""Unit tests for I/O and parsing utilities."""

from __future__ import annotations

import pytest

from isr import load_episode_arrays


def test_load_episode_arrays_success():
    # Construct a dummy dataset representing the JSON schema
    dummy_data = {
        "info": {"name": "test_env"},
        "text": "test instruction",
        "data": [
            {
                "idx": 0,
                "states": {
                    "left_ee": {"qpos": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.1]},
                    "left_arm": {"qpos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "timestamp": 1000.0},
                }
            },
            {
                "idx": 1,
                "states": {
                    "left_ee": {"qpos": [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.2]},
                    "left_arm": {"qpos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "timestamp": 2000.0},
                }
            }
        ]
    }

    positions, timestamps, gripper, frame_indices = load_episode_arrays(dummy_data)

    assert positions.shape == (2, 3)
    assert positions[0].tolist() == [1.0, 2.0, 3.0]
    assert positions[1].tolist() == [4.0, 5.0, 6.0]

    assert timestamps.shape == (2,)
    assert timestamps[0] == 1000.0
    assert timestamps[1] == 2000.0

    assert gripper.shape == (2,)
    assert gripper[0] == pytest.approx(0.1)
    assert gripper[1] == pytest.approx(0.2)

    assert frame_indices == [0, 1]


def test_load_episode_arrays_fallback_gripper():
    # Construct a dummy dataset where ee_qpos doesn't have 8 elements, fallback to arm_qpos
    dummy_data = {
        "data": [
            {
                "idx": 10,
                "states": {
                    "left_ee": {"qpos": [1.0, 2.0, 3.0]}, # short
                    "left_arm": {"qpos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], "timestamp": 10.0},
                }
            }
        ]
    }

    _, _, gripper, frame_indices = load_episode_arrays(dummy_data)
    assert gripper[0] == pytest.approx(0.5)
    assert frame_indices == [10]
