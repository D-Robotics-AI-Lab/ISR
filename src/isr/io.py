"""Input/output utilities for reading and parsing trajectory episode files."""

from __future__ import annotations

import numpy as np


def load_episode_arrays(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Extract position, timestamp, gripper and frame-index arrays from raw episode dict."""
    positions: list[list[float]] = []
    timestamps: list[float] = []
    gripper: list[float] = []
    frame_indices: list[int] = []

    for frame in data["data"]:
        states = frame["states"]
        ee_qpos = states["left_ee"]["qpos"]
        arm_qpos = states["left_arm"]["qpos"]

        positions.append(ee_qpos[:3])
        timestamps.append(float(states["left_arm"]["timestamp"]))
        gripper.append(float(ee_qpos[7] if len(ee_qpos) > 7 else arm_qpos[6] if len(arm_qpos) > 6 else 0.0))
        frame_indices.append(int(frame["idx"]))

    return (
        np.array(positions, dtype=float),
        np.array(timestamps, dtype=float),
        np.array(gripper, dtype=float),
        frame_indices,
    )
