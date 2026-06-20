"""Trajectory kinematics utilities for ISR resampling."""

from __future__ import annotations

import numpy as np


def compute_velocities(
    positions: np.ndarray,
    times: np.ndarray | None = None,
    method: str = "forward",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute velocity vectors and speed magnitudes for a trajectory."""
    positions = np.asarray(positions, dtype=float)
    num_frames, _ = positions.shape

    if num_frames <= 1:
        velocities = np.zeros_like(positions)
        speeds = np.zeros((num_frames,), dtype=float)
        return velocities, speeds

    if times is not None:
        times = np.asarray(times, dtype=float)
        if times.shape != (num_frames,):
            raise ValueError(f"times shape {times.shape} does not match positions shape {positions.shape}")
        dt = np.maximum(times[1:] - times[:-1], 1e-9)
    else:
        dt = np.ones((num_frames - 1,), dtype=float)

    velocities = np.zeros_like(positions, dtype=float)

    if method == "forward":
        velocities[:-1] = (positions[1:] - positions[:-1]) / dt[:, None]
        velocities[-1] = (positions[-1] - positions[-2]) / dt[-1]
    elif method == "backward":
        velocities[0] = (positions[1] - positions[0]) / dt[0]
        velocities[1:] = (positions[1:] - positions[:-1]) / dt[:, None]
    elif method == "central":
        velocities[0] = (positions[1] - positions[0]) / dt[0]
        for i in range(1, num_frames - 1):
            velocities[i] = (positions[i + 1] - positions[i - 1]) / (dt[i - 1] + dt[i])
        velocities[-1] = (positions[-1] - positions[-2]) / dt[-1]
    else:
        raise ValueError("method must be 'forward', 'backward', or 'central'")

    speeds = np.linalg.norm(velocities, axis=1)
    return velocities, speeds


def compute_acceleration_magnitudes_from_positions(
    positions: np.ndarray,
    times: np.ndarray | None = None,
) -> np.ndarray:
    """Compute acceleration magnitudes from position samples."""
    velocities, _ = compute_velocities(positions, times=times, method="forward")
    return _compute_acceleration_magnitudes_from_velocities(
        velocities,
        times=times,
        method="mean_dt",
    )


def _compute_acceleration_magnitudes_from_velocities(
    velocities: np.ndarray,
    times: np.ndarray | None = None,
    method: str = "backward",
) -> np.ndarray:
    velocities = np.asarray(velocities, dtype=float)
    num_frames, _ = velocities.shape

    if num_frames <= 1:
        return np.zeros((num_frames,), dtype=float)

    if times is not None:
        times = np.asarray(times, dtype=float)
        if times.shape != (num_frames,):
            raise ValueError(f"times shape {times.shape} does not match velocities shape {velocities.shape}")
        dt = np.maximum(times[1:] - times[:-1], 1e-9)
    else:
        dt = np.ones((num_frames - 1,), dtype=float)

    accelerations = np.zeros_like(velocities, dtype=float)

    if method == "forward":
        accelerations[:-1] = (velocities[1:] - velocities[:-1]) / dt[:, None]
        accelerations[-1] = accelerations[-2]
    elif method == "backward":
        accelerations[0] = (velocities[1] - velocities[0]) / dt[0]
        accelerations[1:] = (velocities[1:] - velocities[:-1]) / dt[:, None]
    elif method == "mean_dt":
        if num_frames > 2:
            mean_dt = 0.5 * (dt[:-1] + dt[1:])
            accelerations[1:-1] = (velocities[1:-1] - velocities[:-2]) / mean_dt[:, None]
            accelerations[0] = accelerations[1]
            accelerations[-1] = accelerations[-2]
    elif method == "central":
        accelerations[0] = (velocities[1] - velocities[0]) / dt[0]
        for i in range(1, num_frames - 1):
            accelerations[i] = (velocities[i + 1] - velocities[i - 1]) / (dt[i - 1] + dt[i])
        accelerations[-1] = (velocities[-1] - velocities[-2]) / dt[-1]
    else:
        raise ValueError("method must be 'forward', 'backward', 'mean_dt', or 'central'")

    return np.linalg.norm(accelerations, axis=1)
