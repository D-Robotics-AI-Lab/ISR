"""Unit tests for trajectory kinematics utilities."""

from __future__ import annotations

import numpy as np
import pytest

from isr.kinematics import compute_acceleration_magnitudes_from_positions, compute_velocities


# ---------------------------------------------------------------------------
# compute_velocities
# ---------------------------------------------------------------------------

def test_single_frame_velocity_is_zero():
    positions = np.array([[1.0, 2.0, 3.0]])
    vels, speeds = compute_velocities(positions)
    assert vels.shape == (1, 3)
    assert speeds.shape == (1,)
    np.testing.assert_array_equal(vels, np.zeros((1, 3)))
    np.testing.assert_array_equal(speeds, np.zeros(1))


def test_uniform_motion_constant_velocity():
    """Uniform motion along X: velocity should be constant."""
    n = 10
    positions = np.column_stack([np.arange(n, dtype=float), np.zeros(n), np.zeros(n)])
    vels, speeds = compute_velocities(positions, method="forward")
    # All speeds should be ~1.0 (unit steps, unit time)
    np.testing.assert_allclose(speeds[:-1], 1.0, atol=1e-10)


def test_stationary_trajectory_zero_speed():
    positions = np.zeros((10, 3))
    _, speeds = compute_velocities(positions)
    np.testing.assert_array_equal(speeds, np.zeros(10))


def test_velocity_shape_matches_positions():
    positions = np.random.rand(15, 3)
    vels, speeds = compute_velocities(positions)
    assert vels.shape == positions.shape
    assert speeds.shape == (15,)


def test_velocity_with_timestamps():
    """Providing timestamps should scale velocities correctly."""
    positions = np.column_stack([np.array([0.0, 2.0, 4.0]), np.zeros(3), np.zeros(3)])
    times = np.array([0.0, 2.0, 4.0])   # dt=2 each step
    vels, speeds = compute_velocities(positions, times=times, method="forward")
    # displacement=2, dt=2 → speed=1
    np.testing.assert_allclose(speeds[0], 1.0, atol=1e-10)


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method must be"):
        compute_velocities(np.random.rand(5, 3), method="invalid")


def test_timestamps_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_velocities(np.random.rand(5, 3), times=np.arange(3, dtype=float))


# ---------------------------------------------------------------------------
# compute_acceleration_magnitudes_from_positions
# ---------------------------------------------------------------------------

def test_acceleration_shape():
    positions = np.random.rand(20, 3)
    acc = compute_acceleration_magnitudes_from_positions(positions)
    assert acc.shape == (20,)


def test_acceleration_non_negative():
    positions = np.random.rand(20, 3)
    acc = compute_acceleration_magnitudes_from_positions(positions)
    assert np.all(acc >= 0.0)


def test_constant_velocity_low_acceleration():
    """Uniform linear motion should produce near-zero acceleration."""
    n = 20
    positions = np.column_stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)])
    acc = compute_acceleration_magnitudes_from_positions(positions)
    np.testing.assert_allclose(acc[1:-1], 0.0, atol=1e-10)


def test_single_frame_acceleration_is_zero():
    acc = compute_acceleration_magnitudes_from_positions(np.zeros((1, 3)))
    assert acc.shape == (1,)
    np.testing.assert_array_equal(acc, np.zeros(1))
