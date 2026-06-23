"""Unit tests for isr_resample_trajectory — core boundary cases."""

from __future__ import annotations

import numpy as np
import pytest

from isr import isr_resample_trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _straight_line(n: int, length: float = 1.0) -> np.ndarray:
    """Return n points uniformly spaced along the X axis."""
    return np.column_stack([np.linspace(0, length, n), np.zeros(n), np.zeros(n)])


# ---------------------------------------------------------------------------
# Edge cases: empty / single / two frames
# ---------------------------------------------------------------------------

def test_empty_trajectory():
    result = isr_resample_trajectory(np.zeros((0, 3)))
    assert result == [], "Empty input should return an empty list"


def test_single_frame():
    result = isr_resample_trajectory(np.zeros((1, 3)))
    assert result == [0], "Single-frame input should return [0]"


def test_two_frames_always_kept():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = isr_resample_trajectory(positions)
    assert 0 in result and 1 in result, "Both frames of a 2-frame trajectory must be kept"


# ---------------------------------------------------------------------------
# Boundary indices are always included
# ---------------------------------------------------------------------------

def test_first_and_last_always_included():
    positions = _straight_line(50, length=2.0)
    result = isr_resample_trajectory(positions, d_target=0.1)
    assert result[0] == 0, "First frame must always be selected"
    assert result[-1] == 49, "Last frame must always be selected"


def test_result_is_sorted():
    positions = _straight_line(30, length=1.5)
    result = isr_resample_trajectory(positions, d_target=0.1)
    assert result == sorted(result), "Returned indices must be in ascending order"


def test_result_contains_no_duplicates():
    positions = _straight_line(30, length=1.5)
    result = isr_resample_trajectory(positions, d_target=0.1)
    assert len(result) == len(set(result)), "Returned indices must be unique"


# ---------------------------------------------------------------------------
# Straight-line trajectory: spacing should be roughly d_target
# ---------------------------------------------------------------------------

def test_straight_line_spacing():
    """For a uniform straight line, selected frames should be ~d_target apart."""
    n = 100
    total_length = 1.0
    d_target = 0.1
    positions = _straight_line(n, length=total_length)
    result = isr_resample_trajectory(positions, d_target=d_target, lambda_acc=0.0)

    selected = positions[result]
    dists = np.linalg.norm(np.diff(selected, axis=0), axis=1)
    # Allow generous tolerance — DP is approximate
    assert dists.mean() == pytest.approx(d_target, rel=0.5), (
        f"Mean segment length {dists.mean():.4f} should be near d_target={d_target}"
    )


def test_large_d_target_keeps_fewer_frames():
    positions = _straight_line(50, length=1.0)
    result_fine = isr_resample_trajectory(positions, d_target=0.05, lambda_acc=0.0)
    result_coarse = isr_resample_trajectory(positions, d_target=0.25, lambda_acc=0.0)
    assert len(result_coarse) <= len(result_fine), (
        "Larger d_target should select fewer or equal frames"
    )


# ---------------------------------------------------------------------------
# Gripper forced sampling
# ---------------------------------------------------------------------------

def test_gripper_change_frames_are_forced():
    """Frames around a gripper state change must appear in the output."""
    n = 50
    positions = _straight_line(n, length=1.0)
    gripper = np.zeros(n)
    gripper[20:] = 1.0          # state change at frame 19→20

    result = isr_resample_trajectory(
        positions,
        d_target=0.5,           # coarse — would normally skip frame 20
        gripper=gripper,
        gripper_expand_ranges=False,
    )
    # Both sides of the transition must be forced
    assert 19 in result or 20 in result, (
        "At least one frame adjacent to the gripper change must be selected"
    )


def test_gripper_expand_ranges_includes_gap():
    """With expand_ranges=True, frames in the gap between two gripper events are filled in."""
    n = 60
    positions = _straight_line(n, length=1.0)
    gripper = np.zeros(n)
    gripper[10] = 1.0           # brief spike at frame 10
    gripper[15] = 1.0           # another at frame 15 (gap = 4 < default max_gap=20)

    result = isr_resample_trajectory(
        positions,
        d_target=1.0,           # very coarse
        gripper=gripper,
        gripper_expand_ranges=True,
        gripper_expand_max_gap=20,
    )
    # The continuous range 10..15 should all be included
    for idx in range(10, 16):
        assert idx in result, f"Frame {idx} should be in expanded gripper range"


# ---------------------------------------------------------------------------
# Timestamps (times parameter)
# ---------------------------------------------------------------------------

def test_with_timestamps_does_not_crash():
    positions = _straight_line(30, length=1.0)
    times = np.linspace(0.0, 3.0, 30)
    result = isr_resample_trajectory(positions, d_target=0.1, times=times)
    assert len(result) >= 2


def test_result_indices_within_bounds():
    positions = _straight_line(20, length=1.0)
    result = isr_resample_trajectory(positions, d_target=0.1)
    assert all(0 <= idx < 20 for idx in result), "All indices must be valid frame indices"


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------

def test_wrong_ndim_raises():
    with pytest.raises(ValueError, match="positions must have shape"):
        isr_resample_trajectory(np.zeros((10,)))   # 1-D, not 2-D
