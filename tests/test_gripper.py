"""Unit tests for gripper-change sampling utilities."""

from __future__ import annotations

import numpy as np
import pytest

from isr.gripper import expand_forced_sample_ranges, find_forced_gripper_sample_indices


# ---------------------------------------------------------------------------
# find_forced_gripper_sample_indices
# ---------------------------------------------------------------------------

def test_no_gripper_returns_empty():
    result = find_forced_gripper_sample_indices(
        num_frames=10, gripper=None
    )
    assert result == []


def test_constant_gripper_returns_empty():
    gripper = np.ones(10)
    result = find_forced_gripper_sample_indices(
        num_frames=10, gripper=gripper
    )
    assert result == []


def test_single_gripper_change_returns_both_neighbours():
    gripper = np.zeros(10)
    gripper[5:] = 1.0           # change between frame 4 and 5
    result = find_forced_gripper_sample_indices(
        num_frames=10, gripper=gripper
    )
    assert 4 in result and 5 in result


def test_multiple_gripper_changes():
    gripper = np.zeros(20)
    gripper[5:8] = 1.0          # changes at 4→5 and 7→8
    gripper[15:] = 1.0          # change at 14→15
    result = find_forced_gripper_sample_indices(
        num_frames=20, gripper=gripper
    )
    assert 4 in result and 5 in result
    assert 7 in result and 8 in result
    assert 14 in result and 15 in result


def test_gripper_length_mismatch_returns_empty():
    """If gripper length != num_frames, return empty (safe fallback)."""
    gripper = np.ones(5)
    result = find_forced_gripper_sample_indices(
        num_frames=10, gripper=gripper
    )
    assert result == []


def test_result_is_sorted_and_unique():
    gripper = np.zeros(30)
    gripper[10:12] = 1.0
    gripper[20:22] = 1.0
    result = find_forced_gripper_sample_indices(
        num_frames=30, gripper=gripper
    )
    assert result == sorted(set(result))



# ---------------------------------------------------------------------------
# expand_forced_sample_ranges
# ---------------------------------------------------------------------------

def test_empty_forced_indices_returns_empty():
    result = expand_forced_sample_ranges([], num_frames=20, max_gap=5)
    assert result == []


def test_single_index_returns_itself():
    result = expand_forced_sample_ranges([7], num_frames=20, max_gap=5)
    assert result == [7]


def test_close_indices_merged_into_range():
    # gap between 3 and 7 is 4, within max_gap=5 → should expand to 3..7
    result = expand_forced_sample_ranges([3, 7], num_frames=20, max_gap=5)
    assert result == list(range(3, 8))


def test_far_indices_kept_separate():
    # gap between 3 and 15 is 12, exceeds max_gap=5 → two separate singletons
    result = expand_forced_sample_ranges([3, 15], num_frames=20, max_gap=5)
    assert 3 in result and 15 in result
    # Should NOT include intermediate frames 4..14
    assert 9 not in result


def test_expanded_indices_within_bounds():
    result = expand_forced_sample_ranges([0, 4], num_frames=10, max_gap=10)
    assert all(0 <= idx < 10 for idx in result)


def test_out_of_bounds_indices_filtered():
    result = expand_forced_sample_ranges([8, 9], num_frames=10, max_gap=5)
    assert all(idx < 10 for idx in result)
