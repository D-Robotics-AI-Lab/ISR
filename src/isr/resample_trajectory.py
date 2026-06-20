"""ISR trajectory resampling via dynamic programming."""

from __future__ import annotations

import numpy as np

from .gripper import expand_forced_sample_ranges, find_forced_gripper_sample_indices
from .kinematics import compute_acceleration_magnitudes_from_positions


def isr_resample_trajectory(
    positions,
    d_target: float = 0.05,
    lambda_dist: float = 1.0,
    lambda_acc: float = 0.01,
    *,
    times: np.ndarray | None = None,
    gripper: np.ndarray | None = None,
    ignore_short_first_segment: bool = False,
    gripper_expand_ranges: bool = True,
    gripper_expand_max_gap: int = 20,
) -> list[int]:
    """Resample a position trajectory with ISR costs.

    Args:
        positions: Position trajectory, usually an array of shape [T, 3].
        d_target: Target spatial distance between neighboring sampled frames.
        lambda_dist: Weight for the spatial-distance term.
        lambda_acc: Weight for the accumulated acceleration term.
        times: Optional timestamps with shape [T].
        gripper: Optional scalar gripper sequence with shape [T].
        ignore_short_first_segment: Do not penalize the first segment if it is
            shorter than d_target.
        gripper_expand_ranges: Expand clustered gripper-change frames into
            continuous forced sampling ranges.
        gripper_expand_max_gap: Maximum gap for grouping gripper-change frames.

    Returns:
        Sorted resampled frame indices in the input trajectory frame space.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2:
        raise ValueError(f"positions must have shape [T, D], got {positions.shape}")

    num_frames = int(positions.shape[0])
    if num_frames == 0:
        return []
    if num_frames == 1:
        return [0]

    d_target = float(d_target)
    lambda_dist = float(lambda_dist)
    lambda_acc = float(lambda_acc)

    boundary_indices = [0, num_frames - 1]
    gripper_forced_indices = find_forced_gripper_sample_indices(
        positions=positions,
        num_frames=num_frames,
        gripper=gripper,
    )

    if gripper_forced_indices and gripper_expand_ranges:
        gripper_forced_indices = expand_forced_sample_ranges(
            gripper_forced_indices,
            num_frames=num_frames,
            max_gap=gripper_expand_max_gap,
        )
        print(f"Expanded gripper-forced sample indices: {gripper_forced_indices}")

    if lambda_acc > 0.0:
        acc_mag = compute_acceleration_magnitudes_from_positions(positions, times=times)
        acc_abs_prefix = np.zeros((num_frames + 1,), dtype=float)
        acc_abs_prefix[1:] = np.cumsum(acc_mag)
    else:
        acc_abs_prefix = None

    def _segment_cost(k: int, i: int) -> float:
        dist = float(np.linalg.norm(positions[i] - positions[k]))
        if ignore_short_first_segment and k == 0 and dist <= d_target:
            return 0.0
        if lambda_acc > 0.0 and acc_abs_prefix is not None:
            info_acc = float(acc_abs_prefix[i] - acc_abs_prefix[k])
            return (lambda_dist * dist + lambda_acc * info_acc - d_target) ** 2
        return (lambda_dist * dist - d_target) ** 2

    costs = np.full((num_frames,), float("inf"), dtype=float)
    prev = np.full((num_frames,), -1, dtype=int)
    costs[0] = 0.0

    for i in range(1, num_frames):
        best_cost = float("inf")
        best_prev = -1
        for k in range(0, i):
            if costs[k] == float("inf"):
                continue
            total_cost = costs[k] + _segment_cost(k, i)
            if total_cost < best_cost:
                best_cost = total_cost
                best_prev = k
        costs[i] = best_cost
        prev[i] = best_prev

    resampled_indices = _backtrack_resampled_indices(prev, end_idx=num_frames - 1)
    resampled_indices = sorted(set(resampled_indices + boundary_indices + gripper_forced_indices))

    print(f"Resampled frame indices: {resampled_indices}")
    return resampled_indices


def _backtrack_resampled_indices(prev: np.ndarray, *, end_idx: int) -> list[int]:
    resampled_indices: list[int] = []
    idx = int(end_idx)
    while idx >= 0:
        resampled_indices.append(idx)
        if idx == 0:
            break
        idx = int(prev[idx])
        if idx < 0:
            return [0, int(end_idx)]
    resampled_indices.reverse()
    return resampled_indices
