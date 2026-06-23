"""Gripper-change sampling utilities for ISR resampling."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def find_forced_gripper_sample_indices(
    *,
    num_frames: int,
    gripper: np.ndarray | None,
    threshold: float = 0.05,
) -> list[int]:
    """Find frames that should be kept around gripper state changes."""
    gripper_forced_indices: list[int] = []

    if gripper is not None:
        gripper = np.asarray(gripper, dtype=float)
        if gripper.shape[0] == num_frames:
            for i in range(num_frames - 1):
                if abs(float(gripper[i + 1]) - float(gripper[i])) > threshold:
                    gripper_forced_indices.append(i)
                    gripper_forced_indices.append(i + 1)
            gripper_forced_indices = sorted(set(gripper_forced_indices))
            if gripper_forced_indices:
                logger.info(f"Gripper-forced sample indices (|delta| > {threshold}): {gripper_forced_indices}")
    return gripper_forced_indices


def expand_forced_sample_ranges(
    forced_indices: list[int],
    *,
    num_frames: int,
    max_gap: int,
) -> list[int]:
    """Expand nearby forced sample indices into continuous ranges."""
    forced_indices = sorted(set(forced_indices))
    if not forced_indices:
        return []

    max_gap = max(1, int(max_gap))
    expanded: list[int] = []
    cluster_start = forced_indices[0]
    cluster_end = forced_indices[0]

    for idx in forced_indices[1:]:
        if idx - cluster_end <= max_gap:
            cluster_end = idx
        else:
            expanded.extend(range(cluster_start, cluster_end + 1))
            cluster_start = cluster_end = idx
    expanded.extend(range(cluster_start, cluster_end + 1))
    return sorted(set(i for i in expanded if 0 <= i < num_frames))
