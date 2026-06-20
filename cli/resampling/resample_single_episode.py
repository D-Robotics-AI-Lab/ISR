"""Resample one episode data.json file with ISR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def load_episode_arrays(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
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


def save_selected_json(data: dict, selected_frame_indices: list[int], output_path: Path) -> None:
    frames_by_idx = {int(frame["idx"]): frame for frame in data["data"]}
    output_data = {
        "info": data.get("info", {}),
        "text": data.get("text", ""),
        "data": [frames_by_idx[idx] for idx in selected_frame_indices],
    }
    for new_idx, frame in enumerate(output_data["data"]):
        frame["idx"] = new_idx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample one episode with ISR")
    parser.add_argument("--data", required=True, help="Path to episode data.json")
    parser.add_argument("--output_json", default=None, help="Optional output data.json with resampled frames")
    parser.add_argument("--d_target", type=float, default=0.05, help="Target spacing between sampled frames")
    parser.add_argument("--lambda_dist", type=float, default=1.0, help="Distance cost weight")
    parser.add_argument("--lambda_acc", type=float, default=0.01, help="Acceleration cost weight")
    args = parser.parse_args()

    data_path = Path(args.data)
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    from isr import isr_resample_trajectory

    positions, timestamps, gripper, frame_indices = load_episode_arrays(data)
    if len(positions) < 2:
        raise ValueError(f"Expected at least two frames in {data_path}")

    print("=" * 60)
    print(f"ISR resampling: {data_path}")

    resampled_indices = isr_resample_trajectory(
        positions=positions,
        d_target=args.d_target,
        lambda_dist=args.lambda_dist,
        lambda_acc=args.lambda_acc,
        times=timestamps,
        gripper=gripper,
    )
    selected_frame_indices = [frame_indices[idx] for idx in resampled_indices]

    print("=" * 60)
    print(f"Total frames in file: {len(data.get('data', []))}")
    print(f"ISR resampled frame count: {len(resampled_indices)}")
    print(f"Compression: {len(resampled_indices) / len(positions) * 100:.2f}%")

    if args.output_json:
        output_path = Path(args.output_json)
        save_selected_json(data, selected_frame_indices, output_path)
        print(f"Saved selected data: {output_path}")


if __name__ == "__main__":
    main()
