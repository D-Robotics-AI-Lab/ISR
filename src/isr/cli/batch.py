"""
Batch trajectory resampling for ISR.

The input directory is expected to contain episode folders named
``episode_*``. Each episode should contain ``data.json`` and ``colors/`` and
optionally contain ``depths/`` and ``audios/``.
"""


from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


from isr import load_episode_arrays


def _copy_indexed_files(
    source_dir: Path,
    target_dir: Path,
    selected_indices: list[int],
    extensions: tuple[str, ...],
) -> None:
    if not source_dir.is_dir():
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    for new_idx, original_idx in enumerate(selected_indices):
        for ext in extensions:
            source_file = source_dir / f"{original_idx:06d}{ext}"
            if source_file.exists():
                shutil.copy2(source_file, target_dir / f"{new_idx:06d}{ext}")
                break


def process_episode(
    episode_dir: str | Path,
    output_episode_dir: str | Path,
    d_target: float,
    lambda_dist: float,
    lambda_acc: float,
) -> tuple[int, int] | None:
    """Resample one episode with ISR and copy selected frame assets."""

    episode_dir = Path(episode_dir)
    output_episode_dir = Path(output_episode_dir)
    data_path = episode_dir / "data.json"
    if not data_path.exists():
        print(f"  [SKIP] Missing {data_path}")
        return None
    if data_path.stat().st_size == 0:
        print(f"  [SKIP] {data_path} is empty")
        return None

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    positions, timestamps, gripper, frame_indices = load_episode_arrays(data)

    if len(positions) < 2:
        print("  [SKIP] Fewer than two frames")
        return None


    from isr import isr_resample_trajectory

    resampled_indices = isr_resample_trajectory(
        positions=positions,
        d_target=d_target,
        lambda_dist=lambda_dist,
        lambda_acc=lambda_acc,
        times=timestamps,
        gripper=gripper,
    )

    selected_data_indices = [frame_indices[idx] for idx in resampled_indices]
    import copy
    output_episode_dir.mkdir(parents=True, exist_ok=True)
    frames_by_idx = {int(frame["idx"]): frame for frame in data["data"]}

    output_data = {
        "info": data.get("info", {}),
        "text": data.get("text", ""),
        "data": [copy.deepcopy(frames_by_idx[idx]) for idx in selected_data_indices],
    }

    for new_idx, frame in enumerate(output_data["data"]):
        frame["idx"] = new_idx
        for modality in ("colors", "depths"):
            if modality not in frame or not isinstance(frame[modality], dict):
                continue
            for camera_name, camera_info in frame[modality].items():
                if isinstance(camera_info, dict) and "path" in camera_info:
                    ext = os.path.splitext(camera_info["path"])[1]
                    camera_info["path"] = f"{modality}/{camera_name}/{new_idx:06d}{ext}"

    with (output_episode_dir / "data.json").open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    colors_dir = episode_dir / "colors"
    if colors_dir.is_dir():
        for camera_dir in colors_dir.iterdir():
            if camera_dir.is_dir():
                _copy_indexed_files(
                    camera_dir,
                    output_episode_dir / "colors" / camera_dir.name,
                    selected_data_indices,
                    (".jpg", ".png"),
                )

    depths_dir = episode_dir / "depths"
    if depths_dir.is_dir():
        for camera_dir in depths_dir.iterdir():
            if camera_dir.is_dir():
                _copy_indexed_files(
                    camera_dir,
                    output_episode_dir / "depths" / camera_dir.name,
                    selected_data_indices,
                    (".png", ".npy", ".jpg"),
                )

    _copy_indexed_files(
        episode_dir / "audios",
        output_episode_dir / "audios",
        selected_data_indices,
        (".wav", ".mp3", ".flac"),
    )

    original_count = len(data.get("data", []))
    resampled_count = len(output_data["data"])
    print(
        f"  [OK] {episode_dir.name}: {original_count} -> {resampled_count} frames "
        f"({resampled_count / original_count * 100:.1f}%)"
    )
    return original_count, resampled_count


def main() -> None:
    import logging

    parser = argparse.ArgumentParser(description="Batch ISR trajectory resampling")
    parser.add_argument("--input_dir", required=True, help="Input root containing episode_* folders")
    parser.add_argument("--output_dir", required=True, help="Output root for resampled episodes")
    parser.add_argument("--d_target", type=float, default=0.05, help="Target spacing between sampled frames")
    parser.add_argument("--lambda_dist", type=float, default=1.0, help="Distance cost weight")
    parser.add_argument("--lambda_acc", type=float, default=0.01, help="Acceleration cost weight")
    parser.add_argument("--log_path", default=None, help="Optional JSON summary path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging of resampler details")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s"
    )

    episode_dirs = sorted(glob.glob(os.path.join(args.input_dir, "episode_*")))
    episode_dirs = [Path(path) for path in episode_dirs if Path(path).is_dir()]
    if not episode_dirs:
        print(f"No episode_* directories found under {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(episode_dirs)} episodes")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"d_target: {args.d_target}")
    print(f"lambda_dist: {args.lambda_dist}, lambda_acc: {args.lambda_acc}")
    print("=" * 60)

    results: list[tuple[str, int, int]] = []
    for episode_dir in episode_dirs:
        output_episode_dir = Path(args.output_dir) / episode_dir.name
        print(f"\nProcessing {episode_dir.name} ...")
        try:
            counts = process_episode(
                episode_dir,
                output_episode_dir,
                args.d_target,
                args.lambda_dist,
                args.lambda_acc,
            )
        except Exception as exc:  # keep batch jobs moving, but report the failed episode
            print(f"  [FAIL] {episode_dir.name}: {exc}")
            continue

        if counts is not None:
            results.append((episode_dir.name, counts[0], counts[1]))

    print("\n" + "=" * 60)
    if results:
        original_counts = np.array([r[1] for r in results], dtype=float)
        resampled_counts = np.array([r[2] for r in results], dtype=float)
        stats = {
            "method": "ISR",
            "params": {
                "d_target": args.d_target,
                "lambda_dist": args.lambda_dist,
                "lambda_acc": args.lambda_acc,
            },
            "episodes": [
                {"name": name, "original_frames": original, "resampled_frames": resampled}
                for name, original, resampled in results
            ],
            "original_frames_mean": float(np.mean(original_counts)),
            "original_frames_std": float(np.std(original_counts)),
            "resampled_frames_mean": float(np.mean(resampled_counts)),
            "resampled_frames_std": float(np.std(resampled_counts)),
            "compression_ratio_mean": float(np.mean(resampled_counts / original_counts)),
        }
        print(f"Original frames: mean={stats['original_frames_mean']:.1f}, std={stats['original_frames_std']:.1f}")
        print(f"Resampled frames: mean={stats['resampled_frames_mean']:.1f}, std={stats['resampled_frames_std']:.1f}")
        print(f"Mean compression ratio: {stats['compression_ratio_mean'] * 100:.1f}%")

        if args.log_path:
            log_path = Path(args.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4, ensure_ascii=False)
            print(f"Saved summary: {log_path}")

    print("Done.")


if __name__ == "__main__":
    main()
