"""
Visualize original and ISR-resampled trajectories from episode data.json files.

Example::

    isr-visualize \\
        --original /path/to/raw/episode/data.json \\
        --resampled /path/to/isr/episode/data.json \\
        --save outputs/trajectory_compare.png
"""


from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "isr_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "isr_cache"))

signal.signal(signal.SIGINT, lambda *_: sys.exit(0))


def load_ee_positions(data_path: str | Path) -> np.ndarray:
    """Load left end-effector xyz positions from an episode data.json file."""
    with Path(data_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    positions = []
    for frame in data.get("data", []):
        states = frame.get("states", {})
        if "left_ee" in states:
            qpos = states["left_ee"].get("qpos", [])
            if len(qpos) >= 3:
                positions.append(qpos[:3])

    if not positions:
        raise ValueError(f"No left_ee xyz positions found in {data_path}")
    return np.array(positions, dtype=float)


def set_equal_axes(ax, points: np.ndarray) -> None:
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
    if max_range <= 0:
        max_range = 1.0
    mid = (points.max(axis=0) + points.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ISR trajectory resampling")
    parser.add_argument("--original", required=True, help="Original episode data.json")
    parser.add_argument("--resampled", default=None, help="Resampled episode data.json")
    parser.add_argument("--title", default="ISR Trajectory Visualization", help="Figure title")
    parser.add_argument("--save", default=None, help="Output image path. If omitted, show an interactive window.")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    original_positions = load_ee_positions(args.original)
    print(f"Original trajectory: {len(original_positions)} frames")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False

    ax.scatter(
        original_positions[:, 0],
        original_positions[:, 1],
        original_positions[:, 2],
        color="#4169E1",
        s=24,
        alpha=0.65,
        label=f"Original ({len(original_positions)} pts)",
    )

    all_points = original_positions
    if args.resampled:
        resampled_positions = load_ee_positions(args.resampled)
        print(f"Resampled trajectory: {len(resampled_positions)} frames")
        all_points = np.vstack([original_positions, resampled_positions])

        ax.plot(
            resampled_positions[:, 0],
            resampled_positions[:, 1],
            resampled_positions[:, 2],
            color="#D62728",
            linewidth=1.6,
            alpha=0.85,
        )
        ax.scatter(
            resampled_positions[:, 0],
            resampled_positions[:, 1],
            resampled_positions[:, 2],
            color="#D62728",
            s=54,
            alpha=0.95,
            zorder=20,
            edgecolors="none",
            label=f"ISR resampled ({len(resampled_positions)} pts)",
        )
        start = resampled_positions[0]
        end = resampled_positions[-1]
    else:
        start = original_positions[0]
        end = original_positions[-1]

    text_style = {"fontsize": 10, "fontweight": "bold", "color": "black"}
    ax.text(start[0], start[1], start[2], "  Start", **text_style, zorder=20)
    ax.text(end[0], end[1], end[2], "  End", **text_style, zorder=20)

    ax.set_xlabel("X", fontsize=13)
    ax.set_ylabel("Y", fontsize=13)
    ax.set_zlabel("Z", fontsize=13)
    ax.xaxis.set_major_locator(MaxNLocator(8))
    ax.yaxis.set_major_locator(MaxNLocator(8))
    ax.zaxis.set_major_locator(MaxNLocator(8))
    ax.set_title(args.title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=12, markerscale=2.0)
    set_equal_axes(ax, all_points)
    ax.view_init(elev=30, azim=-145)

    fig.canvas.mpl_connect("key_press_event", lambda event: plt.close("all") if event.key == "q" else None)
    plt.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    else:
        print("Press q to close the window.")
        plt.show()


if __name__ == "__main__":
    main()
