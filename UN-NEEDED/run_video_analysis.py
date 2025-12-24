# run_video_analysis.py

import argparse
import json
from pathlib import Path

from pose_estimation import detect_moves_and_visualize


def load_hold_positions(holds_path: Path):
    """
    Optional: load hold positions from a JSON file.
    Expected format:
    {
      "H1": [x, y],
      "H2": [x, y],
      ...
    }
    If the file does not exist or is not provided, returns {}.
    """
    if holds_path is None:
        return {}

    if not holds_path.exists():
        print(f"Warning: holds file {holds_path} not found. Continuing with no holds.")
        return {}

    with open(holds_path, "r") as f:
        data = json.load(f)

    hold_positions = {}
    for hold_id, coords in data.items():
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            hold_positions[hold_id] = (int(coords[0]), int(coords[1]))
    return hold_positions


def main():
    parser = argparse.ArgumentParser(description="Run climbing move analysis on a video.")
    parser.add_argument(
        "--video",
        "-v",
        required=True,
        help="Path to the input climbing video (mp4, mov, etc).",
    )
    parser.add_argument(
    "--holds",
    "-H",
    default=None,
    help="Optional path to a JSON file with hold positions.",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    holds_path = Path(args.holds) if args.holds else None
    hold_positions = load_hold_positions(holds_path)

    print(f"Using video: {video_path}")
    print(f"Hold positions loaded: {len(hold_positions)}")

    moves = detect_moves_and_visualize(str(video_path), hold_positions)

    climb_data = {
        "video_path": str(video_path),
        "moves": moves,
    }

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    climb_json_path = output_dir / "climb_data.json"

    with open(climb_json_path, "w") as f:
        json.dump(climb_data, f, indent=2)

    print(f"\nSaved climb_data.json to {climb_json_path}")
    print("Now you can run: python analyze_moves.py")


if __name__ == "__main__":
    main()
