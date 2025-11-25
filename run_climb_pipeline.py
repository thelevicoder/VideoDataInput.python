# run_climb_pipeline.py
#
# One command to:
# 1. Detect holds
# 2. Detect and classify moves
# 3. Create annotated video

from pathlib import Path
import argparse

from hold_detection import build_holds_json_from_video
from move_detector import detect_and_classify_moves
from annotate_video_with_moves import annotate_video_with_moves


def main():
    parser = argparse.ArgumentParser(
        description="Full climbing video pipeline: holds, moves, annotation"
    )
    parser.add_argument(
        "--video",
        "-v",
        required=True,
        help="Path to climbing video file",
    )
    parser.add_argument(
        "--holds",
        "-H",
        default=None,
        help="Optional existing holds JSON. If omitted, holds are auto detected.",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default="output",
        help="Output directory for data and annotated video",
    )

    args = parser.parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.out_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get holds JSON
    if args.holds:
        holds_json_path = Path(args.holds)
        if not holds_json_path.exists():
            raise FileNotFoundError(f"Holds JSON not found: {holds_json_path}")
        print(f"Using existing holds file: {holds_json_path}")
    else:
        print("No holds JSON provided, auto detecting holds")
        holds_json_path_str = build_holds_json_from_video(
            str(video_path),
            output_json=str(out_dir / "hold_positions_auto.json"),
            debug_image_out=str(out_dir / "holds_debug.jpg"),
        )
        holds_json_path = Path(holds_json_path_str)

    # Step 2 and 3: detect and classify moves
    climb_data_path_str = detect_and_classify_moves(
        str(video_path),
        str(holds_json_path),
        output_dir=str(out_dir),
    )

    # Step 4: annotate video
    annotated_video_path = out_dir / "annotated_climb.mp4"
    annotate_video_with_moves(
        str(video_path),
        climb_data_path_str,
        str(annotated_video_path),
    )


if __name__ == "__main__":
    main()
