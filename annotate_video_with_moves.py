# annotate_video_with_moves.py
#
# Create an annotated video from climb_data.json moves.

import json
from pathlib import Path

import cv2


def annotate_video_with_moves(
    video_path: str,
    climb_data_path: str,
    output_path: str = "output/annotated_climb.mp4",
):
    video_path = Path(video_path)
    climb_data_path = Path(climb_data_path)
    output_path = Path(output_path)

    if not climb_data_path.exists():
        raise FileNotFoundError(f"climb_data.json not found: {climb_data_path}")

    with climb_data_path.open("r", encoding="utf8") as f:
        climb_data = json.load(f)

    moves = climb_data.get("moves", [])
    moves = sorted(moves, key=lambda m: m["frame_index"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = climb_data.get("fps", 30.0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Annotating video: {video_path}")
    print(f"Using climb data: {climb_data_path}")
    print(f"Saving annotated video to: {output_path}")

    frame_idx = 0
    move_idx = 0
    current_label = "none"
    current_conf = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        # Advance move pointer if needed
        while move_idx < len(moves) and frame_idx >= moves[move_idx]["frame_index"]:
            m = moves[move_idx]
            current_label = m["type"]
            current_conf = m.get("confidence", 0.0)
            move_idx += 1

        label_text = f"{current_label}"
        if current_label != "none":
            label_text = f"Move: {current_label} ({current_conf:.2f})"

        cv2.putText(
            frame,
            label_text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Frame {frame_idx}",
            (15, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

    cap.release()
    writer.release()

    print("Done writing annotated video")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create annotated climbing video from climb_data.json"
    )
    parser.add_argument("--video", "-v", required=True, help="Path to video file")
    parser.add_argument(
        "--data",
        "-d",
        default="output/climb_data.json",
        help="Path to climb_data.json",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="output/annotated_climb.mp4",
        help="Output annotated video path",
    )

    args = parser.parse_args()
    annotate_video_with_moves(args.video, args.data, args.out)
