# annotate_video_with_moves.py
#
# Create a new video with the model's predicted move type
# overlaid on each frame.

import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from move_classifier.model_inference import PoseMoveClassifier


def annotate_video(
    video_path: Path,
    output_path: Path,
    min_confidence: float = 0.4,
):
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    classifier = PoseMoveClassifier()

    frame_idx = 0

    print(f"Annotating video: {video_path}")
    print(f"Saving to: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        label_text = "no_pose"
        conf_text = ""
        color = (0, 0, 255)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            move_label, probs = classifier.predict_from_landmarks(landmarks)
            top_conf = float(np.max(probs))

            if top_conf >= min_confidence:
                label_text = move_label
                conf_text = f"{top_conf:.2f}"
                color = (0, 255, 0)
            else:
                label_text = "low_conf"
                conf_text = f"{top_conf:.2f}"
                color = (0, 255, 255)

        # Draw label and confidence
        text = f"{label_text}"
        if conf_text:
            text += f" ({conf_text})"

        cv2.putText(
            frame,
            text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )

        # Optional frame index
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

        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames")

    cap.release()
    writer.release()
    pose.close()

    print(f"Done. Annotated video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate a climbing video with model predicted move types."
    )
    parser.add_argument(
        "--video",
        "-v",
        required=True,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="output/annotated_climb.mp4",
        help="Path for output annotated video file.",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.4,
        help="Minimum confidence to trust a prediction.",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.out)

    annotate_video(video_path, output_path, min_confidence=args.min_conf)


if __name__ == "__main__":
    main()
