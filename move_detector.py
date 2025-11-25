# move_detector.py
#
# Detects moves from a climbing video, saves snapshots, and classifies each move.

import cv2
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import mediapipe as mp

from move_classifier.model_inference import PoseMoveClassifier


mp_pose = mp.solutions.pose
POSE_LANDMARKS = mp_pose.PoseLandmark

LIMB_LANDMARKS = {
    "lefthand": POSE_LANDMARKS.LEFT_WRIST,
    "righthand": POSE_LANDMARKS.RIGHT_WRIST,
    "leftfoot": POSE_LANDMARKS.LEFT_ANKLE,
    "rightfoot": POSE_LANDMARKS.RIGHT_ANKLE,
}


def _compute_limb_positions(results, width: int, height: int) -> Dict[str, Tuple[int, int] | None]:
    limb_positions: Dict[str, Tuple[int, int] | None] = {}

    for limb, lm_enum in LIMB_LANDMARKS.items():
        lm = results.pose_landmarks.landmark[lm_enum]
        if lm.visibility > 0.3:
            x_px = int(lm.x * width)
            y_px = int(lm.y * height)
            limb_positions[limb] = (x_px, y_px)
        else:
            limb_positions[limb] = None

    return limb_positions


def _movement_since(last_positions, current_positions) -> float:
    if last_positions is None:
        return 9999.0

    dists = []
    for limb, curr in current_positions.items():
        prev = last_positions.get(limb) if last_positions else None
        if curr is None or prev is None:
            continue
        dists.append(np.linalg.norm(np.array(curr) - np.array(prev)))

    if not dists:
        return 0.0
    return float(max(dists))


def _assign_limbs_to_holds(
    limb_positions: Dict[str, Tuple[int, int] | None],
    holds: Dict[str, List[int]],
    threshold: int = 60,
) -> Dict[str, str]:
    assignments: Dict[str, str] = {}

    for limb, pos in limb_positions.items():
        if pos is None:
            assignments[limb] = "hanging"
            continue

        best_hold = "hanging"
        best_dist = float("inf")
        for hold_id, coords in holds.items():
            hx, hy = coords
            dist = np.linalg.norm(np.array(pos) - np.array([hx, hy]))
            if dist < best_dist:
                best_dist = dist
                best_hold = hold_id

        assignments[limb] = best_hold if best_dist <= threshold else "hanging"

    return assignments


def detect_and_classify_moves(
    video_path: str,
    holds_json_path: str,
    output_dir: str = "output",
) -> str:
    """
    Uses MediaPipe pose and your classifier to detect moves and classify each one.
    Saves snapshots to output/moves and climb_data.json to output/.
    Returns the path to climb_data.json.
    """
    video_path = Path(video_path)
    holds_json_path = Path(holds_json_path)
    output_root = Path(output_dir)
    moves_dir = output_root / "moves"

    if not holds_json_path.exists():
        raise FileNotFoundError(f"Holds JSON not found: {holds_json_path}")

    with holds_json_path.open("r", encoding="utf8") as f:
        holds = json.load(f)

    # Clean moves folder
    if moves_dir.exists():
        for item in moves_dir.iterdir():
            if item.is_file():
                item.unlink()
            else:
                import shutil
                shutil.rmtree(item)
    else:
        moves_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    classifier = PoseMoveClassifier()

    frame_idx = 0
    last_saved_frame = -999
    last_saved_positions = None

    MIN_CONFIDENCE = 0.5
    MOVE_PIXEL_THRESHOLD = 55.0
    COOLDOWN_FRAMES = 10

    moves: List[Dict] = []

    print(f"Processing video for moves: {video_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        height, width = frame.shape[:2]

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            continue

        limb_positions = _compute_limb_positions(results, width, height)
        movement = _movement_since(last_saved_positions, limb_positions)

        # Only consider big enough movements and cooldown
        if movement < MOVE_PIXEL_THRESHOLD:
            continue
        if frame_idx - last_saved_frame < COOLDOWN_FRAMES:
            continue

        # Classify this pose
        label, probs = classifier.predict_from_landmarks(
            results.pose_landmarks.landmark
        )
        top_conf = float(np.max(probs))
        if top_conf < MIN_CONFIDENCE:
            continue

        assignments = _assign_limbs_to_holds(limb_positions, holds)

        move_idx = len(moves)
        snapshot_path = moves_dir / f"move_{move_idx+1:02d}.jpg"
        cv2.imwrite(str(snapshot_path), frame)

        move_data = {
            "move_index": move_idx,
            "frame_index": frame_idx,
            "time_seconds": frame_idx / fps,
            "type": label,
            "confidence": top_conf,
            "assignments": assignments,
            "snapshot_path": str(snapshot_path),
            "probs": probs.tolist(),
        }
        moves.append(move_data)
        last_saved_frame = frame_idx
        last_saved_positions = dict(limb_positions)

        print(
            f"Saved move {move_idx+1:02d} at frame {frame_idx}, "
            f"type={label}, conf={top_conf:.2f}, movement={movement:.1f}"
        )

    cap.release()
    pose.close()

    climb_data = {
        "video_path": str(video_path),
        "holds_json": str(holds_json_path),
        "fps": fps,
        "moves": moves,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    climb_data_path = output_root / "climb_data.json"
    with climb_data_path.open("w", encoding="utf8") as f:
        json.dump(climb_data, f, indent=2)

    print(f"\nDetected {len(moves)} moves")
    print(f"Saved climb_data.json to: {climb_data_path}")

    return str(climb_data_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect and classify moves from a climbing video"
    )
    parser.add_argument("--video", "-v", required=True, help="Path to video file")
    parser.add_argument(
        "--holds",
        "-H",
        required=True,
        help="Path to holds JSON file",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default="output",
        help="Output directory (default: output)",
    )

    args = parser.parse_args()
    detect_and_classify_moves(args.video, args.holds, output_dir=args.out_dir)
