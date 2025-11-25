# pose_estimation.py (ML based move detection with stability and movement thresholds)

import cv2
import numpy as np
import mediapipe as mp
import os
import json
from pathlib import Path
import shutil
from typing import List, Dict, Tuple
from move_classifier.model_inference import PoseMoveClassifier


mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
POSE_LANDMARKS = mp_pose.PoseLandmark

# Load your trained classifier once
move_classifier = PoseMoveClassifier()

# Limb definitions
LIMB_NAMES = {
    "lefthand": POSE_LANDMARKS.LEFT_WRIST,
    "righthand": POSE_LANDMARKS.RIGHT_WRIST,
    "leftfoot": POSE_LANDMARKS.LEFT_ANKLE,
    "rightfoot": POSE_LANDMARKS.RIGHT_ANKLE,
}

LIMB_COLORS = {
    "lefthand": (0, 0, 255),
    "righthand": (0, 255, 0),
    "leftfoot": (255, 0, 0),
    "rightfoot": (0, 255, 255),
}


def assign_limbs_to_holds(
    limb_positions: Dict[str, Tuple[int, int]],
    hold_positions: Dict[str, Tuple[int, int]],
    threshold: int = 60,
) -> Dict[str, str]:
    """Assign each limb to the nearest hold within a pixel threshold, else 'hanging'."""
    assignments = {}
    for limb, pos in limb_positions.items():
        if pos is None:
            assignments[limb] = "hanging"
            continue

        min_dist = float("inf")
        assigned_hold = "hanging"

        for hold_id, hold_pos in hold_positions.items():
            dist = np.linalg.norm(np.array(pos) - np.array(hold_pos))
            if dist < min_dist:
                min_dist = dist
                assigned_hold = hold_id

        assignments[limb] = assigned_hold if min_dist <= threshold else "hanging"

    return assignments


def draw_limbs_on_frame(
    frame,
    limb_positions: Dict[str, Tuple[int, int]],
    assignments: Dict[str, str],
    move_idx: int,
    move_type: str,
    confidence: float | None,
):
    """Draw limb markers, assignments, and move label on a frame."""
    annotated = frame.copy()

    for limb, pos in limb_positions.items():
        if pos is None:
            continue
        color = LIMB_COLORS.get(limb, (255, 255, 255))
        cv2.circle(annotated, pos, 8, color, -1)
        cv2.putText(
            annotated,
            f"{limb}: {assignments.get(limb, 'hanging')}",
            (pos[0] + 5, pos[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    label_text = f"Move {move_idx+1}: {move_type}"
    if confidence is not None:
        label_text += f" ({confidence:.2f})"

    cv2.putText(
        annotated,
        label_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def _compute_movement_pixels(
    last_positions: Dict[str, Tuple[int, int]] | None,
    current_positions: Dict[str, Tuple[int, int]],
) -> float:
    """Estimate how much the body moved since the last saved move, in pixels."""
    if last_positions is None:
        return 9999.0  # treat first move as big movement

    dists = []
    for limb, curr in current_positions.items():
        prev = last_positions.get(limb)
        if curr is None or prev is None:
            continue
        d = np.linalg.norm(np.array(curr) - np.array(prev))
        dists.append(d)

    if not dists:
        return 0.0

    # Use max limb displacement as the movement measure
    return float(max(dists))


def detect_moves_and_visualize(
    video_path: str,
    hold_positions: Dict[str, Tuple[int, int]],
) -> List[Dict[str, str]]:
    """
    Run pose detection and move classification over a video.
    Returns a list of move dicts that get saved to climb_data.json.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Clean output/moves before running
    moves_dir = Path("output/moves")
    if moves_dir.exists():
        for item in moves_dir.iterdir():
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)
    else:
        moves_dir.mkdir(parents=True, exist_ok=True)

    all_moves: List[Dict[str, str]] = []

    # State for move detection
    frame_idx = 0
    last_saved_frame = -999
    last_saved_positions: Dict[str, Tuple[int, int]] | None = None

    # For stability: require same label several frames in a row
    stable_label: str | None = None
    stable_count: int = 0

    # Tuning knobs
    MIN_CONFIDENCE = 0.5           # how sure the model must be to trust a label
    STABLE_FRAMES = 4              # require label to be seen this many frames in a row
    COOLDOWN_FRAMES = 12           # minimum frames between saved moves
    MOVE_PIXEL_THRESHOLD = 50.0    # required max limb displacement between moves

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            continue

        height, width = frame.shape[:2]

        # Get limb pixel positions
        limb_positions: Dict[str, Tuple[int, int]] = {}
        for limb, landmark_enum in LIMB_NAMES.items():
            lm = results.pose_landmarks.landmark[landmark_enum]
            if lm.visibility > 0.3:
                x_px = int(lm.x * width)
                y_px = int(lm.y * height)
                limb_positions[limb] = (x_px, y_px)
            else:
                limb_positions[limb] = None

        curr_assignment = assign_limbs_to_holds(limb_positions, hold_positions)

        # Model prediction
        model_label, probs = move_classifier.predict_from_landmarks(
            results.pose_landmarks.landmark
        )
        top_conf = float(np.max(probs))

        # Only consider labels above confidence threshold
        if top_conf >= MIN_CONFIDENCE:
            if model_label == stable_label:
                stable_count += 1
            else:
                stable_label = model_label
                stable_count = 1
        else:
            # Low confidence breaks stability
            stable_label = None
            stable_count = 0

        # Decide whether to save this as a move
        should_save = False

        if (
            stable_label is not None
            and stable_count >= STABLE_FRAMES
            and (frame_idx - last_saved_frame) >= COOLDOWN_FRAMES
        ):
            # Check that the climber actually moved enough since the last saved move
            movement = _compute_movement_pixels(last_saved_positions, limb_positions)
            if movement >= MOVE_PIXEL_THRESHOLD or last_saved_frame < 0:
                should_save = True

        if should_save:
            move = {
                "frame_index": frame_idx,
                "type": stable_label,
                "confidence": top_conf,
                "assignments": curr_assignment,
                "move_probs": probs.tolist(),
            }

            all_moves.append(move)

            move_idx = len(all_moves) - 1
            annotated = draw_limbs_on_frame(
                frame,
                limb_positions,
                curr_assignment,
                move_idx,
                stable_label,
                top_conf,
            )
            cv2.imwrite(f"output/moves/move_{move_idx+1:02d}.jpg", annotated)

            last_saved_frame = frame_idx
            last_saved_positions = dict(limb_positions)

    cap.release()
    pose.close()

    return all_moves
