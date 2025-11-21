# pose_estimation.py (updated with better move type classification)

import cv2
import numpy as np
import mediapipe as mp
import os
import json
from typing import List, Dict, Tuple

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
POSE_LANDMARKS = mp_pose.PoseLandmark

LIMB_NAMES = {
    "lefthand": POSE_LANDMARKS.LEFT_WRIST,
    "righthand": POSE_LANDMARKS.RIGHT_WRIST,
    "leftfoot": POSE_LANDMARKS.LEFT_ANKLE,
    "rightfoot": POSE_LANDMARKS.RIGHT_ANKLE,
}

LIMB_COLORS = {
    "lefthand": (0, 0, 255),       # Red
    "righthand": (0, 255, 0),     # Green
    "leftfoot": (255, 0, 0),      # Blue
    "rightfoot": (0, 255, 255),   # Yellow
}

def assign_limbs_to_holds(limb_positions: Dict[str, Tuple[int, int]], hold_positions: Dict[str, Tuple[int, int]], threshold: int = 60) -> Dict[str, str]:
    assignments = {}
    for limb, pos in limb_positions.items():
        if pos is None:
            assignments[limb] = "hanging"
            continue
        min_dist = float('inf')
        assigned_hold = "hanging"
        for hold_id, hold_pos in hold_positions.items():
            dist = np.linalg.norm(np.array(pos) - np.array(hold_pos))
            if dist < min_dist:
                min_dist = dist
                assigned_hold = hold_id
        if min_dist <= threshold:
            assignments[limb] = assigned_hold
        else:
            assignments[limb] = "hanging"
    return assignments

def classify_move(prev_assignment, curr_assignment, limb_positions) -> str:
    def safe_y(limb):
        pos = limb_positions.get(limb)
        return pos[1] if pos else 9999 if 'foot' in limb else 0

    changed_limbs = [limb for limb in curr_assignment if curr_assignment[limb] != prev_assignment[limb]]
    if not changed_limbs:
        return None

    left_hand_y = safe_y("lefthand")
    right_hand_y = safe_y("righthand")
    left_foot_y = safe_y("leftfoot")
    right_foot_y = safe_y("rightfoot")

    # Heel Hook: foot much higher than hip level and bent posture
    if "leftfoot" in changed_limbs and left_foot_y < left_hand_y:
        return "heel_hook"
    if "rightfoot" in changed_limbs and right_foot_y < right_hand_y:
        return "heel_hook"

    # Flag: one foot far to the side of the other
    if abs(left_foot_y - right_foot_y) < 50:  # feet horizontally aligned
        left_foot_x = limb_positions.get("leftfoot", (0,))[0]
        right_foot_x = limb_positions.get("rightfoot", (0,))[0]
        if abs(left_foot_x - right_foot_x) > 150:
            return "flag"

    # Smear: foot is hanging (not on hold) but visible
    for limb in ["leftfoot", "rightfoot"]:
        if limb in changed_limbs and curr_assignment[limb] == "hanging" and limb_positions.get(limb):
            return "smear"

    return "reach"


def draw_limbs_on_frame(frame, limb_positions: Dict[str, Tuple[int, int]], assignments: Dict[str, str], move_idx: int, move_type: str):
    annotated = frame.copy()
    for limb, pos in limb_positions.items():
        if pos is None:
            continue
        color = LIMB_COLORS.get(limb, (255, 255, 255))
        cv2.circle(annotated, pos, 8, color, -1)
        cv2.putText(annotated, f"{limb}: {assignments[limb]}", (pos[0]+5, pos[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(annotated, f"Move {move_idx+1}: {move_type}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated

def detect_moves_and_visualize(video_path: str, hold_positions: Dict[str, Tuple[int, int]]) -> List[Dict[str, str]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
    os.makedirs("output/moves", exist_ok=True)

    all_moves = []
    prev_assignment = None
    move_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            height, width = frame.shape[:2]
            limb_positions = {}
            for limb, landmark_enum in LIMB_NAMES.items():
                lm = results.pose_landmarks.landmark[landmark_enum]
                if lm.visibility > 0.3:
                    x_px = int(lm.x * width)
                    y_px = int(lm.y * height)
                    limb_positions[limb] = (x_px, y_px)
                else:
                    limb_positions[limb] = None

            curr_assignment = assign_limbs_to_holds(limb_positions, hold_positions)

            if prev_assignment is None:
                move_type = "start"
                should_save = True
            else:
                move_type = classify_move(prev_assignment, curr_assignment, limb_positions)
                should_save = move_type is not None

            if should_save:
                move = {"type": move_type, **curr_assignment}
                all_moves.append(move)
                annotated = draw_limbs_on_frame(frame, limb_positions, curr_assignment, move_idx, move_type)
                cv2.imwrite(f"output/moves/move_{move_idx+1:02d}.jpg", annotated)
                move_idx += 1
                prev_assignment = curr_assignment

    cap.release()
    pose.close()

    return all_moves
