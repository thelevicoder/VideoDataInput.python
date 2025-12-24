# move_detector_simple.py
#
# SIMPLE, TUNABLE move detector
# Easy to adjust sensitivity with clear parameters at the top

import cv2
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import mediapipe as mp
from move_classifier.model_inference import PoseMoveClassifier

mp_pose = mp.solutions.pose
POSE_LANDMARKS = mp_pose.PoseLandmark

# ==============================================================================
# TUNING PARAMETERS - ADJUST THESE TO FIX SENSITIVITY
# ==============================================================================

# How close must limb be to hold to count as "on hold" (pixels)
# Higher = stricter (limb must be closer)
# Lower = more lenient (limb can be farther)
HOLD_ASSIGNMENT_DISTANCE = 70

# How long must limb stay on same hold to be "stable" (frames)
# Higher = fewer moves detected (requires longer stability)
# Lower = more moves detected (shorter stability OK)
STABILITY_FRAMES = 6

# Minimum distance between holds to count as a move (pixels)  
# Higher = fewer moves (only big movements)
# Lower = more moves (small movements count)
MIN_MOVE_DISTANCE = 50.0

# Minimum time between consecutive moves for same limb (frames)
# Higher = fewer moves (prevents rapid succession)
# Lower = more moves (allows quick moves)
COOLDOWN_FRAMES = 15

# ==============================================================================

LIMB_LANDMARKS = {
    "lefthand": POSE_LANDMARKS.LEFT_WRIST,
    "righthand": POSE_LANDMARKS.RIGHT_WRIST,
    "leftfoot": POSE_LANDMARKS.LEFT_ANKLE,
    "rightfoot": POSE_LANDMARKS.RIGHT_ANKLE,
}

LIMBS = ["lefthand", "righthand", "leftfoot", "rightfoot"]


def _compute_limb_positions(results, width: int, height: int) -> Dict[str, Tuple[int, int] | None]:
    """Extract limb positions from MediaPipe pose."""
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


def _assign_limbs_to_holds(
    limb_positions: Dict[str, Tuple[int, int] | None],
    holds: Dict[str, List[int]],
) -> Dict[str, str]:
    """Assign each limb to nearest hold within threshold."""
    assignments: Dict[str, str] = {}
    
    for limb, pos in limb_positions.items():
        if pos is None:
            assignments[limb] = "no_pose"
            continue
        
        best_hold = "air"
        best_dist = float("inf")
        
        for hold_id, coords in holds.items():
            hx, hy = coords
            dist = np.linalg.norm(np.array(pos) - np.array([hx, hy]))
            if dist < best_dist:
                best_dist = dist
                best_hold = hold_id
        
        if best_dist <= HOLD_ASSIGNMENT_DISTANCE:
            assignments[limb] = best_hold
        else:
            assignments[limb] = "air"
    
    return assignments


def _detect_moves_simple(
    assignments_series: List[Dict[str, str]],
    holds: Dict[str, List[int]],
) -> List[Dict]:
    """
    Simple move detection:
    - Track each limb independently
    - Require stability on old hold
    - Require stability on new hold  
    - Require sufficient distance
    - Enforce cooldown between moves
    """
    moves = []
    
    for limb in LIMBS:
        # Track current stable hold for this limb
        current_hold = None
        stable_count = 0
        last_move_frame = -999
        
        for frame_idx, assignments in enumerate(assignments_series):
            limb_hold = assignments.get(limb, "no_pose")
            
            # Skip non-holds
            if limb_hold in ("air", "no_pose"):
                current_hold = None
                stable_count = 0
                continue
            
            # Same hold - increment stability
            if limb_hold == current_hold:
                stable_count += 1
                continue
            
            # Different hold - check if we should record a move
            if current_hold is not None and current_hold != limb_hold:
                # Check if old hold was stable
                if stable_count >= STABILITY_FRAMES:
                    # Check distance between holds
                    old_coords = holds.get(current_hold)
                    new_coords = holds.get(limb_hold)
                    
                    if old_coords and new_coords:
                        dist = np.linalg.norm(
                            np.array(old_coords) - np.array(new_coords)
                        )
                        
                        # Check if distance is significant
                        if dist >= MIN_MOVE_DISTANCE:
                            # Check cooldown
                            if (frame_idx - last_move_frame) >= COOLDOWN_FRAMES:
                                moves.append({
                                    "frame_index": frame_idx + 1,  # 1-based
                                    "limb": limb,
                                    "from_hold": current_hold,
                                    "to_hold": limb_hold,
                                    "hold_distance_px": float(dist),
                                })
                                last_move_frame = frame_idx
            
            # Start tracking new hold
            current_hold = limb_hold
            stable_count = 1
    
    # Sort by frame
    moves.sort(key=lambda m: m["frame_index"])
    return moves


def detect_and_classify_moves(
    video_path: str,
    holds_json_path: str,
    output_dir: str = "output",
) -> str:
    """
    Simple move detection and classification.
    """
    video_path = Path(video_path)
    holds_json_path = Path(holds_json_path)
    output_root = Path(output_dir)
    moves_dir = output_root / "moves"
    
    if not holds_json_path.exists():
        raise FileNotFoundError(f"Holds JSON not found: {holds_json_path}")
    
    with holds_json_path.open("r", encoding="utf8") as f:
        holds = json.load(f)
    
    print(f"[move_detection] Loaded {len(holds)} holds")
    
    # Clean moves folder
    if moves_dir.exists():
        import shutil
        for item in moves_dir.iterdir():
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)
    moves_dir.mkdir(parents=True, exist_ok=True)
    
    # First pass - collect assignments
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[move_detection] Video: {width}x{height} @ {fps:.1f}fps")
    print(f"[move_detection] Settings:")
    print(f"  - Hold assignment distance: {HOLD_ASSIGNMENT_DISTANCE}px")
    print(f"  - Stability frames: {STABILITY_FRAMES}")
    print(f"  - Min move distance: {MIN_MOVE_DISTANCE}px")
    print(f"  - Cooldown frames: {COOLDOWN_FRAMES}")
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    assignments_series: List[Dict[str, str]] = []
    
    print("[move_detection] Pass 1: Analyzing video...")
    frame_count = 0
    
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            assignments = {limb: "no_pose" for limb in LIMBS}
        else:
            positions = _compute_limb_positions(results, w, h)
            assignments = _assign_limbs_to_holds(positions, holds)
        
        assignments_series.append(assignments)
    
    cap.release()
    pose.close()
    
    print(f"[move_detection] Processed {frame_count} frames")
    
    # Detect moves
    move_events = _detect_moves_simple(assignments_series, holds)
    
    print(f"[move_detection] Detected {len(move_events)} moves")
    
    # Second pass - classify and save
    cap = cv2.VideoCapture(str(video_path))
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    classifier = PoseMoveClassifier()
    
    moves: List[Dict] = []
    
    print("[move_detection] Pass 2: Classifying moves...")
    
    for move_idx, ev in enumerate(move_events):
        target_frame = ev["frame_index"]
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
        ok, frame = cap.read()
        if not ok:
            continue
        
        h, w = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            label = "no_pose"
            probs = np.zeros((1,), dtype=float)
            conf = 0.0
        else:
            label, probs = classifier.predict_from_landmarks(
                results.pose_landmarks.landmark
            )
            conf = float(np.max(probs))
        
        snapshot = moves_dir / f"move_{move_idx+1:02d}.jpg"
        cv2.imwrite(str(snapshot), frame)
        
        move_data = {
            "move_index": move_idx,
            "frame_index": int(target_frame),
            "time_seconds": target_frame / fps,
            "limb": ev["limb"],
            "from_hold": ev["from_hold"],
            "to_hold": ev["to_hold"],
            "hold_distance_px": ev["hold_distance_px"],
            "type": label,
            "confidence": conf,
            "snapshot_path": str(snapshot),
            "probs": probs.tolist(),
        }
        moves.append(move_data)
        
        print(
            f"  Move {move_idx+1}: {ev['limb']} "
            f"{ev['from_hold']}→{ev['to_hold']} "
            f"({ev['hold_distance_px']:.0f}px) - {label}"
        )
    
    cap.release()
    pose.close()
    
    # Save results
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
    
    print(f"\n[move_detection] ✅ Complete! Saved to {climb_data_path}")
    return str(climb_data_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v", required=True)
    parser.add_argument("--holds", "-H", required=True)
    parser.add_argument("--out-dir", "-o", default="output")
    
    args = parser.parse_args()
    detect_and_classify_moves(args.video, args.holds, args.out_dir)
