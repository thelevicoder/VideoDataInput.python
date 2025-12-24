# move_detector_improved.py
#
# IMPROVED move detection with better filtering for real climbing moves.
#
# Key improvements:
# 1. Higher thresholds for minimum movement distance
# 2. Longer stability requirements before/after moves
# 3. Velocity-based move detection
# 4. Better filtering of jitter and micro-movements
# 5. Adaptive thresholding based on video resolution

import cv2
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

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

LIMBS = ["lefthand", "righthand", "leftfoot", "rightfoot"]

# IMPROVED PARAMETERS - Much stricter for real moves
ASSIGN_THRESHOLD = 80           # Increased from 60 - limb must be closer to hold
MIN_STABLE_FRAMES = 8           # Increased from 3 - require longer stability
MIN_HOLD_SWITCH_DIST = 60.0     # Increased from 8.0 - significant movement only
MIN_MOVE_VELOCITY = 15.0        # New: minimum pixels/frame to count as move
MAX_OSCILLATION_RATIO = 0.3     # New: filter out back-and-forth jitter

# Classifier confidence
MIN_CONFIDENCE_FOR_TAG = 0.0
LOW_CONFIDENCE_PREFIX = "uncertain_"


def _compute_limb_positions(results, width: int, height: int) -> Dict[str, Tuple[int, int] | None]:
    """Extract limb positions from MediaPipe pose results."""
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
    threshold: int = ASSIGN_THRESHOLD,
) -> Dict[str, str]:
    """
    Assign limbs to holds with stricter distance threshold.
    """
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

        assignments[limb] = best_hold if best_dist <= threshold else "air"

    return assignments


def _calculate_limb_trajectory(
    assignments_series: List[Dict[str, str]],
    limb_positions_series: List[Dict[str, Tuple[int, int] | None]],
    limb: str
) -> List[Dict]:
    """
    Calculate movement trajectory for a limb including velocity and stability.
    Returns list of trajectory points with metadata.
    """
    trajectory = []
    
    for i, (assignments, positions) in enumerate(zip(assignments_series, limb_positions_series)):
        pos = positions.get(limb)
        hold = assignments.get(limb)
        
        trajectory.append({
            'frame': i,
            'hold': hold,
            'position': pos,
            'velocity': None,  # Will calculate
            'stable': False,   # Will mark stable regions
        })
    
    # Calculate velocities
    for i in range(1, len(trajectory)):
        prev_pos = trajectory[i-1]['position']
        curr_pos = trajectory[i]['position']
        
        if prev_pos is not None and curr_pos is not None:
            velocity = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
            trajectory[i]['velocity'] = velocity
        else:
            trajectory[i]['velocity'] = 0.0
    
    return trajectory


def _find_stable_segments(trajectory: List[Dict]) -> List[Dict]:
    """
    Find segments where limb is stable on a hold.
    Requires:
    - Same hold for MIN_STABLE_FRAMES consecutive frames
    - Low velocity throughout the segment
    - Not "air" or "no_pose"
    """
    segments = []
    current_hold = None
    start_idx = None
    stable_count = 0
    
    VELOCITY_THRESHOLD = 5.0  # Max velocity for "stable"
    
    for i, point in enumerate(trajectory):
        hold = point['hold']
        velocity = point['velocity'] or 0.0
        
        # Check if we're on a real hold with low velocity
        is_stable = (
            hold not in ("air", "no_pose") and
            velocity < VELOCITY_THRESHOLD
        )
        
        if is_stable and hold == current_hold:
            # Continue current stable segment
            stable_count += 1
        elif is_stable:
            # Start new potential stable segment
            current_hold = hold
            start_idx = i
            stable_count = 1
        else:
            # Unstable - save previous segment if long enough
            if stable_count >= MIN_STABLE_FRAMES and current_hold is not None:
                segments.append({
                    'hold': current_hold,
                    'start': start_idx,
                    'end': i - 1,
                    'length': stable_count
                })
            current_hold = None
            start_idx = None
            stable_count = 0
    
    # Save final segment
    if stable_count >= MIN_STABLE_FRAMES and current_hold is not None:
        segments.append({
            'hold': current_hold,
            'start': start_idx,
            'end': len(trajectory) - 1,
            'length': stable_count
        })
    
    return segments


def _detect_moves_from_segments(
    segments_by_limb: Dict[str, List[Dict]],
    holds: Dict[str, List[int]],
    limb_positions_series: List[Dict[str, Tuple[int, int] | None]],
) -> List[Dict]:
    """
    Detect real moves between stable segments.
    
    A move is valid if:
    1. Transition between two different holds
    2. Holds are far enough apart (MIN_HOLD_SWITCH_DIST)
    3. Sufficient velocity during transition
    4. Not just oscillation/jitter
    """
    events: List[Dict] = []
    
    for limb, segs in segments_by_limb.items():
        segs = sorted(segs, key=lambda s: s["start"])
        
        for i in range(len(segs) - 1):
            s1 = segs[i]
            s2 = segs[i + 1]
            h1 = s1["hold"]
            h2 = s2["hold"]
            
            if h1 == h2:
                continue
            
            # Check distance between holds
            c1 = _hold_center(h1, holds)
            c2 = _hold_center(h2, holds)
            
            if c1 is None or c2 is None:
                continue
            
            hold_distance = float(np.linalg.norm(c1 - c2))
            
            if hold_distance < MIN_HOLD_SWITCH_DIST:
                continue
            
            # Check velocity during transition
            transition_start = s1['end']
            transition_end = s2['start']
            
            if transition_end <= transition_start:
                continue
            
            # Calculate average velocity during transition
            velocities = []
            for frame_idx in range(transition_start, min(transition_end + 1, len(limb_positions_series))):
                if frame_idx > 0 and frame_idx < len(limb_positions_series):
                    prev_pos = limb_positions_series[frame_idx - 1].get(limb)
                    curr_pos = limb_positions_series[frame_idx].get(limb)
                    
                    if prev_pos and curr_pos:
                        vel = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
                        velocities.append(vel)
            
            if not velocities:
                continue
            
            avg_velocity = np.mean(velocities)
            max_velocity = np.max(velocities)
            
            # Require significant velocity to count as real move
            if max_velocity < MIN_MOVE_VELOCITY:
                continue
            
            # Check for oscillation pattern (back and forth)
            # Look ahead to next segment to see if it goes back to h1
            is_oscillation = False
            if i + 2 < len(segs):
                h3 = segs[i + 2]["hold"]
                if h3 == h1:
                    # Went from h1 -> h2 -> h1, likely oscillation
                    # But allow if each segment was long enough (real repositioning)
                    if s2['length'] < MIN_STABLE_FRAMES * 2:
                        is_oscillation = True
            
            if is_oscillation:
                continue
            
            # Use midpoint of transition as representative frame
            move_frame = (transition_start + transition_end) // 2
            
            events.append({
                "frame_index": move_frame + 1,  # 1-based
                "limb": limb,
                "from_hold": h1,
                "to_hold": h2,
                "hold_distance_px": hold_distance,
                "avg_velocity": avg_velocity,
                "max_velocity": max_velocity,
            })
    
    # Sort all events by time
    events.sort(key=lambda e: e["frame_index"])
    return events


def _hold_center(hold_id: str, holds: Dict[str, List[int]]) -> Optional[np.ndarray]:
    """Get hold center coordinates."""
    coords = holds.get(hold_id)
    if coords is None:
        return None
    return np.array(coords, dtype=float)


def detect_and_classify_moves(
    video_path: str,
    holds_json_path: str,
    output_dir: str = "output",
) -> str:
    """
    Improved move detection and classification.
    
    Returns:
        Path to climb_data.json
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
    
    # First pass - collect all frame data
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Adjust thresholds based on resolution
    global MIN_HOLD_SWITCH_DIST, ASSIGN_THRESHOLD, MIN_MOVE_VELOCITY
    resolution_scale = np.sqrt(width * height) / 1000.0
    MIN_HOLD_SWITCH_DIST = max(50.0, 60.0 * resolution_scale)
    ASSIGN_THRESHOLD = max(60, int(80 * resolution_scale))
    MIN_MOVE_VELOCITY = max(10.0, 15.0 * resolution_scale)
    
    print(f"[move_detection] Video resolution: {width}x{height}")
    print(f"[move_detection] Adjusted MIN_HOLD_SWITCH_DIST: {MIN_HOLD_SWITCH_DIST:.1f} px")
    print(f"[move_detection] Adjusted ASSIGN_THRESHOLD: {ASSIGN_THRESHOLD} px")
    print(f"[move_detection] Adjusted MIN_MOVE_VELOCITY: {MIN_MOVE_VELOCITY:.1f} px/frame")
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    assignments_series: List[Dict[str, str]] = []
    limb_positions_series: List[Dict[str, Tuple[int, int] | None]] = []
    
    print(f"[move_detection] First pass: analyzing motion...")
    
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        
        frame_count += 1
        height, width = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            assignments = {limb: "no_pose" for limb in LIMBS}
            positions = {limb: None for limb in LIMBS}
        else:
            positions = _compute_limb_positions(results, width, height)
            assignments = _assign_limbs_to_holds(positions, holds)
        
        assignments_series.append(assignments)
        limb_positions_series.append(positions)
    
    total_frames = len(assignments_series)
    cap.release()
    pose.close()
    
    print(f"[move_detection] Processed {total_frames} frames")
    
    # Analyze trajectories and find stable segments for each limb
    print(f"[move_detection] Analyzing limb trajectories...")
    
    segments_by_limb: Dict[str, List[Dict]] = {}
    for limb in LIMBS:
        trajectory = _calculate_limb_trajectory(
            assignments_series,
            limb_positions_series,
            limb
        )
        stable_segs = _find_stable_segments(trajectory)
        segments_by_limb[limb] = stable_segs
        
        print(f"[move_detection]   {limb}: {len(stable_segs)} stable segments")
    
    # Detect moves from stable segments
    move_events = _detect_moves_from_segments(
        segments_by_limb,
        holds,
        limb_positions_series
    )
    
    print(f"[move_detection] Detected {len(move_events)} distinct moves")
    
    # Second pass - classify each move and save snapshots
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not reopen video file: {video_path}")
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    classifier = PoseMoveClassifier()
    
    moves: List[Dict] = []
    
    print(f"[move_detection] Second pass: classifying moves...")
    
    for move_idx, ev in enumerate(move_events):
        target_frame = ev["frame_index"]
        if target_frame < 1 or target_frame > total_frames:
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[move_detection] Warning - could not read frame {target_frame}")
            continue
        
        height, width = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            label = "no_pose"
            probs = np.zeros((1,), dtype=float)
            top_conf = 0.0
        else:
            label, probs = classifier.predict_from_landmarks(
                results.pose_landmarks.landmark
            )
            top_conf = float(np.max(probs))
            if top_conf < MIN_CONFIDENCE_FOR_TAG:
                label = LOW_CONFIDENCE_PREFIX + label
        
        snapshot_path = moves_dir / f"move_{move_idx+1:02d}.jpg"
        cv2.imwrite(str(snapshot_path), frame)
        
        move_data = {
            "move_index": move_idx,
            "frame_index": int(target_frame),
            "time_seconds": target_frame / fps,
            "limb": ev["limb"],
            "from_hold": ev["from_hold"],
            "to_hold": ev["to_hold"],
            "hold_distance_px": ev["hold_distance_px"],
            "avg_velocity_px_per_frame": ev["avg_velocity"],
            "max_velocity_px_per_frame": ev["max_velocity"],
            "type": label,
            "confidence": top_conf,
            "snapshot_path": str(snapshot_path),
            "probs": probs.tolist(),
        }
        moves.append(move_data)
        
        print(
            f"Move {move_idx+1:02d}: frame {target_frame}, "
            f"limb={ev['limb']}, {ev['from_hold']} -> {ev['to_hold']}, "
            f"dist={ev['hold_distance_px']:.1f}px, vel={ev['avg_velocity']:.1f}px/f, "
            f"type={label}, conf={top_conf:.2f}"
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
    
    print(f"\n[move_detection] ✅ Detected {len(moves)} real moves")
    print(f"[move_detection] Saved to: {climb_data_path}")
    
    return str(climb_data_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Improved move detection for climbing videos"
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
