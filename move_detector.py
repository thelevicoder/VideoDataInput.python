# move_detector.py
#
# Geometry based move detection for training data.
#
# A move is recorded whenever a single limb (hand or foot) changes from one
# detected hold ("contour_x") to another detected hold, after smoothing out
# short jitter segments.
#
# Pipeline:
#   1) First pass: for every frame run pose, assign each limb to nearest hold.
#   2) For each limb, compress assignments into stable segments on each hold.
#   3) Any transition seg_i.hold -> seg_{i+1}.hold (both real holds) is a move.
#   4) Second pass: for each move frame, run classifier and save snapshot.
#
# This file is focused on generating clean, structured move data that can be
# used as training labels for route grade prediction.

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

# Assignment parameters
ASSIGN_THRESHOLD = 60          # max distance hand/foot to hold center in pixels
MIN_SEG_FRAMES = 3             # min frames for a segment to be kept
MIN_HOLD_SWITCH_DIST = 8.0     # min distance between hold centers to count as new hold

# Classifier
MIN_CONFIDENCE_FOR_TAG = 0.0   # never drop moves because of confidence
LOW_CONFIDENCE_PREFIX = "uncertain_"


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


def _assign_limbs_to_holds(
    limb_positions: Dict[str, Tuple[int, int] | None],
    holds: Dict[str, List[int]],
    threshold: int = ASSIGN_THRESHOLD,
) -> Dict[str, str]:
    """
    For each limb, assign nearest hold center if within threshold.
    Otherwise:
        "air"      - pose detected but not near any hold
        "no_pose"  - pose missing for this limb
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


def _build_segments(assignments_series: List[Dict[str, str]], limb: str) -> List[Dict]:
    """
    Build raw segments for one limb:
    [
      { "hold": hold_id, "start": start_frame_idx0, "end": end_frame_idx0 },
      ...
    ]
    with zero based indices.
    """
    segments: List[Dict] = []

    if not assignments_series:
        return segments

    current_hold = assignments_series[0][limb]
    start_idx = 0

    for i in range(1, len(assignments_series)):
        val = assignments_series[i][limb]
        if val == current_hold:
            continue

        segments.append({"hold": current_hold, "start": start_idx, "end": i - 1})
        current_hold = val
        start_idx = i

    segments.append({"hold": current_hold, "start": start_idx, "end": len(assignments_series) - 1})
    return segments


def _filter_stable_segments(segments: List[Dict]) -> List[Dict]:
    """
    Keep only segments where:
      - hold is a real contour
      - segment length >= MIN_SEG_FRAMES
    """
    stable: List[Dict] = []
    for seg in segments:
        hold = seg["hold"]
        length = seg["end"] - seg["start"] + 1
        if hold in ("air", "no_pose"):
            continue
        if length < MIN_SEG_FRAMES:
            continue
        seg["length"] = length
        stable.append(seg)
    return stable


def _hold_center(hold_id: str, holds: Dict[str, List[int]]) -> Optional[np.ndarray]:
    coords = holds.get(hold_id)
    if coords is None:
        return None
    return np.array(coords, dtype=float)


def _detect_moves_from_segments(
    segments_by_limb: Dict[str, List[Dict]],
    holds: Dict[str, List[int]],
) -> List[Dict]:
    """
    Produce one move per limb segment transition:
    [
      {
        "frame_index": int (1 based),
        "limb": "lefthand",
        "from_hold": "contour_1",
        "to_hold": "contour_3",
        "hold_distance_px": float
      },
      ...
    ]
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

            c1 = _hold_center(h1, holds)
            c2 = _hold_center(h2, holds)
            if c1 is not None and c2 is not None:
                dist = float(np.linalg.norm(c1 - c2))
            else:
                dist = 0.0

            # optional filter on tiny distance switches
            if dist < MIN_HOLD_SWITCH_DIST:
                continue

            # choose representative frame as first frame of new segment
            frame_idx_1based = s2["start"] + 1

            events.append(
                {
                    "frame_index": frame_idx_1based,
                    "limb": limb,
                    "from_hold": h1,
                    "to_hold": h2,
                    "hold_distance_px": dist,
                }
            )

    # sort all limbs' events by time
    events.sort(key=lambda e: e["frame_index"])
    return events


def detect_and_classify_moves(
    video_path: str,
    holds_json_path: str,
    output_dir: str = "output",
) -> str:
    """
    Full offline move detection and classification.

    Returns:
        Path to climb_data.json

    Output JSON has:
      - video_path
      - holds_json
      - fps
      - moves: list of
          {
            move_index,
            frame_index,
            time_seconds,
            limb,
            from_hold,
            to_hold,
            hold_distance_px,
            type,              # classifier label (possibly "uncertain_*")
            confidence,        # classifier max probability
            snapshot_path,
            probs,             # full probability vector
          }
    """
    video_path = Path(video_path)
    holds_json_path = Path(holds_json_path)
    output_root = Path(output_dir)
    moves_dir = output_root / "moves"

    if not holds_json_path.exists():
        raise FileNotFoundError(f"Holds JSON not found: {holds_json_path}")

    with holds_json_path.open("r", encoding="utf8") as f:
        holds = json.load(f)

    # clean moves folder
    if moves_dir.exists():
        for item in moves_dir.iterdir():
            if item.is_file():
                item.unlink()
            else:
                import shutil
                shutil.rmtree(item)
    else:
        moves_dir.mkdir(parents=True, exist_ok=True)

    # first pass - store assignments and limb positions for each frame
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

    assignments_series: List[Dict[str, str]] = []

    print(f"[move_detection] First pass on {video_path}")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        height, width = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            assignments = {limb: "no_pose" for limb in LIMBS}
        else:
            limb_positions = _compute_limb_positions(results, width, height)
            assignments = _assign_limbs_to_holds(limb_positions, holds)

        assignments_series.append(assignments)

    total_frames = len(assignments_series)
    cap.release()
    pose.close()

    print(f"[move_detection] Processed {total_frames} frames for assignments")

    # segments per limb
    segments_by_limb: Dict[str, List[Dict]] = {}
    for limb in LIMBS:
        raw_segments = _build_segments(assignments_series, limb)
        stable = _filter_stable_segments(raw_segments)
        segments_by_limb[limb] = stable

    # move events from segments
    move_events = _detect_moves_from_segments(segments_by_limb, holds)
    print(f"[move_detection] Found {len(move_events)} limb hold switches")

    # second pass - classify each move frame and save snapshot
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

    for move_idx, ev in enumerate(move_events):
        target_frame = ev["frame_index"]
        if target_frame < 1 or target_frame > total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[move_detection] Warning - could not read frame {target_frame} for move {move_idx}")
            continue

        height, width = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            print(f"[move_detection] No pose at frame {target_frame} for move {move_idx}, skipping classifier")
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
            "type": label,
            "confidence": top_conf,
            "snapshot_path": str(snapshot_path),
            "probs": probs.tolist(),
        }
        moves.append(move_data)

        print(
            f"Saved move {move_idx+1:02d} at frame {target_frame}, "
            f"limb={ev['limb']}, {ev['from_hold']} -> {ev['to_hold']}, "
            f"dist={ev['hold_distance_px']:.1f}, type={label}, conf={top_conf:.2f}"
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
        description="Detect limb hold switches and classify moves from a climbing video"
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
