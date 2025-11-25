# move_detector.py
#
# Offline move detection:
# 1. First pass: run pose and assign limbs to holds for every frame.
# 2. Segment each hand's timeline into stable "on hold" segments.
# 3. Moves are transitions between stable hand segments with different holds.
# 4. Second pass: classify each move frame with PoseMoveClassifier and save snapshots.

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

HANDS = ["lefthand", "righthand"]

# assignment and segmentation parameters
ASSIGN_THRESHOLD = 60            # px, max distance hand to hold center
MIN_SEGMENT_FRAMES = 3           # min frames on a hold to count as stable segment
MERGE_MOVE_WINDOW_FRAMES = 3     # merge hand moves that happen within this many frames
MIN_CONFIDENCE = 0.5             # classifier confidence
MIN_HOLD_SWITCH_DIST = 10.0      # px center distance to consider it a real switch


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
    For each limb, assign nearest hold if within threshold. Otherwise "air" or "no_pose".
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
    [{ "hold": hold_id, "start": frame_idx0, "end": frame_idx1 }, ...]
    using zero based frame indices.
    """
    segments: List[Dict] = []
    current_hold: Optional[str] = None
    start_idx: Optional[int] = None

    for i, ass in enumerate(assignments_series):
        val = ass.get(limb, "no_pose")

        if current_hold is None:
            current_hold = val
            start_idx = i
            continue

        if val == current_hold:
            continue

        # close current segment
        segments.append({"hold": current_hold, "start": start_idx, "end": i - 1})
        current_hold = val
        start_idx = i

    if current_hold is not None and start_idx is not None:
        segments.append({"hold": current_hold, "start": start_idx, "end": len(assignments_series) - 1})

    return segments


def _hold_center(hold_id: str, holds: Dict[str, List[int]]) -> Optional[np.ndarray]:
    if hold_id in ("air", "no_pose"):
        return None
    coords = holds.get(hold_id)
    if coords is None:
        return None
    return np.array(coords, dtype=float)


def _detect_hand_moves_from_segments(
    segments_by_hand: Dict[str, List[Dict]],
    holds: Dict[str, List[int]],
) -> List[Dict]:
    """
    Given stable segments for each hand, produce raw move events:
    [{ "frame_index": int (1 based), "limb": str, "from": hold_id, "to": hold_id }]
    """
    raw_events: List[Dict] = []

    for hand, segs in segments_by_hand.items():
        if not segs:
            continue

        # sort just in case
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
                if dist < MIN_HOLD_SWITCH_DIST:
                    # tiny switch on same macro feature, ignore
                    continue

            frame_index_1based = s2["start"] + 1  # first frame of new segment, 1 based
            raw_events.append(
                {
                    "frame_index": frame_index_1based,
                    "limb": hand,
                    "from": h1,
                    "to": h2,
                }
            )

    # merge events from different hands that are close in time
    raw_events.sort(key=lambda e: e["frame_index"])
    merged: List[Dict] = []

    current_group: Optional[Dict] = None

    for ev in raw_events:
        if current_group is None:
            current_group = {
                "frame_index": ev["frame_index"],
                "changes": {ev["limb"]: {"from": ev["from"], "to": ev["to"]}},
            }
            continue

        if ev["frame_index"] - current_group["frame_index"] <= MERGE_MOVE_WINDOW_FRAMES:
            # merge into current group
            current_group["changes"][ev["limb"]] = {"from": ev["from"], "to": ev["to"]}
            # keep frame index near the earliest
        else:
            merged.append(current_group)
            current_group = {
                "frame_index": ev["frame_index"],
                "changes": {ev["limb"]: {"from": ev["from"], "to": ev["to"]}},
            }

    if current_group is not None:
        merged.append(current_group)

    return merged


def detect_and_classify_moves(
    video_path: str,
    holds_json_path: str,
    output_dir: str = "output",
) -> str:
    """
    Full offline move detection and classification.
    Returns path to climb_data.json.
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

    # first pass: assignments for every frame
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

    frame_idx = 0
    assignments_series: List[Dict[str, str]] = []

    print(f"First pass: computing assignments for each frame of {video_path}")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        height, width = frame.shape[:2]

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            # no pose, all limbs set to "no_pose"
            assignments = {limb: "no_pose" for limb in LIMB_LANDMARKS}
        else:
            limb_positions = _compute_limb_positions(results, width, height)
            assignments = _assign_limbs_to_holds(limb_positions, holds)

        assignments_series.append(assignments)

    cap.release()
    pose.close()

    total_frames = len(assignments_series)
    print(f"[move_detection] Processed {total_frames} frames for assignments")

    # build stable segments for each hand
    segments_by_hand: Dict[str, List[Dict]] = {}
    for hand in HANDS:
        raw_segments = _build_segments(assignments_series, hand)
        stable_segments: List[Dict] = []

        for seg in raw_segments:
            hold = seg["hold"]
            length = seg["end"] - seg["start"] + 1
            if hold in ("air", "no_pose"):
                continue
            if length < MIN_SEGMENT_FRAMES:
                continue
            stable_segments.append(seg)

        segments_by_hand[hand] = stable_segments

    # detect moves from segments
    move_events = _detect_hand_moves_from_segments(segments_by_hand, holds)
    print(f"[move_detection] Found {len(move_events)} candidate move events from hand segments")

    # second pass: classify each move frame and save snapshots
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
        # seek to frame (0 based index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame - 1))
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[move_detection] Warning: could not read frame {target_frame} for move {move_idx}")
            continue

        height, width = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # tiny search around if pose missing here
        if not results.pose_landmarks:
            found = False
            for delta in [-2, -1, 1, 2]:
                alt_frame = target_frame + delta
                if alt_frame < 1 or alt_frame > total_frames:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, alt_frame - 1)
                ok2, frame2 = cap.read()
                if not ok2 or frame2 is None:
                    continue
                image_rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                results2 = pose.process(image_rgb2)
                if results2.pose_landmarks:
                    target_frame = alt_frame
                    frame = frame2
                    results = results2
                    found = True
                    break
            if not found:
                print(f"[move_detection] No pose around frame {ev['frame_index']} for move {move_idx}, skipping")
                continue

        label, probs = classifier.predict_from_landmarks(
            results.pose_landmarks.landmark
        )
        top_conf = float(np.max(probs))
        if top_conf < MIN_CONFIDENCE:
            print(f"[move_detection] Low confidence {top_conf:.2f} at frame {target_frame}, skipping move")
            continue

        # snapshot
        snapshot_path = moves_dir / f"move_{move_idx+1:02d}.jpg"
        cv2.imwrite(str(snapshot_path), frame)

        # assignments at that frame (from first pass)
        assignment_at_frame = assignments_series[target_frame - 1]

        move_data = {
            "move_index": move_idx,
            "frame_index": int(target_frame),
            "time_seconds": target_frame / fps,
            "type": label,
            "confidence": top_conf,
            "hand_changes": ev["changes"],           # which hands moved between which holds
            "assignments": assignment_at_frame,      # all limbs
            "snapshot_path": str(snapshot_path),
            "probs": probs.tolist(),
        }
        moves.append(move_data)

        print(
            f"Saved move {move_idx+1:02d} at frame {target_frame}, "
            f"type={label}, conf={top_conf:.2f}, hand_changes={ev['changes']}"
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
