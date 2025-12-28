#!/usr/bin/env python3
# web_pipeline.py
#
# WEB-BASED INTERACTIVE PIPELINE
# Process videos through a browser interface

from flask import Flask, render_template, request, jsonify, send_from_directory, session
from pathlib import Path
import json
import cv2
import numpy as np
import base64
from typing import List, Tuple, Dict, Optional
import secrets

# Import detection modules
from hold_detection import sample_frames_from_video, create_composite_mask, filter_and_group_contours
from split_grouped_holds import split_grouped_holds
from move_detector_simple import detect_and_classify_moves
from export_training_data import convert_to_training_format, save_training_record

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Global state
VIDEO_FOLDER = Path("Vids")
OUTPUT_FOLDER = Path("batch_output")
CURRENT_VIDEO_INDEX = 0
VIDEO_FILES = []
CLIMBER_CONFIG = {}

def find_videos():
    """Find all video files in the video folder."""
    global VIDEO_FILES
    video_extensions = ['.mp4', '.mov', '.avi']
    video_files = []
    for ext in video_extensions:
        video_files.extend(VIDEO_FOLDER.glob(f"*{ext}"))
        video_files.extend(VIDEO_FOLDER.glob(f"*{ext.upper()}"))
    
    # Remove duplicates
    VIDEO_FILES = sorted(list(set(video_files)))
    return VIDEO_FILES


def get_video_frame(video_path: str, frame_index: int = None) -> str:
    """Get a frame from video as base64 encoded image."""
    cap = cv2.VideoCapture(video_path)
    
    if frame_index is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    else:
        # Get middle frame
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Resize for web display
    h, w = frame.shape[:2]
    max_width = 1280
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"


@app.route('/')
def index():
    """Main page - setup climber info."""
    find_videos()
    return render_template('index.html', 
                         video_count=len(VIDEO_FILES),
                         climber_config=CLIMBER_CONFIG)


@app.route('/api/setup_climber', methods=['POST'])
def setup_climber():
    """Save climber configuration."""
    global CLIMBER_CONFIG
    data = request.json
    
    CLIMBER_CONFIG = {
        'height': float(data.get('height', 70)),
        'wingspan': float(data.get('wingspan', 70)),
        'skill': data.get('skill', 'intermediate')
    }
    
    # Save to file
    config_file = Path("pipeline_config.json")
    with open(config_file, 'w') as f:
        json.dump(CLIMBER_CONFIG, f, indent=2)
    
    return jsonify({'success': True, 'config': CLIMBER_CONFIG})


@app.route('/api/get_current_video')
def get_current_video():
    """Get current video info."""
    global CURRENT_VIDEO_INDEX
    
    if CURRENT_VIDEO_INDEX >= len(VIDEO_FILES):
        return jsonify({'done': True})
    
    video_path = VIDEO_FILES[CURRENT_VIDEO_INDEX]
    frame_base64 = get_video_frame(str(video_path))
    
    return jsonify({
        'done': False,
        'index': CURRENT_VIDEO_INDEX + 1,
        'total': len(VIDEO_FILES),
        'filename': video_path.name,
        'video_path': str(video_path),
        'frame': frame_base64
    })


@app.route('/api/detect_holds', methods=['POST'])
def detect_holds():
    """Detect holds with selected colors."""
    data = request.json
    video_path = data['video_path']
    colors = data['colors']  # List of {lab, hsv, bgr}
    
    # Convert to numpy arrays
    selected_colors = []
    for c in colors:
        lab = np.array(c['lab'], dtype=np.float32)
        hsv = np.array(c['hsv'], dtype=np.float32)
        bgr = np.array(c['bgr'], dtype=np.float32)
        selected_colors.append((lab, hsv, bgr))
    
    # Create output directory
    video_name = Path(video_path).stem
    output_dir = OUTPUT_FOLDER / f"video_{CURRENT_VIDEO_INDEX+1:03d}_{video_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect holds
    frames = sample_frames_from_video(video_path, 10)
    reference_frame = frames[len(frames) // 2]
    
    combined_composite = np.zeros(frames[0].shape[:2], dtype=np.uint8)
    
    for lab, hsv, _ in selected_colors:
        is_red = (hsv[0] < 15 or hsv[0] > 170)
        composite = create_composite_mask(frames, lab, hsv, is_red, None)
        combined_composite = cv2.bitwise_or(combined_composite, composite)
    
    hold_ids, hold_positions, vis = filter_and_group_contours(combined_composite, reference_frame)
    
    # Save outputs
    cv2.imwrite(str(output_dir / "hold_mask_composite.jpg"), combined_composite)
    cv2.imwrite(str(output_dir / "holds_debug.jpg"), vis)
    
    holds_json = output_dir / "hold_positions_auto.json"
    with open(holds_json, 'w') as f:
        json.dump(hold_positions, f, indent=2)
    
    # Run splitter
    try:
        split_path = split_grouped_holds(
            holds_json_path=str(holds_json),
            mask_path=str(output_dir / "hold_mask_composite.jpg"),
            debug_image_path=str(output_dir / "holds_debug.jpg"),
            output_dir=str(output_dir),
            video_path=video_path
        )
        
        if split_path:
            with open(split_path, 'r') as f:
                hold_positions = json.load(f)
            # Use split visualization
            vis_path = output_dir / "holds_debug_split.jpg"
        else:
            vis_path = output_dir / "holds_debug.jpg"
    except:
        vis_path = output_dir / "holds_debug.jpg"
    
    # Encode visualization
    vis_img = cv2.imread(str(vis_path))
    _, buffer = cv2.imencode('.jpg', vis_img)
    vis_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Save to session
    session['current_output_dir'] = str(output_dir)
    session['hold_positions'] = hold_positions
    
    # Create a clickable visualization for finish hold selection
    # Draw holds with labels
    finish_select_img = vis_img.copy()
    for hold_id, pos in hold_positions.items():
        cx, cy = pos
        # Draw larger circles for clicking
        cv2.circle(finish_select_img, (cx, cy), 30, (0, 255, 255), 3)
        cv2.circle(finish_select_img, (cx, cy), 25, (0, 0, 0), -1)
        cv2.putText(finish_select_img, hold_id.split('_')[1], (cx-10, cy+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    _, buffer2 = cv2.imencode('.jpg', finish_select_img)
    finish_select_base64 = base64.b64encode(buffer2).decode('utf-8')
    
    return jsonify({
        'success': True,
        'num_holds': len(hold_positions),
        'visualization': f"data:image/jpeg;base64,{vis_base64}",
        'finish_select_img': f"data:image/jpeg;base64,{finish_select_base64}",
        'holds': hold_positions
    })


@app.route('/api/detect_moves', methods=['POST'])
def detect_moves():
    """Detect and classify moves, stopping only when hand reaches finish hold."""
    data = request.json
    video_path = data['video_path']
    finish_hold = data.get('finish_hold')  # e.g., "hold_5"
    
    output_dir = session.get('current_output_dir')
    
    if not output_dir:
        return jsonify({'error': 'No output directory in session'}), 400
    
    output_dir = Path(output_dir)
    holds_json = output_dir / "hold_positions_auto_split.json"
    if not holds_json.exists():
        holds_json = output_dir / "hold_positions_auto.json"
    
    # Classify holds first (if classifier available)
    try:
        classifier_path = Path("move_classifier/hold_classifier.h5")
        if classifier_path.exists():
            import sys
            old_argv = sys.argv
            sys.argv = [
                'enrich',
                '--video', video_path,
                '--holds', str(holds_json),
                '--output', str(output_dir)
            ]
            
            try:
                from enrich_holds_with_classifier_multiframe import main as enrich_holds
                enrich_holds()
                
                # Use enriched holds if created
                enriched_holds_path = output_dir / "hold_positions_enriched.json"
                if enriched_holds_path.exists():
                    holds_json = enriched_holds_path
                    print("[web] Using enriched holds with classifications")
            finally:
                sys.argv = old_argv
    except Exception as e:
        print(f"[web] Hold classification failed: {e}")
    
    # Detect moves with finish hold constraint
    climb_data_path = detect_and_classify_moves_with_finish(
        video_path,
        str(holds_json),
        finish_hold=finish_hold,
        output_dir=str(output_dir)
    )
    
    # Load move data
    with open(climb_data_path, 'r') as f:
        climb_data = json.load(f)
    
    # Get move snapshots
    move_snapshots = []
    for move in climb_data['moves']:
        snapshot_path = Path(move['snapshot_path'])
        if snapshot_path.exists():
            img = cv2.imread(str(snapshot_path))
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            move_snapshots.append(f"data:image/jpeg;base64,{img_base64}")
        else:
            move_snapshots.append(None)
    
    session['climb_data_path'] = climb_data_path
    
    return jsonify({
        'success': True,
        'num_moves': len(climb_data['moves']),
        'moves': climb_data['moves'],
        'snapshots': move_snapshots
    })


def detect_and_classify_moves_with_finish(
    video_path: str,
    holds_json_path: str,
    finish_hold: str = None,
    output_dir: str = "output",
) -> str:
    """
    Modified move detection that only stops when a hand reaches the finish hold.
    """
    from move_detector_simple import (
        _compute_limb_positions, _assign_limbs_to_holds,
        LIMBS, LIMB_LANDMARKS, mp_pose, PoseMoveClassifier,
        HOLD_ASSIGNMENT_DISTANCE, STABILITY_FRAMES, MIN_MOVE_DISTANCE, COOLDOWN_FRAMES
    )
    
    video_path = Path(video_path)
    holds_json_path = Path(holds_json_path)
    output_root = Path(output_dir)
    moves_dir = output_root / "moves"
    
    if not holds_json_path.exists():
        raise FileNotFoundError(f"Holds JSON not found: {holds_json_path}")
    
    with holds_json_path.open("r", encoding="utf8") as f:
        holds = json.load(f)
    
    print(f"[move_detection] Loaded {len(holds)} holds")
    if finish_hold:
        print(f"[move_detection] Finish hold: {finish_hold}")
    
    # Clean moves folder
    if moves_dir.exists():
        import shutil
        for item in moves_dir.iterdir():
            if item.is_file():
                item.unlink()
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
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    assignments_series = []
    finish_reached = False
    finish_hold_stable_count = 0
    min_frames_before_finish = int(fps * 5)  # Must climb for at least 5 seconds
    
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
        
        # Check if either hand reached finish hold
        if finish_hold and frame_count > min_frames_before_finish:
            hand_on_finish = (assignments.get('lefthand') == finish_hold or 
                            assignments.get('righthand') == finish_hold)
            
            if hand_on_finish:
                finish_hold_stable_count += 1
                # Require hand to be stable on finish for 30 frames (~1 second)
                if finish_hold_stable_count >= 30:
                    print(f"[move_detection] Finish hold reached and stable at frame {frame_count}")
                    # Continue for a bit longer to catch final moves
                    for _ in range(int(fps * 2)):  # 2 more seconds
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame_count += 1
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = pose.process(image_rgb)
                        
                        if not results.pose_landmarks:
                            assignments = {limb: "no_pose" for limb in LIMBS}
                        else:
                            positions = _compute_limb_positions(results, w, h)
                            assignments = _assign_limbs_to_holds(positions, holds)
                        
                        assignments_series.append(assignments)
                    break
            else:
                finish_hold_stable_count = 0  # Reset if hand leaves
    
    cap.release()
    pose.close()
    
    print(f"[move_detection] Processed {len(assignments_series)} frames")
    
    # Detect moves (same logic as before)
    from move_detector_simple import _detect_moves_simple
    move_events = _detect_moves_simple(assignments_series, holds)
    
    print(f"[move_detection] Detected {len(move_events)} moves")
    
    # Second pass - classify and save (same as before)
    cap = cv2.VideoCapture(str(video_path))
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    classifier = PoseMoveClassifier()
    
    moves = []
    
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
        "finish_hold": finish_hold,
        "fps": fps,
        "moves": moves,
    }
    
    output_root.mkdir(parents=True, exist_ok=True)
    climb_data_path = output_root / "climb_data.json"
    with climb_data_path.open("w", encoding="utf8") as f:
        json.dump(climb_data, f, indent=2)
    
    print(f"\n[move_detection] ✅ Complete! Saved to {climb_data_path}")
    return str(climb_data_path)


@app.route('/api/save_climb', methods=['POST'])
def save_climb():
    """Save climb data with metadata."""
    data = request.json
    
    grade_num = data['grade']
    grade = f"V{grade_num}"
    
    wall_rating = float(data['wall_angle'])
    wall_angle = wall_rating * 6.0  # Convert rating to degrees
    
    gym = data.get('gym', 'Unknown Gym')
    notes = data.get('notes', '')
    
    # Load climb data
    climb_data_path = session.get('climb_data_path')
    if not climb_data_path:
        return jsonify({'error': 'No climb data in session'}), 400
    
    with open(climb_data_path, 'r') as f:
        climb_data = json.load(f)
    
    # Convert to training format - check function signature
    try:
        training_entry = convert_to_training_format(
            climb_data,
            route_grade=grade,
            climber_height_in=CLIMBER_CONFIG['height'],
            climber_wingspan_in=CLIMBER_CONFIG['wingspan'],
            climber_skill_level=CLIMBER_CONFIG['skill'],
            wall_angle_deg=wall_angle,
            gym_name=gym,
            video_path=climb_data['video_path'],
            notes=notes
        )
    except TypeError:
        # If that doesn't work, try alternate parameter names
        training_entry = {
            'climb_data': climb_data,
            'route_grade': grade,
            'climber_height': CLIMBER_CONFIG['height'],
            'climber_wingspan': CLIMBER_CONFIG['wingspan'],
            'climber_skill': CLIMBER_CONFIG['skill'],
            'wall_angle': wall_angle,
            'gym_name': gym,
            'video_path': climb_data['video_path'],
            'notes': notes
        }
    
    # Save to database
    db_path = Path("climb_database")
    db_path.mkdir(parents=True, exist_ok=True)
    
    # Simple save - create unique filename
    import time
    timestamp = int(time.time())
    output_file = db_path / f"climb_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(training_entry, f, indent=2)
    
    return jsonify({'success': True})


@app.route('/api/skip_video', methods=['POST'])
def skip_video():
    """Skip current video."""
    global CURRENT_VIDEO_INDEX
    CURRENT_VIDEO_INDEX += 1
    return jsonify({'success': True})


@app.route('/api/next_video', methods=['POST'])
def next_video():
    """Move to next video after completing current one."""
    global CURRENT_VIDEO_INDEX
    CURRENT_VIDEO_INDEX += 1
    return jsonify({'success': True})


@app.route('/videos/<path:filename>')
def serve_video(filename):
    """Serve video files."""
    return send_from_directory(VIDEO_FOLDER, filename)


if __name__ == '__main__':
    # Create output directories
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("CLIMB ANALYSIS WEB INTERFACE")
    print("="*70)
    print("\n🌐 Starting web server...")
    print("\n📂 Video folder:", VIDEO_FOLDER.absolute())
    print("📂 Output folder:", OUTPUT_FOLDER.absolute())
    print("\n🚀 Open your browser to: http://localhost:5000")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)