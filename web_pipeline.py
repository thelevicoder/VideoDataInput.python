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
VIDEO_FOLDER = Path("my_videos")
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
    
    return jsonify({
        'success': True,
        'num_holds': len(hold_positions),
        'visualization': f"data:image/jpeg;base64,{vis_base64}",
        'holds': hold_positions
    })


@app.route('/api/detect_moves', methods=['POST'])
def detect_moves():
    """Detect and classify moves."""
    data = request.json
    video_path = data['video_path']
    output_dir = session.get('current_output_dir')
    
    if not output_dir:
        return jsonify({'error': 'No output directory in session'}), 400
    
    output_dir = Path(output_dir)
    holds_json = output_dir / "hold_positions_auto_split.json"
    if not holds_json.exists():
        holds_json = output_dir / "hold_positions_auto.json"
    
    # Detect moves
    climb_data_path = detect_and_classify_moves(
        video_path,
        str(holds_json),
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
    
    # Convert to training format
    training_entry = convert_to_training_format(
        climb_data=climb_data,
        grade=grade,
        climber_height=CLIMBER_CONFIG['height'],
        climber_wingspan=CLIMBER_CONFIG['wingspan'],
        climber_skill=CLIMBER_CONFIG['skill'],
        wall_angle=wall_angle,
        gym_name=gym,
        video_path=climb_data['video_path'],
        notes=notes
    )
    
    # Save to database
    db_path = Path("climb_database")
    save_training_record(training_entry, db_path)
    
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