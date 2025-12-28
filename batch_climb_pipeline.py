#!/usr/bin/env python3
# batch_climb_pipeline_v2.py
#
# IMPROVED BATCH PROCESSING PIPELINE
# Uses the new multi-color picker with preview and auto-splitting

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import sys

# Import detection modules
from hold_detection import sample_frames_from_video, create_composite_mask, filter_and_group_contours
try:
    from split_grouped_holds import split_grouped_holds
    SPLITTER_AVAILABLE = True
except ImportError:
    SPLITTER_AVAILABLE = False
    print("⚠️  split_grouped_holds.py not found - hold splitting disabled")

from move_detector_simple import detect_and_classify_moves
from export_training_data import convert_to_training_format, save_training_record


class ClimbPipelineConfig:
    """Stores configuration that persists across videos."""
    def __init__(self):
        self.climber_height = None
        self.climber_wingspan = None
        self.climber_skill = None
        
        self.config_file = Path("pipeline_config.json")
        self.load_config()
    
    def load_config(self):
        """Load saved config if exists."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.climber_height = data.get('climber_height')
                self.climber_wingspan = data.get('climber_wingspan')
                self.climber_skill = data.get('climber_skill')
    
    def save_config(self):
        """Save config to file."""
        data = {
            'climber_height': self.climber_height,
            'climber_wingspan': self.climber_wingspan,
            'climber_skill': self.climber_skill,
        }
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def setup_climber_info(self):
        """Get climber info once at the beginning."""
        print("\n" + "="*70)
        print("CLIMBER INFORMATION (one-time setup)")
        print("="*70)
        
        if self.climber_height is None:
            height_str = input("Climber height in inches (e.g., 70): ").strip()
            self.climber_height = float(height_str) if height_str else 70.0
        else:
            print(f"Using saved height: {self.climber_height} inches")
        
        if self.climber_wingspan is None:
            wingspan_str = input("Climber wingspan in inches (default = height): ").strip()
            self.climber_wingspan = float(wingspan_str) if wingspan_str else self.climber_height
        else:
            print(f"Using saved wingspan: {self.climber_wingspan} inches")
        
        if self.climber_skill is None:
            skill = input("Climber skill level (beginner/intermediate/advanced): ").strip().lower()
            self.climber_skill = skill if skill else "intermediate"
        else:
            print(f"Using saved skill level: {self.climber_skill}")
        
        self.save_config()
        print(f"\n✓ Climber config saved to {self.config_file}")


def multi_color_picker_with_preview(video_path: str) -> Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Interactive multi-color picker with live preview.
    Returns list of (lab, hsv, bgr) tuples, or None if cancelled.
    """
    print("\n" + "="*70)
    print("HOLD COLOR SELECTION WITH PREVIEW")
    print("="*70)
    
    # Get middle frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open {video_path}")
        return None
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Error: Could not read frame")
        return None
    
    print("\n📺 Instructions:")
    print("  • Click on holds to add colors (you can click MULTIPLE)")
    print("  • After each click, you'll see a PREVIEW of detected holds")
    print("  • Press 'c' to CONFIRM and continue")
    print("  • Press 'r' to RESET and start over")
    print("  • Press 'q' to QUIT/SKIP this video")
    print("\n")
    
    # Window setup
    h, w = frame.shape[:2]
    max_width = 1280
    if w > max_width:
        scale = max_width / w
        display_w = max_width
        display_h = int(h * scale)
    else:
        display_w = w
        display_h = h
    
    display = frame.copy()
    selected_colors = []
    preview_mode = False
    
    def show_preview():
        """Generate and show preview of detected holds."""
        nonlocal preview_mode, display
        
        if len(selected_colors) == 0:
            return
        
        print("\n🔍 Generating preview...")
        preview_mode = True
        
        # Sample frames
        frames = sample_frames_from_video(video_path, 5)
        
        # Combine masks from all colors
        combined_composite = np.zeros(frames[0].shape[:2], dtype=np.uint8)
        
        for i, (lab, hsv, _) in enumerate(selected_colors):
            is_red = (hsv[0] < 15 or hsv[0] > 170)
            composite = create_composite_mask(frames, lab, hsv, is_red, None)
            combined_composite = cv2.bitwise_or(combined_composite, composite)
        
        # Detect holds
        hold_ids, hold_positions, vis = filter_and_group_contours(combined_composite, frame)
        
        # Show preview
        display = vis.copy()
        
        # Add color swatches
        swatch_size = 60
        for i, (_, _, bgr) in enumerate(selected_colors):
            x_offset = 20 + (i * (swatch_size + 10))
            y_offset = 20
            
            cv2.rectangle(display, (x_offset, y_offset), 
                         (x_offset + swatch_size, y_offset + swatch_size),
                         bgr.tolist(), -1)
            cv2.rectangle(display, (x_offset, y_offset), 
                         (x_offset + swatch_size, y_offset + swatch_size),
                         (255, 255, 255), 2)
        
        # Info text
        cv2.rectangle(display, (10, h-100), (w-10, h-10), (0, 0, 0), -1)
        cv2.putText(display, f"PREVIEW: {len(hold_positions)} holds detected", 
                   (20, h-70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(display, f"{len(selected_colors)} color(s) | 'c'=confirm | click=add color | 'r'=reset", 
                   (20, h-35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Select Hold Colors", display)
        print(f"✅ Preview: {len(hold_positions)} holds detected")
        print("   Click to add more colors, 'c' to confirm, or 'r' to reset")
    
    def on_mouse(event, x, y, flags, param):
        nonlocal selected_colors, display
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Sample 5x5 region
            patch = frame[max(0, y-2):min(frame.shape[0], y+3),
                         max(0, x-2):min(frame.shape[1], x+3)]
            
            # Average color
            color_bgr = np.mean(patch.reshape(-1, 3), axis=0)
            
            # Convert to LAB and HSV
            bgr_array = color_bgr.reshape(1, 1, 3).astype(np.uint8)
            lab = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2LAB)[0][0]
            hsv = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)[0][0]
            
            lab = lab.astype(np.float32)
            hsv = hsv.astype(np.float32)
            
            # Add to list
            selected_colors.append((lab, hsv, color_bgr))
            
            print(f"✓ Color {len(selected_colors)} added - LAB: {lab.astype(int)}, HSV: {hsv.astype(int)}")
            
            # Show preview
            show_preview()
    
    cv2.namedWindow("Select Hold Colors", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Hold Colors", display_w, display_h)
    cv2.setMouseCallback("Select Hold Colors", on_mouse)
    cv2.imshow("Select Hold Colors", display)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') and len(selected_colors) > 0:
            cv2.destroyAllWindows()
            print(f"\n✅ Confirmed! {len(selected_colors)} color(s) selected")
            return selected_colors
        
        elif key == ord('r'):
            display = frame.copy()
            selected_colors = []
            preview_mode = False
            cv2.imshow("Select Hold Colors", display)
            print("🔄 Reset - click to select colors")
        
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("\n❌ Skipped")
            return None
    
    cv2.destroyAllWindows()
    return None


def detect_holds_multi_color(
    video_path: str,
    selected_colors: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path
) -> Tuple[List[str], Dict]:
    """
    Detect holds using multiple colors.
    Returns (hold_ids, hold_positions).
    """
    print("\n" + "="*70)
    print("HOLD DETECTION (Multi-Color)")
    print("="*70)
    
    # Sample frames
    frames = sample_frames_from_video(video_path, 10)
    reference_frame = frames[len(frames) // 2]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine masks from all colors
    combined_composite = np.zeros(frames[0].shape[:2], dtype=np.uint8)
    
    for i, (lab, hsv, _) in enumerate(selected_colors):
        is_red = (hsv[0] < 15 or hsv[0] > 170)
        print(f"Processing color {i+1}/{len(selected_colors)}...")
        
        composite = create_composite_mask(frames, lab, hsv, is_red, None)
        combined_composite = cv2.bitwise_or(combined_composite, composite)
    
    print("[hold_detection] Combined all color masks")
    
    # Detect holds
    hold_ids, hold_positions, vis = filter_and_group_contours(combined_composite, reference_frame)
    
    # Save outputs
    cv2.imwrite(str(output_dir / "hold_mask_composite.jpg"), combined_composite)
    cv2.imwrite(str(output_dir / "holds_debug.jpg"), vis)
    
    holds_json = output_dir / "hold_positions_auto.json"
    with open(holds_json, 'w') as f:
        json.dump(hold_positions, f, indent=2)
    
    print(f"✅ Detected {len(hold_positions)} holds")
    
    return hold_ids, hold_positions


def get_video_metadata() -> Dict:
    """Get route metadata from user."""
    print("\nEnter route information:")
    
    # Simplified grade input
    grade_num = input("Route grade (just the number, e.g., 0, 2, 4, 8): ").strip()
    if not grade_num:
        print("⚠ Grade is required! Using V0 as default.")
        grade = "V0"
    else:
        # Add V prefix automatically
        grade = f"V{grade_num}"
    
    # Simplified wall angle input
    print("\nWall angle (-5 to +5):")
    print("  -5 = super slabby")
    print("   0 = vertical wall")
    print("  +5 = super overhung")
    wall_rating = input("Enter rating [-5 to +5]: ").strip()
    
    if wall_rating:
        try:
            rating = float(wall_rating)
            # Convert rating to approximate degrees
            # -5 = -30° (slabby), 0 = 0° (vertical), +5 = +30° (overhung)
            wall_angle = rating * 6.0  # Each point = 6 degrees
        except ValueError:
            print("⚠ Invalid rating, using 0 (vertical)")
            wall_angle = 0.0
    else:
        wall_angle = 0.0
    
    gym = input("Gym name (optional): ").strip()
    gym = gym if gym else "Unknown Gym"
    
    notes = input("Notes about this climb (optional): ").strip()
    
    return {
        'grade': grade,
        'wall_angle': wall_angle,
        'gym_name': gym,
        'notes': notes,
    }


def process_single_video(
    video_path: Path,
    config: ClimbPipelineConfig,
    output_base: Path,
    video_index: int,
    total_videos: int
) -> bool:
    """
    Process a single video through the complete pipeline.
    Returns True if successful, False if skipped/failed.
    """
    print("\n" + "="*70)
    print(f"PROCESSING VIDEO {video_index}/{total_videos}")
    print(f"File: {video_path.name}")
    print("="*70)
    
    # Step 1: Multi-color selection with preview
    print("\n[1/6] Hold Color Selection (Multi-Color with Preview)")
    selected_colors = multi_color_picker_with_preview(str(video_path))
    
    if selected_colors is None:
        print("⚠ Skipping video (no colors selected)")
        return False
    
    # Step 2: Get metadata
    print("\n[2/6] Video Metadata")
    metadata = get_video_metadata()
    
    # Create output directory
    video_output = output_base / f"video_{video_index:03d}_{video_path.stem}"
    video_output.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Output directory: {video_output}")
    
    # Step 3: Detect holds
    print("\n[3/6] Detecting Holds...")
    try:
        hold_ids, hold_positions = detect_holds_multi_color(
            str(video_path),
            selected_colors,
            video_output
        )
    except Exception as e:
        print(f"✗ Hold detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3.5: Split grouped holds
    if SPLITTER_AVAILABLE:
        print("\n[3.5/6] Checking for Grouped Holds...")
        try:
            split_path = split_grouped_holds(
                holds_json_path=str(video_output / "hold_positions_auto.json"),
                mask_path=str(video_output / "hold_mask_composite.jpg"),
                debug_image_path=str(video_output / "holds_debug.jpg"),
                output_dir=str(video_output),
                video_path=str(video_path)
            )
            
            # Use split version if created
            if split_path:
                holds_json = Path(split_path)
                with open(holds_json, 'r') as f:
                    hold_positions = json.load(f)
                print(f"✅ Using split holds: {len(hold_positions)} holds")
            else:
                holds_json = video_output / "hold_positions_auto.json"
        except Exception as e:
            print(f"⚠ Hold splitting failed: {e}")
            holds_json = video_output / "hold_positions_auto.json"
    else:
        holds_json = video_output / "hold_positions_auto.json"
    
    # Step 4: Detect and classify moves
    print("\n[4/7] Detecting Moves...")
    try:
        climb_data_path = detect_and_classify_moves(
            str(video_path),
            str(holds_json),
            output_dir=str(video_output)
        )
        print(f"✓ Detected moves, saved to {climb_data_path}")
    except Exception as e:
        print(f"✗ Move detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Classify holds
    print("\n[5/7] Classifying Holds...")
    try:
        # Check if classifier exists
        classifier_path = Path("move_classifier/hold_classifier.h5")
        if classifier_path.exists():
            # Run hold classification
            import sys
            old_argv = sys.argv
            sys.argv = [
                'enrich',
                '--video', str(video_path),
                '--holds', str(holds_json),
                '--output', str(video_output)
            ]
            
            try:
                from enrich_holds_with_classifier_multiframe import main as enrich_holds
                enrich_holds()
                
                # Use enriched holds if created
                enriched_holds_path = video_output / "hold_positions_enriched.json"
                if enriched_holds_path.exists():
                    holds_json = enriched_holds_path
                    print(f"✓ Holds classified and enriched")
                else:
                    print("⚠ Enrichment ran but no output - using original holds")
            finally:
                sys.argv = old_argv
        else:
            print("⚠ Hold classifier not found - skipping hold classification")
            print(f"   Looking for: {classifier_path.absolute()}")
    except Exception as e:
        print(f"⚠ Hold classification failed (non-critical): {e}")
        import traceback
        traceback.print_exc()
    
    # Step 6: Export to training format
    print("\n[6/7] Exporting Training Data...")
    try:
        # Load climb data
        with open(climb_data_path, 'r') as f:
            climb_data = json.load(f)
        
        # Convert to training format
        training_entry = convert_to_training_format(
            climb_data=climb_data,
            grade=metadata['grade'],
            climber_height=config.climber_height,
            climber_wingspan=config.climber_wingspan,
            climber_skill=config.climber_skill,
            wall_angle=metadata['wall_angle'],
            gym_name=metadata['gym_name'],
            video_path=str(video_path),
            notes=metadata['notes']
        )
        
        # Save to database
        db_path = Path("climb_database")
        save_training_record(training_entry, db_path)
        
        print(f"✓ Training data saved to {db_path}")
    except Exception as e:
        print(f"⚠ Training export failed (non-critical): {e}")
    
    # Step 7: Summary
    print("\n[7/7] Video Processing Complete!")
    print("="*70)
    print(f"✅ Output: {video_output}")
    print(f"✅ Grade: {metadata['grade']}")
    print(f"✅ Holds: {len(hold_positions)}")
    print("="*70)
    
    return True


def batch_process_videos(video_folder: str, output_folder: str = "batch_output"):
    """
    Process all videos in a folder through the complete pipeline.
    """
    video_folder = Path(video_folder)
    output_base = Path(output_folder)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Find all video files (case-insensitive, no duplicates)
    video_extensions = ['.mp4', '.mov', '.avi']
    video_files = []
    for ext in video_extensions:
        # Search for both lowercase and uppercase
        video_files.extend(video_folder.glob(f"*{ext}"))
        video_files.extend(video_folder.glob(f"*{ext.upper()}"))
    
    # Remove duplicates (in case filesystem is case-insensitive)
    video_files = list(set(video_files))
    video_files = sorted(video_files)
    
    # Print found videos for debugging
    print("\nFound videos:")
    for vf in video_files:
        print(f"  - {vf.name}")
    
    if len(video_files) == 0:
        print(f"❌ No video files found in {video_folder}")
        return
    
    print("\n" + "="*70)
    print("BATCH CLIMB PROCESSING PIPELINE v2")
    print("="*70)
    print(f"Found {len(video_files)} videos in {video_folder}")
    print(f"Output directory: {output_base}")
    print("="*70)
    
    # Setup climber info once
    config = ClimbPipelineConfig()
    config.setup_climber_info()
    
    # Process each video
    successful = 0
    skipped = 0
    
    for i, video_path in enumerate(video_files, 1):
        result = process_single_video(
            video_path,
            config,
            output_base,
            video_index=i,
            total_videos=len(video_files)
        )
        
        if result:
            successful += 1
        else:
            skipped += 1
    
    # Final summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE!")
    print("="*70)
    print(f"Total videos: {len(video_files)}")
    print(f"✅ Successful: {successful}")
    print(f"⚠ Skipped: {skipped}")
    print(f"\n📁 Output location: {output_base.absolute()}")
    print(f"📊 Training database: climb_database/")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process climbing videos (v2)")
    parser.add_argument("--videos", "-v", required=True,
                       help="Folder containing video files")
    parser.add_argument("--output", "-o", default="batch_output",
                       help="Output folder for processed data")
    
    args = parser.parse_args()
    
    batch_process_videos(args.videos, args.output)