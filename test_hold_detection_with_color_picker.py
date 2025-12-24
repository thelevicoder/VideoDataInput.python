#!/usr/bin/env python3
# test_hold_detection_with_color_picker.py
#
# Test hold detection with manual color selection

import cv2
import numpy as np
import sys
import subprocess
from pathlib import Path

# Import the hold splitter
try:
    from split_grouped_holds import split_grouped_holds
    SPLITTER_AVAILABLE = True
except ImportError:
    SPLITTER_AVAILABLE = False
    print("⚠️  split_grouped_holds.py not found - hold splitting disabled")

def pick_color_from_video(video_path: str):
    """
    Interactive color picker with PREVIEW - see detected holds after each color selection.
    """
    print("\n" + "="*70)
    print("HOLD COLOR SELECTION WITH PREVIEW")
    print("="*70)
    print("Loading video frame...")
    
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
    print("  • Click on a hold to add its color")
    print("  • After each click, you'll see a PREVIEW of detected holds")
    print("  • Press 'c' to CONFIRM and accept the current detection")
    print("  • Press 'r' to RESET and start over")
    print("  • Press 'q' to QUIT")
    print("\n")
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Calculate display size (max 1280 wide, maintain aspect ratio)
    max_width = 1280
    if w > max_width:
        scale = max_width / w
        display_w = max_width
        display_h = int(h * scale)
    else:
        display_w = w
        display_h = h
    
    display = frame.copy()
    selected_colors = []  # List of (lab, hsv, bgr) tuples
    click_positions = []  # Track where user clicked
    preview_mode = False
    preview_image = None
    
    def show_preview():
        """Generate and show preview of detected holds with current colors."""
        nonlocal preview_image, preview_mode
        
        if len(selected_colors) == 0:
            return
        
        print("\n🔍 Generating preview...")
        preview_mode = True
        
        # Import detection functions
        from hold_detection import sample_frames_from_video, create_composite_mask, filter_and_group_contours
        
        # Sample frames quickly (fewer frames for preview)
        frames = sample_frames_from_video(video_path, 5)
        
        # Combine masks from all colors
        combined_composite = np.zeros(frames[0].shape[:2], dtype=np.uint8)
        
        for i, (lab, hsv, _) in enumerate(selected_colors):
            is_red = (hsv[0] < 15 or hsv[0] > 165)
            composite = create_composite_mask(frames, lab, hsv, is_red, None)
            combined_composite = cv2.bitwise_or(combined_composite, composite)
        
        # Detect holds
        hold_ids, hold_positions, vis = filter_and_group_contours(combined_composite, frame)
        
        # Add info overlay
        preview_image = vis.copy()
        
        # Show color swatches
        swatch_size = 60
        for i, (_, _, bgr) in enumerate(selected_colors):
            x_offset = 20 + (i * (swatch_size + 10))
            y_offset = 20
            
            cv2.rectangle(preview_image, (x_offset, y_offset), 
                         (x_offset + swatch_size, y_offset + swatch_size),
                         bgr.tolist(), -1)
            cv2.rectangle(preview_image, (x_offset, y_offset), 
                         (x_offset + swatch_size, y_offset + swatch_size),
                         (255, 255, 255), 2)
        
        # Info text
        cv2.rectangle(preview_image, (10, h-100), (w-10, h-10), (0, 0, 0), -1)
        cv2.putText(preview_image, f"PREVIEW: {len(hold_positions)} holds detected", 
                   (20, h-70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(preview_image, f"{len(selected_colors)} color(s) | 'c'=confirm | click=add color | 'r'=reset", 
                   (20, h-35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Select Hold Color", preview_image)
        print(f"✅ Preview: {len(hold_positions)} holds detected")
        print("   Click to add more colors, 'c' to confirm, or 'r' to reset")
    
    def on_mouse(event, x, y, flags, param):
        nonlocal selected_colors, display, click_positions, preview_mode
        
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
            click_positions.append((x, y))
            
            print(f"✓ Color {len(selected_colors)} added - LAB: {lab.astype(int)}, HSV: {hsv.astype(int)}")
            
            # Show preview immediately
            show_preview()
    
    cv2.namedWindow("Select Hold Color", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Hold Color", display_w, display_h)
    cv2.setMouseCallback("Select Hold Color", on_mouse)
    cv2.imshow("Select Hold Color", display)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') and len(selected_colors) > 0:
            cv2.destroyAllWindows()
            print(f"\n✅ Confirmed! {len(selected_colors)} color(s) selected")
            for i, (lab, hsv, _) in enumerate(selected_colors):
                print(f"   Color {i+1}: LAB {lab.astype(int)}, HSV {hsv.astype(int)}")
            return selected_colors
        
        elif key == ord('r'):
            display = frame.copy()
            selected_colors = []
            click_positions = []
            preview_mode = False
            cv2.imshow("Select Hold Color", display)
            print("🔄 Reset - click to select colors")
        
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("\n❌ Cancelled")
            return None
    
    cv2.destroyAllWindows()
    return None


def run_hold_detection_with_color(video_path: str, selected_colors):
    """
    Run hold_detection.py with multiple colors by creating a temp script.
    """
    # Build LAB and HSV arrays from all selected colors
    lab_values = ", ".join([f"[{c[0][0]}, {c[0][1]}, {c[0][2]}]" for c in selected_colors])
    hsv_values = ", ".join([f"[{c[1][0]}, {c[1][1]}, {c[1][2]}]" for c in selected_colors])
    
    # Create a temporary Python script to run detection
    temp_script = f"""
import sys
import numpy as np
import cv2
from hold_detection import sample_frames_from_video, create_composite_mask, filter_and_group_contours
from pathlib import Path

video_path = sys.argv[1]

# All selected colors
lab_colors = [{lab_values}]
hsv_colors = [{hsv_values}]

print("="*70)
print("HOLD DETECTOR (Multiple Colors)")
print("="*70)
print(f"Detecting {{len(lab_colors)}} color(s):")
for i, (lab, hsv) in enumerate(zip(lab_colors, hsv_colors)):
    lab_arr = np.array(lab, dtype=np.float32)
    hsv_arr = np.array(hsv, dtype=np.float32)
    is_red = (hsv_arr[0] < 15 or hsv_arr[0] > 165)
    print(f"  Color {{i+1}}: LAB {{lab}}, HSV {{hsv}}" + (" (RED)" if is_red else ""))
print("="*70)

# Sample frames
frames = sample_frames_from_video(video_path, 10)
reference_frame = frames[len(frames) // 2]

# Create output directory
output_path = Path("output")
output_path.mkdir(parents=True, exist_ok=True)
debug_dir = output_path / "debug_hold_detection"
debug_dir.mkdir(exist_ok=True)

# Combine masks from all colors
combined_composite = np.zeros(frames[0].shape[:2], dtype=np.uint8)

for i, (lab, hsv) in enumerate(zip(lab_colors, hsv_colors)):
    lab_arr = np.array(lab, dtype=np.float32)
    hsv_arr = np.array(hsv, dtype=np.float32)
    is_red = (hsv_arr[0] < 15 or hsv_arr[0] > 165)
    
    print(f"Processing color {{i+1}}...")
    
    # Create composite for this color
    from hold_detection import create_composite_mask
    composite = create_composite_mask(frames, lab_arr, hsv_arr, is_red, debug_dir)
    
    # Combine with overall mask
    combined_composite = cv2.bitwise_or(combined_composite, composite)

print("[hold_detection] Combined all color masks")

# Detect holds from combined mask
hold_ids, hold_positions, vis = filter_and_group_contours(combined_composite, reference_frame)

# Save outputs
import json
cv2.imwrite(str(output_path / "hold_mask_composite.jpg"), combined_composite)
cv2.imwrite(str(output_path / "holds_debug.jpg"), vis)

holds_json = output_path / "hold_positions_auto.json"
with open(holds_json, "w") as f:
    json.dump(hold_positions, f, indent=2)

print("="*70)
print(f"✓ DETECTED {{len(hold_positions)}} HOLDS")
print("="*70)
print(f"Saved to: {{holds_json}}")
print(f"Visualization: {{output_path / 'holds_debug.jpg'}}")
"""
    
    # Save temp script
    temp_file = Path("temp_detect_multi.py")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(temp_script)
    
    try:
        # Run detection
        print("\n" + "="*70)
        print("RUNNING HOLD DETECTION")
        print("="*70)
        
        result = subprocess.run(
            [sys.executable, "temp_detect_multi.py", video_path],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n" + "="*70)
            print("✅ DETECTION DONE!")
            print("="*70)
            
            # Run hold splitter
            if SPLITTER_AVAILABLE:
                print("\n🔍 Checking for grouped holds...")
                try:
                    split_path = split_grouped_holds(
                        holds_json_path="output/hold_positions_auto.json",
                        mask_path="output/hold_mask_composite.jpg",
                        debug_image_path="output/holds_debug.jpg",
                        output_dir="output",
                        video_path=video_path  # Pass video path for clean frame
                    )
                    
                    if split_path:
                        print("\n" + "="*70)
                        print("📦 CHECK THESE FILES:")
                        print("="*70)
                        print("  📸 output/holds_debug_split.jpg  - FINAL holds (after splitting)")
                        print("  📄 output/hold_positions_auto_split.json - FINAL positions")
                        print("\n  Original (before split):")
                        print("  📸 output/holds_debug.jpg")
                        print("  📄 output/hold_positions_auto.json")
                        print("="*70)
                    else:
                        print("\n" + "="*70)
                        print("📦 CHECK THESE FILES:")
                        print("="*70)
                        print("  📸 output/holds_debug.jpg          - Visual with detected holds")
                        print("  📄 output/hold_positions_auto.json - Hold coordinates")
                        print("  🔍 output/debug_hold_detection/    - All intermediate steps")
                        print("="*70)
                
                except Exception as e:
                    print(f"\n⚠️  Hold splitting failed: {e}")
                    print("\n" + "="*70)
                    print("📦 CHECK THESE FILES (no splitting):")
                    print("="*70)
                    print("  📸 output/holds_debug.jpg          - Visual with detected holds")
                    print("  📄 output/hold_positions_auto.json - Hold coordinates")
                    print("="*70)
            else:
                print("\n" + "="*70)
                print("📦 CHECK THESE FILES:")
                print("="*70)
                print("  📸 output/holds_debug.jpg          - Visual with detected holds")
                print("  📄 output/hold_positions_auto.json - Hold coordinates")
                print("  🔍 output/debug_hold_detection/    - All intermediate steps")
                print("="*70)
        else:
            print("\n❌ Hold detection failed!")
    
    finally:
        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_hold_detection_with_color_picker.py <video_path>")
        print("Example: python test_hold_detection_with_color_picker.py Vids/climbVid3.mov")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    print("\n" + "="*70)
    print("HOLD DETECTION WITH MULTI-COLOR PICKER")
    print("="*70)
    
    # Step 1: Pick colors (can select multiple!)
    selected_colors = pick_color_from_video(video_path)
    
    if selected_colors is None:
        print("\n❌ No colors selected. Exiting.")
        sys.exit(1)
    
    # Step 2: Run detection with all selected colors
    run_hold_detection_with_color(video_path, selected_colors)


if __name__ == "__main__":
    main()