# video_input_cli.py (updated for merged hold detection)

import json
import os
from hold_detection import detect_holds_from_video
from pose_estimation import detect_moves_and_visualize
from auto_climb_pipeline import extract_wall_frame

# Prompt user for basic metadata
user_input = {
    "climber_height": "5'10\"",
    "wingspan_inches": 70,
    "grade": "V4",
    "video_path": "Vids/climbVid.mov"
}

print(f"\n>>\nExtracting wall frame from {user_input['video_path']}...")
image_id = extract_wall_frame(user_input["video_path"])

# Let user click color for sampling
import cv2
import numpy as np

clicked_lab = None
clicked_hsv = None


def click_color(event, x, y, flags, param):
    global clicked_lab, clicked_hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        region = image[max(0, y-5):y+5, max(0, x-5):x+5]
        region_lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        clicked_lab = np.mean(region_lab.reshape(-1, 3), axis=0)
        clicked_hsv = np.mean(region_hsv.reshape(-1, 3), axis=0)
        print("\n🎯 Sampled Hold Color:")
        print("LAB:", clicked_lab.astype(int).tolist())
        print("HSV:", clicked_hsv.astype(int).tolist())
        cv2.destroyAllWindows()

image = cv2.imread(image_id)
cv2.imshow("Click a Hold to Sample Color", image)
cv2.setMouseCallback("Click a Hold to Sample Color", click_color)
print("\n🖱️  Click a hold in the image to sample its color...")
cv2.waitKey(0)

# Step 2: Detect holds from multiple frames
hold_ids, hold_positions = detect_holds_from_video(
    user_input["video_path"],
    ref_lab=clicked_lab,
    ref_hsv=clicked_hsv
)

# Step 3: Detect moves
moves = detect_moves_and_visualize(user_input["video_path"], hold_positions)

# Step 4: Final output data
output = {
    "image_id": image_id,
    "climber_height": user_input["climber_height"],
    "wingspan_inches": user_input["wingspan_inches"],
    "holds": hold_ids,
    "moves": moves,
    "grade": user_input["grade"]
}

os.makedirs("output", exist_ok=True)
with open("output/climb_data.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✅ Auto-analysis complete. Output saved to output/climb_data.json")
