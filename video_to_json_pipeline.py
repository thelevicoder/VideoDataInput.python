# video_to_json_pipeline.py

"""
High-level execution script for the ClimbIQ video-to-JSON pipeline.
User uploads a climbing video and provides height, wingspan, and grade.
System extracts wall image, detects holds, tracks movement, and outputs climb data.
"""

from auto_climb_pipeline import generate_climb_data
from hold_detection import detect_holds
import numpy as np

# Dummy placeholder: in a real UI this would come from the user
user_input = {
    "video_path": "vids/climbVid.mov",
    "climber_height": "5'10\"",
    "wingspan_inches": 70,
    "grade": "V4"
}

# Dummy color references for hold detection
reference_lab = np.array([145, 135, 130])
reference_hsv = np.array([30, 150, 120])

# --- Replace hold detection in pipeline with real model ---
image_id = "wall_example.jpg"
holds = detect_holds(image_id, reference_lab, reference_hsv)

# --- Inject detected holds into climb data ---
from auto_climb_pipeline import detect_moves
moves = detect_moves(user_input["video_path"], holds)

climb_data = {
    "image_id": image_id,
    "climber_height": user_input["climber_height"],
    "wingspan_inches": user_input["wingspan_inches"],
    "holds": holds,
    "moves": moves,
    "grade": user_input["grade"]
}

import json
with open("output/climb_data.json", "w") as f:
    json.dump(climb_data, f, indent=2)

print("Pipeline finished. Data exported to output/climb_data.json")
