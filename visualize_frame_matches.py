# visualize_frame_matches.py (interactive viewer)

"""
Visualize each move step-by-step, overlaying limb placements and holds
on the wall image. Use arrow keys to step through each move.
"""

import cv2
import json
import numpy as np

# Load image and data
image_path = "wall_example.jpg"
json_path = "output/climb_data.json"
hold_coords_path = "output/hold_positions.json"

image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Couldn't load image: {image_path}")

with open(json_path, "r") as f:
    data = json.load(f)

with open(hold_coords_path, "r") as f:
    hold_coords = {k: tuple(v) for k, v in json.load(f).items()}

moves = data["moves"]

# Colors
hold_color = (0, 255, 0)
text_color = (255, 255, 255)
limb_colors = {
    "lefthand": (0, 0, 255),
    "righthand": (0, 255, 255),
    "leftfoot": (255, 0, 0),
    "rightfoot": (255, 0, 255)
}

index = 0

while True:
    display = image.copy()

    # Draw holds
    for hold_id, (x, y) in hold_coords.items():
        cv2.circle(display, (x, y), 15, hold_color, 2)
        cv2.putText(display, hold_id, (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Draw current move
    if 0 <= index < len(moves):
        move = moves[index]
        for limb, hold_id in move.items():
            if limb == "type":
                continue
            if hold_id != "hanging" and hold_id in hold_coords:
                x, y = hold_coords[hold_id]
                color = limb_colors.get(limb, (255, 255, 255))
                cv2.circle(display, (x, y), 6, color, -1)
                cv2.putText(display, f"{limb}", (x + 6, y - 6), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1)

        cv2.putText(display, f"Move {index + 1}/{len(moves)}: {move['type']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    cv2.imshow("ClimbIQ Move Viewer", display)
    key = cv2.waitKey(0)

    if key == 27:  # ESC
        break
    elif key == 81 or key == ord('a'):  # left arrow or 'a'
        index = max(0, index - 1)
    elif key == 83 or key == ord('d'):  # right arrow or 'd'
        index = min(len(moves) - 1, index + 1)

cv2.destroyAllWindows()