# sample_hold_color.py

"""
Interactive color picker to extract reference LAB and HSV values
from a clicked point on the wall image. Use this to detect
the color of specific climbing holds.
"""

import cv2
import numpy as np

image_path = "wall_example.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

region_size = 10  # region radius for averaging


def filter_outliers(region):
    region_reshaped = region.reshape(-1, 3)
    mean_color = np.mean(region_reshaped, axis=0)
    diff = np.linalg.norm(region_reshaped - mean_color, axis=1)
    filtered_pixels = region_reshaped[diff < np.std(diff) * 2]
    return np.mean(filtered_pixels, axis=0) if filtered_pixels.size > 0 else mean_color


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        min_x, max_x = max(0, x - region_size), min(image.shape[1], x + region_size)
        min_y, max_y = max(0, y - region_size), min(image.shape[0], y + region_size)

        region_lab = lab_image[min_y:max_y, min_x:max_x]
        region_hsv = hsv_image[min_y:max_y, min_x:max_x]

        filtered_lab = filter_outliers(region_lab)
        filtered_hsv = filter_outliers(region_hsv)

        print("\n🎯 Sampled Hold Color:")
        print("LAB:", np.round(filtered_lab).astype(int).tolist())
        print("HSV:", np.round(filtered_hsv).astype(int).tolist())
        print("Paste these into video_input_cli.py for reference_lab and reference_hsv")

cv2.imshow("Click a hold to sample color", image)
cv2.setMouseCallback("Click a hold to sample color", on_mouse)
cv2.waitKey(0)
cv2.destroyAllWindows()
