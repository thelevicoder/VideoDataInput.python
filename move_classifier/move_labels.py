# move_labels.py

# Canonical labels (lowercase, snake_case)
CLASS_NAMES = [
    "reach",
    "heel_hook",
    "toe_hook",
    "flag",
    "smear",
    "figure_4",
    "bat_hang",
]

# Map from your folder names (exact) to canonical labels
FOLDER_TO_LABEL = {
    "Reach": "reach",
    "Heel_Hook": "heel_hook",
    "Toe_Hook": "toe_hook",
    "Flag": "flag",
    "Smear": "smear",
    "Figure4": "figure_4",
    "Bat_Hang": "bat_hang",
}

# Integer index for each class (used by the model)
LABEL_TO_INDEX = {label: i for i, label in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL = {i: label for label, i in LABEL_TO_INDEX.items()}
