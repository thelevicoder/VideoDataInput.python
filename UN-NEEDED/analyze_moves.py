import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from move_classifier.move_labels import CLASS_NAMES

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    climb_json_path = OUTPUT_DIR / "climb_data.json"
    if not climb_json_path.exists():
        raise FileNotFoundError(f"Could not find {climb_json_path}. Run the pipeline first.")

    with open(climb_json_path, "r") as f:
        climb_data = json.load(f)

    moves = climb_data.get("moves", [])

    if not moves:
        print("No moves found in climb_data.json")
        return

    # Build rows for table
    move_rows = []
    probs_list = []

    for i, move in enumerate(moves):
        assigns = move.get("assignments", {})
        move_type = move.get("type", "unknown")
        probs = move.get("move_probs", None)

        row = {
            "Move": f"Move {i+1:02d}",
            "PredictedMoveType": move_type,
            "lefthand": assigns.get("lefthand"),
            "righthand": assigns.get("righthand"),
            "leftfoot": assigns.get("leftfoot"),
            "rightfoot": assigns.get("rightfoot"),
        }
        move_rows.append(row)

        if probs is not None:
            probs_list.append(probs)

    df = pd.DataFrame(move_rows)

    # Show moves table
    print("\n=== Detected Moves ===\n")
    print(df)

    # Save CSV of moves
    move_csv_path = OUTPUT_DIR / "move_analysis.csv"
    df.to_csv(move_csv_path, index=False)
    print(f"\nSaved move_analysis.csv to {move_csv_path}")

    # 1) Plot and save counts per move type
    counts = df["PredictedMoveType"].value_counts().sort_index()
    print("\n=== Move Type Counts ===\n")
    print(counts)

    plt.figure()
    counts.plot(kind="bar")
    plt.xlabel("Move type")
    plt.ylabel("Count")
    plt.title("Detected move types")
    plt.tight_layout()

    counts_plot_path = OUTPUT_DIR / "move_counts.png"
    plt.savefig(counts_plot_path)
    plt.close()
    print(f"\nSaved move_counts.png to {counts_plot_path}")

    # 2) Optional: average confidence per class if probabilities were stored
    if probs_list:
        probs_arr = np.array(probs_list)  # shape: (num_moves_with_probs, num_classes)
        if probs_arr.shape[1] == len(CLASS_NAMES):
            mean_conf = probs_arr.mean(axis=0)
            conf_df = pd.DataFrame({
                "move_type": CLASS_NAMES,
                "mean_confidence": mean_conf,
            })
            conf_csv_path = OUTPUT_DIR / "move_confidence_summary.csv"
            conf_df.to_csv(conf_csv_path, index=False)

            print("\n=== Mean confidence per move type ===\n")
            print(conf_df)
            print(f"\nSaved confidence summary to {conf_csv_path}")
        else:
            print(
                f"\nWarning: move_probs length {probs_arr.shape[1]} does not match "
                f"CLASS_NAMES length {len(CLASS_NAMES)}. Skipping confidence summary."
            )
    else:
        print("\nNo move_probs found in moves. Skipping confidence summary.")


if __name__ == "__main__":
    main()
