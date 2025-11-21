import json
import pandas as pd

# Load the climb data
with open("output/climb_data.json", "r") as f:
    climb_data = json.load(f)

# Extract move info
moves = climb_data["moves"]

# Build a dataframe
move_data = []
for i, move in enumerate(moves):
    move_data.append({
        "Move": f"Move {i+1:02d}",
        "lefthand": move["lefthand"],
        "righthand": move["righthand"],
        "leftfoot": move["leftfoot"],
        "rightfoot": move["rightfoot"]
    })

df = pd.DataFrame(move_data)

# Display in terminal
print(df)

# Optional: Save as CSV
df.to_csv("output/move_analysis.csv", index=False)
print("\n✅ Saved move analysis to output/move_analysis.csv")
