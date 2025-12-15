# export_training_data.py
#
# Convert your climb_data.json and enriched holds into training format
# for the grade prediction model.

from pathlib import Path
import json
from typing import Dict, List
import numpy as np


def load_climb_data(output_dir: Path = Path("output")):
    """Load all pipeline outputs."""
    climb_data_path = output_dir / "climb_data.json"
    enriched_holds_path = output_dir / "hold_positions_enriched.json"
    
    if not climb_data_path.exists():
        raise FileNotFoundError(f"climb_data.json not found at {climb_data_path}")
    if not enriched_holds_path.exists():
        raise FileNotFoundError(f"hold_positions_enriched.json not found at {enriched_holds_path}")
    
    with climb_data_path.open("r") as f:
        climb_data = json.load(f)
    
    with enriched_holds_path.open("r") as f:
        enriched_holds = json.load(f)
    
    return climb_data, enriched_holds


def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def extract_route_features(holds: Dict, moves: List, climber_height: float, climber_wingspan: float):
    """Extract ML features from route data."""
    
    if not moves:
        return None
    
    total_holds = len(holds)
    total_moves = len(moves)
    
    # Count hold types
    hold_type_counts = {}
    for hold_id, hold_data in holds.items():
        hold_type = hold_data.get('class', 'unknown')
        hold_type_counts[hold_type] = hold_type_counts.get(hold_type, 0) + 1
    
    # Count move types
    move_type_counts = {}
    for move in moves:
        move_type = move.get('type', 'unknown')
        move_type_counts[move_type] = move_type_counts.get(move_type, 0) + 1
    
    # Calculate distance metrics
    distances = []
    for move in moves:
        from_hold = move.get('from_hold')
        to_hold = move.get('to_hold')
        
        if from_hold in holds and to_hold in holds:
            pos1 = holds[from_hold]['center']
            pos2 = holds[to_hold]['center']
            dist = calculate_distance(pos1, pos2)
            distances.append(dist)
    
    # Normalize distances by climber size (use height as body unit)
    body_unit = climber_height if climber_height > 0 else 70  # default 70 inches
    normalized_distances = [d / body_unit for d in distances] if distances else [0]
    
    # Get hold positions for spatial metrics
    hold_positions = [h['center'] for h in holds.values()]
    if hold_positions:
        xs = [p[0] for p in hold_positions]
        ys = [p[1] for p in hold_positions]
        route_width = max(xs) - min(xs)
        route_height = max(ys) - min(ys)
        hold_density = total_holds / (route_width * route_height) if route_width * route_height > 0 else 0
    else:
        route_width = 0
        route_height = 0
        hold_density = 0
    
    features = {
        # Basic metrics
        'total_holds': total_holds,
        'total_moves': total_moves,
        
        # Hold distribution (percentages)
        'pct_crimp_or_foot': hold_type_counts.get('crimp_or_foot', 0) / total_holds if total_holds > 0 else 0,
        'pct_jug': hold_type_counts.get('jug', 0) / total_holds if total_holds > 0 else 0,
        'pct_pinch': hold_type_counts.get('pinch', 0) / total_holds if total_holds > 0 else 0,
        'pct_pocket': hold_type_counts.get('pocket', 0) / total_holds if total_holds > 0 else 0,
        'pct_sloper': hold_type_counts.get('sloper', 0) / total_holds if total_holds > 0 else 0,
        'pct_volume': hold_type_counts.get('volume', 0) / total_holds if total_holds > 0 else 0,
        'pct_unknown': hold_type_counts.get('unknown', 0) / total_holds if total_holds > 0 else 0,
        
        # Move distribution (percentages)
        'pct_reach': move_type_counts.get('reach', 0) / total_moves if total_moves > 0 else 0,
        'pct_heel_hook': move_type_counts.get('heel_hook', 0) / total_moves if total_moves > 0 else 0,
        'pct_toe_hook': move_type_counts.get('toe_hook', 0) / total_moves if total_moves > 0 else 0,
        'pct_flag': move_type_counts.get('flag', 0) / total_moves if total_moves > 0 else 0,
        'pct_smear': move_type_counts.get('smear', 0) / total_moves if total_moves > 0 else 0,
        'pct_figure_4': move_type_counts.get('figure_4', 0) / total_moves if total_moves > 0 else 0,
        'pct_bat_hang': move_type_counts.get('bat_hang', 0) / total_moves if total_moves > 0 else 0,
        
        # Distance metrics (normalized by body size)
        'avg_move_distance': float(np.mean(normalized_distances)),
        'max_move_distance': float(np.max(normalized_distances)),
        'min_move_distance': float(np.min(normalized_distances)),
        'std_move_distance': float(np.std(normalized_distances)),
        
        # Reach difficulty
        'pct_long_reaches': sum(1 for d in normalized_distances if d > 2.0) / len(normalized_distances) if normalized_distances else 0,
        'max_reach_ratio': float(np.max(normalized_distances)) / (climber_wingspan / body_unit) if climber_wingspan > 0 else 0,
        
        # Spatial characteristics
        'route_height': route_height,
        'route_width': route_width,
        'hold_density': hold_density,
        
        # Climber characteristics
        'climber_height_inches': climber_height,
        'climber_wingspan_inches': climber_wingspan,
    }
    
    return features


def convert_to_training_format(
    climb_data: Dict,
    enriched_holds: Dict,
    route_grade: str,
    climber_height: float = 70,
    climber_wingspan: float = 70,
    wall_angle: float = 0,
    gym_name: str = "Unknown Gym",
    additional_metadata: Dict = None
):
    """
    Convert pipeline output to training format.
    
    Args:
        climb_data: Output from move_detector.py
        enriched_holds: Output from enrich_holds_with_classifier.py
        route_grade: The actual grade (e.g., "V4") - MUST BE PROVIDED!
        climber_height: Climber height in inches
        climber_wingspan: Climber wingspan in inches
        wall_angle: Wall overhang in degrees (0 = vertical, positive = overhang)
        gym_name: Name of the gym
        additional_metadata: Any other info you want to store
    """
    
    moves = climb_data.get('moves', [])
    
    # Extract features
    features = extract_route_features(
        enriched_holds,
        moves,
        climber_height,
        climber_wingspan
    )
    
    if features is None:
        raise ValueError("Could not extract features - no moves found")
    
    # Build training record
    training_record = {
        'route_grade': route_grade,
        'gym_name': gym_name,
        'wall_angle': wall_angle,
        'climber_height': climber_height,
        'climber_wingspan': climber_wingspan,
        
        # Raw data
        'holds': enriched_holds,
        'moves': moves,
        
        # Extracted features for ML
        'features': features,
        
        # Source files
        'video_path': climb_data.get('video_path', ''),
        'fps': climb_data.get('fps', 30),
        
        # Additional metadata
        'metadata': additional_metadata or {}
    }
    
    return training_record


def save_training_record(training_record: Dict, output_path: Path):
    """Save training record to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w') as f:
        json.dump(training_record, f, indent=2)
    
    print(f"✅ Saved training record to {output_path}")


def main():
    """
    Interactive script to export current climb to training database.
    """
    print("\n" + "="*60)
    print("EXPORT CLIMB TO TRAINING DATABASE")
    print("="*60 + "\n")
    
    # Load pipeline outputs
    try:
        climb_data, enriched_holds = load_climb_data()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you've run:")
        print("  1. python run_climb_pipeline.py --video YOUR_VIDEO.mov")
        print("  2. python enrich_holds_with_classifier_multiframe.py")
        return
    
    print(f"✅ Loaded climb data with {len(enriched_holds)} holds and {len(climb_data['moves'])} moves\n")
    
    # Collect metadata from user
    print("Please provide the following information:\n")
    
    route_grade = input("Route Grade (e.g., V0, V2, V4, V7): ").strip().upper()
    if not route_grade:
        print("❌ Route grade is required!")
        return
    
    climber_height = input("Climber Height (inches, default 70): ").strip()
    climber_height = float(climber_height) if climber_height else 70.0
    
    climber_wingspan = input("Climber Wingspan (inches, default 70): ").strip()
    climber_wingspan = float(climber_wingspan) if climber_wingspan else 70.0
    
    wall_angle = input("Wall Angle (degrees, 0=vertical, positive=overhang, default 0): ").strip()
    wall_angle = float(wall_angle) if wall_angle else 0.0
    
    gym_name = input("Gym Name (optional): ").strip()
    gym_name = gym_name if gym_name else "Unknown Gym"
    
    climber_skill = input("Climber Skill Level (beginner/intermediate/advanced, optional): ").strip().lower()
    
    notes = input("Any notes about this climb (optional): ").strip()
    
    # Build metadata
    additional_metadata = {}
    if climber_skill:
        additional_metadata['climber_skill_level'] = climber_skill
    if notes:
        additional_metadata['notes'] = notes
    
    # Convert to training format
    try:
        training_record = convert_to_training_format(
            climb_data,
            enriched_holds,
            route_grade,
            climber_height,
            climber_wingspan,
            wall_angle,
            gym_name,
            additional_metadata
        )
    except Exception as e:
        print(f"❌ Error creating training record: {e}")
        return
    
    # Generate filename
    database_dir = Path("climb_database")
    database_dir.mkdir(exist_ok=True)
    
    # Count existing files to generate ID
    existing_files = list(database_dir.glob("climb_*.json"))
    next_id = len(existing_files) + 1
    
    output_path = database_dir / f"climb_{next_id:03d}_{route_grade}.json"
    
    # Save
    save_training_record(training_record, output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING RECORD SUMMARY")
    print("="*60)
    print(f"Route Grade: {route_grade}")
    print(f"Total Holds: {training_record['features']['total_holds']}")
    print(f"Total Moves: {training_record['features']['total_moves']}")
    print(f"Avg Move Distance: {training_record['features']['avg_move_distance']:.2f} body units")
    print(f"Max Reach: {training_record['features']['max_move_distance']:.2f} body units")
    print(f"\nHold Distribution:")
    print(f"  Jugs: {training_record['features']['pct_jug']*100:.1f}%")
    print(f"  Slopers: {training_record['features']['pct_sloper']*100:.1f}%")
    print(f"  Crimps: {training_record['features']['pct_crimp_or_foot']*100:.1f}%")
    print(f"  Volumes: {training_record['features']['pct_volume']*100:.1f}%")
    print(f"\nMove Distribution:")
    print(f"  Reaches: {training_record['features']['pct_reach']*100:.1f}%")
    print(f"  Heel Hooks: {training_record['features']['pct_heel_hook']*100:.1f}%")
    print(f"  Technical: {training_record['features']['pct_smear']*100:.1f}%")
    print("="*60 + "\n")
    
    print(f"📊 Training database now has {len(existing_files) + 1} climbs")
    print(f"\nNext steps:")
    print(f"  1. Process more videos and export them")
    print(f"  2. Once you have 50+ climbs, run: python train_grade_predictor.py")


if __name__ == "__main__":
    main()