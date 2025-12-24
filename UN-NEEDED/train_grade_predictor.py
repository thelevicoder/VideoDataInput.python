# train_grade_predictor.py
#
# Train a machine learning model to predict climbing route grades
# based on hold types, move sequences, and spatial features.

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def load_training_database(database_dir: Path = Path("climb_database")):
    """Load all training records from the database."""
    json_files = list(database_dir.glob("climb_*.json"))
    
    if not json_files:
        raise FileNotFoundError(
            f"No training data found in {database_dir}. "
            "Run export_training_data.py first!"
        )
    
    records = []
    for json_file in json_files:
        with json_file.open('r') as f:
            records.append(json.load(f))
    
    print(f"✅ Loaded {len(records)} climbing routes from database")
    return records


def prepare_dataset(records):
    """Convert training records into X (features) and y (labels)."""
    
    X_data = []
    y_data = []
    
    for record in records:
        features = record['features']
        grade = record['route_grade']
        
        # Convert features dict to list (in consistent order)
        feature_vector = [
            features['total_holds'],
            features['total_moves'],
            features['pct_crimp_or_foot'],
            features['pct_jug'],
            features['pct_pinch'],
            features['pct_pocket'],
            features['pct_sloper'],
            features['pct_volume'],
            features['pct_unknown'],
            features['pct_reach'],
            features['pct_heel_hook'],
            features['pct_toe_hook'],
            features['pct_flag'],
            features['pct_smear'],
            features['pct_figure_4'],
            features['pct_bat_hang'],
            features['avg_move_distance'],
            features['max_move_distance'],
            features['min_move_distance'],
            features['std_move_distance'],
            features['pct_long_reaches'],
            features['max_reach_ratio'],
            features['route_height'],
            features['route_width'],
            features['hold_density'],
            features['climber_height_inches'],
            features['climber_wingspan_inches'],
            record['wall_angle'],
        ]
        
        X_data.append(feature_vector)
        y_data.append(grade)
    
    # Feature names for interpretability
    feature_names = [
        'total_holds', 'total_moves',
        'pct_crimp_or_foot', 'pct_jug', 'pct_pinch', 'pct_pocket', 'pct_sloper', 'pct_volume', 'pct_unknown',
        'pct_reach', 'pct_heel_hook', 'pct_toe_hook', 'pct_flag', 'pct_smear', 'pct_figure_4', 'pct_bat_hang',
        'avg_move_distance', 'max_move_distance', 'min_move_distance', 'std_move_distance',
        'pct_long_reaches', 'max_reach_ratio',
        'route_height', 'route_width', 'hold_density',
        'climber_height_inches', 'climber_wingspan_inches', 'wall_angle'
    ]
    
    X = pd.DataFrame(X_data, columns=feature_names)
    y = pd.Series(y_data)
    
    return X, y, feature_names


def analyze_dataset(y):
    """Print dataset statistics."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    grade_counts = y.value_counts().sort_index()
    print("\nGrade Distribution:")
    for grade, count in grade_counts.items():
        print(f"  {grade}: {count} climbs ({count/len(y)*100:.1f}%)")
    
    print(f"\nTotal Routes: {len(y)}")
    print(f"Unique Grades: {y.nunique()}")
    
    # Check for imbalance
    if y.value_counts().min() < 5:
        print("\n⚠️  WARNING: Some grades have very few examples (<5).")
        print("   Consider collecting more data for balanced training.")
    
    print("="*60 + "\n")


def train_model(X, y, feature_names):
    """Train and evaluate the grade prediction model."""
    
    # Encode grades to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("Training set size:", len(X_train))
    print("Test set size:", len(X_test))
    print()
    
    # Try multiple models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    best_model = None
    best_score = -1
    best_name = None
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)//2))
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        if test_score > best_score:
            best_score = test_score
            best_model = model
            best_name = name
        
        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            for idx, row in importances.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name} (Test Accuracy: {best_score:.3f})")
    print(f"{'='*60}\n")
    
    # Detailed evaluation of best model
    y_pred = best_model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cmap='Blues'
    )
    plt.title(f'{best_name} - Confusion Matrix')
    plt.ylabel('True Grade')
    plt.xlabel('Predicted Grade')
    plt.tight_layout()
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    print(f"\n✅ Saved confusion matrix to output/confusion_matrix.png")
    plt.close()
    
    return best_model, label_encoder


def save_model(model, label_encoder, feature_names):
    """Save the trained model and metadata."""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / 'grade_predictor.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Saved model to {model_path}")
    
    # Save label encoder
    encoder_path = models_dir / 'grade_label_encoder.pkl'
    joblib.dump(label_encoder, encoder_path)
    print(f"✅ Saved label encoder to {encoder_path}")
    
    # Save feature names
    metadata = {
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'grades': label_encoder.classes_.tolist()
    }
    metadata_path = models_dir / 'grade_predictor_metadata.json'
    with metadata_path.open('w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to {metadata_path}")


def main():
    print("\n" + "="*60)
    print("CLIMBING ROUTE GRADE PREDICTOR - TRAINING")
    print("="*60 + "\n")
    
    # Load training data
    try:
        records = load_training_database()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Check minimum data requirement
    if len(records) < 20:
        print(f"⚠️  WARNING: You only have {len(records)} training examples.")
        print("   Recommended: 50+ for decent accuracy, 200+ for good accuracy")
        response = input("\nContinue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting. Collect more data and try again.")
            return
    
    # Prepare dataset
    X, y, feature_names = prepare_dataset(records)
    
    # Analyze dataset
    analyze_dataset(y)
    
    # Train model
    model, label_encoder = train_model(X, y, feature_names)
    
    # Save model
    save_model(model, label_encoder, feature_names)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\nYou can now predict grades with:")
    print("  python predict_route_grade.py --image wall.jpg --color 0,255,255")
    print("\nTo improve accuracy:")
    print("  - Collect more training data (aim for 200+ routes)")
    print("  - Ensure balanced grade distribution")
    print("  - Include variety of walls, holds, and climbers")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()