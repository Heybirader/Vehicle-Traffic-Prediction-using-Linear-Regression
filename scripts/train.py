#!/usr/bin/env python
"""Main training script - orchestrates the entire ML pipeline.

Usage:
    python scripts/train.py
"""
import sys
from pathlib import Path

# Add project root to path so we can import config and src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from src.preprocessing import load_data, extract_time_features, prepare_training_data, print_data_info
from src.model import train_model, evaluate_model, print_model_performance, print_model_coefficients
from src.predict import generate_future_features, make_predictions, print_predictions, print_prediction_summary


def main():
    """Run the complete ML pipeline."""
    print("\n" + "=" * 70)
    print("VEHICLE TRAFFIC PREDICTION - LINEAR REGRESSION")
    print("=" * 70)
    
    # 1. Load and explore data
    print(f"\nLoading data from: {config.RAW_DATA_FILE}")
    data = load_data(config.RAW_DATA_FILE)
    print_data_info(data)
    
    # 2. Extract time-based features
    print("\nExtracting time-based features...")
    data = extract_time_features(data)
    print(f"Features extracted: {', '.join(config.FEATURES)}")
    print(f"\nSample of extracted features:")
    print(data[['DateTime', 'Vehicles'] + config.FEATURES[:4]].head(10))
    
    # 3. Prepare training data
    print("\nPreparing training data...")
    X, y = prepare_training_data(data, config.FEATURES)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # 4. Train model
    print(f"\nTraining Linear Regression model...")
    print(f"  - Train/test split: {(1-config.TRAIN_TEST_SPLIT)*100:.0f}% / {config.TRAIN_TEST_SPLIT*100:.0f}%")
    print(f"  - Random state: {config.RANDOM_STATE}")
    
    result = train_model(
        X, y,
        test_size=config.TRAIN_TEST_SPLIT,
        random_state=config.RANDOM_STATE
    )
    model = result['model']
    X_train = result['X_train']
    X_test = result['X_test']
    y_train = result['y_train']
    y_test = result['y_test']
    
    # 5. Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    print_model_performance(metrics)
    print_model_coefficients(model, config.FEATURES)
    
    # 6. Make future predictions
    print("\nGenerating future features for prediction...")
    last_date = data['DateTime'].max()
    min_date = data['DateTime'].min()
    
    future_features = generate_future_features(
        last_date,
        config.FORECAST_HOURS,
        min_date,
        config.FEATURES
    )
    
    predictions = make_predictions(model, future_features)
    
    # 7. Print results
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(hours=1),
        periods=config.FORECAST_HOURS,
        freq='H'
    )
    print_predictions(future_dates, predictions)
    print_prediction_summary(data['Vehicles'], predictions)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    import pandas as pd  # Import here for use in main
    main()
