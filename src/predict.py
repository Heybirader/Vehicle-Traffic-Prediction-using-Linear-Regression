"""Prediction utilities for future traffic forecasting."""
import pandas as pd
import numpy as np


def generate_future_features(last_date, forecast_hours, min_date, features):
    """Generate features for future prediction.
    
    Args:
        last_date (datetime): Last date in the training data.
        forecast_hours (int): Number of hours to forecast ahead.
        min_date (datetime): Minimum date in training data (for hours_since_start).
        features (list): List of feature names to generate.
        
    Returns:
        pd.DataFrame: DataFrame with future features ready for prediction.
    """
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(hours=1),
        periods=forecast_hours,
        freq='H'
    )
    
    future_data = pd.DataFrame({
        'DateTime': future_dates,
        'date': [d.day for d in future_dates],
        'weekday': [d.weekday() for d in future_dates],
        'hour': [d.hour for d in future_dates],
        'month': [d.month for d in future_dates],
        'year': [d.year for d in future_dates],
        'dayofyear': [d.dayofyear for d in future_dates],
        'weekofyear': [d.isocalendar().week for d in future_dates],
    })
    
    # Calculate hours_since_start relative to dataset start
    future_data['hours_since_start'] = (
        (future_data['DateTime'] - min_date).dt.total_seconds() / 3600
    )
    
    return future_data[features]


def make_predictions(model, future_features):
    """Make predictions on future features.
    
    Args:
        model: Trained LinearRegression model.
        future_features (pd.DataFrame): Features for prediction.
        
    Returns:
        np.array: Predicted vehicle counts.
    """
    return model.predict(future_features)


def print_predictions(future_dates, predictions):
    """Print formatted predictions.
    
    Args:
        future_dates (pd.Series): DatetimeIndex of future dates.
        predictions (np.array): Predicted vehicle counts.
    """
    print("\n" + "=" * 70)
    print("FUTURE PREDICTIONS")
    print("=" * 70)
    print(f"\nPredictions for next 7 days (starting {future_dates[0]}):")
    print(f"\n{'DateTime':<25} {'Predicted Vehicles':<20}")
    print("-" * 45)
    
    for date, pred in zip(future_dates, predictions):
        print(f"{str(date):<25} {pred:>18.2f}")


def print_prediction_summary(historical_vehicles, predicted_vehicles):
    """Print summary statistics comparing historical vs predicted.
    
    Args:
        historical_vehicles (pd.Series): Historical vehicle counts.
        predicted_vehicles (np.array): Predicted vehicle counts.
    """
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\nHistorical Data:")
    print(f"  - Mean vehicles: {historical_vehicles.mean():.2f}")
    print(f"  - Std deviation: {historical_vehicles.std():.2f}")
    print(f"  - Min: {historical_vehicles.min()}")
    print(f"  - Max: {historical_vehicles.max()}")
    
    print(f"\nFuture Predictions:")
    print(f"  - Mean predicted vehicles: {predicted_vehicles.mean():.2f}")
    print(f"  - Std deviation: {predicted_vehicles.std():.2f}")
    print(f"  - Min predicted: {predicted_vehicles.min():.2f}")
    print(f"  - Max predicted: {predicted_vehicles.max():.2f}")
