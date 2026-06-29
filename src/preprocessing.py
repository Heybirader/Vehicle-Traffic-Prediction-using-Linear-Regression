"""Data preprocessing and feature extraction."""
import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath):
    """Load vehicle traffic data from CSV.
    
    Args:
        filepath (str or Path): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data with DateTime column as datetime.
    """
    data = pd.read_csv(filepath)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    return data


def extract_time_features(data):
    """Extract time-based features from DateTime column.
    
    Features extracted:
    - date: Day of month
    - weekday: Day of week (0 = Monday, 6 = Sunday)
    - hour: Hour of day (0-23)
    - month: Month of year (1-12)
    - year: Calendar year
    - dayofyear: Day number in year (1-366)
    - weekofyear: ISO week number (1-53)
    - hours_since_start: Hours elapsed since dataset start
    
    Args:
        data (pd.DataFrame): DataFrame with 'DateTime' column.
        
    Returns:
        pd.DataFrame: Data with added time features.
    """
    data = data.copy()
    
    data['date'] = data['DateTime'].dt.day
    data['weekday'] = data['DateTime'].dt.weekday
    data['hour'] = data['DateTime'].dt.hour
    data['month'] = data['DateTime'].dt.month
    data['year'] = data['DateTime'].dt.year
    data['dayofyear'] = data['DateTime'].dt.dayofyear
    data['weekofyear'] = data['DateTime'].dt.isocalendar().week
    data['hours_since_start'] = (
        (data['DateTime'] - data['DateTime'].min()).dt.total_seconds() / 3600
    )
    
    return data


def prepare_training_data(data, features):
    """Prepare features (X) and target (y) for model training.
    
    Args:
        data (pd.DataFrame): DataFrame with features and 'Vehicles' target.
        features (list): List of feature column names.
        
    Returns:
        tuple: (X, y) where X is feature matrix and y is target vector.
    """
    X = data[features]
    y = data['Vehicles']
    return X, y


def print_data_info(data):
    """Print summary information about the dataset.
    
    Args:
        data (pd.DataFrame): DataFrame to summarize.
    """
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)
    print(f"\nDataset shape: {data.shape}")
    print(f"Date range: {data['DateTime'].min()} to {data['DateTime'].max()}")
    print(f"\nVehicle counts:")
    print(f"  - Mean: {data['Vehicles'].mean():.2f}")
    print(f"  - Std Dev: {data['Vehicles'].std():.2f}")
    print(f"  - Min: {data['Vehicles'].min()}")
    print(f"  - Max: {data['Vehicles'].max()}")
    print(f"\nFirst few rows:")
    print(data[['DateTime', 'Vehicles']].head(10))
