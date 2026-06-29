"""Configuration for the vehicle traffic prediction project."""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Raw data file
RAW_DATA_FILE = RAW_DATA_DIR / 'vehicles.csv'

# Model paths
MODEL_DIR = PROJECT_ROOT / 'models'
TRAINED_MODEL_FILE = MODEL_DIR / 'traffic_model.pkl'

# Model hyperparameters
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42

# Feature engineering
FEATURES = [
    'date',           # Day of month
    'weekday',        # Day of week (0 = Monday)
    'hour',           # Hour of day (0-23)
    'month',          # Month of year
    'year',           # Calendar year
    'dayofyear',      # Day number in year
    'weekofyear',     # ISO week number
    'hours_since_start'  # Hours elapsed since dataset start
]

# Prediction
FORECAST_HOURS = 168  # 7 days ahead
