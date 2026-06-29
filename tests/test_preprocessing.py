"""Tests for preprocessing module."""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import extract_time_features, prepare_training_data


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions."""
    
    def setUp(self):
        """Create sample data for testing."""
        dates = pd.date_range('2015-01-01', periods=100, freq='H')
        self.data = pd.DataFrame({
            'DateTime': dates,
            'Vehicles': np.random.randint(50, 200, 100)
        })
    
    def test_extract_time_features(self):
        """Test that time features are extracted correctly."""
        result = extract_time_features(self.data)
        
        # Check that all expected features exist
        expected_features = [
            'date', 'weekday', 'hour', 'month', 'year',
            'dayofyear', 'weekofyear', 'hours_since_start'
        ]
        for feature in expected_features:
            self.assertIn(feature, result.columns, f"Feature {feature} not found")
        
        # Check that we have same number of rows
        self.assertEqual(len(result), len(self.data))
        
        # Check that hours_since_start is monotonically increasing
        self.assertTrue(
            result['hours_since_start'].is_monotonic_increasing,
            "hours_since_start should be monotonically increasing"
        )
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        data = extract_time_features(self.data)
        features = ['date', 'hour', 'month', 'year']
        
        X, y = prepare_training_data(data, features)
        
        # Check shapes
        self.assertEqual(X.shape[0], len(data))
        self.assertEqual(X.shape[1], len(features))
        self.assertEqual(len(y), len(data))
        
        # Check that all features are in X
        for feature in features:
            self.assertIn(feature, X.columns)


if __name__ == '__main__':
    unittest.main()
