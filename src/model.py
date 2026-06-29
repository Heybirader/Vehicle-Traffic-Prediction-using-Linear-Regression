"""Model training and evaluation."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def train_model(X, y, test_size=0.2, random_state=42):
    """Train a Linear Regression model.
    
    Args:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        dict: Dictionary containing:
            - 'model': Trained LinearRegression object
            - 'X_train': Training features
            - 'X_test': Testing features
            - 'y_train': Training targets
            - 'y_test': Testing targets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance on training and test sets.
    
    Args:
        model: Trained LinearRegression model.
        X_train, X_test: Training and test features.
        y_train, y_test: Training and test targets.
        
    Returns:
        dict: Dictionary with metrics for train and test sets.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'train': {
            'r2': r2_score(y_train, y_pred_train),
            'mse': mean_squared_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'mae': mean_absolute_error(y_train, y_pred_train)
        },
        'test': {
            'r2': r2_score(y_test, y_pred_test),
            'mse': mean_squared_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae': mean_absolute_error(y_test, y_pred_test)
        }
    }
    
    return metrics


def print_model_performance(metrics):
    """Print model performance metrics.
    
    Args:
        metrics (dict): Dictionary with train/test metrics.
    """
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)
    
    print(f"\nTraining Set:")
    print(f"  - R² Score: {metrics['train']['r2']:.4f}")
    print(f"  - Mean Squared Error: {metrics['train']['mse']:.4f}")
    print(f"  - Root Mean Squared Error: {metrics['train']['rmse']:.4f}")
    print(f"  - Mean Absolute Error: {metrics['train']['mae']:.4f}")
    
    print(f"\nTest Set:")
    print(f"  - R² Score: {metrics['test']['r2']:.4f}")
    print(f"  - Mean Squared Error: {metrics['test']['mse']:.4f}")
    print(f"  - Root Mean Squared Error: {metrics['test']['rmse']:.4f}")
    print(f"  - Mean Absolute Error: {metrics['test']['mae']:.4f}")


def print_model_coefficients(model, features):
    """Print model coefficients for interpretability.
    
    Args:
        model: Trained LinearRegression model.
        features (list): List of feature names.
    """
    print(f"\nModel Coefficients:")
    for feature, coef in zip(features, model.coef_):
        print(f"  - {feature}: {coef:.6f}")
    print(f"  - Intercept: {model.intercept_:.6f}")
