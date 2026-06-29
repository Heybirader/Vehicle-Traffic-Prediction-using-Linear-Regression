import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Load data
data = pd.read_csv('d:/Github/Linear Regression/vehicles.csv')
print("=" * 70)
print("VEHICLE DATA ANALYSIS - LINEAR REGRESSION")
print("=" * 70)
print(f"\nDataset shape: {data.shape}")
print(f"Date range: {data['DateTime'].min()} to {data['DateTime'].max()}")
print("\nFirst few rows:")
print(data.head(10))
print("\nData Info:")
print(data.info())

# Date/Time feature extraction
def get_dom(dt):
    return dt.day

def get_weekday(dt):
    return dt.weekday()

def get_hour(dt):
    return dt.hour

def get_year(dt):
    return dt.year

def get_month(dt):
    return dt.month

def get_dayofyear(dt):
    return dt.dayofyear

def get_weekofyear(dt):
    return dt.isocalendar()[1]

# Convert DateTime and extract features
data['DateTime'] = pd.to_datetime(data['DateTime'])
data['date'] = data['DateTime'].map(get_dom)
data['weekday'] = data['DateTime'].map(get_weekday)
data['hour'] = data['DateTime'].map(get_hour)
data['month'] = data['DateTime'].map(get_month)
data['year'] = data['DateTime'].map(get_year)
data['dayofyear'] = data['DateTime'].map(get_dayofyear)
data['weekofyear'] = data['DateTime'].map(get_weekofyear)

# Add a time index (days since start)
data['days_since_start'] = (data['DateTime'] - data['DateTime'].min()).dt.days
data['hours_since_start'] = (data['DateTime'] - data['DateTime'].min()).dt.total_seconds() / 3600

print("\nExtracted features:")
print(data[['DateTime', 'Vehicles', 'date', 'hour', 'month', 'year', 'hours_since_start']].head(10))

# Prepare features for model
features = ['date', 'weekday', 'hour', 'month', 'year', 'dayofyear', 'weekofyear', 'hours_since_start']
X = data[features]
y = data['Vehicles']

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Model Performance
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)
print(f"\nTraining Set:")
print(f"  - R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"  - Mean Squared Error: {mean_squared_error(y_train, y_pred_train):.4f}")
print(f"  - Root Mean Squared Error: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"  - Mean Absolute Error: {mean_absolute_error(y_train, y_pred_train):.4f}")

print(f"\nTest Set:")
print(f"  - R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"  - Mean Squared Error: {mean_squared_error(y_test, y_pred_test):.4f}")
print(f"  - Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print(f"  - Mean Absolute Error: {mean_absolute_error(y_test, y_pred_test):.4f}")

# Model Coefficients
print(f"\nModel Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"  - {feature}: {coef:.6f}")
print(f"  - Intercept: {model.intercept_:.6f}")

# Future Predictions
print("\n" + "=" * 70)
print("FUTURE PREDICTIONS")
print("=" * 70)

# Generate future dates (next 7 days from the end of data)
last_date = data['DateTime'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=168, freq='H')

# Create future features
future_data = pd.DataFrame({
    'DateTime': future_dates,
    'date': [d.day for d in future_dates],
    'weekday': [d.weekday() for d in future_dates],
    'hour': [d.hour for d in future_dates],
    'month': [d.month for d in future_dates],
    'year': [d.year for d in future_dates],
    'dayofyear': [d.dayofyear for d in future_dates],
    'weekofyear': [d.isocalendar()[1] for d in future_dates],
})

# Calculate hours_since_start for future dates
min_date = data['DateTime'].min()
future_data['hours_since_start'] = (future_data['DateTime'] - min_date).dt.total_seconds() / 3600

# Make predictions
future_predictions = model.predict(future_data[features])
future_data['Predicted_Vehicles'] = future_predictions

print(f"\nPredictions for next 7 days (starting {last_date + pd.Timedelta(hours=1)}):")
print(f"\n{'DateTime':<25} {'Predicted Vehicles':<20}")
print("-" * 45)
for idx, row in future_data.iterrows():
    print(f"{str(row['DateTime']):<25} {row['Predicted_Vehicles']:>18.2f}")

# Summary Statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"\nHistorical Data:")
print(f"  - Mean vehicles: {data['Vehicles'].mean():.2f}")
print(f"  - Std deviation: {data['Vehicles'].std():.2f}")
print(f"  - Min: {data['Vehicles'].min()}")
print(f"  - Max: {data['Vehicles'].max()}")

print(f"\nFuture Predictions:")
print(f"  - Mean predicted vehicles: {future_data['Predicted_Vehicles'].mean():.2f}")
print(f"  - Std deviation: {future_data['Predicted_Vehicles'].std():.2f}")
print(f"  - Min predicted: {future_data['Predicted_Vehicles'].min():.2f}")
print(f"  - Max predicted: {future_data['Predicted_Vehicles'].max():.2f}")



