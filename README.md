# 🚗 Vehicle Traffic Prediction using Linear Regression

A machine learning project that analyzes historical vehicle traffic data and predicts future traffic counts using Linear Regression with time-based feature engineering.

## 📌 Overview

This project uses a time-series vehicle count dataset to:

- Extract meaningful date/time features from raw timestamps
- Train a Linear Regression model on historical traffic patterns
- Evaluate model performance using standard regression metrics
- Forecast vehicle counts for the next 7 days (168 hours)

## 📁 Project Structure

```
vehicle-traffic-prediction/
├── data/
│   ├── raw/                    # Original datasets
│   │   └── vehicles.csv        # (add your dataset here)
│   └── processed/              # Generated processed data
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Data loading & feature extraction
│   ├── model.py                # Model training & evaluation
│   └── predict.py              # Prediction & forecasting logic
├── scripts/
│   └── train.py                # Main entry point (run this!)
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py   # Unit tests
├── config.py                   # Centralized configuration
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🧠 Features Used

| Feature | Description |
|---------|-------------|
| `date` | Day of month (1–31) |
| `weekday` | Day of week (0 = Monday, 6 = Sunday) |
| `hour` | Hour of day (0–23) |
| `month` | Month of year (1–12) |
| `year` | Calendar year |
| `dayofyear` | Day number in the year (1–366) |
| `weekofyear` | ISO week number (1–53) |
| `hours_since_start` | Hours elapsed since dataset start |

## ⚙️ Tech Stack

- **Language:** Python 3.7+
- **Core Libraries:**
  - `pandas` — data manipulation
  - `numpy` — numerical operations
  - `scikit-learn` — Linear Regression, train/test split, metrics

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Heybirader/Vehicle-Traffic-Prediction-using-Linear-Regression.git
cd Vehicle-Traffic-Prediction-using-Linear-Regression
```

### 2. Set up Python environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your dataset

Place your `vehicles.csv` file in the `data/raw/` directory. The CSV must contain at least two columns:

- `DateTime` — timestamp (e.g., `2015-11-01 00:00:00`)
- `Vehicles` — integer vehicle count

**Example:**
```
DateTime,Vehicles
2015-11-01 00:00:00,3354
2015-11-01 01:00:00,2912
...
```

### 5. Run the training pipeline

```bash
python scripts/train.py
```

This will:
1. Load and explore the dataset
2. Extract time-based features
3. Train the Linear Regression model
4. Evaluate performance on train/test sets
5. Print model coefficients for interpretability
6. Generate 7-day traffic forecasts
7. Display summary statistics

## 📊 Model Performance

The model is evaluated using standard regression metrics:

| Metric | Description |
|--------|-------------|
| **R² Score** | Goodness of fit (0-1, higher is better) |
| **MSE** | Mean Squared Error (lower is better) |
| **RMSE** | Root Mean Squared Error (same units as target) |
| **MAE** | Mean Absolute Error (average prediction error) |

Metrics are computed on both training (80%) and test (20%) sets.

## 🔮 Output

After running `python scripts/train.py`, the script outputs:

1. **Dataset Information**
   - Shape, date range, vehicle count statistics

2. **Feature Summary**
   - Extracted time features (date, hour, month, etc.)

3. **Model Performance**
   - R², MSE, RMSE, MAE for train and test sets
   - Model coefficients showing feature importance

4. **Future Predictions**
   - Hourly forecasts for next 7 days

5. **Summary Statistics**
   - Comparison of historical vs predicted traffic patterns

## 🧪 Running Tests

```bash
python -m pytest tests/
# Or:
python -m unittest discover tests/
```

## 💡 Next Steps & Improvements

- [ ] Add visualization with `matplotlib`/`seaborn` (plots of historical vs predicted)
- [ ] Try non-linear models: Random Forest, XGBoost, or neural networks
- [ ] Add lag features (traffic from 1 hour/1 day ago)
- [ ] Implement cross-validation for more robust evaluation
- [ ] Save trained model to disk for reuse
- [ ] Deploy as a web service using Flask or Streamlit
- [ ] Add automated data validation and error handling
- [ ] Track experiments with MLflow or Weights & Biases

## 📁 Dataset Sources

You can find similar datasets on:

- [Kaggle — Traffic Volume Datasets](https://www.kaggle.com/search?q=traffic+volume)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets)

## 📝 License

MIT License (or your preferred license)

## 👤 Author

Heybirader

---

**Questions or suggestions?** Open an issue on GitHub!
