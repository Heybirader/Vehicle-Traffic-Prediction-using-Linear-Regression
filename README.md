🚗 Vehicle Traffic Prediction using Linear Regression

A machine learning project that analyzes historical vehicle traffic data and predicts future traffic counts using Linear Regression with time-based feature engineering.


📌 Overview

This project uses a time-series vehicle count dataset to:


Extract meaningful date/time features from raw timestamps

Train a Linear Regression model on historical traffic patterns

Evaluate model performance using standard regression metrics

Forecast vehicle counts for the next 7 days (168 hours)



📂 Project Structure

├── py.py               # Main script: preprocessing, training, prediction

├── vehicles.csv        # Dataset (not included — see Dataset section)

└── README.md


🧠 Features Used

FeatureDescriptiondateDay of monthweekdayDay of week (0 = Monday)hourHour of day (0–23)monthMonth of yearyearCalendar yeardayofyearDay number in the yearweekofyearISO week numberhours_since_startHours elapsed since the earliest record


⚙️ Tech Stack


Python 3.x

pandas — data loading and manipulation

NumPy — numerical operations

scikit-learn — Linear Regression, train/test split, metrics




🚀 Getting Started

1. Clone the repository

bashgit clone https://github.com/your-username/vehicle-traffic-prediction.git
cd vehicle-traffic-prediction

2. Install dependencies

bashpip install pandas numpy scikit-learn

3. Add the dataset

Place your vehicles.csv file in the project root. The CSV must contain at least:


DateTime — timestamp column (e.g., 2015-11-01 00:00:00)
Vehicles — integer vehicle count


4. Run the script

bashpython py.py


📊 Model Performance

The model is evaluated on both training and test sets using:


R² Score — goodness of fit
MSE — Mean Squared Error
RMSE — Root Mean Squared Error
MAE — Mean Absolute Error



Actual scores will vary depending on your dataset.




🔮 Output

The script prints:


Dataset info and first few rows
Extracted time features
Training and test set metrics
Feature coefficients
Hourly vehicle predictions for the next 7 days
Summary statistics comparing historical vs predicted values



📁 Dataset

This project was built using a vehicle traffic count dataset with hourly records.

You can find similar datasets on:


Kaggle — Traffic Volume Datasets
UCI ML Repository



💡 Potential Improvements


Try Random Forest or XGBoost for better non-linear capture
Add lag features (e.g., traffic 1 hour/1 day ago)
Visualize predictions with matplotlib/seaborn
Deploy as a web app using Flask or Streamlit
