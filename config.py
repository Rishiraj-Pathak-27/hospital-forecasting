"""
Configuration file for Hospital Emergency Prediction System
"""

# Data Generation Parameters
DAYS_OF_HISTORICAL_DATA = 180  # 6 months of data
PREDICTION_HORIZON = 48  # Default: Predict next 48 hours (can be changed dynamically)

# Model Parameters
CHRONOS_MODEL = "amazon/chronos-t5-small"  # Fast for hackathon
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'random_state': 42
}

# Hospital Parameters
BASE_EMERGENCY_ADMISSIONS = 50  # Average per day
ICU_CAPACITY = 20
STAFF_PER_10_PATIENTS = 3  # Ratio
CRITICAL_THRESHOLD = 0.85  # 85% capacity triggers alert

# Feature Engineering
LAG_FEATURES = [1, 7, 14]  # Previous day, week, 2 weeks
ROLLING_WINDOWS = [3, 7, 14]  # Rolling averages

# Output
RESULTS_DIR = "results"
MODELS_DIR = "models"
