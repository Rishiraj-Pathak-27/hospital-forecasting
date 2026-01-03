"""
Improved Model Training Script
Better parameters for higher accuracy
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Improved XGBoost parameters
IMPROVED_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}

def load_data():
    """Load and prepare data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'hospital_data_ml.csv')
    
    if not os.path.exists(data_path):
        data_path = os.path.join(script_dir, '..', 'hospital_data_ml.csv')
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    return df

def train_icu_model(df):
    """Train improved ICU demand model"""
    print("\n" + "="*50)
    print("Training ICU Demand Model")
    print("="*50)
    
    feature_cols = [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'temperature', 'flu_season_index', 'air_quality_index',
        'emergency_admissions_lag_1h', 'emergency_admissions_lag_7h',
        'emergency_admissions_rolling_3h', 'emergency_admissions_rolling_7h',
        'icu_demand_lag_1h', 'icu_demand_lag_7h'
    ]
    
    X = df[feature_cols].copy()
    y = df['icu_demand'].copy()
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    mae_scores = []
    r2_scores = []
    
    print("\nCross-validation results:")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**IMPROVED_PARAMS, verbosity=0)
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        r2 = r2_score(y_val, pred)
        
        mae_scores.append(mae)
        r2_scores.append(r2)
        print(f"  Fold {fold+1}: MAE={mae:.4f}, R²={r2:.4f}")
    
    print(f"\nAverage: MAE={np.mean(mae_scores):.4f} (±{np.std(mae_scores):.4f})")
    print(f"Average: R²={np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    final_model = xgb.XGBRegressor(**IMPROVED_PARAMS, verbosity=0)
    final_model.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop features:")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return final_model, feature_cols

def train_staff_model(df):
    """Train improved staff workload model"""
    print("\n" + "="*50)
    print("Training Staff Workload Model")
    print("="*50)
    
    feature_cols = [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'temperature', 'flu_season_index', 'air_quality_index',
        'emergency_admissions_lag_1h', 'emergency_admissions_lag_7h',
        'emergency_admissions_rolling_3h', 'emergency_admissions_rolling_7h',
        'icu_demand_lag_1h',
        'bed_occupancy'
    ]
    
    X = df[feature_cols].copy()
    y = df['staff_workload'].copy()
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    mae_scores = []
    r2_scores = []
    
    print("\nCross-validation results:")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**IMPROVED_PARAMS, verbosity=0)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        r2 = r2_score(y_val, pred)
        
        mae_scores.append(mae)
        r2_scores.append(r2)
        print(f"  Fold {fold+1}: MAE={mae:.4f}, R²={r2:.4f}")
    
    print(f"\nAverage: MAE={np.mean(mae_scores):.4f} (±{np.std(mae_scores):.4f})")
    print(f"Average: R²={np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    final_model = xgb.XGBRegressor(**IMPROVED_PARAMS, verbosity=0)
    final_model.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop features:")
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return final_model, feature_cols

def main():
    """Main training pipeline"""
    print("="*60)
    print("HOSPITAL PREDICTION - IMPROVED MODEL TRAINING")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Train models
    icu_model, icu_features = train_icu_model(df)
    staff_model, staff_features = train_staff_model(df)
    
    # Save models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    icu_path = os.path.join(models_dir, 'icu_demand_model.pkl')
    staff_path = os.path.join(models_dir, 'staff_workload_model.pkl')
    
    joblib.dump(icu_model, icu_path)
    joblib.dump(staff_model, staff_path)
    
    print("\n" + "="*60)
    print("MODELS SAVED SUCCESSFULLY")
    print("="*60)
    print(f"ICU Model: {icu_path}")
    print(f"Staff Model: {staff_path}")
    
    # Verify
    print("\nVerifying models...")
    icu_loaded = joblib.load(icu_path)
    staff_loaded = joblib.load(staff_path)
    
    # Test prediction
    test_sample = df[icu_features].tail(1)
    icu_pred = icu_loaded.predict(test_sample)
    print(f"ICU test prediction: {icu_pred[0]:.2f}")
    
    test_sample = df[staff_features].tail(1)
    staff_pred = staff_loaded.predict(test_sample)
    print(f"Staff test prediction: {staff_pred[0]:.2f}")
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
