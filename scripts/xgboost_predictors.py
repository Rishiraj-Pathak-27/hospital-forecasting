"""
ICU Demand & Staff Workload Predictors using XGBoost
Fast and accurate for multivariate time series with covariates
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import config
import warnings
warnings.filterwarnings('ignore')

class MultivariatePredictorXGB:
    """Base class for XGBoost-based predictors with external covariates"""
    
    def __init__(self, target_col, feature_cols, name="Predictor"):
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.name = name
        self.model = None
        self.feature_importance = None
        
    def prepare_features(self, df):
        """Prepare feature matrix and target"""
        X = df[self.feature_cols].copy()
        y = df[self.target_col].copy()
        return X, y
    
    def train(self, df, test_size=0.2):
        """Train XGBoost model"""
        print(f"\n=== Training {self.name} ===")
        
        X, y = self.prepare_features(df)
        
        # Split data (time-series aware: no shuffle)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train model
        self.model = xgb.XGBRegressor(**config.XGBOOST_PARAMS, verbosity=0)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"Train MAE: {train_mae:.2f}")
        print(f"Test MAE: {test_mae:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Test R²: {test_r2:.4f}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 important features:")
        print(self.feature_importance.head())
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2
        }
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_next_hours(self, df, hours=48):
        """
        Predict future values (requires forecasting features)
        Note: This is simplified - in production, you'd need to forecast covariates too
        """
        # Use last available data point features
        last_features = df[self.feature_cols].tail(hours).copy()
        predictions = self.predict(last_features)
        return predictions
    
    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

class ICUDemandPredictor(MultivariatePredictorXGB):
    """Predict ICU bed demand"""
    
    def __init__(self):
        feature_cols = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'temperature', 'flu_season_index', 'air_quality_index',
            'emergency_admissions_lag_1h', 'emergency_admissions_lag_7h',
            'emergency_admissions_rolling_3h', 'emergency_admissions_rolling_7h',
            'icu_demand_lag_1h', 'icu_demand_lag_7h'
        ]
        super().__init__(
            target_col='icu_demand',
            feature_cols=feature_cols,
            name="ICU Demand Predictor"
        )

class StaffWorkloadPredictor(MultivariatePredictorXGB):
    """Predict staff workload requirements"""
    
    def __init__(self):
        feature_cols = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'temperature', 'flu_season_index', 'air_quality_index',
            'emergency_admissions_lag_1h', 'emergency_admissions_lag_7h',
            'emergency_admissions_rolling_3h', 'emergency_admissions_rolling_7h',
            'icu_demand_lag_1h',
            'bed_occupancy'
        ]
        super().__init__(
            target_col='staff_workload',
            feature_cols=feature_cols,
            name="Staff Workload Predictor"
        )

def demo():
    """Demo ICU and Staff predictors"""
    print("=== XGBoost Multivariate Predictors Demo ===\n")
    
    # Load ML-ready data
    print("Loading prepared data...")
    df = pd.read_csv('hospital_data_ml.csv')
    print(f"Loaded {len(df)} records with features\n")
    
    # Train ICU Demand Predictor
    icu_predictor = ICUDemandPredictor()
    icu_metrics = icu_predictor.train(df)
    icu_predictor.save_model('models/icu_demand_model.pkl')
    
    # Train Staff Workload Predictor
    staff_predictor = StaffWorkloadPredictor()
    staff_metrics = staff_predictor.train(df)
    staff_predictor.save_model('models/staff_workload_model.pkl')
    
    # Make predictions on last 48 hours
    print("\n=== Making Predictions ===")
    test_data = df.tail(48)
    
    icu_pred = icu_predictor.predict(test_data[icu_predictor.feature_cols])
    staff_pred = staff_predictor.predict(test_data[staff_predictor.feature_cols])
    
    # Create results DataFrame
    results = pd.DataFrame({
        'datetime': pd.to_datetime(test_data['datetime'].values),
        'actual_icu_demand': test_data['icu_demand'].values,
        'predicted_icu_demand': icu_pred,
        'actual_staff_workload': test_data['staff_workload'].values,
        'predicted_staff_workload': staff_pred
    })
    
    print("\nPrediction Results (last 24 hours):")
    print(results.tail(24))
    
    # Calculate errors
    icu_mae = mean_absolute_error(results['actual_icu_demand'], results['predicted_icu_demand'])
    staff_mae = mean_absolute_error(results['actual_staff_workload'], results['predicted_staff_workload'])
    
    print(f"\nICU Demand MAE: {icu_mae:.2f}")
    print(f"Staff Workload MAE: {staff_mae:.2f}")
    
    # Save results
    results.to_csv('predictions_icu_staff.csv', index=False)
    print("\nPredictions saved to predictions_icu_staff.csv")
    
    return icu_predictor, staff_predictor, results

if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)
    demo()
