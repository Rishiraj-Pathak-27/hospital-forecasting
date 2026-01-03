"""
Generate realistic mock hospital data with seasonality and external factors
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config

np.random.seed(42)

def generate_hospital_data(days=180):
    """Generate synthetic hospital data with realistic patterns"""
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    df = pd.DataFrame({'datetime': dates})
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Time-based patterns
    # More admissions in winter (flu season) and evening hours
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * df['month'] / 12 - np.pi/2)  # Peak in Dec-Jan
    hourly_factor = 1 + 0.4 * np.sin(2 * np.pi * (df['hour'] - 6) / 24)  # Peak 6 PM
    weekend_factor = 1 + 0.2 * df['is_weekend']
    
    # External factors (covariates)
    df['temperature'] = 20 + 10 * np.sin(2 * np.pi * df['month'] / 12) + np.random.normal(0, 3, len(df))
    df['flu_season_index'] = (df['month'].isin([11, 12, 1, 2])).astype(float) * 0.7 + np.random.uniform(0, 0.3, len(df))
    df['air_quality_index'] = 50 + 30 * np.random.random(len(df)) + 20 * (df['month'].isin([6, 7, 8])).astype(int)
    
    # Emergency Admissions (main target)
    base_admissions = config.BASE_EMERGENCY_ADMISSIONS / 24  # Per hour
    df['emergency_admissions'] = (
        base_admissions * 
        seasonal_factor * 
        hourly_factor * 
        weekend_factor *
        (1 + 0.2 * df['flu_season_index']) *
        np.random.gamma(2, 0.5, len(df))
    ).round().astype(int)
    
    # ICU Demand (15-20% of emergency admissions need ICU)
    icu_rate = 0.15 + 0.05 * df['flu_season_index']
    df['icu_demand'] = (df['emergency_admissions'] * icu_rate).round().astype(int)
    df['icu_demand'] = df['icu_demand'].clip(upper=config.ICU_CAPACITY)
    
    # Staff Workload (complex: depends on patients + severity)
    severity_factor = 1 + 0.3 * df['flu_season_index'] + 0.2 * (df['hour'] >= 18).astype(int)
    df['staff_workload'] = (
        df['emergency_admissions'] * severity_factor * 
        (config.STAFF_PER_10_PATIENTS / 10)
    ).round().astype(int)
    
    # Current bed occupancy
    df['bed_occupancy'] = np.random.randint(30, 90, len(df))
    
    # Add some random spikes (simulate outbreaks/accidents)
    spike_indices = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
    df.loc[spike_indices, 'emergency_admissions'] *= np.random.uniform(1.5, 2.5, len(spike_indices))
    df.loc[spike_indices, 'icu_demand'] *= np.random.uniform(1.3, 2.0, len(spike_indices))
    
    # Round values
    df['emergency_admissions'] = df['emergency_admissions'].round().astype(int)
    df['icu_demand'] = df['icu_demand'].round().astype(int)
    df['staff_workload'] = df['staff_workload'].round().astype(int)
    
    return df

def add_lag_features(df, target_col, lags):
    """Add lag features for time series"""
    for lag in lags:
        df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
    return df

def add_rolling_features(df, target_col, windows):
    """Add rolling average features"""
    for window in windows:
        df[f'{target_col}_rolling_{window}h'] = df[target_col].rolling(window=window).mean()
    return df

def prepare_ml_features(df):
    """Prepare features for ML models"""
    
    # Add lag features for key targets
    df = add_lag_features(df, 'emergency_admissions', config.LAG_FEATURES)
    df = add_lag_features(df, 'icu_demand', config.LAG_FEATURES)
    
    # Add rolling features
    df = add_rolling_features(df, 'emergency_admissions', config.ROLLING_WINDOWS)
    
    # Drop NaN rows created by lag features
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    # Generate and save data
    print("Generating hospital data...")
    df = generate_hospital_data(config.DAYS_OF_HISTORICAL_DATA)
    
    print(f"Generated {len(df)} hourly records over {config.DAYS_OF_HISTORICAL_DATA} days")
    print(f"\nData shape: {df.shape}")
    print(f"\nSample data:")
    print(df.head())
    
    print(f"\nStatistics:")
    print(df[['emergency_admissions', 'icu_demand', 'staff_workload']].describe())
    
    # Save raw data
    df.to_csv('hospital_data.csv', index=False)
    print(f"\nData saved to hospital_data.csv")
    
    # Prepare ML features
    df_ml = prepare_ml_features(df.copy())
    df_ml.to_csv('hospital_data_ml.csv', index=False)
    print(f"ML-ready data saved to hospital_data_ml.csv")
