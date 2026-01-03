"""
Emergency Admissions Predictor using Amazon Chronos
State-of-the-art pretrained time-series model
"""

import pandas as pd
import numpy as np
import torch
import config
from chronos import ChronosPipeline
import warnings
warnings.filterwarnings('ignore')

class EmergencyAdmissionsPredictor:
    def __init__(self, model_name=config.CHRONOS_MODEL):
        """Initialize Chronos model for emergency admissions prediction"""
        self.model_name = model_name
        self.pipeline = None
        
    def load_model(self):
        """Load pretrained Chronos model"""
        print(f"Loading Chronos model: {self.model_name}...")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map=device,
            torch_dtype=dtype,
        )
        print(f"Model loaded on {device}")
        
    def predict(self, historical_data, prediction_length=48, num_samples=20):
        """
        Predict future emergency admissions
        
        Args:
            historical_data: Array or list of historical admissions (hourly)
            prediction_length: Hours to predict ahead
            num_samples: Number of forecast samples for uncertainty
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if self.pipeline is None:
            self.load_model()
        
        # Convert to tensor
        context = torch.tensor(historical_data, dtype=torch.float32)
        
        # Generate forecast
        forecast = self.pipeline.predict(
            context=context,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )
        
        # Calculate statistics
        forecast_np = forecast.squeeze().numpy()
        
        predictions = {
            'mean': forecast_np.mean(axis=0),
            'median': np.median(forecast_np, axis=0),
            'lower_bound': np.percentile(forecast_np, 10, axis=0),
            'upper_bound': np.percentile(forecast_np, 90, axis=0),
            'std': forecast_np.std(axis=0)
        }
        
        return predictions
    
    def predict_from_dataframe(self, df, context_length=168, prediction_length=48):
        """
        Predict from a pandas DataFrame
        
        Args:
            df: DataFrame with 'emergency_admissions' column
            context_length: Hours of context to use (default: 1 week)
            prediction_length: Hours to predict ahead
        """
        # Get last context_length hours
        historical_data = df['emergency_admissions'].tail(context_length).values
        
        # Predict
        predictions = self.predict(historical_data, prediction_length)
        
        # Create future dates
        last_date = df['datetime'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=prediction_length,
            freq='H'
        )
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'datetime': future_dates,
            'predicted_admissions': predictions['mean'],
            'predicted_admissions_lower': predictions['lower_bound'],
            'predicted_admissions_upper': predictions['upper_bound'],
        })
        
        return pred_df, predictions

def demo():
    """Demo the emergency admissions predictor"""
    print("=== Emergency Admissions Predictor Demo ===\n")
    
    # Load data
    print("Loading hospital data...")
    df = pd.read_csv('hospital_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"Data loaded: {len(df)} hours of data")
    print(f"Latest admission count: {df['emergency_admissions'].iloc[-1]}")
    print(f"Average admissions (last 7 days): {df['emergency_admissions'].tail(168).mean():.2f}\n")
    
    # Initialize predictor
    predictor = EmergencyAdmissionsPredictor()
    
    # Make prediction
    print(f"Predicting next {config.PREDICTION_HORIZON} hours...")
    pred_df, predictions = predictor.predict_from_dataframe(
        df, 
        context_length=168,  # Use 1 week of context
        prediction_length=config.PREDICTION_HORIZON
    )
    
    # Show results
    print("\nPredictions for next 24 hours:")
    print(pred_df.head(24))
    
    # Summary statistics
    print(f"\nSummary for next 48 hours:")
    print(f"Expected total admissions: {predictions['mean'].sum():.0f}")
    print(f"Peak hour prediction: {predictions['mean'].max():.0f} admissions")
    print(f"Low hour prediction: {predictions['mean'].min():.0f} admissions")
    
    # Save predictions
    pred_df.to_csv('predictions_emergency_admissions.csv', index=False)
    print("\nPredictions saved to predictions_emergency_admissions.csv")
    
    return pred_df, predictions

if __name__ == "__main__":
    demo()
