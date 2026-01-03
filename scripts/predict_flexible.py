"""
Flexible Prediction Interface
Predict for any time period: next 24hrs, 48hrs, weekend, week, or custom period
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from scripts.xgboost_predictors import ICUDemandPredictor, StaffWorkloadPredictor
from scripts.resource_optimizer import HospitalResourceOptimizer
import warnings
warnings.filterwarnings('ignore')

class FlexiblePredictor:
    """Flexible prediction system for any time period"""
    
    def __init__(self):
        self.data = None
        self.data_ml = None
        self.icu_predictor = None
        self.staff_predictor = None
        self.optimizer = HospitalResourceOptimizer()
        
        # Load data from data folder
        data_path = os.path.join('data', 'hospital_data.csv')
        data_ml_path = os.path.join('data', 'hospital_data_ml.csv')
        
        if os.path.exists(data_path):
            self.data = pd.read_csv(data_path)
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            print("✓ Loaded existing hospital data")
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        if os.path.exists(data_ml_path):
            self.data_ml = pd.read_csv(data_ml_path)
            print("✓ Loaded ML-ready data")
        else:
            raise FileNotFoundError(f"ML data file not found: {data_ml_path}")
            
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load trained models"""
        if os.path.exists('models/icu_demand_model.pkl') and os.path.exists('models/staff_workload_model.pkl'):
            self.icu_predictor = ICUDemandPredictor()
            self.icu_predictor.load_model('models/icu_demand_model.pkl')
            
            self.staff_predictor = StaffWorkloadPredictor()
            self.staff_predictor.load_model('models/staff_workload_model.pkl')
            
            print("✓ Loaded trained models\n")
        else:
            print("Training models...")
            self.system.train_models()
            self.icu_predictor = self.system.icu_predictor
            self.staff_predictor = self.system.staff_predictor
    
    def get_next_weekend(self):
        """Calculate hours until next weekend and weekend duration"""
        now = self.system.data['datetime'].iloc[-1]
        current_weekday = now.weekday()  # Monday = 0, Sunday = 6
        
        if current_weekday < 5:  # Monday to Friday
            hours_until_weekend = (5 - current_weekday) * 24 - now.hour
            weekend_start = now + timedelta(hours=hours_until_weekend)
        else:  # Already weekend
            hours_until_weekend = 0
            weekend_start = now
            
        # Weekend is Friday 6pm to Sunday 11pm (54 hours)
        weekend_duration = 54
        
        return hours_until_weekend, weekend_duration, weekend_start
    
    def predict_next_24_hours(self):
        """Predict for next 24 hours"""
        return self._predict_period(24, "Next 24 Hours")
    
    def predict_next_48_hours(self):
        """Predict for next 48 hours"""
        return self._predict_period(48, "Next 48 Hours")
    
    def predict_next_week(self):
        """Predict for next week (168 hours)"""
        return self._predict_period(168, "Next Week (7 Days)")
    
    def predict_next_weekend(self):
        """Predict for next weekend"""
        hours_until, duration, weekend_start = self.get_next_weekend()
        
        if hours_until > 0:
            print(f"Next weekend starts in {hours_until} hours ({weekend_start.strftime('%A, %B %d at %I%p')})")
            period_name = f"Next Weekend ({weekend_start.strftime('%b %d')} - {(weekend_start + timedelta(hours=duration)).strftime('%b %d')})"
        else:
            period_name = "Current Weekend"
            
        return self._predict_period(duration, period_name, start_offset=hours_until)
    
    def predict_custom(self, hours, description="Custom Period"):
        """Predict for custom number of hours"""
        return self._predict_period(hours, description)
    
    def _predict_period(self, hours, period_name, start_offset=0):
        """
        Internal method to predict for any period
        
        Args:
            hours: Number of hours to predict
            period_name: Name of the period (for display)
            start_offset: Hours from now to start prediction
        """
        print(f"\n{'='*60}")
        print(f"PREDICTING: {period_name}")
        print(f"{'='*60}\n")
        
        # Create datetime range for future
        last_datetime = self.data['datetime'].iloc[-1]
        future_dates = pd.date_range(
            start=last_datetime + timedelta(hours=start_offset+1),
            periods=hours,
            freq='H'
        )
        
        # Generate future features for prediction
        future_features = pd.DataFrame({
            'datetime': future_dates,
            'hour': future_dates.hour,
            'day_of_week': future_dates.dayofweek,
            'month': future_dates.month,
            'is_weekend': (future_dates.dayofweek >= 5).astype(int),
            'temperature': 20 + 5 * np.sin(2 * np.pi * future_dates.hour / 24),
            'flu_season_index': ((future_dates.month >= 11) | (future_dates.month <= 2)).astype(float),
            'air_quality_index': 60 + 20 * np.random.randn(hours),
            'bed_occupancy': np.random.randint(40, 80, hours)  # Simulate bed occupancy
        })
        
        # Add lag features from historical data (repeat last known values)
        recent_admissions = self.data['emergency_admissions'].tail(24).values
        recent_icu = self.data['icu_demand'].tail(24).values
        
        future_features['emergency_admissions_lag_1h'] = np.full(hours, recent_admissions[-1])
        future_features['emergency_admissions_lag_7h'] = np.full(hours, recent_admissions[-7] if len(recent_admissions) >= 7 else recent_admissions[-1])
        future_features['emergency_admissions_lag_14h'] = np.full(hours, recent_admissions[-14] if len(recent_admissions) >= 14 else recent_admissions[-1])
        future_features['emergency_admissions_rolling_3h'] = np.full(hours, np.mean(recent_admissions[-3:]))
        future_features['emergency_admissions_rolling_7h'] = np.full(hours, np.mean(recent_admissions[-7:]))
        future_features['emergency_admissions_rolling_14h'] = np.full(hours, np.mean(recent_admissions[-14:]))
        
        future_features['icu_demand_lag_1h'] = np.full(hours, recent_icu[-1])
        future_features['icu_demand_lag_7h'] = np.full(hours, recent_icu[-7] if len(recent_icu) >= 7 else recent_icu[-1])
        future_features['icu_demand_lag_14h'] = np.full(hours, recent_icu[-14] if len(recent_icu) >= 14 else recent_icu[-1])
        
        # Generate realistic emergency admissions with patterns
        base_rate = float(self.data['emergency_admissions'].tail(168).mean())
        hourly_pattern = 1 + 0.3 * np.sin(2 * np.pi * (future_dates.hour.values - 18) / 24)  # Peak evening
        day_pattern = 1 + 0.2 * (future_dates.dayofweek.values >= 4).astype(float)  # Higher on weekends
        noise = np.random.normal(0, 0.15, hours)
        emergency_admissions = base_rate * hourly_pattern * day_pattern * (1 + noise)
        emergency_admissions = np.maximum(emergency_admissions, 0.5)  # Minimum 0.5
        
        # Predict ICU and staff using XGBoost models
        icu_predictions = self.icu_predictor.predict(future_features[self.icu_predictor.feature_cols].values)
        staff_predictions = self.staff_predictor.predict(future_features[self.staff_predictor.feature_cols].values)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'datetime': future_dates,
            'predicted_emergency_admissions': emergency_admissions,
            'predicted_icu_demand': icu_predictions,
            'predicted_staff_workload': staff_predictions
        })
        
        # Optimize resources
        optimization = self.optimizer.optimize(
            predicted_admissions=emergency_admissions,
            predicted_icu=icu_predictions,
            predicted_workload=staff_predictions,
            current_occupancy=np.random.randint(50, 70)
        )
        
        # Display summary
        self._display_summary(predictions_df, optimization, period_name)
        
        # Save results to proper folders
        filename_safe = period_name.replace(' ', '_').replace('(', '').replace(')', '')
        
        # Ensure folders exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        predictions_df.to_csv(f'data/predictions_{filename_safe}.csv', index=False)
        
        with open(f'reports/report_{filename_safe}.json', 'w') as f:
            report = {
                'period': period_name,
                'generated_at': datetime.now().isoformat(),
                'hours': hours,
                'start_offset': start_offset,
                'summary': {
                    'total_admissions': int(emergency_admissions.sum()),
                    'peak_admissions': int(emergency_admissions.max()),
                    'total_icu_demand': int(icu_predictions.sum()),
                    'peak_icu_demand': int(icu_predictions.max()),
                    'peak_staff': int(optimization['staff_requirements']['peak_staff']),
                    'status': optimization['preparedness_plan']['status']
                },
                'optimization': {
                    'staff': {
                        'peak': int(optimization['staff_requirements']['peak_staff']),
                        'avg': float(optimization['staff_requirements']['avg_staff']),
                    },
                    'icu': {
                        'max_utilization_pct': float(optimization['bed_assessment']['max_icu_utilization'] * 100),
                        'critical_hours': len(optimization['bed_assessment']['critical_hours'])
                    },
                    'alerts': optimization['bed_assessment']['alerts']
                }
            }
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Saved: predictions_{filename_safe}.csv")
        print(f"✓ Saved: report_{filename_safe}.json\n")
        
        return predictions_df, optimization
    
    def _display_summary(self, predictions_df, optimization, period_name):
        """Display formatted summary"""
        print(f"Period: {period_name}")
        print(f"Duration: {len(predictions_df)} hours")
        print(f"From: {predictions_df['datetime'].iloc[0].strftime('%Y-%m-%d %H:%M')}")
        print(f"To: {predictions_df['datetime'].iloc[-1].strftime('%Y-%m-%d %H:%M')}\n")
        
        print("--- PREDICTIONS ---")
        print(f"  Total Admissions: {int(predictions_df['predicted_emergency_admissions'].sum())}")
        print(f"  Peak Hour Admissions: {int(predictions_df['predicted_emergency_admissions'].max())}")
        print(f"  Total ICU Demand: {int(predictions_df['predicted_icu_demand'].sum())}")
        print(f"  Peak ICU Demand: {int(predictions_df['predicted_icu_demand'].max())}\n")
        
        print("--- RESOURCE REQUIREMENTS ---")
        print(f"  Peak Staff: {optimization['staff_requirements']['peak_staff']} personnel")
        print(f"  Avg Staff/Hour: {optimization['staff_requirements']['avg_staff']:.1f}")
        print(f"  Status: {optimization['preparedness_plan']['status']}\n")
        
        if optimization['bed_assessment']['alerts']:
            print("--- ALERTS ---")
            for alert in optimization['bed_assessment']['alerts']:
                print(f"  [{alert['severity']}] {alert['message']}")


def interactive_menu():
    """Interactive menu for choosing prediction period"""
    print("\n" + "="*60)
    print("HOSPITAL EMERGENCY PREDICTION SYSTEM")
    print("Flexible Time Period Prediction")
    print("="*60 + "\n")
    
    predictor = FlexiblePredictor()
    
    while True:
        print("\n" + "-"*60)
        print("Choose Prediction Period:")
        print("-"*60)
        print("1. Next 24 hours")
        print("2. Next 48 hours (default)")
        print("3. Next weekend")
        print("4. Next week (7 days)")
        print("5. Custom period")
        print("6. Compare all periods")
        print("0. Exit")
        print("-"*60)
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == '1':
            predictor.predict_next_24_hours()
        elif choice == '2':
            predictor.predict_next_48_hours()
        elif choice == '3':
            predictor.predict_next_weekend()
        elif choice == '4':
            predictor.predict_next_week()
        elif choice == '5':
            try:
                hours = int(input("Enter number of hours to predict: "))
                if hours > 0 and hours <= 720:  # Max 30 days
                    description = input("Enter description (optional): ").strip() or f"{hours} Hours"
                    predictor.predict_custom(hours, description)
                else:
                    print("Please enter a value between 1 and 720 hours")
            except ValueError:
                print("Invalid input. Please enter a number.")
        elif choice == '6':
            print("\nGenerating predictions for all periods...\n")
            predictor.predict_next_24_hours()
            predictor.predict_next_48_hours()
            predictor.predict_next_weekend()
            predictor.predict_next_week()
            print("\n✓ All predictions complete!")
        elif choice == '0':
            print("\nExiting. Thank you!")
            break
        else:
            print("Invalid choice. Please try again.")


def quick_demo():
    """Quick demo of all prediction periods"""
    print("\n" + "="*60)
    print("QUICK DEMO: ALL PREDICTION PERIODS")
    print("="*60 + "\n")
    
    predictor = FlexiblePredictor()
    
    # Predict all periods
    print("\n1/4 - Predicting next 24 hours...")
    predictor.predict_next_24_hours()
    
    print("\n2/4 - Predicting next 48 hours...")
    predictor.predict_next_48_hours()
    
    print("\n3/4 - Predicting next weekend...")
    predictor.predict_next_weekend()
    
    print("\n4/4 - Predicting next week...")
    predictor.predict_next_week()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("\nAll predictions saved with filenames:")
    print("  • predictions_Next_24_Hours.csv")
    print("  • predictions_Next_48_Hours.csv")
    print("  • predictions_Next_Weekend_*.csv")
    print("  • predictions_Next_Week_7_Days.csv")
    print("\nCorresponding report files also generated.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        quick_demo()
    else:
        interactive_menu()
