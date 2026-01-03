"""
Simple Command-Line Interface for Quick Predictions
"""

import sys
from predict_flexible import FlexiblePredictor

def main():
    if len(sys.argv) < 2:
        print("\nUsage: python predict.py <period>")
        print("\nAvailable periods:")
        print("  24h       - Next 24 hours")
        print("  48h       - Next 48 hours (default)")
        print("  weekend   - Next weekend")
        print("  week      - Next week (7 days)")
        print("  <number>  - Custom hours (e.g., 72 for 3 days)")
        print("\nExamples:")
        print("  python predict.py 24h")
        print("  python predict.py weekend")
        print("  python predict.py 120    (5 days)")
        return
    
    period = sys.argv[1].lower()
    
    predictor = FlexiblePredictor()
    
    if period == '24h' or period == '24':
        predictor.predict_next_24_hours()
    elif period == '48h' or period == '48':
        predictor.predict_next_48_hours()
    elif period == 'weekend':
        predictor.predict_next_weekend()
    elif period == 'week' or period == '7d':
        predictor.predict_next_week()
    else:
        try:
            hours = int(period)
            if hours > 0 and hours <= 720:
                predictor.predict_custom(hours, f"Next {hours} Hours")
            else:
                print("Error: Hours must be between 1 and 720")
        except ValueError:
            print(f"Error: Unknown period '{period}'")
            print("Use: 24h, 48h, weekend, week, or a number")

if __name__ == "__main__":
    main()
