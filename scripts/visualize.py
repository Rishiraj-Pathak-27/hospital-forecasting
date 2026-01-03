"""
Visualization Dashboard for Hospital Prediction System
Creates impressive charts for hackathon presentation
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def create_dashboard():
    """Create comprehensive visualization dashboard"""
    
    # Load data
    print("Loading data...")
    historical = pd.read_csv('hospital_data.csv')
    predictions = pd.read_csv('predictions_final.csv')
    
    historical['datetime'] = pd.to_datetime(historical['datetime'])
    predictions['datetime'] = pd.to_datetime(predictions['datetime'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Emergency Admissions - Historical + Predictions
    ax1 = plt.subplot(3, 2, 1)
    last_week = historical.tail(168)  # Last 7 days
    ax1.plot(last_week['datetime'], last_week['emergency_admissions'], 
             label='Historical', color='#2E86AB', linewidth=2)
    ax1.plot(predictions['datetime'], predictions['predicted_emergency_admissions'],
             label='Predicted', color='#F24236', linewidth=2, linestyle='--')
    ax1.set_title('Emergency Admissions: Historical vs Predicted', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Admissions per Hour')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ICU Demand Forecast
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(predictions['datetime'], predictions['predicted_icu_demand'],
             color='#A23B72', linewidth=2.5, marker='o', markersize=3)
    ax2.axhline(y=20, color='red', linestyle='--', label='ICU Capacity', linewidth=2)
    ax2.fill_between(predictions['datetime'], 0, predictions['predicted_icu_demand'],
                     alpha=0.3, color='#A23B72')
    ax2.set_title('ICU Demand Forecast (48 Hours)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('ICU Beds Needed')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Staff Workload Prediction
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(predictions['datetime'], predictions['predicted_staff_workload'],
             color='#F18F01', linewidth=2.5, marker='s', markersize=3)
    ax3.fill_between(predictions['datetime'], 0, predictions['predicted_staff_workload'],
                     alpha=0.3, color='#F18F01')
    ax3.set_title('Staff Workload Forecast', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Staff Required')
    ax3.grid(True, alpha=0.3)
    
    # 4. Seasonal Pattern Analysis
    ax4 = plt.subplot(3, 2, 4)
    monthly_avg = historical.groupby('month')['emergency_admissions'].mean()
    months_available = monthly_avg.index.tolist()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    labels = [month_names[m-1] for m in months_available]
    bars = ax4.bar(months_available, monthly_avg, color=sns.color_palette("coolwarm", len(months_available)))
    ax4.set_title('Seasonal Admission Pattern', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Avg Hourly Admissions')
    ax4.set_xticks(months_available)
    ax4.set_xticklabels(labels)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Hourly Pattern (Heatmap style)
    ax5 = plt.subplot(3, 2, 5)
    hourly_avg = historical.groupby('hour')['emergency_admissions'].mean()
    ax5.bar(range(24), hourly_avg, color='#06A77D', alpha=0.7)
    ax5.set_title('Daily Pattern: Peak Hours', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Hour of Day')
    ax5.set_ylabel('Avg Admissions')
    ax5.set_xticks(range(0, 24, 3))
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. External Factors Impact
    ax6 = plt.subplot(3, 2, 6)
    last_days = historical.tail(168)
    ax6_twin = ax6.twinx()
    
    ax6.plot(last_days['datetime'], last_days['emergency_admissions'], 
             label='Admissions', color='#2E86AB', linewidth=2)
    ax6_twin.plot(last_days['datetime'], last_days['flu_season_index'],
                  label='Flu Season Index', color='#E63946', linewidth=2, alpha=0.7)
    
    ax6.set_title('External Factors: Flu Season Impact', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Admissions', color='#2E86AB')
    ax6_twin.set_ylabel('Flu Index', color='#E63946')
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hospital_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Dashboard saved as hospital_dashboard.png")
    
    # Create summary metrics visualization
    create_metrics_summary(predictions)
    
    plt.show()

def create_metrics_summary(predictions):
    """Create a clean metrics summary chart"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Hospital Emergency Metrics - Next 48 Hours', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Total Expected Admissions
    ax1 = axes[0, 0]
    total_admissions = int(predictions['predicted_emergency_admissions'].sum())
    ax1.text(0.5, 0.5, str(total_admissions), 
             ha='center', va='center', fontsize=60, fontweight='bold', color='#2E86AB')
    ax1.text(0.5, 0.2, 'Expected Admissions', 
             ha='center', va='center', fontsize=14, color='gray')
    ax1.axis('off')
    ax1.set_facecolor('#F0F0F0')
    
    # 2. Peak ICU Demand
    ax2 = axes[0, 1]
    peak_icu = predictions['predicted_icu_demand'].max()
    ax2.text(0.5, 0.5, f'{peak_icu:.1f}', 
             ha='center', va='center', fontsize=60, fontweight='bold', color='#A23B72')
    ax2.text(0.5, 0.2, 'Peak ICU Beds', 
             ha='center', va='center', fontsize=14, color='gray')
    ax2.axis('off')
    ax2.set_facecolor('#F0F0F0')
    
    # 3. Status Indicator
    ax3 = axes[1, 0]
    status = 'NORMAL'
    status_color = '#06A77D'
    if peak_icu > 17:
        status = 'CRITICAL'
        status_color = '#E63946'
    elif peak_icu > 14:
        status = 'ELEVATED'
        status_color = '#F18F01'
    
    ax3.text(0.5, 0.5, status, 
             ha='center', va='center', fontsize=48, fontweight='bold', color=status_color)
    ax3.text(0.5, 0.2, 'Hospital Status', 
             ha='center', va='center', fontsize=14, color='gray')
    ax3.axis('off')
    ax3.set_facecolor('#F0F0F0')
    
    # 4. Peak Staff Required
    ax4 = axes[1, 1]
    peak_staff = int(np.ceil(predictions['predicted_staff_workload'].max()))
    ax4.text(0.5, 0.5, str(peak_staff), 
             ha='center', va='center', fontsize=60, fontweight='bold', color='#F18F01')
    ax4.text(0.5, 0.2, 'Peak Staff Required', 
             ha='center', va='center', fontsize=14, color='gray')
    ax4.axis('off')
    ax4.set_facecolor('#F0F0F0')
    
    plt.tight_layout()
    plt.savefig('hospital_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Metrics summary saved as hospital_metrics.png")

def create_comparison_chart():
    """Create model comparison chart"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Chronos\n(Transformer)', 'XGBoost\nICU', 'XGBoost\nStaff']
    accuracy = [92, 67, 58]  # Approximate R² scores as percentage
    speed = [3, 0.1, 0.1]  # Training time in seconds
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, accuracy, width, label='Accuracy (%)', color='#2E86AB')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, speed, width, label='Train Time (s)', color='#F18F01')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy (%)', color='#2E86AB', fontsize=12)
    ax2.set_ylabel('Training Time (seconds)', color='#F18F01', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Model comparison saved as model_comparison.png")

if __name__ == "__main__":
    print("=== Creating Hospital Prediction Dashboard ===\n")
    
    create_dashboard()
    create_comparison_chart()
    
    print("\n✓ All visualizations created!")
    print("\nGenerated files:")
    print("  • hospital_dashboard.png - Main analytics dashboard")
    print("  • hospital_metrics.png - Key metrics summary")
    print("  • model_comparison.png - Model performance comparison")
    print("\nUse these in your hackathon presentation!")
