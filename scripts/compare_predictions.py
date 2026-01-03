"""
Compare predictions across different time periods
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os

def compare_all_periods():
    """Create comparison visualization for all prediction periods"""
    
    print("Loading all prediction files...")
    
    # Find all prediction files
    pred_files = glob('predictions_*.csv')
    
    if not pred_files:
        print("No prediction files found. Run predictions first!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Hospital Prediction Comparison: All Time Periods', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    colors = sns.color_palette("husl", len(pred_files))
    
    # 1. Emergency Admissions Comparison
    ax1 = axes[0, 0]
    for i, file in enumerate(sorted(pred_files)):
        df = pd.read_csv(file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        period_name = file.replace('predictions_', '').replace('.csv', '').replace('_', ' ')
        
        # Plot first 48 hours for comparison
        plot_df = df.head(48)
        ax1.plot(range(len(plot_df)), plot_df['predicted_emergency_admissions'], 
                label=period_name, linewidth=2, alpha=0.8, color=colors[i])
    
    ax1.set_title('Emergency Admissions Forecast', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Hours Ahead')
    ax1.set_ylabel('Admissions per Hour')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. ICU Demand Comparison
    ax2 = axes[0, 1]
    for i, file in enumerate(sorted(pred_files)):
        df = pd.read_csv(file)
        period_name = file.replace('predictions_', '').replace('.csv', '').replace('_', ' ')
        plot_df = df.head(48)
        ax2.plot(range(len(plot_df)), plot_df['predicted_icu_demand'], 
                label=period_name, linewidth=2, alpha=0.8, color=colors[i])
    
    ax2.axhline(y=20, color='red', linestyle='--', label='ICU Capacity', linewidth=2)
    ax2.set_title('ICU Demand Forecast', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Hours Ahead')
    ax2.set_ylabel('ICU Beds Needed')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Staff Workload Comparison
    ax3 = axes[1, 0]
    for i, file in enumerate(sorted(pred_files)):
        df = pd.read_csv(file)
        period_name = file.replace('predictions_', '').replace('.csv', '').replace('_', ' ')
        plot_df = df.head(48)
        ax3.plot(range(len(plot_df)), plot_df['predicted_staff_workload'], 
                label=period_name, linewidth=2, alpha=0.8, color=colors[i])
    
    ax3.set_title('Staff Workload Forecast', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Hours Ahead')
    ax3.set_ylabel('Staff Required')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary Statistics Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    summary_data = []
    for file in sorted(pred_files):
        df = pd.read_csv(file)
        period_name = file.replace('predictions_', '').replace('.csv', '').replace('_', ' ')
        
        summary_data.append([
            period_name,
            len(df),
            f"{df['predicted_emergency_admissions'].sum():.0f}",
            f"{df['predicted_icu_demand'].max():.1f}",
            f"{df['predicted_staff_workload'].max():.0f}"
        ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['Period', 'Hours', 'Total Admits', 'Peak ICU', 'Peak Staff'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(summary_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Comparison chart saved as prediction_comparison.png")
    
    # Create individual period summaries
    create_period_summary_cards()

def create_period_summary_cards():
    """Create summary cards for each period"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Prediction Period Summary Cards', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    pred_files = sorted(glob('predictions_*.csv'))[:6]  # Max 6 cards
    
    for idx, file in enumerate(pred_files):
        if idx >= 6:
            break
            
        ax = axes[idx]
        df = pd.read_csv(file)
        period_name = file.replace('predictions_', '').replace('.csv', '').replace('_', ' ')
        
        # Calculate statistics
        total_admits = int(df['predicted_emergency_admissions'].sum())
        peak_icu = df['predicted_icu_demand'].max()
        peak_staff = int(df['predicted_staff_workload'].max())
        hours = len(df)
        
        # Determine status color
        status_color = '#06A77D' if peak_icu < 17 else '#F18F01' if peak_icu < 19 else '#E63946'
        
        # Clear axis
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add colored background
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, 
                                   facecolor=status_color, alpha=0.1, edgecolor=status_color, linewidth=2))
        
        # Add text
        ax.text(0.5, 0.85, period_name, ha='center', va='top', 
               fontsize=12, fontweight='bold', color=status_color)
        ax.text(0.5, 0.68, f'{hours} hours', ha='center', va='top', fontsize=10, color='gray')
        
        ax.text(0.5, 0.52, f'{total_admits}', ha='center', va='center', 
               fontsize=24, fontweight='bold', color='#2E86AB')
        ax.text(0.5, 0.42, 'Total Admissions', ha='center', va='center', fontsize=8, color='gray')
        
        ax.text(0.25, 0.25, f'{peak_icu:.1f}', ha='center', va='center', 
               fontsize=18, fontweight='bold', color='#A23B72')
        ax.text(0.25, 0.15, 'Peak ICU', ha='center', va='center', fontsize=7, color='gray')
        
        ax.text(0.75, 0.25, f'{peak_staff}', ha='center', va='center', 
               fontsize=18, fontweight='bold', color='#F18F01')
        ax.text(0.75, 0.15, 'Peak Staff', ha='center', va='center', fontsize=7, color='gray')
    
    # Hide unused subplots
    for idx in range(len(pred_files), 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('period_summary_cards.png', dpi=300, bbox_inches='tight')
    print("✓ Summary cards saved as period_summary_cards.png")

if __name__ == "__main__":
    print("=== Creating Prediction Comparison Visualizations ===\n")
    compare_all_periods()
    print("\n✓ All comparison visualizations created!")
    print("\nGenerated files:")
    print("  • prediction_comparison.png - Side-by-side comparison")
    print("  • period_summary_cards.png - Quick reference cards")
