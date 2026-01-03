"""
Hospital Emergency Prediction System - Gradio Interface
Deployable on Hugging Face Spaces
"""

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import prediction modules
from scripts.predict_flexible import FlexiblePredictor

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')

# Initialize predictor at startup
print("Initializing Hospital Prediction System...")
predictor = FlexiblePredictor()
print("System ready!")


def classify_load(predictions_df, hours=48):
    """Classify hospital load as HIGH/MEDIUM/LOW based on time period"""
    try:
        avg_admissions = float(predictions_df['predicted_emergency_admissions'].mean())
        peak_admissions = float(predictions_df['predicted_emergency_admissions'].max())
        avg_icu = float(predictions_df['predicted_icu_demand'].mean())
        peak_icu = float(predictions_df['predicted_icu_demand'].max())
        avg_staff = float(predictions_df['predicted_staff_workload'].mean())
        total_admissions = float(predictions_df['predicted_emergency_admissions'].sum())
    except Exception as e:
        return {
            'classification': "🟡 MEDIUM LOAD",
            'score': 5,
            'recommendation': f"Unable to classify: {str(e)}",
            'avg_admissions': 2.0,
            'peak_admissions': 3.0,
            'avg_icu': 0.5,
            'peak_icu': 1.0,
            'avg_staff': 0.5,
            'icu_utilization': 2.5,
            'total_admissions': 50
        }
    
    # Dynamic thresholds based on time period
    # Shorter periods = stricter thresholds, Longer periods = higher totals expected
    if hours <= 24:
        # 24 hours - focus on hourly averages
        adm_high, adm_med, adm_low = 2.8, 2.2, 1.5
        icu_high, icu_med, icu_low = 0.6, 0.35, 0.15
        peak_high, peak_med, peak_low = 4.0, 3.0, 2.0
        total_high, total_med = 55, 40
    elif hours <= 48:
        # 48 hours - balanced thresholds
        adm_high, adm_med, adm_low = 2.5, 1.9, 1.3
        icu_high, icu_med, icu_low = 0.45, 0.28, 0.12
        peak_high, peak_med, peak_low = 3.8, 2.8, 1.8
        total_high, total_med = 100, 75
    elif hours <= 168:
        # 7 days - weekly patterns matter
        adm_high, adm_med, adm_low = 2.3, 1.7, 1.1
        icu_high, icu_med, icu_low = 0.35, 0.22, 0.10
        peak_high, peak_med, peak_low = 3.5, 2.5, 1.6
        total_high, total_med = 350, 250
    else:
        # 14 days - long-term trends
        adm_high, adm_med, adm_low = 2.1, 1.5, 0.9
        icu_high, icu_med, icu_low = 0.30, 0.18, 0.08
        peak_high, peak_med, peak_low = 3.2, 2.3, 1.4
        total_high, total_med = 680, 500
    
    # Scoring based on dynamic thresholds
    score = 0
    
    # Admission scoring
    if avg_admissions > adm_high:
        score += 3
    elif avg_admissions > adm_med:
        score += 2
    elif avg_admissions > adm_low:
        score += 1
    
    # ICU scoring
    if avg_icu > icu_high:
        score += 3
    elif avg_icu > icu_med:
        score += 2
    elif avg_icu > icu_low:
        score += 1
    
    # Peak scoring
    if peak_admissions > peak_high:
        score += 3
    elif peak_admissions > peak_med:
        score += 2
    elif peak_admissions > peak_low:
        score += 1
    
    # Total admissions scoring (period-specific)
    if total_admissions > total_high:
        score += 2
    elif total_admissions > total_med:
        score += 1
    
    # Staff workload scoring
    if avg_staff > 1.0:
        score += 2
    elif avg_staff > 0.5:
        score += 1
    
    icu_utilization = (avg_icu / 20) * 100
    
    # Classification (13-point scale now)
    if score >= 9:
        load_class = "🔴 HIGH LOAD"
        recommendation = "⚠️ High patient volume expected. Increase staff by 40-50%. Activate overflow protocols."
    elif score >= 5:
        load_class = "🟡 MEDIUM LOAD"
        recommendation = "⚡ Moderate patient volume. Increase staff by 20-30%. Monitor ICU capacity."
    else:
        load_class = "🟢 LOW LOAD"
        recommendation = "✅ Normal patient volume. Standard staffing adequate."
    
    return {
        'classification': load_class,
        'score': score,
        'recommendation': recommendation,
        'avg_admissions': avg_admissions,
        'peak_admissions': peak_admissions,
        'avg_icu': avg_icu,
        'peak_icu': peak_icu,
        'avg_staff': avg_staff,
        'icu_utilization': icu_utilization,
        'total_admissions': total_admissions
    }


def create_visualizations(predictions_df, load_info, time_period, hours):
    """Create comprehensive visualization plots"""
    
    # Convert timestamp to datetime
    predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Hospital Emergency Predictions - {time_period}', fontsize=16, fontweight='bold')
    
    # Color scheme
    colors = {
        'admissions': '#FF6B6B',
        'icu': '#4ECDC4',
        'staff': '#95E1D3',
        'threshold': '#FFA07A'
    }
    
    # 1. Emergency Admissions Over Time
    ax1 = axes[0, 0]
    ax1.plot(predictions_df['timestamp'], 
             predictions_df['predicted_emergency_admissions'], 
             color=colors['admissions'], linewidth=2.5, label='Predicted Admissions')
    ax1.axhline(y=load_info['avg_admissions'], color=colors['threshold'], 
                linestyle='--', linewidth=1.5, label=f'Average: {load_info["avg_admissions"]:.1f}')
    ax1.fill_between(predictions_df['timestamp'], 
                     predictions_df['predicted_emergency_admissions'], 
                     alpha=0.3, color=colors['admissions'])
    ax1.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Admissions per Hour', fontsize=11, fontweight='bold')
    ax1.set_title('Emergency Admissions Forecast', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. ICU Demand Over Time
    ax2 = axes[0, 1]
    ax2.plot(predictions_df['timestamp'], 
             predictions_df['predicted_icu_demand'], 
             color=colors['icu'], linewidth=2.5, label='ICU Demand')
    ax2.axhline(y=load_info['avg_icu'], color=colors['threshold'], 
                linestyle='--', linewidth=1.5, label=f'Average: {load_info["avg_icu"]:.1f}')
    ax2.axhline(y=20, color='red', linestyle=':', linewidth=2, label='Capacity: 20 beds')
    ax2.fill_between(predictions_df['timestamp'], 
                     predictions_df['predicted_icu_demand'], 
                     alpha=0.3, color=colors['icu'])
    ax2.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax2.set_ylabel('ICU Beds Required', fontsize=11, fontweight='bold')
    ax2.set_title('ICU Demand Forecast', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Staff Workload Over Time
    ax3 = axes[1, 0]
    ax3.plot(predictions_df['timestamp'], 
             predictions_df['predicted_staff_workload'], 
             color=colors['staff'], linewidth=2.5, label='Staff Workload')
    ax3.axhline(y=load_info['avg_staff'], color=colors['threshold'], 
                linestyle='--', linewidth=1.5, label=f'Average: {load_info["avg_staff"]:.2f}')
    ax3.fill_between(predictions_df['timestamp'], 
                     predictions_df['predicted_staff_workload'], 
                     alpha=0.3, color=colors['staff'])
    ax3.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Workload Index', fontsize=11, fontweight='bold')
    ax3.set_title('Staff Workload Forecast', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Comparative Bar Chart - Peak vs Average
    ax4 = axes[1, 1]
    metrics = ['Admissions', 'ICU Beds', 'Staff\nWorkload']
    avg_values = [load_info['avg_admissions'], load_info['avg_icu'], load_info['avg_staff']]
    peak_values = [load_info['peak_admissions'], load_info['peak_icu'], 
                   predictions_df['predicted_staff_workload'].max()]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, avg_values, width, label='Average', 
                    color='#6C5CE7', alpha=0.8)
    bars2 = ax4.bar(x + width/2, peak_values, width, label='Peak', 
                    color='#FD79A8', alpha=0.8)
    
    ax4.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax4.set_title('Peak vs Average Comparison', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend(loc='upper right', framealpha=0.9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def create_live_metrics_plot(predictions_df, load_info):
    """Create real-time metrics dashboard visualization"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Real-Time Metrics Dashboard', fontsize=14, fontweight='bold')
    
    # Metrics data
    metrics = [
        ('Total Admissions', load_info['total_admissions'], '#FF6B6B'),
        ('Avg ICU Demand', load_info['avg_icu'], '#4ECDC4'),
        ('Peak Staff', load_info['avg_staff'], '#95E1D3')
    ]
    
    for idx, (label, value, color) in enumerate(metrics):
        ax = axes[idx]
        ax.bar([0], [value], color=color, alpha=0.7, width=0.6)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.text(0, value/2, f'{value:.1f}', 
                ha='center', va='center', fontsize=24, 
                fontweight='bold', color='white')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def predict_hospital_load(time_period):
    """Main prediction function"""
    try:
        # Map time period to hours
        period_mapping = {
            "24 Hours": 24,
            "48 Hours": 48,
            "7 Days": 168,
            "14 Days": 336
        }
        
        hours = period_mapping.get(time_period, 48)
        period_name = f"Next {time_period}"
        
        # Make predictions starting from now
        predictions_df, optimization = predictor._predict_period(
            hours=hours,
            period_name=period_name,
            start_offset=0
        )
        
        # Classify load with period-specific thresholds
        load_info = classify_load(predictions_df, hours)
        
        # Status message
        now = datetime.now()
        end_time = now + timedelta(hours=hours)
        status_msg = f"📊 **{now.strftime('%Y-%m-%d %H:%M')} → {end_time.strftime('%Y-%m-%d %H:%M')}**"
        
        # Create summary
        summary = f"""
## {load_info['classification']}

**Period:** {period_name} | **Duration:** {hours} hours | **Status:** {optimization['preparedness_plan']['status']}

### 📊 Predicted Metrics
| Metric | Value |
|--------|-------|
| Total Admissions | {int(predictions_df['predicted_emergency_admissions'].sum())} |
| Peak Hourly Admissions | {load_info['peak_admissions']:.1f} |
| Average Hourly Admissions | {load_info['avg_admissions']:.1f} |
| Peak ICU Demand | {load_info['peak_icu']:.1f} beds |
| Average ICU Demand | {load_info['avg_icu']:.1f} beds |
| ICU Utilization | {load_info['icu_utilization']:.1f}% |

### 👥 Staffing
| Metric | Value |
|--------|-------|
| Peak Staff | {optimization['staff_requirements']['peak_staff']} |
| Avg Staff/Hour | {optimization['staff_requirements']['avg_staff']:.1f} |

### 💡 Recommendation
{load_info['recommendation']}

**Load Score:** {load_info['score']}/13 | **Period:** {time_period}
"""

        # Alerts (period-adjusted thresholds)
        alerts_list = []
        # Dynamic alert thresholds based on period
        if hours <= 24:
            peak_alert, icu_alert, avg_alert = 3.5, 0.5, 2.5
        elif hours <= 48:
            peak_alert, icu_alert, avg_alert = 3.2, 0.4, 2.2
        elif hours <= 168:
            peak_alert, icu_alert, avg_alert = 3.0, 0.3, 2.0
        else:
            peak_alert, icu_alert, avg_alert = 2.8, 0.25, 1.8
            
        if load_info['peak_admissions'] > peak_alert:
            alerts_list.append("🔴 **CRITICAL:** High admission volume expected!")
        if load_info['peak_icu'] > icu_alert:
            alerts_list.append("🔴 **CRITICAL:** ICU capacity may be exceeded!")
        if load_info['avg_admissions'] > avg_alert:
            alerts_list.append("🟡 **WARNING:** Above-average patient flow")
        if optimization['preparedness_plan']['status'] == 'ELEVATED':
            alerts_list.append("🟡 **WARNING:** Elevated preparedness recommended")
        if not alerts_list:
            alerts_list.append("✅ No alerts - Normal operations expected")
        
        alerts_text = "\n\n".join(alerts_list)

        # Detailed analysis
        detailed = f"""
### Emergency Department
- **Total Admissions:** {int(predictions_df['predicted_emergency_admissions'].sum())}
- **Peak Hour:** {load_info['peak_admissions']:.1f} | **Average:** {load_info['avg_admissions']:.1f}

### ICU Capacity (20 beds)
- **Peak:** {load_info['peak_icu']:.1f} beds ({load_info['peak_icu']/20*100:.1f}%)
- **Average:** {load_info['avg_icu']:.1f} beds
- **Critical Hours:** {len(optimization['bed_assessment']['critical_hours'])}

### Staffing
- **Peak Required:** {optimization['staff_requirements']['peak_staff']}
- **Shifts:** {len(optimization['staff_requirements']['shifts'])}

### Score Breakdown ({time_period})
| Factor | Score |
|--------|-------|
| Admissions | /3 |
| ICU Demand | /3 |
| Peak Volume | /3 |
| Total Volume | /2 |
| Staff Load | /2 |
| **Total** | **{load_info['score']}/13** |

*Thresholds adjusted for {time_period} prediction window*
"""

        # CSV output
        csv_output = predictions_df.to_csv(index=False)
        
        # Save files to outputs folder
        os.makedirs("outputs/predictions", exist_ok=True)
        os.makedirs("outputs/reports", exist_ok=True)
        
        csv_path = f"outputs/predictions/predictions_{time_period.replace(' ', '_')}.csv"
        predictions_df.to_csv(csv_path, index=False)
        
        # JSON report
        report = {
            'period': period_name,
            'generated_at': datetime.now().isoformat(),
            'hours': hours,
            'load_classification': load_info['classification'],
            'load_score': load_info['score'],
            'metrics': {
                'total_admissions': int(predictions_df['predicted_emergency_admissions'].sum()),
                'peak_admissions': float(load_info['peak_admissions']),
                'avg_admissions': float(load_info['avg_admissions']),
                'peak_icu': float(load_info['peak_icu']),
                'avg_icu': float(load_info['avg_icu']),
                'peak_staff': int(optimization['staff_requirements']['peak_staff']),
                'status': optimization['preparedness_plan']['status']
            },
            'recommendation': load_info['recommendation']
        }
        json_output = json.dumps(report, indent=2)
        
        json_path = f"outputs/reports/report_{time_period.replace(' ', '_')}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create visualizations
        main_plot = create_visualizations(predictions_df, load_info, time_period, hours)
        metrics_plot = create_live_metrics_plot(predictions_df, load_info)
        
        # Real-time data text
        realtime_text = f"""
## 📡 Real-Time Prediction Data

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Current Metrics
- **Total Predicted Admissions:** {int(load_info['total_admissions'])}
- **Average Hourly Admissions:** {load_info['avg_admissions']:.2f}
- **Peak Hourly Admissions:** {load_info['peak_admissions']:.2f}
- **Average ICU Demand:** {load_info['avg_icu']:.2f} beds
- **Peak ICU Demand:** {load_info['peak_icu']:.2f} beds
- **ICU Utilization:** {load_info['icu_utilization']:.1f}%
- **Average Staff Workload:** {load_info['avg_staff']:.2f}

### Environmental Factors
- **Prediction Window:** {hours} hours
- **Period:** {time_period}
- **Status:** {optimization['preparedness_plan']['status']}

### Load Classification
- **Classification:** {load_info['classification']}
- **Load Score:** {load_info['score']}/13
- **Recommendation:** {load_info['recommendation']}
"""
        
        return (status_msg, summary, alerts_text, detailed, csv_output, csv_path, 
                json_output, json_path, main_plot, metrics_plot, realtime_text)
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        empty_fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Error generating plot', ha='center', va='center')
        return error_msg, "", "", "", "", None, "", None, empty_fig, empty_fig, ""


# Gradio Interface (compatible with Gradio 3.x)
with gr.Blocks(title="Hospital Emergency Prediction") as app:
    
    gr.Markdown("""
    # 🏥 Hospital Emergency Prediction System
    **AI-powered forecasting for emergency admissions, ICU demand, and staff workload**
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            time_period = gr.Radio(
                choices=["24 Hours", "48 Hours", "7 Days", "14 Days"],
                value="48 Hours",
                label="📅 Prediction Period"
            )
            
            predict_btn = gr.Button("🔮 Predict", variant="primary")
            
            gr.Markdown("""
            ---
            ### About
            - **Admissions**: Patient arrival forecasting
            - **ICU Demand**: Intensive care bed needs
            - **Staffing**: Optimal personnel allocation
            - **Models**: XGBoost trained on 180 days data
            - **Visualizations**: Real-time graphs & metrics
            """)
        
        with gr.Column(scale=2):
            status_output = gr.Markdown("*Select period and click Predict*")
            
            with gr.Tabs():
                with gr.Tab("📊 Summary"):
                    summary_output = gr.Markdown("Results will appear here...")
                
                with gr.Tab("📈 Forecasts"):
                    gr.Markdown("### Hourly Predictions Over Time")
                    main_plot_output = gr.Plot(label="Prediction Graphs")
                
                with gr.Tab("📡 Real-Time Data"):
                    gr.Markdown("### Live Metrics Dashboard")
                    metrics_plot_output = gr.Plot(label="Metrics")
                    realtime_output = gr.Markdown("Real-time data will appear here...")
                
                with gr.Tab("⚠️ Alerts"):
                    alerts_output = gr.Markdown("Alerts will appear here...")
                
                with gr.Tab("📋 Details"):
                    detailed_output = gr.Markdown("Detailed metrics...")
                
                with gr.Tab("💾 Download"):
                    csv_output = gr.Textbox(label="CSV Preview", lines=6)
                    csv_download = gr.File(label="📥 CSV")
                    json_output = gr.Textbox(label="JSON Preview", lines=6)
                    json_download = gr.File(label="📥 JSON")
    
    predict_btn.click(
        fn=predict_hospital_load,
        inputs=[time_period],
        outputs=[status_output, summary_output, alerts_output, detailed_output, 
                 csv_output, csv_download, json_output, json_download,
                 main_plot_output, metrics_plot_output, realtime_output]
    )

if __name__ == "__main__":
    app.launch()
