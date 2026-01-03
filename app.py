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

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import prediction modules
from scripts.predict_flexible import FlexiblePredictor

# Initialize predictor at startup
print("Initializing Hospital Prediction System...")
predictor = FlexiblePredictor()
print("System ready!")


def classify_load(predictions_df):
    """Classify hospital load as HIGH/MEDIUM/LOW"""
    try:
        avg_admissions = float(predictions_df['predicted_emergency_admissions'].mean())
        peak_admissions = float(predictions_df['predicted_emergency_admissions'].max())
        avg_icu = float(predictions_df['predicted_icu_demand'].mean())
        peak_icu = float(predictions_df['predicted_icu_demand'].max())
        avg_staff = float(predictions_df['predicted_staff_workload'].mean())
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
            'icu_utilization': 2.5
        }
    
    # Scoring based on metrics (adjusted thresholds for better sensitivity)
    score = 0
    
    # Admission scoring (lowered thresholds)
    if avg_admissions > 2.5:
        score += 3
    elif avg_admissions > 1.8:
        score += 2
    elif avg_admissions > 1.0:
        score += 1
    
    # ICU scoring (lowered thresholds)
    if avg_icu > 0.5:
        score += 3
    elif avg_icu > 0.25:
        score += 2
    elif avg_icu > 0.1:
        score += 1
    
    # Peak scoring (lowered thresholds)
    if peak_admissions > 3.5:
        score += 3
    elif peak_admissions > 2.5:
        score += 2
    elif peak_admissions > 1.5:
        score += 1
    
    # Staff workload scoring (new factor)
    if avg_staff > 1.0:
        score += 2
    elif avg_staff > 0.5:
        score += 1
    
    icu_utilization = (avg_icu / 20) * 100
    
    # Classification (adjusted score ranges for 11-point scale)
    if score >= 8:
        load_class = "🔴 HIGH LOAD"
        recommendation = "⚠️ High patient volume expected. Increase staff by 40-50%. Activate overflow protocols."
    elif score >= 4:
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
        'icu_utilization': icu_utilization
    }


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
        
        # Classify load
        load_info = classify_load(predictions_df)
        
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

**Load Score:** {load_info['score']}/11
"""

        # Alerts (adjusted thresholds)
        alerts_list = []
        if load_info['peak_admissions'] > 3.0:
            alerts_list.append("🔴 **CRITICAL:** High admission volume expected!")
        if load_info['peak_icu'] > 0.5:
            alerts_list.append("🔴 **CRITICAL:** ICU capacity may be exceeded!")
        if load_info['avg_admissions'] > 2.0:
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

### Score Breakdown
| Factor | Score |
|--------|-------|
| Admissions | {min(3, int(load_info['avg_admissions']/0.9))}/3 |
| ICU | {min(3, int(load_info['avg_icu']/0.15))}/3 |
| Peaks | {min(3, int(load_info['peak_admissions']/1.2))}/3 |
| Staff | {min(2, int(load_info['avg_staff']/0.5))}/2 |
| **Total** | **{load_info['score']}/11** |
"""

        # CSV output
        csv_output = predictions_df.to_csv(index=False)
        
        # Save files
        csv_path = f"predictions_{time_period.replace(' ', '_')}.csv"
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
        
        json_path = f"report_{time_period.replace(' ', '_')}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return status_msg, summary, alerts_text, detailed, csv_output, csv_path, json_output, json_path
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, "", "", "", "", None, "", None


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
            """)
        
        with gr.Column(scale=2):
            status_output = gr.Markdown("*Select period and click Predict*")
            
            with gr.Tabs():
                with gr.Tab("📊 Summary"):
                    summary_output = gr.Markdown("Results will appear here...")
                
                with gr.Tab("⚠️ Alerts"):
                    alerts_output = gr.Markdown("Alerts will appear here...")
                
                with gr.Tab("📈 Details"):
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
                 csv_output, csv_download, json_output, json_download]
    )

if __name__ == "__main__":
    app.launch()
