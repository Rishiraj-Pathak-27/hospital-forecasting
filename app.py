"""
Hospital Emergency Prediction System - Gradio Web Interface
Deployable on Hugging Face Spaces
"""

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Lazy loading - only initialize when first prediction is made
predictor = None

def get_predictor():
    """Lazy load predictor on first use"""
    global predictor
    if predictor is None:
        print("Loading models and data...")
        from scripts.predict_flexible import FlexiblePredictor
        predictor = FlexiblePredictor()
        print("System ready!")
    return predictor

def validate_date_selection(date_str, hours_str):
    """
    Validate date selection and convert to hours ahead
    Returns: (hours, error_message)
    """
    try:
        selected_date = datetime.strptime(date_str, "%Y-%m-%d")
        current_date = datetime.now()
        
        # Calculate hours difference
        delta = selected_date - current_date
        hours_ahead = int(delta.total_seconds() / 3600)
        
        # Validate range
        if hours_ahead < 0:
            return None, "❌ Error: Cannot predict for past dates!"
        
        if hours_ahead > 336:  # Max 2 weeks (14 days)
            return None, "❌ Error: Maximum prediction range is 2 weeks (336 hours)!"
        
        # Validate against allowed options
        if hours_str == "24h":
            if hours_ahead < 1 or hours_ahead > 24:
                return None, "❌ Error: For 24h option, date must be within next 24 hours!"
            return 24, None
            
        elif hours_str == "48h":
            if hours_ahead < 1 or hours_ahead > 48:
                return None, "❌ Error: For 48h option, date must be within next 48 hours!"
            return 48, None
            
        elif hours_str == "week":
            if hours_ahead < 1 or hours_ahead > 168:
                return None, "❌ Error: For 1 week option, date must be within next 7 days!"
            return 168, None
            
        elif hours_str == "2weeks":
            if hours_ahead < 169 or hours_ahead > 336:
                return None, "❌ Error: For week-to-next-week option, date must be 8-14 days ahead!"
            return hours_ahead, None
            
        else:
            return None, "❌ Error: Invalid time period selected!"
            
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def classify_load(predictions_df):
    """
    Automatically classify hospital load as HIGH/MEDIUM/LOW
    Based on predicted admissions and ICU demand
    """
    avg_admissions = predictions_df['predicted_emergency_admissions'].mean()
    peak_admissions = predictions_df['predicted_emergency_admissions'].max()
    avg_icu = predictions_df['predicted_icu_demand'].mean()
    peak_icu = predictions_df['predicted_icu_demand'].max()
    
    # Classification thresholds
    high_admission_threshold = 3.0  # per hour
    high_icu_threshold = 1.0
    medium_admission_threshold = 2.0
    medium_icu_threshold = 0.5
    
    # Scoring system
    score = 0
    
    if peak_admissions > high_admission_threshold:
        score += 3
    elif peak_admissions > medium_admission_threshold:
        score += 2
    else:
        score += 1
        
    if peak_icu > high_icu_threshold:
        score += 3
    elif peak_icu > medium_icu_threshold:
        score += 2
    else:
        score += 1
    
    # Calculate ICU utilization
    icu_utilization = (avg_icu / 20) * 100  # 20 is ICU capacity
    
    if icu_utilization > 70:
        score += 3
    elif icu_utilization > 50:
        score += 2
    else:
        score += 1
    
    # Final classification
    if score >= 8:
        load_class = "🔴 HIGH LOAD"
        load_color = "#FF4444"
        recommendation = "⚠️ High patient volume expected. Increase staff by 40-50%. Activate overflow protocols."
    elif score >= 5:
        load_class = "🟡 MEDIUM LOAD"
        load_color = "#FFB347"
        recommendation = "⚡ Moderate patient volume. Increase staff by 20-30%. Monitor ICU capacity."
    else:
        load_class = "🟢 LOW LOAD"
        load_color = "#44FF44"
        recommendation = "✅ Normal patient volume. Standard staffing adequate."
    
    return {
        'classification': load_class,
        'color': load_color,
        'score': score,
        'recommendation': recommendation,
        'avg_admissions': avg_admissions,
        'peak_admissions': peak_admissions,
        'avg_icu': avg_icu,
        'peak_icu': peak_icu,
        'icu_utilization': icu_utilization
    }

def predict_hospital_load(date_input, time_period, auto_classify):
    """
    Main prediction function for Gradio interface
    """
    try:
        # Lazy load predictor on first use
        pred = get_predictor()
        
        # If auto-classify is enabled, use default 48h prediction
        if auto_classify:
            hours = 48
            period_name = "Next 48 Hours (Auto)"
            status_msg = "📊 Automatic load classification - Next 48 hours"
        else:
            # Validate date selection
            hours, error = validate_date_selection(date_input, time_period)
            if error:
                return error, "", "", "", ""
            
            period_names = {
                "24h": "Next 24 Hours",
                "48h": "Next 48 Hours", 
                "week": "Next Week (7 Days)",
                "2weeks": "Week to Next Week"
            }
            period_name = period_names.get(time_period, f"Next {hours} Hours")
            status_msg = f"✅ Prediction for: {period_name} (starting {date_input})"
        
        # Make predictions
        predictions_df, optimization = pred._predict_period(
            hours=hours,
            period_name=period_name,
            start_offset=0
        )
        
        # Classify load
        load_info = classify_load(predictions_df)
        
        # Create summary
        summary = f"""
## {load_info['classification']}

**Period:** {period_name}  
**Duration:** {hours} hours  
**Status:** {optimization['preparedness_plan']['status']}

### 📊 Predicted Metrics
- **Total Admissions:** {int(predictions_df['predicted_emergency_admissions'].sum())}
- **Peak Hourly Admissions:** {load_info['peak_admissions']:.1f}
- **Average Hourly Admissions:** {load_info['avg_admissions']:.1f}
- **Peak ICU Demand:** {load_info['peak_icu']:.1f} beds
- **Average ICU Demand:** {load_info['avg_icu']:.1f} beds
- **ICU Utilization:** {load_info['icu_utilization']:.1f}%

### 👥 Staffing Requirements
- **Peak Staff Needed:** {optimization['staff_requirements']['peak_staff']} personnel
- **Average Staff/Hour:** {optimization['staff_requirements']['avg_staff']:.1f}
- **Total Staff-Hours (24h):** {optimization['staff_requirements'].get('total_staff_24h', 'N/A')}

### 💡 Recommendation
{load_info['recommendation']}
"""

        # Create alerts section
        alerts_text = ""
        if optimization['bed_assessment']['alerts']:
            alerts_text = "### ⚠️ Alerts\n"
            for alert in optimization['bed_assessment']['alerts']:
                alerts_text += f"- **[{alert['severity']}]** {alert['message']}\n"
                alerts_text += f"  - Action: {alert['action']}\n"
        
        # Create detailed metrics
        detailed = f"""
### Detailed Analysis

**Emergency Department:**
- Total Expected Admissions: {int(predictions_df['predicted_emergency_admissions'].sum())}
- Peak Hour: {load_info['peak_admissions']:.1f} admissions
- Average per Hour: {load_info['avg_admissions']:.1f}

**ICU Capacity:**
- Peak Demand: {load_info['peak_icu']:.1f} / 20 beds ({load_info['icu_utilization']:.1f}%)
- Average Demand: {load_info['avg_icu']:.1f} beds
- Critical Hours: {len(optimization['bed_assessment']['critical_hours'])}

**Resource Optimization:**
- Recommended Shifts: {len(optimization['staff_requirements']['shifts'])}
- Peak Staff: {optimization['staff_requirements']['peak_staff']}
- Status: {optimization['preparedness_plan']['status']}

**Load Classification Score:** {load_info['score']}/9
- Score 8-9: HIGH LOAD 🔴
- Score 5-7: MEDIUM LOAD 🟡
- Score 0-4: LOW LOAD 🟢
"""

        # Convert predictions to CSV format for download
        csv_output = predictions_df.to_csv(index=False)
        
        # Create JSON report
        report = {
            'period': period_name,
            'generated_at': datetime.now().isoformat(),
            'hours': hours,
            'load_classification': load_info['classification'],
            'load_score': load_info['score'],
            'summary': {
                'total_admissions': int(predictions_df['predicted_emergency_admissions'].sum()),
                'peak_admissions': float(load_info['peak_admissions']),
                'avg_admissions': float(load_info['avg_admissions']),
                'peak_icu': float(load_info['peak_icu']),
                'icu_utilization': float(load_info['icu_utilization']),
                'peak_staff': int(optimization['staff_requirements']['peak_staff']),
                'status': optimization['preparedness_plan']['status']
            },
            'recommendation': load_info['recommendation']
        }
        json_output = json.dumps(report, indent=2)
        
        return status_msg, summary, alerts_text, detailed, csv_output, json_output
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nPlease try again or contact support."
        return error_msg, "", "", "", "", ""

# Create Gradio Interface
with gr.Blocks(title="Hospital Emergency Prediction System", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("# 🏥 Hospital Emergency Prediction")
    
    with gr.Row():
        with gr.Column(scale=1):
            auto_mode = gr.Checkbox(
                label="Auto-Classify",
                value=True
            )
            
            date_input = gr.Textbox(
                label="Date (YYYY-MM-DD)",
                value=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            )
            
            time_period = gr.Radio(
                choices=[
                    ("24 Hours", "24h"),
                    ("48 Hours", "48h"),
                    ("7 Days", "week"),
                    ("14 Days", "2weeks")
                ],
                value="48h",
                label="Period"
            )
            
            predict_btn = gr.Button("Predict", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            status_output = gr.Markdown("Click Predict to start")
            
            with gr.Tabs():
                with gr.Tab("📊 Summary"):
                    summary_output = gr.Markdown("Results will appear here...")
                
                with gr.Tab("⚠️ Alerts"):
                    alerts_output = gr.Markdown("Alerts will appear here...")
                
                with gr.Tab("📈 Detailed Analysis"):
                    detailed_output = gr.Markdown("Detailed metrics will appear here...")
                
                with gr.Tab("💾 Download Data"):
                    gr.Markdown("### Download Predictions")
                    csv_output = gr.Textbox(
                        label="CSV Data (Copy or Download)",
                        lines=10,
                        max_lines=20,
                        placeholder="Prediction data in CSV format..."
                    )
                    json_output = gr.Textbox(
                        label="JSON Report (Copy or Download)",
                        lines=10,
                        max_lines=20,
                        placeholder="Full report in JSON format..."
                    )
    
    # Connect button to prediction function
    predict_btn.click(
        fn=predict_hospital_load,
        inputs=[date_input, time_period, auto_mode],
        outputs=[status_output, summary_output, alerts_output, detailed_output, csv_output, json_output]
    )
    


# Launch settings
if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",  # Use localhost instead
        server_port=7860,
        inbrowser=False,  # Don't auto-open browser
        prevent_thread_lock=False
    )
