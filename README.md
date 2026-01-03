---
title: Hospital Emergency Prediction
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "3.50.2"
app_file: app.py
pinned: false
---

# 🏥 Hospital Emergency Prediction System

> **AI-Powered Forecasting for Emergency Admissions, ICU Demand, and Staff Workload**

A comprehensive machine learning system that predicts hospital emergency department metrics using XGBoost models with period-specific thresholds for 24h, 48h, 7-day, and 14-day forecasting.

[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/rishirajpathak/hospital-ai-forecasting)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Rishiraj-Pathak-27/hospital-forecasting)

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [Dataset](#dataset)
- [Input Format](#input-format)
- [Output Format](#output-format)
- [Installation](#installation)
- [Usage](#usage)
- [File Locations](#file-locations)
- [Model Performance](#model-performance)

---

## ✨ Features

- **Multi-Period Forecasting**: 24 hours, 48 hours, 7 days, 14 days
- **Period-Specific Thresholds**: Dynamic classification based on prediction window
- **Three Key Predictions**:
  - Emergency Admissions
  - ICU Bed Demand
  - Staff Workload Requirements
- **Load Classification**: HIGH/MEDIUM/LOW with adaptive scoring (13-point scale)
- **Resource Optimization**: Staff allocation and bed management recommendations
- **Interactive Web Interface**: Built with Gradio for easy deployment
- **Export Capabilities**: CSV and JSON outputs

---

## 📁 Project Structure

```
hospital-ai-forecasting/
├── app.py                          # Main Gradio application
├── config.py                       # Configuration parameters
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── data/                           # Training data
│   ├── hospital_data.csv          # Raw synthetic hospital data (4,321 records)
│   └── hospital_data_ml.csv       # Feature-engineered ML-ready data
│
├── models/                         # Trained models
│   ├── icu_demand_model.pkl       # ICU demand predictor (XGBoost)
│   └── staff_workload_model.pkl   # Staff workload predictor (XGBoost)
│
├── scripts/                        # Python modules
│   ├── predict_flexible.py        # Main prediction engine
│   ├── xgboost_predictors.py      # XGBoost model classes
│   ├── resource_optimizer.py      # Resource optimization logic
│   └── train_models.py            # Model training script
│
└── outputs/                        # Generated files (gitignored)
    ├── predictions/                # CSV prediction files
    │   ├── predictions_24_Hours.csv
    │   ├── predictions_48_Hours.csv
    │   ├── predictions_7_Days.csv
    │   └── predictions_14_Days.csv
    └── reports/                    # JSON report files
        ├── report_24_Hours.json
        ├── report_48_Hours.json
        ├── report_7_Days.json
        └── report_14_Days.json
```

---

## 🤖 Models Used

### 1. **ICU Demand Predictor** (`icu_demand_model.pkl`)

**Model Type**: XGBoost Regressor  
**Source**: Custom trained on synthetic hospital data  
**Framework**: `xgboost==2.0.0`  

**Model Parameters**:
```python
{
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42
}
```

**Input Features** (13):
- Temporal: `hour`, `day_of_week`, `month`, `is_weekend`
- Environmental: `temperature`, `flu_season_index`, `air_quality_index`
- Lagged: `emergency_admissions_lag_1h`, `emergency_admissions_lag_7h`, `icu_demand_lag_1h`, `icu_demand_lag_7h`
- Rolling: `emergency_admissions_rolling_3h`, `emergency_admissions_rolling_7h`

**Output**: ICU bed demand (continuous value, 0-20 beds)

**Performance**:
- MAE: 0.24
- R²: 0.32
- Cross-validation: 5-fold Time Series Split

---

### 2. **Staff Workload Predictor** (`staff_workload_model.pkl`)

**Model Type**: XGBoost Regressor  
**Source**: Custom trained on synthetic hospital data  
**Framework**: `xgboost==2.0.0`  

**Model Parameters**: Same as ICU Demand model

**Input Features** (13):
- Temporal: `hour`, `day_of_week`, `month`, `is_weekend`
- Environmental: `temperature`, `flu_season_index`, `air_quality_index`
- Lagged: `emergency_admissions_lag_1h`, `emergency_admissions_lag_7h`, `icu_demand_lag_1h`
- Rolling: `emergency_admissions_rolling_3h`, `emergency_admissions_rolling_7h`
- Additional: `bed_occupancy`

**Output**: Staff workload index (continuous value)

**Performance**:
- MAE: 0.45
- R²: 0.37
- Cross-validation: 5-fold Time Series Split

---

## 📊 Dataset

### **Training Dataset**: `hospital_data_ml.csv`

**Source**: Synthetically generated using realistic hospital patterns  
**Size**: 4,321 records (180 days of hourly data)  
**Period**: 6 months of historical data  

**Features** (22 columns):

| Feature | Type | Description |
|---------|------|-------------|
| `timestamp` | datetime | Hourly timestamp |
| `hour` | int | Hour of day (0-23) |
| `day_of_week` | int | Day of week (0-6) |
| `month` | int | Month (1-12) |
| `is_weekend` | bool | Weekend indicator |
| `temperature` | float | Temperature (°C) |
| `flu_season_index` | float | Flu season intensity (0-1) |
| `air_quality_index` | float | Air quality (0-500) |
| `emergency_admissions` | int | Emergency admissions |
| `icu_demand` | int | ICU beds needed |
| `staff_workload` | float | Staff workload index |
| `bed_occupancy` | float | Bed occupancy rate |
| `emergency_admissions_lag_1h` | float | 1-hour lag |
| `emergency_admissions_lag_7h` | float | 7-hour lag |
| `emergency_admissions_rolling_3h` | float | 3-hour rolling mean |
| `emergency_admissions_rolling_7h` | float | 7-hour rolling mean |
| `icu_demand_lag_1h` | float | 1-hour lag |
| `icu_demand_lag_7h` | float | 7-hour lag |
| ... | ... | ... |

**Data Generation**: 
- Base admissions: 50 per day with hourly variations
- Peak hours: 10 AM - 2 PM, 6 PM - 9 PM
- Weekend surge: +20% admissions
- Seasonal patterns: Flu season, holidays

---

## 📥 Input Format

### **Web Interface Input**

**Method**: Radio button selection  
**Options**:
- `"24 Hours"`
- `"48 Hours"`
- `"7 Days"`
- `"14 Days"`

### **Programmatic Input**

```python
from app import predict_hospital_load

# Function signature
result = predict_hospital_load(time_period: str)

# Example
status, summary, alerts, details, csv_data, csv_file, json_data, json_file = predict_hospital_load("48 Hours")
```

**Parameters**:
- `time_period` (str): One of `["24 Hours", "48 Hours", "7 Days", "14 Days"]`

---

## 📤 Output Format

### **1. CSV Output** (`predictions_[period].csv`)

**Location**: `outputs/predictions/`  
**Format**: CSV with hourly predictions

**Columns**:
```csv
timestamp,predicted_emergency_admissions,predicted_icu_demand,predicted_staff_workload
2026-01-03 20:00:00,2.1,0.3,0.8
2026-01-03 21:00:00,2.3,0.4,0.9
...
```

**Schema**:
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Prediction timestamp |
| `predicted_emergency_admissions` | float | Expected admissions |
| `predicted_icu_demand` | float | Expected ICU beds |
| `predicted_staff_workload` | float | Staff workload index |

---

### **2. JSON Output** (`report_[period].json`)

**Location**: `outputs/reports/`  
**Format**: JSON report with metadata and metrics

**Structure**:
```json
{
  "period": "Next 48 Hours",
  "generated_at": "2026-01-03T23:15:00",
  "hours": 48,
  "load_classification": "🟡 MEDIUM LOAD",
  "load_score": 6,
  "metrics": {
    "total_admissions": 105,
    "peak_admissions": 3.2,
    "avg_admissions": 2.2,
    "peak_icu": 0.5,
    "avg_icu": 0.3,
    "peak_staff": 5,
    "status": "NORMAL"
  },
  "recommendation": "⚡ Moderate patient volume. Increase staff by 20-30%. Monitor ICU capacity."
}
```

**Fields**:
- `period` (str): Prediction period name
- `generated_at` (str): ISO timestamp of report generation
- `hours` (int): Number of hours predicted
- `load_classification` (str): Load level with emoji (🔴/🟡/🟢)
- `load_score` (int): Score out of 13
- `metrics` (object): Key performance indicators
- `recommendation` (str): Action items for hospital staff

---

### **3. Web Interface Output**

**Tabs**:

1. **📊 Summary**: Overview with load classification and key metrics
2. **⚠️ Alerts**: Critical warnings and recommendations
3. **📈 Details**: Detailed breakdown by department
4. **💾 Download**: CSV/JSON file previews and download buttons

---

## 🔧 Installation

### **Prerequisites**
- Python 3.10+
- pip package manager

### **Steps**

1. **Clone Repository**
```bash
git clone https://github.com/Rishiraj-Pathak-27/hospital-forecasting.git
cd hospital-forecasting
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

**Dependencies**:
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==2.0.0
gradio==3.50.2
joblib==1.3.2
```

3. **Verify Models**
```bash
python -c "import os; print('Models:', os.listdir('models/'))"
# Should show: ['icu_demand_model.pkl', 'staff_workload_model.pkl']
```

---

## 🚀 Usage

### **Web Interface**

```bash
python app.py
```

Open browser to `http://127.0.0.1:7860`

### **Programmatic Usage**

```python
from scripts.predict_flexible import FlexiblePredictor

# Initialize predictor
predictor = FlexiblePredictor()

# Make prediction
predictions_df, optimization = predictor._predict_period(
    hours=48,
    period_name="Next 48 Hours",
    start_offset=0
)

# Access results
print(predictions_df.head())
print(optimization['staff_requirements'])
```

### **Retrain Models**

```bash
python scripts/train_models.py
```

---

## 📍 File Locations

### **Core Files**
- Main Application: `app.py`
- Configuration: `config.py`
- Dependencies: `requirements.txt`

### **Data Files**
- Raw Data: `data/hospital_data.csv` (4,321 rows)
- ML Data: `data/hospital_data_ml.csv` (4,321 rows, 22 features)

### **Model Files**
- ICU Model: `models/icu_demand_model.pkl` (Size: ~2.6 MB)
- Staff Model: `models/staff_workload_model.pkl` (Size: ~2.6 MB)

### **Script Files**
- Prediction Engine: `scripts/predict_flexible.py`
- Model Classes: `scripts/xgboost_predictors.py`
- Optimization: `scripts/resource_optimizer.py`
- Training: `scripts/train_models.py`

### **Output Files** (Generated at runtime)
- Predictions: `outputs/predictions/predictions_[24_Hours|48_Hours|7_Days|14_Days].csv`
- Reports: `outputs/reports/report_[24_Hours|48_Hours|7_Days|14_Days].json`

---

## 📈 Model Performance

### **ICU Demand Model**

| Metric | Value |
|--------|-------|
| Mean Absolute Error | 0.24 beds |
| R² Score | 0.32 |
| RMSE | 0.31 beds |
| Cross-Validation MAE | 0.24 ± 0.04 |

**Top Features**:
1. `emergency_admissions_rolling_3h` (18.9%)
2. `icu_demand_lag_1h` (14.2%)
3. `emergency_admissions_lag_1h` (11.2%)

### **Staff Workload Model**

| Metric | Value |
|--------|-------|
| Mean Absolute Error | 0.45 |
| R² Score | 0.37 |
| Cross-Validation MAE | 0.45 ± 0.02 |

**Top Features**:
1. `emergency_admissions_rolling_3h` (27.6%)
2. `icu_demand_lag_1h` (17.6%)
3. `emergency_admissions_lag_1h` (10.4%)

---

## 🎯 Period-Specific Thresholds

The system uses adaptive thresholds based on prediction window:

| Period | Admission High | Admission Med | ICU High | ICU Med | Total High | Total Med |
|--------|---------------|---------------|----------|---------|------------|-----------|
| **24h** | 2.8/hr | 2.2/hr | 0.6 | 0.35 | 55 | 40 |
| **48h** | 2.5/hr | 1.9/hr | 0.45 | 0.28 | 100 | 75 |
| **7d** | 2.3/hr | 1.7/hr | 0.35 | 0.22 | 350 | 250 |
| **14d** | 2.1/hr | 1.5/hr | 0.30 | 0.18 | 680 | 500 |

**Scoring System** (13-point scale):
- Admissions: 0-3 points
- ICU Demand: 0-3 points
- Peak Volume: 0-3 points
- Total Volume: 0-2 points
- Staff Load: 0-2 points

**Classification**:
- 🔴 HIGH LOAD: Score ≥ 9
- 🟡 MEDIUM LOAD: Score 5-8
- 🟢 LOW LOAD: Score ≤ 4

---

## 🛠️ Technology Stack

- **ML Framework**: XGBoost 2.0.0
- **Web Framework**: Gradio 3.50.2
- **Data Processing**: Pandas 2.0.3, NumPy 1.24.3
- **ML Tools**: Scikit-learn 1.3.0
- **Model Persistence**: Joblib 1.3.2
- **Language**: Python 3.10+

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👨‍💻 Author

**Rishiraj Pathak**  
- GitHub: [@Rishiraj-Pathak-27](https://github.com/Rishiraj-Pathak-27)
- HuggingFace: [@rishirajpathak](https://huggingface.co/rishirajpathak)

---

## 🙏 Acknowledgments

- XGBoost library for efficient gradient boosting
- Gradio for simple web interface creation
- Synthetic data generation based on real-world hospital patterns

---

## 📞 Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/Rishiraj-Pathak-27/hospital-forecasting/issues)
- Visit the [HuggingFace Space](https://huggingface.co/spaces/rishirajpathak/hospital-ai-forecasting)

---

**Last Updated**: January 3, 2026
