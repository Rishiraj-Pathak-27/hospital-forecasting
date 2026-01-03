---
title: Hospital Emergency Prediction
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# Hospital Emergency Prediction System

> **AI-Powered Forecasting System for Emergency Admissions, ICU Demand, and Staff Workload**

A comprehensive machine learning system that predicts hospital emergency department metrics for any time period (24 hours, weekends, weeks) using state-of-the-art time-series models and multivariate analysis.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Pretrained Models Used](#pretrained-models-used)
3. [Dataset Information](#dataset-information)
4. [Input Format](#input-format)
5. [Output Format](#output-format)
6. [Project Structure](#project-structure)
7. [Quick Start](#quick-start)
8. [Usage Examples](#usage-examples)

---

## 🎯 Overview

This system solves critical hospital management challenges:

- **Predicts Emergency Admissions** - Forecasts patient volume
- **Predicts ICU Demand** - Anticipates intensive care bed requirements
- **Predicts Staff Workload** - Optimizes staff allocation
- **Generates Resource Plans** - Provides actionable recommendations

**Key Capabilities:**
- Flexible time periods: 24h, 48h, weekend, week, or custom
- Handles external factors: weather, flu season, air quality
- Automated alerts when capacity exceeds thresholds
- Production-ready outputs (CSV, JSON)

---

## 🤖 Pretrained Models Used

### 1. **Amazon Chronos (T5-Small)** - Emergency Admissions
- **Type:** Transformer-based time-series foundation model
- **Architecture:** Based on Google's T5 (Text-to-Text Transfer Transformer)
- **Pretrained On:** 100+ billion time-series data points
- **Source:** [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
- **Purpose:** Zero-shot forecasting of emergency admissions
- **Model Size:** 20M parameters (small variant for speed)
- **Input:** Historical admission time-series (168 hours context)
- **Output:** Probabilistic forecasts with confidence intervals
- **Fallback:** Historical moving average if unavailable

### 2. **XGBoost** - ICU Demand Predictor
- **Library:** XGBoost 3.1.2 (Gradient Boosting)
- **Training:** Custom trained on 180 days synthetic data
- **Features:** 13 multivariate features
  - Temporal: hour, day_of_week, month, is_weekend
  - Historical: lag features (1h, 7h), rolling averages (3h, 7h)
  - External: temperature, flu_season_index, air_quality
  - Hospital: emergency_admissions_lag, icu_demand_lag
- **Performance:** MAE: 0.25 beds, R²: 0.32
- **Training Time:** <5 seconds

### 3. **XGBoost** - Staff Workload Predictor
- **Library:** XGBoost 3.1.2
- **Training:** Custom trained on 180 days synthetic data
- **Features:** 13 multivariate features + bed_occupancy
- **Performance:** MAE: 0.43 staff units, R²: 0.42
- **Training Time:** <5 seconds

---

## 📊 Dataset Information

### Data Source
**Synthetic Hospital Data** - Generated using realistic patterns:
- Hospital admission statistics
- CDC flu season data
- Weather correlations
- Day-of-week and hourly variations

### Dataset Size
- **Duration:** 180 days (6 months)
- **Granularity:** Hourly records
- **Total Records:** 4,320 hours
- **Features:** 22 features after engineering

### Data Columns

**Raw Data (hospital_data.csv):**
```
datetime, hour, day_of_week, month, is_weekend,
temperature, flu_season_index, air_quality_index,
emergency_admissions, icu_demand, staff_workload, bed_occupancy
```

**ML Data (hospital_data_ml.csv):**
```
All raw columns +
emergency_admissions_lag_1h, _lag_7h, _lag_14h,
emergency_admissions_rolling_3h, _rolling_7h, _rolling_14h,
icu_demand_lag_1h, icu_demand_lag_7h
```

### Location
- **Raw:** `data/hospital_data.csv`
- **ML-ready:** `data/hospital_data_ml.csv`

---

## 📥 Input Format

### Method 1: Simple Command Line

```bash
python predict.py <period>
```

**Options:**
- `24h` - Next 24 hours
- `48h` - Next 48 hours
- `weekend` - Next weekend
- `week` - Next week (168 hours)
- `<number>` - Custom hours (e.g., `72`)

**Examples:**
```bash
python predict.py 24h
python predict.py weekend
python predict.py 120    # 5 days
```

### Method 2: Interactive Menu

```bash
python scripts/predict_flexible.py
```

### Method 3: Python API

```python
from scripts.predict_flexible import FlexiblePredictor

predictor = FlexiblePredictor()
predictions, optimization = predictor.predict_next_24_hours()
```

---

## 📤 Output Format

### Output Files

Each prediction generates **2 files**:

#### 1. CSV File (Predictions)
**Location:** `data/predictions_<Period>.csv`

**Format:**
```csv
datetime,predicted_emergency_admissions,predicted_icu_demand,predicted_staff_workload
2026-01-03 20:01:54,1.85,0.14,0.50
2026-01-03 21:01:54,1.85,0.03,0.17
```

**Columns:**
- `datetime` - Hourly timestamp
- `predicted_emergency_admissions` - Expected admissions (float)
- `predicted_icu_demand` - ICU beds needed (float)
- `predicted_staff_workload` - Staff units required (float)

#### 2. JSON File (Report)
**Location:** `reports/report_<Period>.json`

**Format:**
```json
{
  "period": "Next 24 Hours",
  "generated_at": "2026-01-03T19:08:45",
  "hours": 24,
  "summary": {
    "total_admissions": 44,
    "peak_staff": 5,
    "status": "NORMAL"
  },
  "optimization": {
    "staff": {"peak": 5, "avg": 5.0},
    "icu": {"max_utilization_pct": 4.5},
    "alerts": []
  }
}
```

**Status Values:** `NORMAL` | `ELEVATED` | `CRITICAL`

### Visualizations
**Location:** `visualizations/`

| File | Description |
|------|-------------|
| `hospital_dashboard.png` | 6-panel analytics |
| `hospital_metrics.png` | Key metrics cards |
| `prediction_comparison.png` | Period comparison |

**Generate with:** `python scripts/visualize.py`

### Models
**Location:** `models/`
- `icu_demand_model.pkl` - Trained XGBoost (~500KB)
- `staff_workload_model.pkl` - Trained XGBoost (~500KB)

---

## 📂 Project Structure

```
gfgML/
│
├── 📁 data/                          # All CSV files
│   ├── hospital_data.csv             # Historical data (4,320 rows)
│   ├── hospital_data_ml.csv          # ML features (4,307 rows)
│   ├── predictions_Next_24_Hours.csv
│   ├── predictions_Next_48_Hours.csv
│   ├── predictions_Current_Weekend.csv
│   └── predictions_Next_Week_7_Days.csv
│
├── 📁 reports/                       # All JSON reports
│   ├── report_Next_24_Hours.json
│   ├── report_Next_48_Hours.json
│   └── report_Next_Week_7_Days.json
│
├── 📁 visualizations/                # All PNG charts
│   ├── hospital_dashboard.png
│   ├── hospital_metrics.png
│   └── prediction_comparison.png
│
├── 📁 models/                        # Trained models
│   ├── icu_demand_model.pkl
│   └── staff_workload_model.pkl
│
├── 📁 scripts/                       # Python modules
│   ├── data_generator.py
│   ├── emergency_admissions_predictor.py
│   ├── xgboost_predictors.py
│   ├── resource_optimizer.py
│   ├── predict_flexible.py
│   ├── predict.py
│   └── visualize.py
│
├── 📄 main.py                        # Main pipeline
├── 📄 config.py                      # Configuration
├── 📄 requirements.txt               # Dependencies
└── 📄 README.md                      # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- pandas, numpy, scikit-learn
- xgboost, matplotlib, seaborn
- torch (optional, for Chronos)

### First Run

```bash
# 2. Generate data and train models
python main.py
```

**This will:**
- ✅ Generate 180 days of hospital data
- ✅ Create ML features
- ✅ Train XGBoost models
- ✅ Generate 48-hour forecast
- ✅ Create resource plan
- ✅ Save models to `models/`

**Time:** ~30 seconds

### Daily Use

```bash
# Predict next 24 hours
python predict.py 24h

# Predict weekend
python predict.py weekend

# Predict next week
python predict.py week
```

---

## 💡 Usage Examples

### Example 1: Daily Planning

```bash
python predict.py 24h
```

**Output:**
- `data/predictions_Next_24_Hours.csv` (24 rows)
- `reports/report_Next_24_Hours.json`

**Use:** Schedule today's staff based on hourly predictions

### Example 2: Weekend Preparation

```bash
python predict.py weekend
```

**Output:**
```
Period: Next Weekend
Duration: 54 hours
Total Admissions: 99
Peak Staff: 7
Status: NORMAL
```

**Use:** Plan weekend staffing and ICU capacity

### Example 3: Weekly Planning

```bash
python predict.py week
```

**Output:**
- 168 hours of predictions
- Daily patterns visible
- Staff requirements by shift

### Example 4: Custom Period

```bash
python predict.py 72    # 3 days
```

### Example 5: Compare All

```bash
python scripts/predict_flexible.py --demo
```

Generates all periods + comparison charts

---

## 🔧 Configuration

Edit `config.py`:

```python
DAYS_OF_HISTORICAL_DATA = 180  # Historical data length
PREDICTION_HORIZON = 48        # Default prediction hours
BASE_EMERGENCY_ADMISSIONS = 50 # Average per day
ICU_CAPACITY = 20              # Total ICU beds
CRITICAL_THRESHOLD = 0.85      # Alert threshold
```

---

## 📊 Model Performance

| Model | Metric | Value | Speed |
|-------|--------|-------|-------|
| Chronos | Admissions | ~92% | 3s |
| XGBoost | ICU (R²) | 0.32 | 0.1s |
| XGBoost | Staff (R²) | 0.42 | 0.1s |

---

## 🎯 Key Features

- ✅ Flexible time periods (24h to weeks)
- ✅ State-of-the-art models (Chronos + XGBoost)
- ✅ External factors (weather, flu season)
- ✅ Automated alerts (>85% capacity)
- ✅ Multiple outputs (CSV, JSON, PNG)
- ✅ Fast (<30s training, <1s prediction)
- ✅ Production-ready structure

---

## 🏆 Hackathon Highlights

1. **Innovation:** Amazon Chronos foundation model
2. **Complete:** Prediction + Optimization + Alerts
3. **Practical:** 3 key hospital metrics
4. **Flexible:** Any time period
5. **Fast:** 30s train, 0.1s predict
6. **Clean:** Organized file structure

---

## 🔍 Troubleshooting

**Q: Chronos warning?**  
A: Normal! Uses XGBoost fallback automatically

**Q: No data files?**  
A: Run `python main.py` first

**Q: Want more data?**  
A: Change `DAYS_OF_HISTORICAL_DATA` in config.py

---

**Last Updated:** January 3, 2026
