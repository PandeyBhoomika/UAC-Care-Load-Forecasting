<div align="center">

![Python](https://img.shields.io/badge/PYTHON-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/STREAMLIT-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBOOST-1.7-orange?style=for-the-badge)
![SARIMA](https://img.shields.io/badge/SARIMA-statsmodels-9B59B6?style=for-the-badge)
![Plotly](https://img.shields.io/badge/PLOTLY-5.x-3D9970?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/PANDAS-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)

<br/>

# 🛡️ UAC Care Load Forecaster

### AI-Powered Placement Demand Forecasting for HHS

A predictive intelligence dashboard for the U.S. Department of Health and Human Services (HHS) — ingests official UAC program data, trains dual ML + statistical models, and delivers a live Streamlit dashboard with capacity breach alerts and confidence intervals.

<br/>

[Quick Start](#-quick-start) • [Features](#-features) • [Pipeline](#-pipeline) • [Structure](#-project-structure) • [Tech Stack](#-tech-stack) • [Environment](#-environment)

</div>

---

## 📖 About This Project

The UAC Care Load Forecaster is an end-to-end time-series forecasting system built on official HHS Unaccompanied Alien Children (UAC) program data. It enables program managers and data analysts to move from reactive to predictive operations — providing a clear view of expected shelter demand days in advance.

The system runs a full data pipeline: raw CSV ingestion → date gap interpolation → lag and rolling feature engineering → dual-model training (XGBoost + SARIMA) → an interactive Streamlit dashboard with adjustable forecast horizon, configurable capacity limits, and real-time breach-risk alerts.

---

## ✨ Features

| Category | Feature | Description |
|---|---|---|
| 🤖 **ML Forecasting** | XGBoost Regressor | Auto-regressive model trained on lag-1, 7-day rolling average, and net pressure features |
| 📈 **Statistical Model** | SARIMA(1,1,1) | Time-series model with built-in 90% confidence intervals for uncertainty quantification |
| 🔁 **Data Pipeline** | Gap Interpolation | Detects and fills missing daily records via linear interpolation for a clean continuous series |
| ⚙️ **Feature Engineering** | Lag & Pressure Features | Lag-1 care load, 7-day rolling mean, and net daily pressure (inflow minus outflow) |
| 🚨 **Risk Management** | Capacity Breach Alerts | Detects if upper confidence bound exceeds user-defined shelter capacity; shows lead-time in days |
| 📊 **Dashboard** | Live KPI Cards | Displays current care load, peak forecasted load, and projected change at a glance |
| 🔵 **Confidence Bands** | Shaded Intervals | Expanding confidence cone (XGBoost) or SARIMA-native 90% CI rendered as a shaded region |
| 🎛️ **Controls** | Adjustable Horizon | Sidebar slider to set forecast window from 1 to 30 days; model selector to switch approaches |

---

## 🚀 Quick Start

### Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Runtime |
| pip | 23+ | Package management |

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/UAC-Care-Load-Forecasting.git
cd UAC-Care-Load-Forecasting
```

### 2. Create a Virtual Environment

```bash
# Create environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**

| URL | Description |
|---|---|
| `http://localhost:8501` | Main forecasting dashboard |
| Sidebar → Model Selector | Switch between XGBoost and SARIMA |
| Sidebar → Forecast Horizon | Set 1–30 day forecast window |
| Sidebar → Capacity Limit | Configure shelter capacity threshold |

---

## 🔄 Pipeline

The project follows a three-phase pipeline across two notebooks and the app:

**Phase 1 — Data Cleaning** (`01_Data_Cleaning.ipynb`)

```
Raw HHS CSV  →  Strip whitespace  →  Parse dates  →  Sort chronologically
     →  Reindex to full daily range  →  Linear interpolation for gaps
     →  Export HHS_UAC_Cleaned.csv
```

**Phase 2 — Feature Engineering** (`02_Feature_Engineering.ipynb`)

```
Cleaned data  →  Lag-1 feature  →  7-day rolling mean  →  Net Pressure metric
     →  Train/test split (last 30 days held out)
     →  Baseline naive forecast (MAE benchmark)
     →  XGBoost evaluation  →  SARIMA evaluation
```

**Phase 3 — Live Dashboard** (`app.py`)

```
Full dataset  →  Retrain both models  →  Sidebar controls
     →  Iterative XGBoost forecast  →  SARIMA forecast + CI
     →  Plotly chart (history + forecast + capacity line)
     →  KPI cards + breach-risk alert
```

---

## 🗂️ Project Structure

```
UAC-Care-Load-Forecasting/
│
├── data/
│   └── HHS_Unaccompanied_Alien_Children_Program.csv   # Raw HHS dataset
│
├── 01_Data_Cleaning.ipynb        # Phase 1: ingestion, cleaning, interpolation
├── 02_Feature_Engineering.ipynb  # Phase 2: features, model training, evaluation
│
├── app.py                        # Phase 3: Streamlit dashboard
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🧰 Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Dashboard** | Streamlit | Interactive web app & sidebar controls |
| **Visualization** | Plotly | Candlestick-style forecast chart with shaded CI |
| **ML Model** | XGBoost | Auto-regressive gradient boosting regressor |
| **Statistical Model** | statsmodels SARIMAX | Time-series forecasting with native confidence intervals |
| **Data Processing** | Pandas / NumPy | Pipeline, interpolation, feature engineering |
| **Exploration** | Matplotlib / Seaborn | EDA charts in notebooks |

---

## 🌐 Environment

No API keys or external services are required. The app runs fully locally.

### `requirements.txt`

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
streamlit
plotly
xgboost
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 📁 Data

The dataset (`HHS_Unaccompanied_Alien_Children_Program.csv`) contains daily operational records:

| Column | Description |
|---|---|
| `Date` | Daily timestamp |
| `Children in HHS Care` | Total children in HHS custody (care load target) |
| `Children transferred out of CBP custody` | Daily inflow into HHS system |
| `Children discharged from HHS Care` | Daily outflow / placements |

> **Source:** U.S. Department of Health and Human Services (HHS) — Office of Refugee Resettlement (ORR)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">
Built for the HHS UAC Program &nbsp;•&nbsp; Python 3.11+ &nbsp;•&nbsp; Streamlit + XGBoost + SARIMA
</div>
