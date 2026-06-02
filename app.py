import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up the Page
st.set_page_config(page_title="HHS Care Load Forecaster", layout="wide")
st.title("🛡️ UAC Care Load & Placement Demand Forecaster")
st.markdown("Predictive intelligence for the U.S. Department of Health and Human Services (HHS).")

# ==========================================
# 1. DATA PIPELINE (Cached for speed)
# ==========================================
@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_csv('data/HHS_Unaccompanied_Alien_Children_Program.csv')
    except:
        df = pd.read_csv('../data/HHS_Unaccompanied_Alien_Children_Program.csv')
        
    df.columns = df.columns.str.strip()
    df['Children in HHS Care'] = df['Children in HHS Care'].replace({',': ''}, regex=True)
    df['Children in HHS Care'] = pd.to_numeric(df['Children in HHS Care'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').dropna(subset=['Date']).set_index('Date')
    
    full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df_clean = df.reindex(full_date_range).interpolate(method='linear')
    
    df_features = df_clean.copy()
    df_features['Care_Load_Lag1'] = df_features['Children in HHS Care'].shift(1)
    df_features['Care_Load_Roll7'] = df_features['Children in HHS Care'].rolling(window=7).mean()
    df_features['Net_Pressure'] = df_features['Children transferred out of CBP custody'] - df_features['Children discharged from HHS Care']
    return df_features.dropna()

df = load_and_prep_data()

# ==========================================
# 2. TRAIN MODELS (Cached)
# ==========================================
@st.cache_resource
def train_models(df):
    # Train XGBoost
    feature_cols = ['Care_Load_Lag1', 'Care_Load_Roll7', 'Net_Pressure']
    X = df[feature_cols]
    y = df['Children in HHS Care']
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X, y)
    
    # Train SARIMA (Lightweight version for web app)
    sarima_model = SARIMAX(y, order=(1, 1, 1)).fit(disp=False)
    
    # Calculate historical variance for Confidence Intervals
    std_dev = y.tail(30).std()
    
    return xgb_model, sarima_model, std_dev

xgb_model, sarima_model, std_dev = train_models(df)

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Forecasting Parameters")
selected_model = st.sidebar.selectbox("Select Predictive Model", ["Machine Learning (XGBoost)", "Statistical (SARIMA)"])
horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=14)

st.sidebar.header("🚨 Risk Management")
max_capacity = st.sidebar.number_input("Shelter Capacity Limit", value=4500, step=100)

# ==========================================
# 4. FORECASTING ENGINE
# ==========================================
last_day = df.iloc[-1]
current_load = last_day['Children in HHS Care']
future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, horizon + 1)]

predictions = []
upper_bound = []
lower_bound = []

if selected_model == "Machine Learning (XGBoost)":
    temp_load = current_load
    for i in range(horizon):
        # Default assuming pressure remains constant for baseline ML forecast
        simulated_pressure = last_day['Net_Pressure']
        features = pd.DataFrame([[temp_load, temp_load, simulated_pressure]], columns=['Care_Load_Lag1', 'Care_Load_Roll7', 'Net_Pressure'])
        pred = xgb_model.predict(features)[0]
        predictions.append(pred)
        temp_load = pred
        # Expanding confidence interval over time
        upper_bound.append(pred + (std_dev * (0.2 * (i+1))))
        lower_bound.append(pred - (std_dev * (0.2 * (i+1))))

else: # SARIMA
    sarima_forecast = sarima_model.get_forecast(steps=horizon)
    predictions = sarima_forecast.predicted_mean.values
    conf_int = sarima_forecast.conf_int(alpha=0.1) # 90% confidence
    lower_bound = conf_int.iloc[:, 0].values
    upper_bound = conf_int.iloc[:, 1].values

# ==========================================
# 5. DASHBOARD VISUALS (Plotly)
# ==========================================
st.subheader(f"Projected Care Demand: {horizon}-Day Outlook")

fig = go.Figure()

# Plot History
history = df.tail(45)
fig.add_trace(go.Scatter(x=history.index, y=history['Children in HHS Care'], mode='lines', name='Historical Load', line=dict(color='black', width=2)))

# Plot Forecast
fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines+markers', name=f'{selected_model} Forecast', line=dict(color='blue', dash='dash')))

# Plot Confidence Intervals (The Shaded Region)
fig.add_trace(go.Scatter(x=future_dates + future_dates[::-1], y=list(upper_bound) + list(lower_bound)[::-1], fill='toself', fillcolor='rgba(0, 0, 255, 0.15)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=True, name='Confidence Interval'))

# Plot Capacity Line
fig.add_trace(go.Scatter(x=history.index.to_list() + future_dates, y=[max_capacity]*(len(history)+horizon), mode='lines', name='Capacity Limit', line=dict(color='red', width=2, dash='dot')))

fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Children in HHS Care", hovermode="x unified", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. KEY PERFORMANCE INDICATORS (KPIs)
# ==========================================
st.markdown("### 📊 Operational Risk Indicators")
col1, col2, col3 = st.columns(3)

peak_forecast = max(predictions)
breach_days = [future_dates[i] for i, val in enumerate(upper_bound) if val > max_capacity]

col1.metric("Current Care Load", f"{int(current_load):,}")
col2.metric(f"Peak Forecasted Load", f"{int(peak_forecast):,}", f"{int(peak_forecast - current_load):,} projected change")

if breach_days:
    lead_time = (breach_days[0] - df.index[-1]).days
    col3.error(f"⚠️ Capacity Breach Risk in {lead_time} Days!")
else:
    col3.success("✅ Capacity Sufficient (No Breach Detected)")