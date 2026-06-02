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
st.set_page_config(page_title="HHS Care Load Forecaster", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# CUSTOM CSS: ANIMATIONS & DARK GLASSMORPHISM
# ==========================================
st.markdown("""
    <style>
    /* 1. Page Load Animation */
    @keyframes slideInUp {
        0% { opacity: 0; transform: translateY(40px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .block-container {
        animation: slideInUp 0.8s ease-out;
    }

    /* 2. Glowing Glassmorphism Cards for KPIs */
    div[data-testid="metric-container"] {
        background: rgba(31, 40, 51, 0.6);
        border: 1px solid rgba(0, 242, 254, 0.1);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    /* 3. Hover Animations */
    div[data-testid="metric-container"]:hover {
        transform: translateY(-8px) scale(1.02);
        border: 1px solid rgba(0, 242, 254, 0.8);
        box-shadow: 0 10px 30px rgba(0, 242, 254, 0.2);
    }

    /* Customize Streamlit Headers to look sleeker */
    h1, h2, h3 {
        font-weight: 300 !important;
        letter-spacing: 1px;
    }
    
    /* Make the divider glow */
    hr {
        border-top: 1px solid rgba(0, 242, 254, 0.3);
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ UAC Predictive Forecaster")
st.markdown("*Advanced Intelligence & Scenario Modeling for HHS Operations*")
st.divider()

# ==========================================
# 1. DATA PIPELINE (Cached)
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
    feature_cols = ['Care_Load_Lag1', 'Care_Load_Roll7', 'Net_Pressure']
    X = df[feature_cols]
    y = df['Children in HHS Care']
    
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X, y)
    
    sarima_model = SARIMAX(y, order=(1, 1, 1)).fit(disp=False)
    std_dev = y.tail(30).std()
    
    return xgb_model, sarima_model, std_dev

xgb_model, sarima_model, std_dev = train_models(df)

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("## ⚙️ Model Parameters")
selected_model = st.sidebar.selectbox("Algorithm Selection", ["XGBoost (Machine Learning)", "SARIMA (Statistical)"])
horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=14)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

with st.sidebar.expander("🚨 Risk Management Controls", expanded=True):
    max_capacity = st.number_input("Shelter Capacity Limit", value=4500, step=100)
    st.caption("Triggers early warning system if forecasted load exceeds this threshold.")

# ==========================================
# 4. FORECASTING ENGINE
# ==========================================
last_day = df.iloc[-1]
current_load = last_day['Children in HHS Care']
future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, horizon + 1)]

predictions = []
upper_bound = []
lower_bound = []

if "XGBoost" in selected_model:
    temp_load = current_load
    for i in range(horizon):
        simulated_pressure = last_day['Net_Pressure']
        features = pd.DataFrame([[temp_load, temp_load, simulated_pressure]], columns=['Care_Load_Lag1', 'Care_Load_Roll7', 'Net_Pressure'])
        pred = xgb_model.predict(features)[0]
        predictions.append(pred)
        temp_load = pred
        upper_bound.append(pred + (std_dev * (0.2 * (i+1))))
        lower_bound.append(pred - (std_dev * (0.2 * (i+1))))

else: # SARIMA
    sarima_forecast = sarima_model.get_forecast(steps=horizon)
    predictions = sarima_forecast.predicted_mean.values
    conf_int = sarima_forecast.conf_int(alpha=0.1)
    lower_bound = conf_int.iloc[:, 0].values
    upper_bound = conf_int.iloc[:, 1].values

# ==========================================
# 5. KEY PERFORMANCE INDICATORS (Animated)
# ==========================================
st.markdown("### 📊 Live System Metrics")
col1, col2, col3 = st.columns(3)

peak_forecast = max(predictions)
breach_days = [future_dates[i] for i, val in enumerate(upper_bound) if val > max_capacity]

col1.metric("Current Active Care Load", f"{int(current_load):,}")
col2.metric(f"Peak Forecast ({horizon} Days)", f"{int(peak_forecast):,}", f"{int(peak_forecast - current_load):,} projected change")

if breach_days:
    lead_time = (breach_days[0] - df.index[-1]).days
    col3.error(f"⚠️ Capacity Breach in {lead_time} Days!")
else:
    col3.success("✅ Capacity Sufficient")

st.markdown("<br><br>", unsafe_allow_html=True)

# ==========================================
# 6. DASHBOARD VISUALS (Dark Mode Plotly)
# ==========================================
st.markdown(f"### 📈 Projected Demand Trajectory")

fig = go.Figure()

# Plot History (Sleek Grey Line)
history = df.tail(45)
fig.add_trace(go.Scatter(x=history.index, y=history['Children in HHS Care'], mode='lines', name='Historical Load', line=dict(color='#8f9bba', width=2)))

# Plot Forecast (Neon Cyan)
fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines+markers', name=f'{selected_model.split()[0]} Forecast', line=dict(color='#00f2fe', width=3, dash='dash')))

# Plot Confidence Intervals (Subtle Cyan Glow)
fig.add_trace(go.Scatter(x=future_dates + future_dates[::-1], y=list(upper_bound) + list(lower_bound)[::-1], fill='toself', fillcolor='rgba(0, 242, 254, 0.1)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=True, name='Confidence Bound'))

# Plot Capacity Line (Neon Red/Pink)
fig.add_trace(go.Scatter(x=history.index.to_list() + future_dates, y=[max_capacity]*(len(history)+horizon), mode='lines', name='Critical Capacity Limit', line=dict(color='#ff0844', width=2, dash='dot')))

# Apply Dark Premium Styling
fig.update_layout(
    height=550, 
    xaxis_title="", 
    yaxis_title="Total Children in Care", 
    hovermode="x unified", 
    template="plotly_dark",  # Forces Dark Mode chart
    paper_bgcolor="rgba(0,0,0,0)", 
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=40, b=0)
)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
fig.update_xaxes(showgrid=False)

st.plotly_chart(fig, use_container_width=True)