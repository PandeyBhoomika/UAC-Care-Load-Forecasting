import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAC Predictive Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# MASTER CSS – Cinematic Dark / Neon Aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root tokens ── */
:root {
    --cyan:    #00f2fe;
    --magenta: #fc00ff;
    --gold:    #ffc857;
    --red:     #ff2a6d;
    --green:   #00ff9d;
    --bg0:     #04050a;
    --bg1:     #0b0d15;
    --bg2:     #111622;
    --glass:   rgba(255,255,255,0.03);
    --border:  rgba(0,242,254,0.12);
}

/* ── Full dark background ── */
[data-testid="stAppViewContainer"] {
    background: var(--bg0);
    background-image:
        radial-gradient(ellipse 80% 60% at 20% -10%, rgba(0,242,254,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 110%, rgba(252,0,255,0.05) 0%, transparent 55%);
}
[data-testid="stHeader"]           { background: transparent !important; }
[data-testid="stSidebar"]          { background: var(--bg1) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] > div    { padding-top: 1.5rem; }
.stMainBlockContainer              { padding: 2rem 3rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar              { width: 6px; height: 6px; }
::-webkit-scrollbar-track        { background: var(--bg1); }
::-webkit-scrollbar-thumb        { background: rgba(0,242,254,0.3); border-radius: 3px; }

/* ── Typography ── */
* { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; letter-spacing: 0.06em; }

/* ── Page-enter animation ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes scanline {
    0%   { top: -4px; }
    100% { top: 100%; }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 18px rgba(0,242,254,0.25); }
    50%       { box-shadow: 0 0 36px rgba(0,242,254,0.55), 0 0 70px rgba(0,242,254,0.15); }
}
@keyframes numberTick {
    from { opacity: 0; transform: scale(0.7); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes borderFlow {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.main .block-container {
    animation: fadeUp 0.75s cubic-bezier(0.16,1,0.3,1) both;
}

/* ── Hero banner ── */
.hero-banner {
    position: relative;
    overflow: hidden;
    padding: 2.8rem 3rem 2.2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(0,242,254,0.07) 0%, rgba(252,0,255,0.04) 100%);
    border: 1px solid var(--border);
    margin-bottom: 2rem;
    animation: fadeUp 0.6s ease both;
}
.hero-banner::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 20px;
    background: linear-gradient(270deg, var(--cyan), var(--magenta), var(--cyan));
    background-size: 400% 400%;
    animation: borderFlow 6s ease infinite;
    mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    mask-composite: exclude;
    -webkit-mask-composite: xor;
    padding: 1px;
    pointer-events: none;
}
/* Scanline effect */
.hero-banner::after {
    content: '';
    position: absolute;
    left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, rgba(0,242,254,0.5), transparent);
    animation: scanline 3.5s linear infinite;
    pointer-events: none;
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--cyan) 0%, #ffffff 50%, var(--magenta) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 0.9rem;
    font-weight: 300;
    color: rgba(255,255,255,0.5);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.hero-badge {
    display: inline-block;
    padding: 3px 14px;
    border: 1px solid rgba(0,255,157,0.4);
    border-radius: 50px;
    color: var(--green);
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em;
    margin-top: 1.2rem;
    background: rgba(0,255,157,0.07);
}

/* ── Section divider ── */
.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), var(--magenta), transparent);
    border: none;
    margin: 2rem 0;
    opacity: 0.4;
}

/* ── Section label ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--cyan), transparent);
    opacity: 0.3;
}

/* ── KPI Cards ── */
div[data-testid="metric-container"] {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.4rem 1.6rem !important;
    backdrop-filter: blur(14px);
    transition: transform 0.35s cubic-bezier(0.175,0.885,0.32,1.275),
                box-shadow 0.35s ease,
                border-color 0.35s ease;
    animation: pulseGlow 4s ease-in-out infinite;
    position: relative;
    overflow: hidden;
}
div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--cyan), var(--magenta));
    border-radius: 16px 16px 0 0;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-8px) scale(1.025) !important;
    border-color: rgba(0,242,254,0.55) !important;
    box-shadow: 0 20px 50px rgba(0,242,254,0.18), 0 0 0 1px rgba(0,242,254,0.2) !important;
    animation: none;
}
div[data-testid="metric-container"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.45) !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 2rem !important;
    color: var(--cyan) !important;
    animation: numberTick 0.8s cubic-bezier(0.34,1.56,0.64,1) both;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Alert cards ── */
.alert-card {
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    gap: 12px;
    animation: fadeIn 0.5s ease both;
    position: relative;
    overflow: hidden;
}
.alert-card.danger {
    background: rgba(255,42,109,0.1);
    border: 1px solid rgba(255,42,109,0.4);
    color: #ff7aa5;
    box-shadow: 0 0 30px rgba(255,42,109,0.12) inset;
}
.alert-card.safe {
    background: rgba(0,255,157,0.07);
    border: 1px solid rgba(0,255,157,0.3);
    color: var(--green);
    box-shadow: 0 0 30px rgba(0,255,157,0.08) inset;
}
.alert-icon { font-size: 1.4rem; }

/* ── Chart container ── */
.chart-wrapper {
    background: rgba(11,13,21,0.7);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    animation: fadeUp 0.9s ease both;
    position: relative;
    overflow: hidden;
}
.chart-wrapper::after {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(0,242,254,0.025) 0%, transparent 60%);
    pointer-events: none;
}

/* ── Sidebar elements ── */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.55);
    text-transform: uppercase;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(0,242,254,0.06) !important;
    border: 1px solid rgba(0,242,254,0.2) !important;
    border-radius: 10px !important;
    color: var(--cyan) !important;
    font-family: 'JetBrains Mono', monospace;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div {
    background: linear-gradient(90deg, var(--cyan), var(--magenta)) !important;
}
[data-testid="stSidebar"] .stNumberInput input {
    background: rgba(0,242,254,0.06) !important;
    border: 1px solid rgba(0,242,254,0.2) !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Sidebar logo area ── */
.sidebar-logo {
    text-align: center;
    padding: 1rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.sidebar-logo .logo-icon {
    font-size: 3rem;
    display: block;
    filter: drop-shadow(0 0 16px rgba(0,242,254,0.8));
    animation: pulseGlow 3s ease-in-out infinite;
}
.sidebar-logo .logo-text {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--cyan);
    letter-spacing: 0.2em;
    margin-top: 8px;
}
.sidebar-logo .logo-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: rgba(255,255,255,0.3);
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* ── Sidebar section header ── */
.sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(0,242,254,0.6);
    padding: 0.8rem 0 0.4rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.8rem;
}

/* ── Info table in sidebar ── */
.info-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.78rem;
}
.info-row .label { color: rgba(255,255,255,0.4); font-family: 'JetBrains Mono', monospace; }
.info-row .value { color: var(--cyan); font-family: 'JetBrains Mono', monospace; font-weight: 600; }

/* ── Bottom analysis panel ── */
.analysis-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.2rem;
    margin-top: 1.5rem;
}
.analysis-card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    animation: fadeUp 1s ease both;
    transition: transform 0.3s ease, border-color 0.3s ease;
}
.analysis-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0,242,254,0.35);
}
.analysis-card .card-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.72rem;
    color: var(--cyan);
    letter-spacing: 0.15em;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
}
.analysis-card .card-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.6rem;
    color: white;
    font-weight: 600;
}
.analysis-card .card-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.35);
    margin-top: 4px;
}
.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 50px;
    font-size: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.08em;
    margin-top: 8px;
}
.tag-cyan    { background: rgba(0,242,254,0.1);  border: 1px solid rgba(0,242,254,0.3);  color: var(--cyan); }
.tag-gold    { background: rgba(255,200,87,0.1); border: 1px solid rgba(255,200,87,0.3); color: var(--gold); }
.tag-magenta { background: rgba(252,0,255,0.08); border: 1px solid rgba(252,0,255,0.3); color: var(--magenta); }
.tag-green   { background: rgba(0,255,157,0.08); border: 1px solid rgba(0,255,157,0.3); color: var(--green); }
.tag-red     { background: rgba(255,42,109,0.1); border: 1px solid rgba(255,42,109,0.3); color: var(--red); }

/* ── Progress bar ── */
.progress-bar-wrap {
    margin: 0.6rem 0;
}
.progress-label {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.5);
    margin-bottom: 4px;
}
.progress-track {
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 3px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 1.5s cubic-bezier(0.16,1,0.3,1);
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: rgba(255,255,255,0.18);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_csv('data/HHS_Unaccompanied_Alien_Children_Program.csv')
    except Exception:
        df = pd.read_csv('../data/HHS_Unaccompanied_Alien_Children_Program.csv')

    df.columns = df.columns.str.strip()
    df['Children in HHS Care'] = pd.to_numeric(
        df['Children in HHS Care'].astype(str).str.replace(',', '', regex=False), errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').dropna(subset=['Date']).set_index('Date')

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_range).interpolate(method='linear')

    df['Care_Load_Lag1']  = df['Children in HHS Care'].shift(1)
    df['Care_Load_Roll7'] = df['Children in HHS Care'].rolling(7).mean()
    df['Net_Pressure']    = (df['Children transferred out of CBP custody']
                             - df['Children discharged from HHS Care'])
    return df.dropna()


@st.cache_resource
def train_models(df):
    feat   = ['Care_Load_Lag1', 'Care_Load_Roll7', 'Net_Pressure']
    X, y   = df[feat], df['Children in HHS Care']
    xgb_m  = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
    xgb_m.fit(X, y)
    sar_m  = SARIMAX(y, order=(1, 1, 1)).fit(disp=False)
    std_d  = float(y.tail(30).std())
    return xgb_m, sar_m, std_d


df                          = load_and_prep_data()
xgb_model, sarima_model, std_dev = train_models(df)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🛡️</span>
        <div class="logo-text">UAC INTEL</div>
        <div class="logo-sub">HHS Forecasting Engine v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">⚙️ Algorithm</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Model Selection",
        ["XGBoost (Machine Learning)", "SARIMA (Statistical)"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-section">📅 Horizon</div>', unsafe_allow_html=True)
    horizon = st.slider("Forecast Days", min_value=1, max_value=30, value=14, label_visibility="collapsed")

    st.markdown('<div class="sidebar-section">🚨 Capacity Threshold</div>', unsafe_allow_html=True)
    max_capacity = st.number_input("Shelter Capacity Limit", value=4500, step=100, label_visibility="collapsed")

    # Live dataset stats
    st.markdown('<div class="sidebar-section">📊 Dataset Stats</div>', unsafe_allow_html=True)
    n_days = len(df)
    hist_min  = int(df['Children in HHS Care'].min())
    hist_max  = int(df['Children in HHS Care'].max())
    hist_mean = int(df['Children in HHS Care'].mean())

    st.markdown(f"""
    <div class="info-row"><span class="label">Records</span><span class="value">{n_days:,}</span></div>
    <div class="info-row"><span class="label">Historical Min</span><span class="value">{hist_min:,}</span></div>
    <div class="info-row"><span class="label">Historical Max</span><span class="value">{hist_max:,}</span></div>
    <div class="info-row"><span class="label">Historical Avg</span><span class="value">{hist_mean:,}</span></div>
    <div class="info-row"><span class="label">Forecast Horizon</span><span class="value">{horizon}d</span></div>
    <div class="info-row"><span class="label">Capacity Limit</span><span class="value">{max_capacity:,}</span></div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FORECASTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
last_day     = df.iloc[-1]
current_load = float(last_day['Children in HHS Care'])
future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, horizon + 1)]
predictions, upper_bound, lower_bound = [], [], []

if "XGBoost" in selected_model:
    temp = current_load
    for i in range(horizon):
        feats = pd.DataFrame([[temp, temp, last_day['Net_Pressure']]],
                             columns=['Care_Load_Lag1', 'Care_Load_Roll7', 'Net_Pressure'])
        pred = float(xgb_model.predict(feats)[0])
        predictions.append(pred)
        temp = pred
        upper_bound.append(pred + std_dev * 0.2 * (i + 1))
        lower_bound.append(pred - std_dev * 0.2 * (i + 1))
else:
    fc        = sarima_model.get_forecast(steps=horizon)
    predictions   = fc.predicted_mean.tolist()
    ci            = fc.conf_int(alpha=0.1)
    lower_bound   = ci.iloc[:, 0].tolist()
    upper_bound   = ci.iloc[:, 1].tolist()

peak_forecast = max(predictions)
breach_days   = [future_dates[i] for i, v in enumerate(upper_bound) if v > max_capacity]
capacity_pct  = min(int(current_load / max_capacity * 100), 100)

# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">UAC PREDICTIVE INTELLIGENCE</div>
    <div class="hero-subtitle">Advanced Scenario Modeling · HHS Office of Refugee Resettlement</div>
    <span class="hero-badge">● LIVE SYSTEM ACTIVE</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">01 — System Metrics</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Care Load",       f"{int(current_load):,}")
c2.metric(f"Peak in {horizon}d",     f"{int(peak_forecast):,}",
          f"{int(peak_forecast - current_load):+,}")
c3.metric("Capacity Utilisation",    f"{capacity_pct}%",
          "Critical" if capacity_pct > 85 else ("Elevated" if capacity_pct > 65 else "Normal"))
c4.metric("Forecast Model",          selected_model.split()[0], f"{horizon}-day window")

st.markdown("<br>", unsafe_allow_html=True)

# Alert card
if breach_days:
    lead = (breach_days[0] - df.index[-1]).days
    st.markdown(f"""
    <div class="alert-card danger">
        <span class="alert-icon">⚠️</span>
        <div><strong>CAPACITY BREACH PROJECTED</strong><br>
        Upper confidence band exceeds {max_capacity:,} limit in <strong>{lead} days</strong>
        ({breach_days[0].strftime('%b %d, %Y')}). Recommend surge planning.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="alert-card safe">
        <span class="alert-icon">✅</span>
        <div><strong>CAPACITY SUFFICIENT</strong><br>
        Forecasted demand remains within the {max_capacity:,} shelter capacity limit
        across the full {horizon}-day horizon.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN FORECAST CHART
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">02 — Demand Trajectory</div>', unsafe_allow_html=True)

history = df.tail(60)

fig = go.Figure()

# Shaded history background
fig.add_vrect(
    x0=history.index[0], x1=history.index[-1],
    fillcolor="rgba(255,255,255,0.015)", line_width=0,
    annotation_text="Historical Window", annotation_position="top left",
    annotation_font_color="rgba(255,255,255,0.2)", annotation_font_size=10
)

# Historical line with area
fig.add_trace(go.Scatter(
    x=history.index, y=history['Children in HHS Care'],
    mode='lines', name='Historical Load',
    line=dict(color='#8f9bba', width=2),
    fill='tozeroy', fillcolor='rgba(143,155,186,0.06)'
))

# Confidence interval
fig.add_trace(go.Scatter(
    x=future_dates + future_dates[::-1],
    y=list(upper_bound) + list(lower_bound)[::-1],
    fill='toself', fillcolor='rgba(0,242,254,0.08)',
    line=dict(color='rgba(0,0,0,0)'),
    hoverinfo='skip', showlegend=True, name='Confidence Band'
))

# Forecast line
fig.add_trace(go.Scatter(
    x=future_dates, y=predictions,
    mode='lines+markers', name=f'{selected_model.split()[0]} Forecast',
    line=dict(color='#00f2fe', width=3),
    marker=dict(size=5, color='#00f2fe',
                line=dict(color='rgba(0,242,254,0.3)', width=6)),
))

# Capacity line
fig.add_trace(go.Scatter(
    x=history.index.tolist() + future_dates,
    y=[max_capacity] * (len(history) + horizon),
    mode='lines', name='Capacity Limit',
    line=dict(color='#ff2a6d', width=2, dash='dot')
))

# Vertical "today" marker - BUG FIXED HERE (Split line and annotation)
fig.add_vline(
    x=df.index[-1], 
    line_width=1, 
    line_dash="dash",
    line_color="rgba(255,200,87,0.5)"
)

fig.add_annotation(
    x=df.index[-1],
    y=1.0, 
    yref="paper", 
    text="NOW",
    showarrow=False,
    font=dict(color="#ffc857", size=11, family='JetBrains Mono'),
    yshift=10
)

fig.update_layout(
    height=480,
    margin=dict(l=0, r=0, t=20, b=0),
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
                font=dict(size=11, family='JetBrains Mono')),
    yaxis=dict(title='Children in HHS Care', showgrid=True,
               gridwidth=1, gridcolor='rgba(255,255,255,0.05)',
               tickfont=dict(family='JetBrains Mono', size=11)),
    xaxis=dict(showgrid=False, tickfont=dict(family='JetBrains Mono', size=11)),
    font=dict(family='Inter'),
)

st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECONDARY CHARTS: Dual-panel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">03 — System Analysis</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    # Rolling net pressure
    fig2 = go.Figure()
    press = df['Net_Pressure'].tail(90)
    colors = ['#ff2a6d' if v > 0 else '#00ff9d' for v in press]
    fig2.add_trace(go.Bar(
        x=press.index, y=press.values,
        marker_color=colors, name='Net Pressure',
        marker_line_width=0
    ))
    fig2.add_trace(go.Scatter(
        x=press.index, y=press.rolling(14).mean(),
        mode='lines', name='14-Day MA',
        line=dict(color='#ffc857', width=2)
    ))
    fig2.update_layout(
        title=dict(text='Net Intake Pressure (90 days)', font=dict(family='Orbitron', size=13, color='rgba(255,255,255,0.7)')),
        height=320, margin=dict(l=0, r=0, t=40, b=0),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(font=dict(size=10, family='JetBrains Mono')),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                   tickfont=dict(family='JetBrains Mono', size=10)),
        xaxis=dict(showgrid=False, tickfont=dict(family='JetBrains Mono', size=10)),
        bargap=0.15
    )
    st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    # 7-day rolling avg vs raw
    fig3 = go.Figure()
    raw  = df['Children in HHS Care'].tail(90)
    roll = raw.rolling(7).mean()
    fig3.add_trace(go.Scatter(
        x=raw.index, y=raw.values, mode='lines',
        name='Daily Load', line=dict(color='rgba(143,155,186,0.4)', width=1)
    ))
    fig3.add_trace(go.Scatter(
        x=roll.index, y=roll.values, mode='lines',
        name='7-Day MA', line=dict(color='#00f2fe', width=2.5),
        fill='tonexty', fillcolor='rgba(0,242,254,0.04)'
    ))
    fig3.update_layout(
        title=dict(text='Care Load Smoothing (90 days)', font=dict(family='Orbitron', size=13, color='rgba(255,255,255,0.7)')),
        height=320, margin=dict(l=0, r=0, t=40, b=0),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(font=dict(size=10, family='JetBrains Mono')),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                   tickfont=dict(family='JetBrains Mono', size=10)),
        xaxis=dict(showgrid=False, tickfont=dict(family='JetBrains Mono', size=10)),
    )
    st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">04 — Intelligence Summary</div>', unsafe_allow_html=True)

avg_forecast = np.mean(predictions)
trend_delta  = predictions[-1] - predictions[0]
trend_dir    = "↑ Rising" if trend_delta > 0 else "↓ Falling"
trend_tag    = "tag-red" if trend_delta > 0 else "tag-green"
volatility   = np.std(predictions)
model_tag    = "XGBoost ML" if "XGBoost" in selected_model else "SARIMA Stats"
model_color  = "tag-cyan" if "XGBoost" in selected_model else "tag-magenta"
util_color   = "tag-red" if capacity_pct > 85 else ("tag-gold" if capacity_pct > 65 else "tag-green")

# Capacity progress bar
bar_color = "#ff2a6d" if capacity_pct > 85 else ("#ffc857" if capacity_pct > 65 else "#00ff9d")

st.markdown(f"""
<div class="analysis-grid">
  <div class="analysis-card">
    <div class="card-title">📈 Forecast Trend</div>
    <div class="card-value">{trend_dir}</div>
    <div class="card-sub">Net change over {horizon}-day horizon: {int(trend_delta):+,}</div>
    <span class="tag {trend_tag}">{int(abs(trend_delta)):,} children delta</span>
  </div>
  <div class="analysis-card">
    <div class="card-title">📊 Forecast Volatility</div>
    <div class="card-value">{volatility:,.0f}</div>
    <div class="card-sub">Std. deviation across forecast window</div>
    <span class="tag tag-gold">±{volatility/avg_forecast*100:.1f}% relative uncertainty</span>
  </div>
  <div class="analysis-card">
    <div class="card-title">🏛️ Capacity Utilisation</div>
    <div class="card-value">{capacity_pct}%</div>
    <div class="progress-bar-wrap">
      <div class="progress-label"><span>Current Load</span><span>{int(current_load):,} / {max_capacity:,}</span></div>
      <div class="progress-track">
        <div class="progress-fill" style="width:{capacity_pct}%; background:{bar_color};"></div>
      </div>
    </div>
    <span class="tag {util_color}">{'Critical' if capacity_pct>85 else ('Elevated' if capacity_pct>65 else 'Nominal')}</span>
  </div>
  <div class="analysis-card">
    <div class="card-title">🤖 Active Model</div>
    <div class="card-value">{model_tag}</div>
    <div class="card-sub">Average {horizon}-day projected load: {int(avg_forecast):,}</div>
    <span class="tag {model_color}">{selected_model.split()[0]}</span>
    &nbsp;<span class="tag tag-cyan">{horizon}-day window</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FORECAST DATA TABLE (collapsible)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📋 Raw Forecast Data", expanded=False):
    table_df = pd.DataFrame({
        'Date':        [d.strftime('%Y-%m-%d') for d in future_dates],
        'Forecast':    [int(p) for p in predictions],
        'Lower Bound': [int(l) for l in lower_bound],
        'Upper Bound': [int(u) for u in upper_bound],
        'Breach Risk': ['🔴 YES' if u > max_capacity else '🟢 NO' for u in upper_bound]
    })
    st.dataframe(table_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    UAC Predictive Intelligence Engine · HHS Office of Refugee Resettlement · Powered by XGBoost & SARIMA
</div>
""", unsafe_allow_html=True)
