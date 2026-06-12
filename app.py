import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; color: #e6edf3; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 13px !important; }
    [data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 24px !important; }
    [data-testid="stMetricDelta"] { font-size: 13px !important; }

    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #58a6ff;
        border-bottom: 2px solid #21262d;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }

    /* Cards */
    .card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
    }

    /* Signal badges */
    .signal-buy {
        background-color: #1a4731;
        color: #3fb950;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid #3fb950;
    }
    .signal-sell {
        background-color: #4a1d1d;
        color: #f85149;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid #f85149;
    }
    .signal-hold {
        background-color: #2d2a1a;
        color: #e3b341;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid #e3b341;
    }

    /* Title */
    .main-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #3fb950);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .sub-title {
        color: #8b949e;
        font-size: 15px;
        margin-bottom: 24px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b22;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        border-radius: 6px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #58a6ff !important;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #21262d;
        border: 1px solid #30363d;
        color: #e6edf3;
    }

    /* Divider */
    hr { border-color: #21262d; }

    /* Table */
    .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── DATA PATHS ────────────────────────────────────────────────
BASE = r"C:\Users\sarin\Downloads\nifty50_project"
FEATURES_PATH  = os.path.join(BASE, "data", "features_data.csv")
METRICS_PATH   = os.path.join(BASE, "data", "stock_metrics.csv")
PORTFOLIO_PATH = os.path.join(BASE, "data", "portfolios.csv")
ANOMALY_PATH   = os.path.join(BASE, "data", "anomaly_data.csv")
MODEL_PATH     = os.path.join(BASE, "models", "xgb_predictor.pkl")

# ── LOAD DATA ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df         = pd.read_csv(FEATURES_PATH,  parse_dates=['date'])
    metrics    = pd.read_csv(METRICS_PATH)
    portfolios = pd.read_csv(PORTFOLIO_PATH)
    anomalies  = pd.read_csv(ANOMALY_PATH,   parse_dates=['date'])
    anomalies  = anomalies.loc[:, ~anomalies.columns.duplicated()]
    return df, metrics, portfolios, anomalies

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df, metrics_df, portfolios_df, anomalies_df = load_data()
model = load_model()

TICKERS = sorted(df['ticker'].unique().tolist())
FEATURES = [
    'MA_7','MA_21','MA_50','EMA_12','EMA_26',
    'MACD','MACD_signal','MACD_hist',
    'RSI','BB_width','BB_pos',
    'momentum_5','momentum_10','momentum_21',
    'volatility_21','volume_ratio',
    'hl_spread','close_vs_open',
    'price_vs_MA50','price_vs_MA200'
]

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 NIFTY-50 Intelligence")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📊 Stock Analysis",
         "🤖 AI Predictor",
         "💼 Portfolio Builder",
         "⚠️ Risk Assessment",
         "🔍 Anomaly Detection"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption(f"📅 2000 – 2021")
    st.caption(f"🏢 {len(TICKERS)} Companies")
    st.caption(f"📋 {len(df):,} Records")

# ── HELPER: PLOTLY THEME ──────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='#161b22',
    plot_bgcolor='#0d1117',
    font=dict(color='#e6edf3', size=12),
    xaxis=dict(gridcolor='#21262d', zerolinecolor='#30363d'),
    yaxis=dict(gridcolor='#21262d', zerolinecolor='#30363d'),
    margin=dict(l=40, r=20, t=40, b=40)
)

# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="main-title">NIFTY-50 Investment Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-powered investment analytics · Historical data 2000–2021 · 50 NSE companies</div>', unsafe_allow_html=True)

    # KPI Row
    best  = metrics_df.nlargest(1, 'sharpe_ratio').iloc[0]
    worst = metrics_df.nsmallest(1, 'max_drawdown').iloc[0]
    avg_ret = metrics_df['annual_return'].mean()
    avg_vol = metrics_df['annual_vol'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Sharpe Ratio",  f"{best['sharpe_ratio']:.2f}",  best['ticker'])
    c2.metric("Avg Annual Return",  f"{avg_ret*100:.1f}%",          "Across 50 stocks")
    c3.metric("Avg Volatility",     f"{avg_vol*100:.1f}%",          "Annualised")
    c4.metric("Worst Max Drawdown", f"{worst['max_drawdown']*100:.1f}%", worst['ticker'])

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📈 Risk vs Return</div>', unsafe_allow_html=True)
        fig = px.scatter(
            metrics_df,
            x='annual_vol', y='annual_return',
            color='sharpe_ratio',
            text='ticker',
            color_continuous_scale='RdYlGn',
            labels={'annual_vol':'Annual Volatility','annual_return':'Annual Return','sharpe_ratio':'Sharpe'},
            title='All 50 Stocks — Risk vs Return'
        )
        fig.update_traces(textposition='top center', textfont_size=9, marker_size=10)
        fig.update_layout(**PLOT_LAYOUT, height=420,
                          coloraxis_colorbar=dict(title='Sharpe'))
        fig.update_xaxes(tickformat='.0%')
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🏆 Top 10 Stocks by Sharpe Ratio</div>', unsafe_allow_html=True)
        top10 = metrics_df.nlargest(10, 'sharpe_ratio')
        fig2  = go.Figure(go.Bar(
            x=top10['sharpe_ratio'],
            y=top10['ticker'],
            orientation='h',
            marker_color='#58a6ff',
            text=[f"{v:.2f}" for v in top10['sharpe_ratio']],
            textposition='outside',
            textfont=dict(color='#e6edf3')
        ))
        fig2.update_layout(**PLOT_LAYOUT, height=420, title='Top 10 by Sharpe Ratio',
                           xaxis_title='Sharpe Ratio', yaxis_title='')
        st.plotly_chart(fig2, use_container_width=True)

    # Sector heatmap of returns
    st.markdown('<div class="section-header">📊 Annual Returns — All Stocks</div>', unsafe_allow_html=True)
    ret_sorted = metrics_df.sort_values('annual_return', ascending=False)
    colors_bar  = ['#3fb950' if v > 0 else '#f85149' for v in ret_sorted['annual_return']]
    fig3 = go.Figure(go.Bar(
        x=ret_sorted['ticker'],
        y=ret_sorted['annual_return'] * 100,
        marker_color=colors_bar,
        text=[f"{v*100:.1f}%" for v in ret_sorted['annual_return']],
        textposition='outside',
        textfont=dict(size=9, color='#e6edf3')
    ))
    fig3.update_layout(**PLOT_LAYOUT, height=350,
                       title='Annualised Return per Stock (%)',
                       yaxis_title='Return (%)', xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — STOCK ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "📊 Stock Analysis":
    st.markdown('<div class="main-title">Stock Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.selectbox("Select Stock", TICKERS, index=TICKERS.index('RELIANCE') if 'RELIANCE' in TICKERS else 0)
    with col2:
        period = st.selectbox("Period", ["Full History", "Last 5 Years", "Last 3 Years", "Last 1 Year"])

    stock = df[df['ticker'] == ticker].copy().sort_values('date')

    if period == "Last 1 Year":
        stock = stock[stock['date'] >= stock['date'].max() - pd.DateOffset(years=1)]
    elif period == "Last 3 Years":
        stock = stock[stock['date'] >= stock['date'].max() - pd.DateOffset(years=3)]
    elif period == "Last 5 Years":
        stock = stock[stock['date'] >= stock['date'].max() - pd.DateOffset(years=5)]

    # KPIs
    latest   = stock.iloc[-1]
    earliest = stock.iloc[0]
    total_ret = (latest['close'] - earliest['close']) / earliest['close']
    m = metrics_df[metrics_df['ticker'] == ticker].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Price",   f"₹{latest['close']:,.2f}")
    c2.metric("Total Return",    f"{total_ret*100:.1f}%")
    c3.metric("Sharpe Ratio",    f"{m['sharpe_ratio']:.2f}")
    c4.metric("Max Drawdown",    f"{m['max_drawdown']*100:.1f}%")
    c5.metric("RSI (Latest)",    f"{latest['RSI']:.1f}")

    # Candlestick + Indicators
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.18, 0.18, 0.14],
        vertical_spacing=0.03,
        subplot_titles=[f'{ticker} — Price & Bollinger Bands', 'MACD', 'RSI (14)', 'Volume']
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=stock['date'], open=stock['open'],
        high=stock['high'], low=stock['low'], close=stock['close'],
        name='Price',
        increasing_line_color='#3fb950',
        decreasing_line_color='#f85149'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=stock['date'], y=stock['MA_21'],
        line=dict(color='#58a6ff', width=1), name='MA 21'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['date'], y=stock['MA_50'],
        line=dict(color='#e3b341', width=1), name='MA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['date'], y=stock['BB_upper'],
        line=dict(color='#8b949e', width=0.5, dash='dash'), name='BB Upper',
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['date'], y=stock['BB_lower'],
        line=dict(color='#8b949e', width=0.5, dash='dash'), name='BB Lower',
        fill='tonexty', fillcolor='rgba(139,148,158,0.08)', showlegend=False), row=1, col=1)

    # MACD
    macd_colors = ['#3fb950' if v >= 0 else '#f85149' for v in stock['MACD_hist']]
    fig.add_trace(go.Bar(x=stock['date'], y=stock['MACD_hist'],
        marker_color=macd_colors, name='MACD Hist', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['date'], y=stock['MACD'],
        line=dict(color='#58a6ff', width=1.2), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['date'], y=stock['MACD_signal'],
        line=dict(color='#f85149', width=1.2), name='Signal'), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=stock['date'], y=stock['RSI'],
        line=dict(color='#bc8cff', width=1.2), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_color='#f85149', line_dash='dash', line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_color='#3fb950', line_dash='dash', line_width=1, row=3, col=1)

    # Volume
    vol_colors = ['#3fb950' if c >= o else '#f85149'
                  for c, o in zip(stock['close'], stock['open'])]
    fig.add_trace(go.Bar(x=stock['date'], y=stock['volume'],
        marker_color=vol_colors, name='Volume', showlegend=False), row=4, col=1)

    fig.update_layout(
        **PLOT_LAYOUT, height=750,
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # RSI Signal
    rsi_val = latest['RSI']
    if rsi_val > 70:
        st.markdown(f'<span class="signal-sell">⚠️ RSI {rsi_val:.1f} — Overbought Zone</span>', unsafe_allow_html=True)
    elif rsi_val < 30:
        st.markdown(f'<span class="signal-buy">✅ RSI {rsi_val:.1f} — Oversold Zone (Potential Buy)</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="signal-hold">⏸ RSI {rsi_val:.1f} — Neutral Zone</span>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 3 — AI PREDICTOR
# ════════════════════════════════════════════════════════════
elif page == "🤖 AI Predictor":
    st.markdown('<div class="main-title">AI Stock Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">XGBoost model trained on 20 technical indicators — predicts next-day price direction</div>', unsafe_allow_html=True)

    ticker = st.selectbox("Select Stock to Predict", TICKERS,
                          index=TICKERS.index('RELIANCE') if 'RELIANCE' in TICKERS else 0)

    stock = df[df['ticker'] == ticker].sort_values('date')
    latest_row = stock.dropna(subset=FEATURES).iloc[-1]

    X_pred = pd.DataFrame([latest_row[FEATURES].values], columns=FEATURES)
    pred       = model.predict(X_pred)[0]
    pred_proba = model.predict_proba(X_pred)[0]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="section-header">🎯 Prediction</div>', unsafe_allow_html=True)
        direction = "📈 UP" if pred == 1 else "📉 DOWN"
        conf      = pred_proba[pred] * 100
        color     = "#3fb950" if pred == 1 else "#f85149"

        st.markdown(f"""
        <div class="card" style="text-align:center; border-color:{color};">
            <div style="font-size:48px; margin-bottom:8px;">{direction}</div>
            <div style="color:#8b949e; font-size:14px;">Next Day Direction</div>
            <div style="font-size:28px; font-weight:700; color:{color}; margin-top:12px;">
                {conf:.1f}% Confidence
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_proba[1] * 100,
            title={'text': "Probability UP (%)", 'font': {'color': '#e6edf3'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#8b949e'},
                'bar': {'color': '#58a6ff'},
                'steps': [
                    {'range': [0,  40], 'color': '#4a1d1d'},
                    {'range': [40, 60], 'color': '#2d2a1a'},
                    {'range': [60, 100],'color': '#1a4731'},
                ],
                'threshold': {
                    'line': {'color': '#f0f0f0', 'width': 3},
                    'thickness': 0.75,
                    'value': 50
                }
            },
            number={'font': {'color': '#e6edf3'}}
        ))
        fig_gauge.update_layout(paper_bgcolor='#161b22', font_color='#e6edf3',
                                height=280, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">📊 Current Technical Signals</div>', unsafe_allow_html=True)

        signals = []
        rsi = latest_row['RSI']
        if rsi > 70:
            signals.append(("RSI", f"{rsi:.1f}", "SELL", "Overbought"))
        elif rsi < 30:
            signals.append(("RSI", f"{rsi:.1f}", "BUY", "Oversold"))
        else:
            signals.append(("RSI", f"{rsi:.1f}", "HOLD", "Neutral"))

        macd = latest_row['MACD']
        sig  = latest_row['MACD_signal']
        if macd > sig:
            signals.append(("MACD", f"{macd:.2f}", "BUY", "MACD above Signal"))
        else:
            signals.append(("MACD", f"{macd:.2f}", "SELL", "MACD below Signal"))

        p50  = latest_row['price_vs_MA50']
        p200 = latest_row['price_vs_MA200']
        signals.append(("Price vs MA50",  f"{p50*100:.1f}%",  "BUY" if p50>0  else "SELL", "Above MA50"  if p50>0  else "Below MA50"))
        signals.append(("Price vs MA200", f"{p200*100:.1f}%", "BUY" if p200>0 else "SELL", "Above MA200" if p200>0 else "Below MA200"))

        mom5 = latest_row['momentum_5']
        signals.append(("5-Day Momentum", f"{mom5*100:.1f}%", "BUY" if mom5>0 else "SELL", "Positive" if mom5>0 else "Negative"))

        bb_pos = latest_row['BB_pos']
        if bb_pos > 0.8:
            signals.append(("Bollinger Band", f"{bb_pos:.2f}", "SELL", "Near Upper Band"))
        elif bb_pos < 0.2:
            signals.append(("Bollinger Band", f"{bb_pos:.2f}", "BUY", "Near Lower Band"))
        else:
            signals.append(("Bollinger Band", f"{bb_pos:.2f}", "HOLD", "Mid Band"))

        signal_df = pd.DataFrame(signals, columns=['Indicator','Value','Signal','Reason'])
        for _, row in signal_df.iterrows():
            badge = f'<span class="signal-{row["Signal"].lower()}">{row["Signal"]}</span>'
            st.markdown(
                f'<div class="card" style="padding:12px 16px;">'
                f'<b style="color:#e6edf3">{row["Indicator"]}</b>'
                f'<span style="color:#8b949e; margin:0 12px">|</span>'
                f'<span style="color:#58a6ff">{row["Value"]}</span>'
                f'<span style="color:#8b949e; margin:0 12px">|</span>'
                f'{badge}'
                f'<span style="color:#8b949e; font-size:12px; margin-left:12px">{row["Reason"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Feature importance mini-chart
        st.markdown('<div class="section-header">🔍 Top Features Used by Model</div>', unsafe_allow_html=True)
        imp_df = pd.DataFrame({'feature': FEATURES, 'importance': model.feature_importances_})
        imp_df = imp_df.nlargest(10, 'importance').sort_values('importance')
        fig_imp = go.Figure(go.Bar(
            x=imp_df['importance'], y=imp_df['feature'],
            orientation='h', marker_color='#58a6ff',
            text=[f"{v:.3f}" for v in imp_df['importance']],
            textposition='outside', textfont=dict(color='#e6edf3', size=10)
        ))
        fig_imp.update_layout(**PLOT_LAYOUT, height=320,
                              title='Top 10 Feature Importances (XGBoost)',
                              xaxis_title='Importance')
        st.plotly_chart(fig_imp, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 4 — PORTFOLIO BUILDER
# ════════════════════════════════════════════════════════════
elif page == "💼 Portfolio Builder":
    st.markdown('<div class="main-title">Portfolio Builder</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Three investor profiles — Conservative, Balanced, Aggressive</div>', unsafe_allow_html=True)

    profile_map = {
        "🛡️ Conservative (Low Risk)": "conservative",
        "⚖️ Balanced (Moderate Risk)": "balanced",
        "🚀 Aggressive (High Growth)": "aggressive"
    }
    profile_label = st.radio("Select Investor Profile", list(profile_map.keys()), horizontal=True)
    profile = profile_map[profile_label]

    port = portfolios_df[portfolios_df['profile'] == profile].copy()

    # Portfolio KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks",          f"{len(port)}")
    c2.metric("Avg Annual Return", f"{port['annual_return'].mean()*100:.1f}%")
    c3.metric("Avg Sharpe",      f"{port['sharpe_ratio'].mean():.2f}")
    c4.metric("Avg Max Drawdown",f"{port['max_drawdown'].mean()*100:.1f}%")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Pie chart
        pie_colors = px.colors.sequential.Blues[2:] if profile == 'conservative' else \
                     px.colors.sequential.Greens[2:] if profile == 'balanced' else \
                     px.colors.sequential.Reds[2:]

        fig_pie = go.Figure(go.Pie(
            labels=port['ticker'],
            values=port['weight'],
            hole=0.45,
            marker=dict(colors=px.colors.qualitative.Set3),
            textfont=dict(size=11, color='#e6edf3'),
            hovertemplate='<b>%{label}</b><br>Weight: %{percent}<extra></extra>'
        ))
        fig_pie.add_annotation(
            text=f"{profile.title()}<br>Portfolio",
            x=0.5, y=0.5, font_size=13,
            font_color='#e6edf3', showarrow=False
        )
        fig_pie.update_layout(**PLOT_LAYOUT, height=400,
                              title=f'{profile.title()} Portfolio Allocation',
                              showlegend=True,
                              legend=dict(bgcolor='#161b22', font=dict(size=10)))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Bar chart of weights
        port_sorted = port.sort_values('weight', ascending=True)
        fig_bar = go.Figure(go.Bar(
            x=port_sorted['weight'] * 100,
            y=port_sorted['ticker'],
            orientation='h',
            marker_color='#58a6ff',
            text=[f"{w*100:.1f}%" for w in port_sorted['weight']],
            textposition='outside',
            textfont=dict(color='#e6edf3')
        ))
        fig_bar.update_layout(**PLOT_LAYOUT, height=400,
                              title='Portfolio Weights (%)',
                              xaxis_title='Weight (%)')
        st.plotly_chart(fig_bar, use_container_width=True)

    # Holdings table
    st.markdown('<div class="section-header">📋 Holdings Detail</div>', unsafe_allow_html=True)
    display_port = port[['ticker','weight','annual_return','annual_vol','sharpe_ratio','max_drawdown']].copy()
    display_port.columns = ['Ticker','Weight','Annual Return','Annual Vol','Sharpe Ratio','Max Drawdown']
    display_port['Weight']        = display_port['Weight'].map('{:.1%}'.format)
    display_port['Annual Return'] = display_port['Annual Return'].map('{:.1%}'.format)
    display_port['Annual Vol']    = display_port['Annual Vol'].map('{:.1%}'.format)
    display_port['Sharpe Ratio']  = display_port['Sharpe Ratio'].map('{:.2f}'.format)
    display_port['Max Drawdown']  = display_port['Max Drawdown'].map('{:.1%}'.format)
    st.dataframe(display_port, use_container_width=True, hide_index=True)

    # Cumulative return simulation
    st.markdown('<div class="section-header">📈 Portfolio vs Individual Stocks</div>', unsafe_allow_html=True)
    close_pivot = df.pivot_table(index='date', columns='ticker', values='close')
    ret_pivot   = close_pivot.pct_change().dropna()

    valid_tickers = [t for t in port['ticker'] if t in ret_pivot.columns]
    weights_arr   = port.set_index('ticker').loc[valid_tickers, 'weight'].values
    weights_arr   = weights_arr / weights_arr.sum()

    port_ret = ret_pivot[valid_tickers].dot(weights_arr)
    cum_port = (1 + port_ret).cumprod()

    fig_cum = go.Figure()
    for t in valid_tickers[:5]:
        cum_t = (1 + ret_pivot[t]).cumprod()
        fig_cum.add_trace(go.Scatter(
            x=cum_t.index, y=cum_t.values,
            name=t, line=dict(width=0.8), opacity=0.5
        ))
    fig_cum.add_trace(go.Scatter(
        x=cum_port.index, y=cum_port.values,
        name=f'📊 {profile.title()} Portfolio',
        line=dict(width=3, color='#f0f0f0')
    ))
    fig_cum.update_layout(**PLOT_LAYOUT, height=400,
                          title='Cumulative Return (₹1 invested)',
                          yaxis_title='Portfolio Value',
                          legend=dict(bgcolor='#161b22', font=dict(size=10)))
    st.plotly_chart(fig_cum, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 5 — RISK ASSESSMENT
# ════════════════════════════════════════════════════════════
elif page == "⚠️ Risk Assessment":
    st.markdown('<div class="main-title">Risk Assessment</div>', unsafe_allow_html=True)

    ticker = st.selectbox("Select Stock", TICKERS,
                          index=TICKERS.index('RELIANCE') if 'RELIANCE' in TICKERS else 0)

    stock  = df[df['ticker'] == ticker].sort_values('date').copy()
    m      = metrics_df[metrics_df['ticker'] == ticker].iloc[0]
    ret    = stock['daily_return'].dropna()

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Annual Return",    f"{m['annual_return']*100:.1f}%")
    c2.metric("Annual Volatility",f"{m['annual_vol']*100:.1f}%")
    c3.metric("Sharpe Ratio",     f"{m['sharpe_ratio']:.2f}")
    c4.metric("Sortino Ratio",    f"{m['sortino_ratio']:.2f}")
    c5.metric("Max Drawdown",     f"{m['max_drawdown']*100:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        # Drawdown chart
        cumret  = (1 + ret).cumprod()
        rollmax = cumret.cummax()
        drawdown = ((cumret - rollmax) / rollmax) * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=stock['date'].iloc[len(stock)-len(drawdown):],
            y=drawdown.values,
            fill='tozeroy', fillcolor='rgba(248,81,73,0.2)',
            line=dict(color='#f85149', width=1),
            name='Drawdown'
        ))
        fig_dd.update_layout(**PLOT_LAYOUT, height=350,
                             title=f'{ticker} — Drawdown Over Time (%)',
                             yaxis_title='Drawdown (%)')
        st.plotly_chart(fig_dd, use_container_width=True)

    with col2:
        # Return distribution
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=ret.clip(-0.15, 0.15) * 100,
            nbinsx=80,
            marker_color='#58a6ff',
            opacity=0.75,
            name='Daily Returns'
        ))
        fig_hist.add_vline(x=0, line_color='white', line_dash='dash')
        fig_hist.add_vline(x=ret.quantile(0.05)*100, line_color='#f85149',
                           line_dash='dot', annotation_text='VaR 5%',
                           annotation_font_color='#f85149')
        fig_hist.update_layout(**PLOT_LAYOUT, height=350,
                               title=f'{ticker} — Daily Return Distribution',
                               xaxis_title='Daily Return (%)',
                               yaxis_title='Frequency')
        st.plotly_chart(fig_hist, use_container_width=True)

    # Rolling Sharpe
    st.markdown('<div class="section-header">📉 Rolling 252-Day Sharpe Ratio</div>', unsafe_allow_html=True)
    rolling_ret = ret.rolling(252).mean() * 252
    rolling_vol = ret.rolling(252).std()  * np.sqrt(252)
    rolling_sharpe = (rolling_ret - 0.06) / (rolling_vol + 1e-10)

    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(
        x=stock['date'].iloc[len(stock)-len(rolling_sharpe):],
        y=rolling_sharpe.values,
        line=dict(color='#58a6ff', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(88,166,255,0.1)',
        name='Rolling Sharpe'
    ))
    fig_rs.add_hline(y=0, line_color='#f85149', line_dash='dash')
    fig_rs.add_hline(y=1, line_color='#3fb950', line_dash='dash',
                     annotation_text='Good (1.0)', annotation_font_color='#3fb950')
    fig_rs.update_layout(**PLOT_LAYOUT, height=320,
                         title='Rolling 1-Year Sharpe Ratio',
                         yaxis_title='Sharpe Ratio')
    st.plotly_chart(fig_rs, use_container_width=True)

    # Risk comparison table
    st.markdown('<div class="section-header">📊 Risk Comparison — All Stocks</div>', unsafe_allow_html=True)
    risk_table = metrics_df.copy()
    risk_table['annual_return'] = risk_table['annual_return'].map('{:.1%}'.format)
    risk_table['annual_vol']    = risk_table['annual_vol'].map('{:.1%}'.format)
    risk_table['sharpe_ratio']  = risk_table['sharpe_ratio'].map('{:.2f}'.format)
    risk_table['sortino_ratio'] = risk_table['sortino_ratio'].map('{:.2f}'.format)
    risk_table['max_drawdown']  = risk_table['max_drawdown'].map('{:.1%}'.format)
    risk_table.columns          = ['Ticker','Annual Return','Annual Vol','Sharpe','Sortino','Max Drawdown','Days']
    st.dataframe(risk_table[['Ticker','Annual Return','Annual Vol','Sharpe','Sortino','Max Drawdown']],
                 use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# PAGE 6 — ANOMALY DETECTION
# ════════════════════════════════════════════════════════════
elif page == "🔍 Anomaly Detection":
    st.markdown('<div class="main-title">Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Isolation Forest detects unusual market events — volatility spikes, crashes, volume surges</div>', unsafe_allow_html=True)

    # KPIs
    total      = len(anomalies_df)
    n_anom     = anomalies_df['is_anomaly'].sum()
    n_crashes  = anomalies_df['extreme_crash'].sum()
    n_surges   = anomalies_df['extreme_surge'].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",    f"{total:,}")
    c2.metric("Anomalies Found",  f"{n_anom:,}", f"{n_anom/total*100:.1f}% of records")
    c3.metric("Extreme Crashes",  f"{n_crashes:,}")
    c4.metric("Extreme Surges",   f"{n_surges:,}")

    # Market anomaly timeline
    st.markdown('<div class="section-header">📅 Market-Wide Anomaly Timeline</div>', unsafe_allow_html=True)
    timeline = (anomalies_df[anomalies_df['is_anomaly']==1]
                .groupby('date').size().reset_index())
    timeline.columns = ['date','count']

    fig_tl = go.Figure()
    fig_tl.add_trace(go.Bar(
        x=timeline['date'], y=timeline['count'],
        marker_color='#f85149', opacity=0.7, name='Anomalies'
    ))
    events = {'2001-09-11':'9/11', '2008-10-01':'2008 Crisis',
              '2016-11-08':'Demonetisation', '2020-03-23':'COVID Crash'}
    for d, label in events.items():
        dt = pd.Timestamp(d)
        if dt >= timeline['date'].min():
            fig_tl.add_vline(x=dt, line_color='white', line_dash='dash', line_width=1.5)
            fig_tl.add_annotation(x=dt, y=timeline['count'].max()*0.85,
                                  text=label, showarrow=False,
                                  font=dict(color='#e3b341', size=10),
                                  bgcolor='#161b22', bordercolor='#e3b341')
    fig_tl.update_layout(**PLOT_LAYOUT, height=320,
                         title='Number of Stocks with Anomalies per Day',
                         xaxis_title='Date', yaxis_title='Count')
    st.plotly_chart(fig_tl, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Anomalies per stock
        st.markdown('<div class="section-header">🏢 Anomalies per Stock</div>', unsafe_allow_html=True)
        per_stock = (anomalies_df[anomalies_df['is_anomaly']==1]
                     .groupby('ticker').size().sort_values(ascending=True).reset_index())
        per_stock.columns = ['ticker','count']
        fig_ps = go.Figure(go.Bar(
            x=per_stock['count'], y=per_stock['ticker'],
            orientation='h', marker_color='#f85149',
            text=per_stock['count'], textposition='outside',
            textfont=dict(color='#e6edf3', size=9)
        ))
        fig_ps.update_layout(**PLOT_LAYOUT, height=500,
                             title='Anomaly Count per Stock',
                             xaxis_title='Count')
        st.plotly_chart(fig_ps, use_container_width=True)

    with col2:
        # Anomaly type breakdown
        st.markdown('<div class="section-header">📊 Anomaly Type Breakdown</div>', unsafe_allow_html=True)
        type_data = {
            'Volatility Spike': anomalies_df['volatility_spike'].sum(),
            'Volume Spike'    : anomalies_df['volume_spike'].sum(),
            'Extreme Crash'   : anomalies_df['extreme_crash'].sum(),
            'Extreme Surge'   : anomalies_df['extreme_surge'].sum()
        }
        fig_types = go.Figure(go.Bar(
            x=list(type_data.keys()),
            y=list(type_data.values()),
            marker_color=['#e3b341','#58a6ff','#f85149','#3fb950'],
            text=list(type_data.values()),
            textposition='outside',
            textfont=dict(color='#e6edf3')
        ))
        fig_types.update_layout(**PLOT_LAYOUT, height=300,
                                title='Anomaly Types Detected',
                                yaxis_title='Count')
        st.plotly_chart(fig_types, use_container_width=True)

        # Top anomalies table
        st.markdown('<div class="section-header">🚨 Most Extreme Anomalies</div>', unsafe_allow_html=True)
        top_anom = (anomalies_df[anomalies_df['is_anomaly']==1]
                    .sort_values('anomaly_raw')
                    .head(10)[['date','ticker','close','daily_return','volume_ratio']]
                    .copy())
        top_anom['daily_return'] = top_anom['daily_return'].map('{:.1%}'.format)
        top_anom['volume_ratio'] = top_anom['volume_ratio'].map('{:.1f}x'.format)
        top_anom['close']        = top_anom['close'].map('₹{:,.2f}'.format)
        top_anom.columns         = ['Date','Ticker','Close','Return','Vol Ratio']
        st.dataframe(top_anom, use_container_width=True, hide_index=True)

    # Single stock anomaly chart
    st.markdown('<div class="section-header">🔍 Anomaly Explorer — Single Stock</div>', unsafe_allow_html=True)
    sel_ticker = st.selectbox("Select Stock", TICKERS,
                              index=TICKERS.index('RELIANCE') if 'RELIANCE' in TICKERS else 0,
                              key='anom_ticker')

    stock_anom  = anomalies_df[anomalies_df['ticker'] == sel_ticker].sort_values('date')
    anom_only   = stock_anom[stock_anom['is_anomaly'] == 1]

    fig_sa = go.Figure()
    fig_sa.add_trace(go.Scatter(
        x=stock_anom['date'], y=stock_anom['close'],
        line=dict(color='#58a6ff', width=1),
        name='Price'
    ))
    fig_sa.add_trace(go.Scatter(
        x=anom_only['date'], y=anom_only['close'],
        mode='markers',
        marker=dict(color='#f85149', size=8, symbol='x',
                    line=dict(color='white', width=1)),
        name='Anomaly'
    ))
    fig_sa.update_layout(**PLOT_LAYOUT, height=380,
                         title=f'{sel_ticker} — Price with Anomalies Highlighted',
                         yaxis_title='Price (₹)',
                         legend=dict(bgcolor='#161b22'))
    st.plotly_chart(fig_sa, use_container_width=True)