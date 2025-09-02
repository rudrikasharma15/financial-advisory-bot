import sys
import os
import time
from typing import Dict, Any, Optional, Tuple, List

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page config must be the first Streamlit call
st.set_page_config(
    page_title="📊 Artha.ai",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        padding-top: 2rem;
    }
    
    /* Custom font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Cards styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    .metric-change {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .positive { color: #38a169; }
    .negative { color: #e53e3e; }
    .neutral { color: #718096; }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: scale(1.02);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    
    .feature-desc {
        opacity: 0.9;
        line-height: 1.5;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: border-color 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Success/Warning/Error styling */
    .stSuccess, .stWarning, .stError {
        border-radius: 8px;
        border: none;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stSuccess {
        background: linear-gradient(90deg, #48bb78, #38a169);
        color: white;
    }
    
    .stWarning {
        background: linear-gradient(90deg, #ed8936, #dd6b20);
        color: white;
    }
    
    .stError {
        background: linear-gradient(90deg, #f56565, #e53e3e);
        color: white;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(90deg, #2d3748 0%, #4a5568 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f7fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* Navigation active state */
    .nav-active {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.2rem 0;
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Path safety: make sure local folder is importable
# ------------------------------------------------------------
try:
    _this_file = __file__
except NameError:
    _this_file = ""

_base_dir = os.path.dirname(os.path.abspath(_this_file)) if _this_file else os.getcwd()
if _base_dir not in sys.path:
    sys.path.append(_base_dir)

# Import your login system
try:
    from auth import auth_component
    # 1. Login / Signup first
    auth_status = auth_component()
    if not auth_status:
        st.markdown("""
        <div class="main-header">
            <h1>🚪 Welcome to Artha.ai</h1>
            <p>Please login to access your financial dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
except ImportError:
    st.warning("Auth module not found. Proceeding without authentication.")

# --- IMPT: Error Handling for logic.py ---
try:
    from logic import (
        fetch_stock_data, compute_rsi, get_general_financial_advice,
        calculate_savings_goal, get_stock_data, add_technical_indicators,
        get_mock_macro_features, prepare_model, predict_stocks, fetch_stock_news,
        get_advice, calculate_risk, get_strategy
    )
    _LOGIC_LOADED = True
except ImportError:
    _LOGIC_LOADED = False
    
    # Dummy implementations for development/testing when logic.py is absent
    def fetch_stock_data(*args, **kwargs): 
        return pd.DataFrame()
    def compute_rsi(*args, **kwargs): 
        return pd.Series([50.0, 55.0])
    def get_general_financial_advice(*args, **kwargs): 
        return "This is dummy financial advice because `logic.py` could not be loaded."
    def calculate_savings_goal(target_amount, years, annual_return):
        if years <= 0:
            return {'monthly_saving': target_amount / 12 if target_amount > 0 else 0.0} 
        
        monthly_rate = (annual_return / 100) / 12
        num_months = years * 12
        
        if num_months == 0: 
            monthly_saving = target_amount
        elif monthly_rate == 0:
            monthly_saving = target_amount / num_months
        else:
            factor = ((1 + monthly_rate)**num_months - 1) / monthly_rate
            monthly_saving = target_amount / factor if factor != 0 else target_amount / num_months
        
        return {
            'target_amount': target_amount,
            'years': years,
            'annual_return': annual_return,
            'monthly_saving': monthly_saving
        }
    def get_stock_data(*args, **kwargs): 
        return pd.DataFrame({'Date': pd.to_datetime(['2023-01-01', '2023-01-02']), 
                           'AAPL': [150, 152], 'MSFT': [250, 255]}).set_index('Date')
    def add_technical_indicators(df, symbols): 
        return df
    def get_mock_macro_features(*args, **kwargs): 
        return pd.DataFrame()
    def prepare_model(*args, **kwargs): 
        return None
    def predict_stocks(model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size):
        dummy_predicted = [155.0, 156.0]
        dummy_actual = [153.0, 154.0]
        return {'AAPL': {'predicted': dummy_predicted, 'actual': dummy_actual}, 
                'MSFT': {'predicted': [258.0, 260.0], 'actual': [256.0, 257.0]}}, {}
    def fetch_stock_news(*args, **kwargs): 
        return "No news available (dummy data)."
    def get_advice(*args, **kwargs): 
        return "Generic advice (dummy data)."
    def calculate_risk(*args, **kwargs): 
        return 5.0
    def get_strategy(*args, **kwargs): 
        return "General strategy (dummy data)."

# Configure caching
@st.cache_data(ttl=300, show_spinner=False)
def load_stock_data(symbols: List[str]) -> Optional[pd.DataFrame]:
    """Load stock data with caching and error handling."""
    if not _LOGIC_LOADED: 
        return get_stock_data(symbols)
    
    if not symbols:
        return None
    try:
        with st.spinner("📡 Fetching market data..."):
            return get_stock_data(symbols)
    except Exception as e:
        st.error(f"⚠️ Error fetching stock data: {str(e)}")
        return None

@st.cache_data(ttl=300, show_spinner=False)
def process_stock_data(stock_data: pd.DataFrame, symbols: List[str]) -> Tuple[Any, Dict]:
    """Process stock data with caching."""
    if not _LOGIC_LOADED: 
        dummy_results, _ = predict_stocks(None,None,None,None,None,None,None,None) 
        return stock_data, dummy_results
    
    if stock_data is None or stock_data.empty:
        return None, {}
    
    stock_data_copy = stock_data.copy()
    stock_data_processed = add_technical_indicators(stock_data_copy, symbols)
    macro = get_mock_macro_features(stock_data_processed.index)
    model_result = prepare_model(symbols, stock_data_processed, macro)
    
    if not model_result:
        return stock_data_processed, {}
    
    model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size = model_result
    results, evaluation = predict_stocks(model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size)
    
    return stock_data_processed, results

# State initialization
if "dashboard_run" not in st.session_state:
    st.session_state["dashboard_run"] = False
if "planner_results" not in st.session_state:
    st.session_state["planner_results"] = None

# Enhanced Sidebar Navigation
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
    <h2 style="color: white; margin: 0; font-weight: 700;">📊 Artha.ai</h2>
    <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;">Your Financial Assistant</p>
</div>
""", unsafe_allow_html=True)

tab_options = st.sidebar.radio(
    "🔎 Navigate",
    [
        "🏠 Home",
        "📊 Stock Dashboard", 
        "💬 Finance Bot",
        "🎯 Goal Planner",
        "💼 Portfolio Tracker",
        "💸 SIP Calculator"
    ]
)

# Navigation status
nav_status = {
    "🏠 Home": "🏠 Home Page",
    "📊 Stock Dashboard": "📊 Stock Analysis", 
    "💬 Finance Bot": "🤖 AI Assistant",
    "🎯 Goal Planner": "🎯 Financial Goals",
    "💼 Portfolio Tracker": "📂 Portfolio Management", 
    "💸 SIP Calculator": "💸 Investment Calculator"
}

st.sidebar.success(f"✅ {nav_status[tab_options]}")

# Home Tab with enhanced design
if tab_options == "🏠 Home":
    # Hero Section
    st.markdown("""
    <div class="main-header">
        <h1>🏠 Welcome to Artha.ai</h1>
        <p>Your intelligent financial companion powered by AI and deep learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Stock Predictions</div>
            <div class="feature-desc">Advanced ML models predict future stock prices with technical analysis and risk assessment</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Goal Planning</div>
            <div class="feature-desc">Smart savings calculator with inflation adjustment and growth projections</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI Financial Advisor</div>
            <div class="feature-desc">Get personalized financial advice powered by Gemini AI for informed decisions</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💼</div>
            <div class="feature-title">Portfolio Tracking</div>
            <div class="feature-desc">Real-time portfolio monitoring with performance analytics and allocation insights</div>
        </div>
        """, unsafe_allow_html=True)

# Stock Dashboard Tab with enhanced UI
elif tab_options == "📊 Stock Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Stock Analysis & Predictions</h1>
        <p>AI-powered stock analysis with technical indicators and price predictions</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state variables
    if "symbols_input" not in st.session_state:
        st.session_state.symbols_input = "AAPL, MSFT"
    if "last_symbols" not in st.session_state:
        st.session_state.last_symbols = ""
    if "stock_data" not in st.session_state:
        st.session_state.stock_data = None
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "symbols" not in st.session_state:
        st.session_state.symbols = []

    # Input section with enhanced styling
    st.markdown("### 📥 Stock Selection")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbols_input = st.text_input(
            "Enter stock symbols (comma-separated)",
            value=st.session_state.symbols_input,
            placeholder="e.g., AAPL, GOOGL, MSFT, TSLA",
            help="Enter valid stock ticker symbols separated by commas"
        )
    
    with col2:
        analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")
    
    with col3:
        refresh_btn = st.button("🔄 Refresh", use_container_width=True)
        if refresh_btn:
            load_stock_data.clear()
            process_stock_data.clear()
            st.session_state.stock_data = None
            st.session_state.results = {}
            st.rerun()

    # Update stored input if changed
    if symbols_input != st.session_state.symbols_input:
        st.session_state.symbols_input = symbols_input

    # Fetch & process data
    symbols_changed = st.session_state.symbols_input != st.session_state.last_symbols
    if analyze_btn or (symbols_changed and st.session_state.stock_data is None):
        st.session_state.last_symbols = st.session_state.symbols_input
        symbols = [s.strip().upper() for s in st.session_state.symbols_input.split(",") if s.strip()]
        
        if not symbols:
            st.warning("⚠️ Please enter at least one stock symbol")
            st.stop()

        with st.spinner("📊 Analyzing market data..."):
            stock_data = load_stock_data(symbols)
            if stock_data is not None and not stock_data.empty:
                stock_data, results = process_stock_data(stock_data, symbols)
                if results:
                    st.session_state.stock_data = stock_data
                    st.session_state.results = results
                    st.session_state.symbols = symbols
                    st.success("✅ Analysis complete!")
                else:
                    st.warning("⚠️ Prediction failed. Displaying available data.")
                    st.session_state.stock_data = stock_data
                    st.session_state.results = {}
            else:
                st.error("⚠️ Unable to fetch stock data. Please verify symbols and try again.")
                st.session_state.stock_data = None
                st.session_state.results = {}

    # Display results with enhanced design
    if st.session_state.stock_data is not None and st.session_state.results:
        st.markdown("### 📊 Analysis Results")
        tabs = st.tabs([f"📊 {symbol}" for symbol in st.session_state.symbols])

        for idx, symbol in enumerate(st.session_state.symbols):
            with tabs[idx]:
                results = st.session_state.results
                stock_data = st.session_state.stock_data

                if symbol not in results or not isinstance(results[symbol], dict) or \
                   'predicted' not in results[symbol] or 'actual' not in results[symbol] or \
                   len(results[symbol]['predicted']) == 0 or len(results[symbol]['actual']) == 0:
                    st.error(f"⚠️ No sufficient prediction data available for {symbol}")
                    if symbol in stock_data.columns and not stock_data.empty:
                        st.markdown("#### Raw Price Data")
                        fig = px.line(stock_data, y=symbol, title=f"{symbol} Price History")
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    continue

                predicted = results[symbol]['predicted']
                actual = results[symbol]['actual']

                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                if len(actual) > 0 and len(predicted) > 0:
                    current_price = actual[-1]
                    predicted_price = predicted[-1]
                    change_percentage = ((predicted_price - current_price) / current_price) * 100
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">${current_price:.2f}</div>
                            <div class="metric-label">Current Price</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">${predicted_price:.2f}</div>
                            <div class="metric-label">Predicted Price</div>
                            <div class="metric-change {'positive' if change_percentage > 0 else 'negative'}">
                                {change_percentage:+.2f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Technical indicators
                    if symbol in stock_data.columns and not stock_data.empty:
                        rsi = compute_rsi(stock_data[symbol])
                        rsi_value = rsi.dropna().iloc[-1] if not rsi.empty else 50
                    else:
                        rsi_value = 50
                    
                    risk = calculate_risk(symbol, stock_data, results)
                    
                    with col3:
                        rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                        rsi_color = "negative" if rsi_value > 70 else "positive" if rsi_value < 30 else "neutral"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{rsi_value:.1f}</div>
                            <div class="metric-label">RSI</div>
                            <div class="metric-change {rsi_color}">{rsi_status}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        risk_color = "negative" if risk > 7 else "neutral" if risk > 4 else "positive"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{risk:.1f}/10</div>
                            <div class="metric-label">Risk Score</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Price prediction chart
                    st.markdown("#### 📈 Price Prediction Chart")
                    chart_index_start = max(0, len(stock_data.index) - len(predicted))
                    chart_data = pd.DataFrame({
                        "Actual": actual,
                        "Predicted": predicted
                    }, index=stock_data.index[chart_index_start:chart_index_start + len(predicted)])
                    
                    fig = px.line(chart_data, title=f"{symbol} - Actual vs Predicted Prices")
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_title="Date",
                        yaxis_title="Price ($)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Additional analysis sections
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    with st.expander(f"📊 Technical Analysis - {symbol}"):
                        if not rsi.empty:
                            fig_rsi = px.line(rsi, title="RSI History")
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                            fig_rsi.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                            )
                            st.plotly_chart(fig_rsi, use_container_width=True)
                        else:
                            st.info("RSI data not available")
                
                with col_right:
                    with st.expander(f"🗞️ Latest News - {symbol}"):
                        news_content = fetch_stock_news(symbol)
                        if news_content and news_content != "No news available (dummy data).":
                            st.markdown(news_content)
                        else:
                            st.info("No recent news available")

                # AI Analysis section
                st.markdown("#### 🤖 AI Analysis")
                query_key = f"query_{symbol}"
                advice_key = f"advice_{symbol}"

                if query_key not in st.session_state:
                    st.session_state[query_key] = ""

                with st.form(key=f"ai_form_{symbol}"):
                    col_query, col_submit = st.columns([3, 1])
                    with col_query:
                        query = st.text_input(
                            f"Ask AI about {symbol}:",
                            value=st.session_state.get(query_key, ""),
                            placeholder=f"e.g., Should I invest in {symbol} now?"
                        )
                    with col_submit:
                        submitted = st.form_submit_button("Get Advice", use_container_width=True)

                    if submitted:
                        if query.strip():
                            st.session_state[query_key] = query
                            try:
                                with st.spinner("🤔 Analyzing..."):
                                    advice = get_general_financial_advice(
                                        query,
                                        [symbol],
                                        st.session_state.stock_data,
                                        st.session_state.results
                                    )
                                    st.session_state[advice_key] = advice
                            except Exception as e:
                                st.error(f"AI analysis error: {e}")
                                st.session_state[advice_key] = f"Error: {e}"
                        else:
                            st.warning("Please enter a question")

                if advice_key in st.session_state and st.session_state[advice_key]:
                    st.markdown("##### 💡 AI Recommendation")
                    st.info(st.session_state[advice_key])

                # Download section
                if 'chart_data' in locals() and isinstance(chart_data, pd.DataFrame) and not chart_data.empty:
                    rsi_for_download = rsi.iloc[-len(predicted):] if not rsi.empty and len(rsi) >= len(predicted) else [None]*len(predicted)
                    
                    download_data = pd.DataFrame({
                        "Date": stock_data.index[chart_index_start:chart_index_start + len(predicted)],
                        "Predicted": predicted,
                        "Actual": actual,
                        "RSI": rsi_for_download
                    }).to_csv(index=False)
                    
                    st.download_button(
                        label=f"📥 Download {symbol} Analysis",
                        data=download_data,
                        file_name=f"{symbol}_analysis.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# Finance Bot Tab
elif tab_options == "💬 Finance Bot":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Financial Advisor</h1>
        <p>Get personalized financial advice powered by advanced AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat interface
    st.markdown("### 💬 Ask Your Financial Questions")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "What would you like to know?", 
            key="general_query",
            placeholder="e.g., How should I diversify my portfolio?"
        )
    with col2:
        get_advice_btn = st.button("Get Advice", use_container_width=True, type="primary")
    
    if get_advice_btn:
        if query:
            try:
                with st.spinner("🤔 AI is thinking..."):
                    advice = get_general_financial_advice(query)
                    st.session_state["advice"] = advice
                    st.success("✅ Analysis complete!")
            except Exception as e:
                st.error(f"Error getting advice: {e}")
                st.session_state["advice"] = f"Error: {e}"
        else:
            st.warning("Please enter a question")
            st.session_state["advice"] = ""
    
    if "advice" in st.session_state and st.session_state["advice"]:
        st.markdown("### 💡 AI Recommendation")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Financial Advisor Says:</div>
            <div style="margin-top: 1rem; line-height: 1.6;">
                {st.session_state['advice']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Sample questions
    st.markdown("### 💭 Sample Questions")
    sample_questions = [
        "What's the best investment strategy for beginners?",
        "How much should I save for retirement?",
        "Should I invest in stocks or bonds?",
        "What are the tax implications of my investments?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(question, key=f"sample_{i}", use_container_width=True):
                st.session_state["general_query"] = question

# Helper function for Goal Planner
def _simulate_goal_growth(current_savings: float, monthly_saving: float, years: int, annual_return: float) -> pd.DataFrame:
    monthly_rate = (annual_return / 100) / 12
    data = []
    current_balance = float(current_savings)
    
    for year_num in range(1, years + 1):
        start_balance_year = current_balance
        contributions_this_year = 0
        interest_this_year = 0
        
        for month in range(12):
            monthly_interest = current_balance * monthly_rate
            current_balance += monthly_interest
            interest_this_year += monthly_interest
            
            current_balance += monthly_saving
            contributions_this_year += monthly_saving
            
        end_balance_year = current_balance
        
        data.append({
            'Year': year_num,
            'Starting Balance': start_balance_year,
            'Annual Contributions': contributions_this_year,
            'Interest Earned': interest_this_year,
            'Ending Balance': end_balance_year
        })
        
    return pd.DataFrame(data)

# Goal Planner Tab
if tab_options == "🎯 Goal Planner":
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Financial Goal Planner</h1>
        <p>Plan your financial future with smart savings strategies</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📝 Define Your Financial Goal")

    # Input form
    col1, col2 = st.columns(2)
    with col1:
        goal_name = st.text_input(
            "Goal Name",
            value=st.session_state.get('planner_goal_name', ""),
            placeholder="e.g., Dream Vacation, Emergency Fund"
        )
        target_amount = st.number_input(
            "Target Amount (₹)",
            min_value=1000.0,
            value=100000.0,
            format="%.2f"
        )
    
    with col2:
        current_savings = st.number_input(
            "Current Savings (₹)",
            min_value=0.0,
            value=10000.0,
            format="%.2f"
        )
        years = st.slider("Time Horizon (years)", 1, 30, 10)

    col3, col4 = st.columns(2)
    with col3:
        annual_return = st.slider("Expected Annual Return (%)", 1, 20, 8)
    
    with col4:
        inflation_enabled = st.checkbox("Adjust for Inflation", value=False)
        inflation_rate = st.slider("Inflation Rate (%)", 0, 10, 5) if inflation_enabled else 0

    # Calculate button
    if st.button("🚀 Calculate My Plan", type="primary", use_container_width=True):
        # Calculations
        real_return_no_inflation = annual_return
        real_return_with_inflation = annual_return - inflation_rate if inflation_enabled else annual_return
        
        monthly_saving_no_inflation = calculate_savings_goal(target_amount, years, real_return_no_inflation).get('monthly_saving', 0.0)
        monthly_saving_with_inflation = calculate_savings_goal(target_amount, years, real_return_with_inflation).get('monthly_saving', 0.0)
        
        growth_df_no_inflation = _simulate_goal_growth(current_savings, monthly_saving_no_inflation, years, real_return_no_inflation)
        growth_df_with_inflation = _simulate_goal_growth(current_savings, monthly_saving_with_inflation, years, real_return_with_inflation)
        
        # Display results
        st.markdown("### 📊 Your Investment Plan")
        
        tab1, tab2 = st.tabs(["Without Inflation", "With Inflation"])
        
        with tab1:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">₹{monthly_saving_no_inflation:,.0f}</div>
                    <div class="metric-label">Monthly Investment Needed</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                final_amount = growth_df_no_inflation['Ending Balance'].iloc[-1] if not growth_df_no_inflation.empty else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">₹{final_amount:,.0f}</div>
                    <div class="metric-label">Final Amount</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                total_contributions = monthly_saving_no_inflation * years * 12 + current_savings
                interest_earned = final_amount - total_contributions
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">₹{interest_earned:,.0f}</div>
                    <div class="metric-label">Interest Earned</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Growth chart
            if not growth_df_no_inflation.empty:
                fig = px.bar(growth_df_no_inflation, x='Year', y=['Annual Contributions', 'Interest Earned'],
                           title="Annual Growth Breakdown")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if inflation_enabled:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">₹{monthly_saving_with_inflation:,.0f}</div>
                        <div class="metric-label">Monthly Investment Needed</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    final_amount_inf = growth_df_with_inflation['Ending Balance'].iloc[-1] if not growth_df_with_inflation.empty else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">₹{final_amount_inf:,.0f}</div>
                        <div class="metric-label">Final Amount</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_c:
                    real_purchasing_power = final_amount_inf / ((1 + inflation_rate/100) ** years)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">₹{real_purchasing_power:,.0f}</div>
                        <div class="metric-label">Real Purchasing Power</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if not growth_df_with_inflation.empty:
                    fig_inf = px.bar(growth_df_with_inflation, x='Year', y=['Annual Contributions', 'Interest Earned'],
                               title="Annual Growth Breakdown (Inflation Adjusted)")
                    fig_inf.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_inf, use_container_width=True)
            else:
                st.info("Enable inflation adjustment to see inflation-adjusted projections")

# Portfolio Tracker Tab
if tab_options == "💼 Portfolio Tracker":
    st.markdown("""
    <div class="main-header">
        <h1>💼 Portfolio Tracker</h1>
        <p>Monitor your investments and track performance in real-time</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize portfolio
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = []

    # Add holding form
    st.markdown("### ➕ Add Investment")
    with st.form("add_holding"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbol = st.text_input("Symbol", placeholder="e.g., AAPL")
        with col2:
            units = st.number_input("Shares", min_value=0.0, format="%.2f")
        with col3:
            buy_price = st.number_input("Buy Price", min_value=0.0, format="%.2f")
        with col4:
            buy_date = st.date_input("Purchase Date")
        
        col_submit, col_clear = st.columns([1, 1])
        with col_submit:
            add_clicked = st.form_submit_button("Add to Portfolio", use_container_width=True, type="primary")
        with col_clear:
            if st.form_submit_button("Clear Portfolio", use_container_width=True):
                st.session_state.portfolio = []
                st.rerun()
        
        if add_clicked and symbol.strip() and units > 0 and buy_price > 0:
            st.session_state.portfolio.append({
                "symbol": symbol.strip().upper(),
                "units": units,
                "buy_price": buy_price,
                "buy_date": str(buy_date)
            })
            st.success(f"✅ {symbol.strip().upper()} added to portfolio!")
            st.rerun()

    # Display portfolio
    if st.session_state.portfolio:
        df = pd.DataFrame(st.session_state.portfolio)
        symbols = list(df["symbol"].unique())
        
        try:
            with st.spinner("📡 Fetching current prices..."):
                prices_df = get_stock_data(symbols)
            
            if prices_df is None or prices_df.empty:
                st.warning("Could not fetch current prices")
                latest_prices = {sym: 0 for sym in symbols}
            else:
                latest_prices = {}
                for symbol in symbols:
                    if symbol in prices_df.columns:
                        latest_prices[symbol] = prices_df[symbol].iloc[-1]
                    else:
                        latest_prices[symbol] = 0
        except Exception as e:
            st.error(f"Error fetching prices: {e}")
            latest_prices = {sym: 0 for sym in symbols}

        # Calculate metrics
        df["Current Price"] = df["symbol"].map(latest_prices)
        df["Current Value"] = df["Current Price"] * df["units"]
        df["Investment Cost"] = df["buy_price"] * df["units"]
        df["Gain/Loss"] = df["Current Value"] - df["Investment Cost"]
        df["% Change"] = ((df["Current Price"] - df["buy_price"]) / df["buy_price"]) * 100

        total_value = df["Current Value"].sum()
        total_cost = df["Investment Cost"].sum()
        total_gain_loss = total_value - total_cost
        total_return_pct = ((total_value / total_cost) - 1) * 100 if total_cost > 0 else 0

        # Portfolio summary
        st.markdown("### 📊 Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">₹{total_cost:,.0f}</div>
                <div class="metric-label">Total Invested</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">₹{total_value:,.0f}</div>
                <div class="metric-label">Current Value</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            gain_loss_color = "positive" if total_gain_loss >= 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">₹{total_gain_loss:,.0f}</div>
                <div class="metric-label">Total Gain/Loss</div>
                <div class="metric-change {gain_loss_color}">{total_return_pct:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            num_holdings = len(df)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{num_holdings}</div>
                <div class="metric-label">Holdings</div>
            </div>
            """, unsafe_allow_html=True)

        # Holdings table
        st.markdown("### 📋 Holdings Details")
        if total_value > 0:
            df["Weight %"] = (df["Current Value"] / total_value * 100).round(1)
        else:
            df["Weight %"] = 0
        
        # Format the dataframe for display
        display_df = df.copy()
        display_df["Current Price"] = display_df["Current Price"].apply(lambda x: f"₹{x:.2f}")
        display_df["buy_price"] = display_df["buy_price"].apply(lambda x: f"₹{x:.2f}")
        display_df["Current Value"] = display_df["Current Value"].apply(lambda x: f"₹{x:,.0f}")
        display_df["Investment Cost"] = display_df["Investment Cost"].apply(lambda x: f"₹{x:,.0f}")
        display_df["Gain/Loss"] = display_df["Gain/Loss"].apply(lambda x: f"₹{x:,.0f}")
        display_df["% Change"] = display_df["% Change"].apply(lambda x: f"{x:+.1f}%")
        display_df["Weight %"] = display_df["Weight %"].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Portfolio allocation chart
        if total_value > 0:
            st.markdown("### 🥧 Portfolio Allocation")
            fig_pie = px.pie(df, values="Current Value", names="symbol", 
                           title="Portfolio Distribution by Value")
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Download portfolio data
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📥 Download Portfolio Data",
            csv_data,
            "portfolio_data.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("Add some holdings above to start tracking your portfolio")

# SIP Calculator Tab
if tab_options == "💸 SIP Calculator":
    st.markdown("""
    <div class="main-header">
        <h1>💸 SIP & Lumpsum Calculator</h1>
        <p>Compare investment strategies and visualize your wealth growth</p>
    </div>
    """, unsafe_allow_html=True)

    # Investment type selection
    st.markdown("### 💡 Choose Investment Strategy")
    investment_type = st.radio("Investment Type:", ["SIP (Systematic Investment Plan)", "Lumpsum Investment"], horizontal=True)

    col1, col2 = st.columns(2)
    
    with col1:
        if "SIP" in investment_type:
            amount = st.number_input("Monthly SIP Amount (₹)", min_value=500.0, value=5000.0, step=100.0)
        else:
            amount = st.number_input("Lumpsum Amount (₹)", min_value=10000.0, value=100000.0, step=1000.0)
        
        duration_years = st.slider("Investment Duration (Years)", 1, 40, 15)
    
    with col2:
        annual_return = st.slider("Expected Annual Return (%)", 1, 25, 12)
        
        # Step-up option for SIP
        if "SIP" in investment_type:
            step_up = st.checkbox("Annual Step-up")
            step_up_rate = st.slider("Step-up Rate (%)", 0, 20, 10) if step_up else 0
        else:
            step_up = False
            step_up_rate = 0

    # Calculate returns
    if st.button("📊 Calculate Returns", type="primary", use_container_width=True):
        r = annual_return / 100
        n = duration_years

        if "SIP" in investment_type:
            if step_up:
                # Calculate SIP with step-up
                total_invested = 0
                current_sip = amount
                fv = 0
                
                for year in range(n):
                    # Calculate FV for current year SIP
                    remaining_years = n - year
                    year_fv = current_sip * 12 * (((1 + r/12) ** (remaining_years * 12) - 1) / (r/12)) * (1 + r/12)
                    fv += year_fv
                    total_invested += current_sip * 12
                    current_sip *= (1 + step_up_rate/100)  # Step up for next year
            else:
                # Regular SIP calculation
                fv = amount * (((1 + r/12) ** (n * 12) - 1) * (1 + r/12)) / (r/12)
                total_invested = amount * n * 12
        else:
            # Lumpsum calculation
            fv = amount * ((1 + r) ** n)
            total_invested = amount

        wealth_gained = fv - total_invested

        # Display results
        st.markdown("### 📈 Investment Results")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">₹{total_invested:,.0f}</div>
                <div class="metric-label">Total Investment</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">₹{fv:,.0f}</div>
                <div class="metric-label">Maturity Value</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            returns_multiple = fv / total_invested if total_invested > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">₹{wealth_gained:,.0f}</div>
                <div class="metric-label">Wealth Gained</div>
                <div class="metric-change positive">{returns_multiple:.1f}x Returns</div>
            </div>
            """, unsafe_allow_html=True)

        # Visualization
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Pie chart showing investment vs returns
            fig_pie = go.Figure(data=[go.Pie(
                labels=["Amount Invested", "Wealth Gained"],
                values=[total_invested, wealth_gained],
                hole=.6,
                marker=dict(colors=["#667eea", "#764ba2"])
            )])
            fig_pie.update_layout(
                title="Investment Composition",
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_chart2:
            # Year-wise growth chart
            growth_data = []
            current_value = 0
            current_invested = 0
            
            for year in range(1, n + 1):
                if "SIP" in investment_type:
                    current_invested += amount * 12
                    # Simplified calculation for visualization
                    current_value = current_invested * ((1 + r) ** (year/2))  # Approximate mid-year investment
                else:
                    current_invested = amount
                    current_value = amount * ((1 + r) ** year)
                
                growth_data.append({
                    'Year': year,
                    'Invested': current_invested,
                    'Value': current_value
                })
            
            growth_df = pd.DataFrame(growth_data)
            fig_line = px.line(growth_df, x='Year', y=['Invested', 'Value'],
                             title="Investment Growth Over Time")
            fig_line.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_line, use_container_width=True)

# Footer
st.markdown("""
<div class="footer">
    <h3>📊 Artha.ai - Your Financial Companion</h3>
    <p>Empowering smart financial decisions through AI and advanced analytics</p>
    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
        <small>Developed with ❤️ by Rudrika Sharma & Team | Powered by Deep Learning & Gemini AI</small>
    </div>
</div>
""", unsafe_allow_html=True)