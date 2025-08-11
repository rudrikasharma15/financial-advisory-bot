
import streamlit as st
import pandas as pd
import time
from PIL import Image
from typing import Dict, Any, Optional, Tuple, List
from logic import (
    fetch_stock_data, compute_rsi, get_general_financial_advice,
    calculate_savings_goal, get_stock_data, add_technical_indicators,
    get_mock_macro_features, prepare_model, predict_stocks, fetch_stock_news,
    get_advice, calculate_risk, get_strategy
)

# Configure caching
@st.cache_data(ttl=300, show_spinner=False)
def load_stock_data(symbols: List[str]) -> Optional[pd.DataFrame]:
    """Load stock data with caching and error handling."""
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
    if stock_data is None or stock_data.empty:
        return None, {}
    
    stock_data = add_technical_indicators(stock_data.copy(), symbols)
    macro = get_mock_macro_features(stock_data.index)
    model_result = prepare_model(symbols, stock_data, macro)
    
    if not model_result:
        return None, {}
    
    model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size = model_result
    results, evaluation = predict_stocks(model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size)
    
    return stock_data, results

# Config and Branding
st.set_page_config(page_title="📊 Financial Advisory Bot", page_icon="💼", layout="wide")
st.markdown('<style>.css-1d391kg{padding-top:0rem;}</style>', unsafe_allow_html=True)
st.markdown('<h1 style="text-align :center; color:#2E86C1;">🤖 Financial Chatbot Assistant</h1>', unsafe_allow_html=True)

# State
if "dashboard_run" not in st.session_state:
    st.session_state["dashboard_run"] = False

# Sidebar Navigation
tab_options = st.sidebar.radio("🔎 Navigate", ["🏠 Home", "📊 Stock Dashboard", "💬 Finance Bot", "🎯 Goal Planner"])

# Home Tab
if tab_options == "🏠 Home":
    st.markdown("## 🏠 Welcome")
    st.markdown("""
    <div style='font-size:18px;'>
        Welcome to the <b>Financial Advisory Bot</b>. This tool helps you:
        <ul>
            <li>💹 Predict future stock prices using deep learning</li>
            <li>📈 Analyze RSI, trends, and risks</li>
            <li>🧠 Get personalized advice via Gemini AI</li>
            <li>🎯 Plan your savings based on your financial goals</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Stock Dashboard Tab
elif tab_options == "📊 Stock Dashboard":
    st.markdown("## 📈 Stock Analysis & Predictions")

    # --- Initialize session state variables ---
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

    # --- Stock symbols input ---
    symbols_input = st.text_input(
        "📥 Enter stock symbols (comma-separated)",
        value=st.session_state.symbols_input,
        help="E.g., AAPL, GOOGL, MSFT"
    )

    # Update stored input if changed
    if symbols_input != st.session_state.symbols_input:
        st.session_state.symbols_input = symbols_input

    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_btn = st.button("🔍 Analyze", use_container_width=True)
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            load_stock_data.clear()
            process_stock_data.clear()
            st.session_state.stock_data = None
            st.session_state.results = {}
            st.experimental_rerun()

    # --- Fetch & process data only if Analyze is clicked or symbols changed ---
    symbols_changed = st.session_state.symbols_input != st.session_state.last_symbols
    if analyze_btn or (symbols_changed and st.session_state.stock_data is None):
        st.session_state.last_symbols = st.session_state.symbols_input
        symbols = [s.strip().upper() for s in st.session_state.symbols_input.split(",") if s.strip()]
        if not symbols:
            st.warning("⚠️ Please enter at least one stock symbol")
            st.stop()

        with st.spinner("📊 Analyzing market data..."):
            stock_data = load_stock_data(symbols)
            if stock_data is not None:
                stock_data, results = process_stock_data(stock_data, symbols)
                if results:
                    st.session_state.stock_data = stock_data
                    st.session_state.results = results
                    st.session_state.symbols = symbols
            else:
                st.error("⚠️ Unable to fetch stock data. Please try again.")

    # --- Display results if available ---
    if st.session_state.stock_data is not None and st.session_state.results:
        tabs = st.tabs([f"📊 {symbol}" for symbol in st.session_state.symbols])

        for idx, symbol in enumerate(st.session_state.symbols):
            with tabs[idx]:
                results = st.session_state.results
                stock_data = st.session_state.stock_data

                if symbol not in results:
                    st.error(f"⚠️ No prediction data available for {symbol}")
                    continue

                predicted = results[symbol]['predicted']
                actual = results[symbol]['actual']

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("### Price Prediction")
                    change_percentage = ((predicted[-1] - actual[-1]) / actual[-1]) * 100
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("📈 Current Price", f"${actual[-1]:.2f}")
                    with m2:
                        st.metric("🔮 Predicted", f"${predicted[-1]:.2f}", f"{change_percentage:+.2f}%")

                    chart_data = pd.DataFrame({
                        "Predicted": predicted,
                        "Actual": actual
                    }, index=stock_data.index[-len(predicted):])
                    st.line_chart(chart_data, use_container_width=True)

                with col2:
                    st.markdown("### 📊 Technical Analysis")
                    rsi = compute_rsi(stock_data[symbol])
                    rsi_value = rsi.dropna().iloc[-1] if not rsi.empty else 50
                    trend = "📈 Uptrend" if predicted[-1] > actual[-1] * 1.01 else "📉 Downtrend" if predicted[-1] < actual[-1] * 0.99 else "➡️ Neutral"
                    rsi_status = "🔥 Overbought" if rsi_value > 70 else "❄️ Oversold" if rsi_value < 30 else "⚖️ Neutral"
                    risk = calculate_risk(symbol, stock_data, results)
                    strategy = get_strategy(get_advice(predicted), risk)

                    st.metric("📊 RSI", f"{rsi_value:.1f}", rsi_status.split()[-1])
                    st.metric("📈 Trend", trend.split()[-1])
                    st.metric("⚠️ Risk", f"{risk:.1f}/10")
                    st.info(strategy)

                with st.expander(f"📊 RSI History - {symbol}"):
                    st.line_chart(rsi)

                with st.expander(f"🗞️ Latest News - {symbol}"):
                    st.markdown(fetch_stock_news(symbol))

                st.download_button(
                    label="📥 Download Data",
                    data=pd.DataFrame({
                        "Date": stock_data.index,
                        "Predicted": predicted,
                        "Actual": actual,
                        "RSI": rsi
                    }).to_csv(index=False),
                    file_name=f"{symbol}_prediction.csv",
                    mime="text/csv"
                )

                # --- Gemini Q&A ---
                query_key = f"query_{symbol}"
                advice_key = f"advice_{symbol}"

                if query_key not in st.session_state:
                    st.session_state[query_key] = ""

                with st.form(key=f"form_{symbol}"):
                    query = st.text_input(
                        f"🤖 Ask Gemini about {symbol}:",
                        value=st.session_state.get(query_key, "")
                    )
                    submitted = st.form_submit_button(f"Get Advice for {symbol}")

                    if submitted:
                        if query.strip():
                            st.session_state[query_key] = query
                            try:
                                with st.spinner("🤔 Analyzing your question..."):
                                    advice = get_general_financial_advice(
                                        query,
                                        [symbol],
                                        st.session_state.stock_data,
                                        st.session_state.results
                                    )
                                    st.session_state[advice_key] = advice
                            except Exception as e:
                                st.error(f"Gemini error: {e}")
                        else:
                            st.warning("Please enter a question.")

                if advice_key in st.session_state and st.session_state[advice_key]:
                    st.markdown("### 💡 Gemini's Advice")
                    st.markdown(st.session_state[advice_key])

                if st.button(f"Clear Advice for {symbol}"):
                    st.session_state.pop(advice_key, None)
                    st.session_state.pop(query_key, None)
                    st.experimental_rerun()


# Finance Bot Tab
elif tab_options == "💬 Finance Bot":
    st.subheader("💬 Ask Gemini Finance Bot")
    query = st.text_input("🔍 Ask a financial question", key="general_query")
    if st.button("Get Advice"):
        if query:
            try:
                advice = get_general_financial_advice(query)
                st.session_state["advice"] = advice
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a query.")
    if "advice" in st.session_state:
        st.markdown(f"🧠 Gemini says:\n\n{st.session_state['advice']}")

# Goal Planner Tab
elif tab_options == "🎯 Goal Planner":
    st.markdown("## 🎯 Financial Goal Planner")
    target_amount = st.number_input("🎯 Target Amount (₹)", min_value=1000.0, value=100000.0)
    years = st.slider("📆 Duration (years)", 1, 40, 10)
    annual_return = st.slider("📈 Expected Annual Return (%)", 0, 15, 7)
    if st.button("Calculate Plan"):
        result = calculate_savings_goal(target_amount, years, annual_return)
        st.success(
            f"To reach ₹{result['target_amount']} in {result['years']} years at {result['annual_return']}% return, "
            f"save ₹{result['monthly_saving']:.2f} monthly."
        )
