
import streamlit as st
import pandas as pd
from PIL import Image
from logic import (
    fetch_stock_data, compute_rsi, get_general_financial_advice,
    calculate_savings_goal, get_stock_data, add_technical_indicators,
    get_mock_macro_features, prepare_model, predict_stocks, fetch_stock_news,
    get_advice, calculate_risk, get_strategy
)

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
    symbols_input = st.text_input("📥 Enter stock symbols (comma-separated)", "AAPL, MSFT", help="E.g., AAPL, GOOGL, MSFT")
    start_btn = st.button("🔍 Analyze")

    if start_btn:
        st.session_state["dashboard_run"] = True

    if st.session_state["dashboard_run"]:
        symbols = [s.strip().upper() for s in symbols_input.split(",")]
        stock_data = get_stock_data(symbols)

        if stock_data is not None:
            stock_data = add_technical_indicators(stock_data, symbols)
            macro = get_mock_macro_features(stock_data.index)
            model_result = prepare_model(symbols, stock_data, macro)

            if model_result:
                model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size = model_result
                results, evaluation = predict_stocks(model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size)
                st.session_state["results"] = results
                st.session_state["symbols"] = symbols
                st.session_state["stock_data"] = stock_data

                for symbol in symbols:
                    st.markdown(f"### 📊 {symbol}")
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        predicted = results[symbol]['predicted']
                        actual = results[symbol]['actual']
                        change_percentage = ((predicted[-1] - actual[-1]) / actual[-1]) * 100
                        st.metric("📉 Predicted Price", f"₹{predicted[-1]:.2f}", f"{change_percentage:+.2f}%")
                        st.metric("🎯 Actual Price", f"₹{actual[-1]:.2f}")
                        st.line_chart(pd.DataFrame({"Predicted": predicted, "Actual": actual}))

                    with col2:
                        st.markdown("#### 🧮 Technical Analysis")
                        rsi = compute_rsi(stock_data[symbol])
                        rsi_value = rsi.dropna().iloc[-1]
                        trend = "📈 Uptrend" if predicted[-1] > actual[-1] else "📉 Downtrend"
                        rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                        risk = calculate_risk(symbol, stock_data, results)
                        strategy = get_strategy(get_advice(predicted), risk)
                        st.markdown(f"- **Trend**: {trend}\n- **RSI**: {rsi_value:.2f} ({rsi_status})\n- **Risk Score**: {risk:.2f}\n- **Strategy**: {strategy}")

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

                    query_key = f"query_{symbol}"
                    advice_key = f"advice_{symbol}"
                    query = st.text_input(f"🤖 Ask Gemini about {symbol}:", key=query_key)

                    if st.button(f"Get Advice for {symbol}"):
                        if query:
                            try:
                                advice = get_general_financial_advice(query, [symbol], stock_data, results)
                                st.session_state[advice_key] = advice
                                st.rerun()
                            except Exception as e:
                                st.error(f"Gemini error: {e}")
                        else:
                            st.warning("Please enter a question.")

                    if advice_key in st.session_state:
                        st.markdown(f"🧠 Gemini's Advice:\n\n{st.session_state[advice_key]}")

        else:
            st.error("⚠️ Unable to fetch stock data. Please try again.")

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
