import streamlit as st
import pandas as pd
import time
from PIL import Image
from typing import Dict, Any, Optional, Tuple, List
import plotly.express as px  # Added for new plots
import plotly.graph_objects as go  # Added for new plots
import numpy as np  # Import numpy for isfinite check
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
    st.error("Error: `logic.py` not found or functions missing. Providing dummy functions. Please ensure `logic.py` is in the same directory and contains all required functions for full functionality.")
    _LOGIC_LOADED = False
    
    # Dummy implementations for development/testing when logic.py is absent
    def fetch_stock_data(*args, **kwargs): return pd.DataFrame()
    def compute_rsi(*args, **kwargs): return pd.Series([50.0, 55.0])
    def get_general_financial_advice(*args, **kwargs): return "This is dummy financial advice because `logic.py` could not be loaded."
    def calculate_savings_goal(target_amount, years, annual_return):
        if years <= 0: return {'monthly_saving': target_amount / 12 if target_amount > 0 else 0.0}
        monthly_rate = (annual_return / 100) / 12
        num_months = years * 12
        factor = ((1 + monthly_rate)**num_months - 1) / monthly_rate
        monthly_saving = target_amount / factor if factor != 0 else target_amount / num_months
        return {'target_amount': target_amount, 'years': years, 'annual_return': annual_return, 'monthly_saving': monthly_saving}
    def get_stock_data(*args, **kwargs): return pd.DataFrame({'Date': pd.to_datetime(['2023-01-01', '2023-01-02']), 'AAPL': [150, 152], 'MSFT': [250, 255]}).set_index('Date')
    def add_technical_indicators(df, symbols): return df
    def get_mock_macro_features(*args, **kwargs): return pd.DataFrame()
    def prepare_model(*args, **kwargs): return None
    def predict_stocks(model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size):
        dummy_predicted = [155.0, 156.0]
        dummy_actual = [153.0, 154.0]
        return {'AAPL': {'predicted': dummy_predicted, 'actual': dummy_actual}, 'MSFT': {'predicted': [258.0, 260.0], 'actual': [256.0, 257.0]}}, {}
    def fetch_stock_news(*args, **kwargs): return "No news available (dummy data)."
    def get_advice(*args, **kwargs): return "Generic advice (dummy data)."
    def calculate_risk(*args, **kwargs): return 5.0
    def get_strategy(*args, **kwargs): return "General strategy (dummy data)."

# Configure caching
@st.cache_data(ttl=300, show_spinner=False)
def load_stock_data(symbols: List[str]) -> Optional[pd.DataFrame]:
    if not _LOGIC_LOADED: return get_stock_data(symbols)
    if not symbols: return None
    try:
        with st.spinner("📡 Fetching market data..."):
            return get_stock_data(symbols)
    except Exception as e:
        st.error(f"⚠️ Error fetching stock data: {str(e)}")
        return None

@st.cache_data(ttl=300, show_spinner=False)
def process_stock_data(stock_data: pd.DataFrame, symbols: List[str]) -> Tuple[Any, Dict]:
    if not _LOGIC_LOADED: 
        dummy_results, _ = predict_stocks(None, None, None, None, None, None, None, None)
        return stock_data, dummy_results
    if stock_data is None or stock_data.empty: return None, {}
    stock_data_copy = stock_data.copy()
    stock_data_processed = add_technical_indicators(stock_data_copy, symbols)
    macro = get_mock_macro_features(stock_data_processed.index)
    model_result = prepare_model(symbols, stock_data_processed, macro)
    if not model_result: return stock_data_processed, {}
    model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size = model_result
    results, evaluation = predict_stocks(model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size)
    return stock_data_processed, results

# Config and Branding
st.set_page_config(page_title="📊 Financial Advisory Bot", page_icon="💼", layout="wide")
st.markdown('<style>.css-1d391kg{padding-top:0rem;}</style>', unsafe_allow_html=True)
st.markdown('<h1 style="text-align :center; color:#2E86C1;">🤖 Financial Chatbot Assistant</h1>', unsafe_allow_html=True)

# State
if "dashboard_run" not in st.session_state:
    st.session_state["dashboard_run"] = False
if "planner_results" not in st.session_state:
    st.session_state["planner_results"] = None

# Sidebar Navigation
tab_options = st.sidebar.radio("🔎 Navigate", ["🏠 Home", "📊 Stock Dashboard", "💬 Finance Bot", "🎯 Goal Planner", "⚠️ Risk Assessment"])

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

    symbols_input = st.text_input(
        "📥 Enter stock symbols (comma-separated)",
        value=st.session_state.symbols_input,
        help="E.g., AAPL, GOOGL, MSFT"
    )
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
            st.rerun()

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
                else:
                    st.warning("⚠️ Prediction failed for the entered symbols. Displaying raw data if available.")
                    st.session_state.stock_data = stock_data
                    st.session_state.results = {}
            else:
                st.error("⚠️ Unable to fetch stock data or data is empty. Please try again or check symbols.")
                st.session_state.stock_data = None
                st.session_state.results = {}

    if st.session_state.stock_data is not None and st.session_state.results:
        tabs = st.tabs([f"📊 {symbol}" for symbol in st.session_state.symbols])
        for idx, symbol in enumerate(st.session_state.symbols):
            with tabs[idx]:
                results = st.session_state.results
                stock_data = st.session_state.stock_data
                if symbol not in results or not isinstance(results[symbol], dict) or \
                   'predicted' not in results[symbol] or 'actual' not in results[symbol] or \
                   len(results[symbol]['predicted']) == 0 or len(results[symbol]['actual']) == 0:
                    st.error(f"⚠️ No sufficient prediction data available for {symbol}. Raw data may be displayed below.")
                    if symbol in stock_data.columns and not stock_data.empty:
                        st.write(f"Displaying raw price data for {symbol}:")
                        st.line_chart(stock_data[[symbol]])
                    continue

                predicted = results[symbol]['predicted']
                actual = results[symbol]['actual']
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("### Price Prediction")
                    if len(actual) > 0 and len(predicted) > 0:
                        change_percentage = ((predicted[-1] - actual[-1]) / actual[-1]) * 100
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("📈 Current Price", f"${actual[-1]:.2f}")
                        with m2:
                            st.metric("🔮 Predicted", f"${predicted[-1]:.2f}", f"{change_percentage:+.2f}%")
                        chart_index_start = max(0, len(stock_data.index) - len(predicted))
                        chart_data = pd.DataFrame({
                            "Predicted": predicted,
                            "Actual": actual
                        }, index=stock_data.index[chart_index_start:chart_index_start + len(predicted)])
                        st.line_chart(chart_data, use_container_width=True)
                    else:
                        st.info("Not enough valid data for price prediction chart for this symbol.")
                with col2:
                    st.markdown("### 📊 Technical Analysis")
                    if symbol in stock_data.columns and not stock_data.empty:
                        rsi = compute_rsi(stock_data[symbol])
                        rsi_value = rsi.dropna().iloc[-1] if not rsi.empty else 50
                    else:
                        rsi = pd.Series([])
                        rsi_value = 50
                        st.warning(f"No valid stock data for {symbol} to compute RSI.")
                    trend = "➡️ Neutral"
                    if len(predicted) > 0 and len(actual) > 0:
                        trend = "📈 Uptrend" if predicted[-1] > actual[-1] * 1.01 else "📉 Downtrend" if predicted[-1] < actual[-1] * 0.99 else "➡️ Neutral"
                    rsi_status = "🔥 Overbought" if rsi_value > 70 else "❄️ Oversold" if rsi_value < 30 else "⚖️ Neutral"
                    risk = calculate_risk(symbol, stock_data, results)
                    strategy = get_strategy(get_advice(predicted), risk)
                    st.metric("📊 RSI", f"{rsi_value:.1f}", rsi_status.split()[0])
                    st.metric("📈 Trend", trend.split()[0])
                    st.metric("⚠️ Risk", f"{risk:.1f}/10")
                    st.info(strategy)
                with st.expander(f"📊 RSI History - {symbol}"):
                    if not rsi.empty:
                        st.line_chart(rsi)
                    else:
                        st.info("RSI data not available for this symbol.")
                with st.expander(f"🗞️ Latest News - {symbol}"):
                    news_content = fetch_stock_news(symbol)
                    if news_content and news_content != "No news available (dummy data).":
                        st.markdown(news_content)
                    else:
                        st.info("No news available for this symbol.")
                if 'chart_data' in locals() and isinstance(chart_data, pd.DataFrame) and not chart_data.empty:
                    rsi_for_download = rsi.iloc[-len(predicted):] if not rsi.empty and len(rsi) >= len(predicted) else [None]*len(predicted)
                    st.download_button(
                        label="📥 Download Data",
                        data=pd.DataFrame({
                            "Date": stock_data.index[chart_index_start:chart_index_start + len(predicted)],
                            "Predicted": predicted,
                            "Actual": actual,
                            "RSI": rsi_for_download
                        }).to_csv(index=False),
                        file_name=f"{symbol}_prediction.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No data to download yet for this symbol (predictions might be missing).")
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
                                st.session_state[advice_key] = f"Error getting advice: {e}"
                        else:
                            st.warning("Please enter a question.")
                            st.session_state[advice_key] = ""
                if advice_key in st.session_state and st.session_state[advice_key]:
                    st.markdown("### 💡 Gemini's Advice")
                    st.markdown(st.session_state[advice_key])
                if st.button(f"Clear Advice for {symbol}", key=f"clear_advice_{symbol}"):
                    st.session_state.pop(advice_key, None)
                    st.session_state.pop(query_key, None)
                    st.rerun()

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
                st.session_state["advice"] = f"Error getting advice: {e}"
        else:
            st.warning("Please enter a query.")
            st.session_state["advice"] = ""
    if "advice" in st.session_state:
        st.markdown(f"🧠 Gemini says:\n\n{st.session_state['advice']}")

# --- Helper function for Goal Planner growth simulation ---
def _simulate_goal_growth(
    current_savings: float,
    monthly_saving: float,
    years: int,
    annual_return: float
) -> pd.DataFrame:
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
    st.markdown("## 🎯 Financial Goal Planner: Your Journey to Financial Freedom")
    st.markdown("Set your financial aspirations and let's calculate a dynamic plan to achieve them! See how your savings grow and explore different scenarios.")
    st.subheader("📝 Define Your Goal")
    col_goal_name, col_current_savings = st.columns(2)
    with col_goal_name:
        goal_name = st.text_input(
            "✨ What are you saving for?",
            value=st.session_state.get('planner_goal_name', "My Dream Goal"),
            placeholder="e.g., Dream Vacation, Retirement, New Car",
            key="planner_goal_name_input"
        )
    with col_current_savings:
        current_savings = st.number_input(
            "💸 Current Savings (₹)",
            min_value=0.0,
            value=st.session_state.get('planner_current_savings', 0.0),
            format="%.2f",
            key="planner_current_savings_input",
            help="The amount you currently have saved towards this goal."
        )
    col_target_amount, col_years, col_return = st.columns(3)
    with col_target_amount:
        target_amount = st.number_input(
            "🎯 Target Amount (₹)",
            min_value=1000.0,
            value=st.session_state.get('planner_target_amount', 1000000.0),
            format="%.2f",
            key="planner_target_amount_input",
            help="The total amount you want to save."
        )
    with col_years:
        years = st.slider(
            "📆 Duration (years)",
            1, 50,
            st.session_state.get('planner_years', 10),
            key="planner_years_input",
            help="The number of years you have to reach your goal."
        )
    with col_return:
        annual_return = st.slider(
            "📈 Expected Annual Return (%)",
            0, 20,
            st.session_state.get('planner_annual_return', 7),
            key="planner_annual_return_input",
            help="The average annual interest rate you expect on your savings/investments."
        )
    st.session_state['planner_goal_name'] = goal_name
    st.session_state['planner_current_savings'] = current_savings
    st.session_state['planner_target_amount'] = target_amount
    st.session_state['planner_years'] = years
    st.session_state['planner_annual_return'] = annual_return
    run_calculation = st.button("🚀 Calculate My Plan", type="primary")
    if 'show_what_if' not in st.session_state:
        st.session_state.show_what_if = False
    if st.session_state["planner_results"] is not None or st.session_state.show_what_if:
        st.session_state.show_what_if = st.checkbox(
            "🔮 Explore 'What-If' Scenarios?",
            value=st.session_state.show_what_if,
            help="Compare your base plan with an adjusted monthly saving scenario."
        )
    if run_calculation:
        if target_amount <= current_savings:
            st.success(f"🎉 Great news! Your current savings of **₹{current_savings:,.2f}** already meet or exceed your target of **₹{target_amount:,.2f}** for **{goal_name}**!")
            st.info("No additional monthly savings required if your goal is just to reach this amount. Consider setting a higher target!")
            st.session_state["planner_results"] = None
            st.session_state.show_what_if = False
            st.stop()
        if years <= 0:
            st.error("Duration must be at least 1 year.")
            st.session_state["planner_results"] = None
            st.session_state.show_what_if = False
            st.stop()
        try:
            with st.spinner("Calculating your financial journey..."):
                base_monthly_saving_calc = calculate_savings_goal(target_amount, years, annual_return)
                base_monthly_saving = base_monthly_saving_calc.get('monthly_saving', 0.0)
                base_monthly_saving = max(0.0, base_monthly_saving)
                growth_df_base = _simulate_goal_growth(current_savings, base_monthly_saving, years, annual_return)
                final_balance_base = growth_df_base['Ending Balance'].iloc[-1] if not growth_df_base.empty else current_savings
                if final_balance_base < target_amount * 0.99999:
                    st.warning(f"⚠️ **{goal_name}** might be challenging! With ₹{base_monthly_saving:,.2f} monthly, you'll reach approximately **₹{final_balance_base:,.2f}** in {years} years. Consider increasing monthly savings, extending duration, or adjusting your target.")
                    base_is_achievable = False
                else:
                    st.success(f"✅ Your plan for **{goal_name}** looks achievable! With **₹{base_monthly_saving:,.2f}** saved monthly, you'll reach **₹{final_balance_base:,.2f}** or more.")
                    base_is_achievable = True
                st.session_state["planner_results"] = {
                    'goal_name': goal_name,
                    'current_savings': current_savings,
                    'target_amount': target_amount,
                    'years': years,
                    'annual_return': annual_return,
                    'base_monthly_saving': base_monthly_saving,
                    'growth_df_base': growth_df_base,
                    'final_balance_base': final_balance_base,
                    'base_is_achievable': base_is_achievable
                }
        except Exception as e:
            st.error(f"⚠️ An error occurred during calculation: {e}. Please check your inputs. Ensure your `logic.py`'s `calculate_savings_goal` function is robust.")
            st.session_state["planner_results"] = None
            st.session_state.show_what_if = False
    if st.session_state["planner_results"] is not None:
        results = st.session_state["planner_results"]
        goal_name = results['goal_name']
        target_amount = results['target_amount']
        years = results['years']
        annual_return = results['annual_return']
        current_savings = results['current_savings']
        base_monthly_saving = results['base_monthly_saving']
        growth_df_base = results['growth_df_base']
        final_balance_base = results['final_balance_base']
        base_is_achievable = results['base_is_achievable']
        st.subheader(f"📊 Your Plan for: {goal_name}")
        col_summary_1, col_summary_2, col_summary_3 = st.columns(3)
        with col_summary_1:
            st.metric("Required Monthly Saving", f"₹{base_monthly_saving:,.2f}")
        with col_summary_2:
            st.metric("Projected Final Balance", f"₹{final_balance_base:,.2f}")
        with col_summary_3:
            total_contributed_base = current_savings + (base_monthly_saving * years * 12)
            total_interest_base = final_balance_base - total_contributed_base
            st.metric("Total Interest Earned", f"₹{max(0, total_interest_base):,.2f}")
        fig = px.line(
            growth_df_base,
            x='Year',
            y='Ending Balance',
            title=f'Projected Savings Growth for {goal_name}',
            labels={'Ending Balance': 'Balance (₹)'},
            line_shape="spline"
        )
        fig.add_hline(y=target_amount, line_dash="dot", line_color="red",
                      annotation_text=f"Target: ₹{target_amount:,.0f}",
                      annotation_position="bottom right",
                      annotation_font_color="red")
        fig.update_traces(name='Base Plan', line=dict(color='#00d4aa', width=3), mode='lines+markers', marker=dict(size=6))
        fig.update_layout(
            hovermode="x unified",
            xaxis_title="Year",
            yaxis_title="Balance (₹)",
            template="plotly_dark",
            font=dict(color='white'),
            height=450,
            showlegend=True
        )
        growth_dfs_to_display = {'Base Plan': growth_df_base}
        if st.session_state.show_what_if:
            st.subheader("🔮 'What-If' Scenario: Adjusting Your Monthly Contribution")
            min_slider_value = -base_monthly_saving if base_monthly_saving > 0 else 0.0
            monthly_saving_adjustment = st.slider(
                "Adjust Monthly Saving (₹)",
                min_value=float(min_slider_value),
                max_value=100000.0,
                value=0.0,
                step=100.0,
                format="₹%.2f",
                key="what_if_monthly_adjustment",
                help="Adjust your monthly saving to see its impact. A negative value means saving less than the base plan."
            )
            what_if_monthly_saving = max(0.0, base_monthly_saving + monthly_saving_adjustment)
            growth_df_what_if = _simulate_goal_growth(current_savings, what_if_monthly_saving, years, annual_return)
            final_balance_what_if = growth_df_what_if['Ending Balance'].iloc[-1] if not growth_df_what_if.empty else current_savings
            fig.add_trace(go.Scatter(
                x=growth_df_what_if['Year'],
                y=growth_df_what_if['Ending Balance'],
                mode='lines+markers',
                name='What-If Plan',
                line=dict(color='#a855f7', width=3),
                marker=dict(size=6)
            ))
            fig.update_layout(title=f'Projected Savings Growth for {goal_name} (Base vs. What-If)')
            growth_dfs_to_display['What-If Plan'] = growth_df_what_if
            st.write(f"With an adjusted monthly saving of **₹{what_if_monthly_saving:,.2f}**:")
            col_what_if_sum_1, col_what_if_sum_2 = st.columns(2)
            with col_what_if_sum_1:
                delta_balance = final_balance_what_if - final_balance_base
                st.metric("New Projected Final Balance", f"₹{final_balance_what_if:,.2f}", delta=delta_balance)
                if np.isfinite(delta_balance) and delta_balance != 0:
                    delta_balance_str = f"{abs(delta_balance):,.2f}"
                    sign = "+" if delta_balance > 0 else "-"
                    st.markdown(f"<small style='text-align: right; margin-top: -10px;'>({sign}₹{delta_balance_str} vs Base)</small>", unsafe_allow_html=True)
                elif not np.isfinite(delta_balance):
                    st.markdown(f"<small style='text-align: right; margin-top: -10px;'>(Difference vs Base: Not a finite number)</small>", unsafe_allow_html=True)
            with col_what_if_sum_2:
                what_if_total_contributed = current_savings + (what_if_monthly_saving * years * 12)
                what_if_total_interest = final_balance_what_if - what_if_total_contributed
                delta_interest = max(0, what_if_total_interest) - max(0, total_interest_base)
                st.metric("New Total Interest Earned", f"₹{max(0, what_if_total_interest):,.2f}", delta=delta_interest)
                if np.isfinite(delta_interest) and delta_interest != 0:
                    delta_interest_str = f"{abs(delta_interest):,.2f}"
                    sign = "+" if delta_interest > 0 else "-"
                    st.markdown(f"<small style='text-align: right; margin-top: -10px;'>({sign}₹{delta_interest_str} vs Base)</small>", unsafe_allow_html=True)
                elif not np.isfinite(delta_interest):
                    st.markdown(f"<small style='text-align: right; margin-top: -10px;'>(Difference vs Base: Not a finite number)</small>", unsafe_allow_html=True)
            if final_balance_what_if >= target_amount * 0.99999 and final_balance_base < target_amount * 0.99999:
                st.info(f"✨ **Fantastic!** By saving **₹{abs(monthly_saving_adjustment):,.2f}** {'more' if monthly_saving_adjustment > 0 else 'less'} per month, your 'What-If' plan now **meets** your target of ₹{target_amount:,.2f}!")
            elif final_balance_what_if < target_amount * 0.99999 and final_balance_base >= target_amount * 0.99999:
                st.warning(f"😔 Oh no! By saving **₹{abs(monthly_saving_adjustment):,.2f}** {'less' if monthly_saving_adjustment < 0 else 'more'} per month, your 'What-If' plan now **falls short** of your target of ₹{target_amount:,.2f}. Careful with those adjustments!")
            elif monthly_saving_adjustment > 0:
                st.info(f"🚀 Great progress! Saving an extra ₹{monthly_saving_adjustment:,.2f} monthly boosts your final balance to ₹{final_balance_what_if:,.2f}.")
            elif monthly_saving_adjustment < 0 and what_if_monthly_saving > 0:
                st.warning(f"📉 Be careful! Saving ₹{abs(monthly_saving_adjustment):,.2f} less monthly reduces your final balance to ₹{final_balance_what_if:,.2f}.")
            elif what_if_monthly_saving == 0 and monthly_saving_adjustment < 0:
                st.warning(f"🛑 You've stopped monthly contributions in the 'What-If' scenario. Your goal might be delayed or not reached.")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 📅 Detailed Yearly Breakdown")
        combined_df_data = []
        for year_num in range(1, years + 1):
            row_base = growth_df_base[growth_df_base['Year'] == year_num].iloc[0]
            base_ending = row_base['Ending Balance']
            base_cont = row_base['Annual Contributions']
            base_interest = row_base['Interest Earned']
            row_data = {
                'Year': year_num,
                'Base Plan - Contributions': f"₹{base_cont:,.2f}",
                'Base Plan - Interest': f"₹{base_interest:,.2f}",
                'Base Plan - Ending Balance': f"₹{base_ending:,.2f}"
            }
            if st.session_state.show_what_if:
                row_what_if = growth_df_what_if[growth_df_what_if['Year'] == year_num].iloc[0]
                what_if_ending = row_what_if['Ending Balance']
                what_if_cont = row_what_if['Annual Contributions']
                what_if_interest = row_what_if['Interest Earned']
                row_data.update({
                    'What-If Plan - Contributions': f"₹{what_if_cont:,.2f}",
                    'What-If Plan - Interest': f"₹{what_if_interest:,.2f}",
                    'What-If Plan - Ending Balance': f"₹{what_if_ending:,.2f}"
                })
            combined_df_data.append(row_data)
        st.dataframe(pd.DataFrame(combined_df_data), use_container_width=True)
    if st.session_state["planner_results"] is not None or (
        st.session_state.get('planner_goal_name') != "My Dream Goal" or
        st.session_state.get('planner_current_savings') != 0.0 or
        st.session_state.get('planner_target_amount') != 1000000.0 or
        st.session_state.get('planner_years') != 10 or
        st.session_state.get('planner_annual_return') != 7 or
        st.session_state.show_what_if is True
    ):
        if st.button("🔄 Reset Goal Planner", key="reset_goal_plan_button"):
            st.session_state["planner_results"] = None
            st.session_state['planner_goal_name'] = "My Dream Goal"
            st.session_state['planner_current_savings'] = 0.0
            st.session_state['planner_target_amount'] = 1000000.0
            st.session_state['planner_years'] = 10
            st.session_state['planner_annual_return'] = 7
            st.session_state.show_what_if = False
            st.rerun()

# Risk Assessment Tab
elif tab_options == "⚠️ Risk Assessment":
    st.title("⚠️ Risk Assessment Module")
    st.write("Analyze the risk distribution of your portfolio based on stock volatility and correlation.")

    # User input for stocks and weights
    stocks = st.multiselect("Select Stocks", ["AAPL", "GOOGL", "MSFT", "TSLA"], default=["AAPL", "GOOGL"])
    weights = []
    for stock in stocks:
        weight = st.number_input(f"Weight for {stock} (0-1, sum to 1)", min_value=0.0, max_value=1.0, value=1.0/len(stocks))
        weights.append(weight)
    if not stocks:
        st.warning("⚠️ Please select at least one stock.")
    elif not weights:
        st.error("⚠️ Weights are not properly set. Please enter weights for all selected stocks.")
    elif not np.isclose(sum(weights), 1.0, atol=0.01):
        st.error("Weights must sum to approximately 1.0")
    else:
        with st.spinner("📊 Calculating risk data..."):
            # Calculate risk scores
            risk_scores = {}
            returns_data = {}
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data
            for stock in stocks:
                ticker = yf.Ticker(stock)
                df = ticker.history(start=start_date, end=end_date)['Close']
                if df.empty:
                    st.error(f"⚠️ No data available for {stock}. Skipping this stock.")
                    continue
                returns = df.pct_change().dropna()
                volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
                risk_scores[stock] = volatility * weights[stocks.index(stock)]
                returns_data[stock] = returns

            # Correlation matrix
            correlation_matrix = pd.DataFrame(returns_data).corr() if returns_data else pd.DataFrame()

            # Display risk scores
            st.subheader("Risk Scores")
            if risk_scores:
                for stock, score in risk_scores.items():
                    st.write(f"{stock}: {score:.4f}")
            else:
                st.warning("⚠️ No risk scores calculated due to data issues.")

            # Heatmap
            st.subheader("Risk Distribution Heatmap")
            if not correlation_matrix.empty:
                plt.figure(figsize=(8, 6))
                sns.heatmap(correlation_matrix, annot=True, cmap="YlOrRd", vmin=-1, vmax=1)
                st.pyplot(plt)
            else:
                st.warning("⚠️ No correlation data available for heatmap.")

            # Automatic Correlation Analysis (New Feature)
            st.subheader("Correlation Analysis")
            if not correlation_matrix.empty:
                # Calculate average correlation (upper triangle, excluding diagonal)
                avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
                if np.isnan(avg_correlation):
                    st.info("No significant correlation data to analyze. Your portfolio might have only one stock.")
                elif avg_correlation > 0.7:
                    st.warning(f"High average correlation ({avg_correlation:.2f}). This means your stocks tend to move together, increasing risk. Consider adding more diverse stocks to spread the risk.")
                elif avg_correlation > 0.4:
                    st.info(f"Moderate average correlation ({avg_correlation:.2f}). Your portfolio has some overlap, but it's okay. To lower risk further, add stocks that move independently.")
                else:
                    st.success(f"Low average correlation ({avg_correlation:.2f}). Great job! Your portfolio is well-diversified, which helps protect against market swings.")
            else:
                st.warning("⚠️ No correlation data to analyze.")

            # Risk level summary
            total_risk = sum(risk_scores.values()) if risk_scores else 0.0
            risk_level = "Low" if total_risk < 0.1 else "Moderate" if total_risk < 0.2 else "High"
            st.subheader("Risk Summary")
            st.write(f"Overall Risk Level: **{risk_level}** (Total Risk Score: {total_risk:.4f})")


# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.02); border-radius: 10px; margin-top: 30px;">
    <p style="color: #888; margin: 0;">
        📊 <strong>Financial Advisory Bot</strong> | Empowering Your Financial Decisions<br>
        Powered by Deep Learning & Gemini AI<br>
        Developed by Your Name/Organization
    </p>
    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
        <small style="color: #666;">
            Features: Stock Prediction | Technical Analysis | AI Chatbot | Dynamic Goal Planner | Risk Assessment
        </small>
    </div>
</div>
""", unsafe_allow_html=True)