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
    layout="wide"
)

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
from auth import auth_component


# ------------------------------------------------------------
# 1. Login / Signup first
# ------------------------------------------------------------
auth_status = auth_component()

if not auth_status:
    st.warning("Please login to access the app 🚪")
    st.stop()

            
import streamlit as st
import pandas as pd
import time
from PIL import Image
from typing import Dict, Any, Optional, Tuple, List
import plotly.express as px  # Added for new plots
import plotly.graph_objects as go  # Added for new plots
import numpy as np # Import numpy for isfinite check

# --- IMPT: Error Handling for logic.py ---
# Ensure your logic.py file is in the same directory as this app.py
# and contains ALL the functions listed below.
# If logic.py is missing or functions are not defined there,
# this try-except block will provide dummy functions to allow the app to run,
# but the core financial/stock logic will not work as intended.
try:
    from logic import (
        fetch_stock_data, compute_rsi, get_general_financial_advice,
        calculate_savings_goal, get_stock_data, add_technical_indicators,
        get_mock_macro_features, prepare_model, predict_stocks, fetch_stock_news,
        get_advice, calculate_risk, get_strategy
    )
    # Flag to indicate if real logic is loaded
    _LOGIC_LOADED = True
except ImportError:
    # st.error("Error: `logic.py` not found or functions missing. Providing dummy functions. Please ensure `logic.py` is in the same directory and contains all required functions for full functionality.")
    _LOGIC_LOADED = False
    
    # Dummy implementations for development/testing when logic.py is absent
    def fetch_stock_data(*args, **kwargs): 
        # st.warning("Dummy fetch_stock_data called."); 
        return pd.DataFrame()
    def compute_rsi(*args, **kwargs): 
        # st.warning("Dummy compute_rsi called."); 
        return pd.Series([50.0, 55.0]) # Return a series for charting
    def get_general_financial_advice(*args, **kwargs): 
        # st.warning("Dummy get_general_financial_advice called."); 
        return "This is dummy financial advice because `logic.py` could not be loaded."
    def calculate_savings_goal(target_amount, years, annual_return):
        # st.warning("Dummy calculate_savings_goal called.");
        # Simplified dummy calculation for monthly saving needed to reach target_amount from ZERO
        if years <= 0: # Avoid division by zero
            return {'monthly_saving': target_amount / 12 if target_amount > 0 else 0.0} 
        
        monthly_rate = (annual_return / 100) / 12
        num_months = years * 12
        
        if num_months == 0: 
            monthly_saving = target_amount # If duration is zero, need to save all immediately
        elif monthly_rate == 0: # No interest earned
            monthly_saving = target_amount / num_months
        else:
            # Future Value of an Ordinary Annuity (PMT formula)
            # PMT = FV * r / ((1 + r)^n - 1)
            # This is a basic approximation for dummy purposes.
            factor = ((1 + monthly_rate)**num_months - 1) / monthly_rate
            monthly_saving = target_amount / factor if factor != 0 else target_amount / num_months
        
        return {
            'target_amount': target_amount,
            'years': years,
            'annual_return': annual_return,
            'monthly_saving': monthly_saving
        }
    def get_stock_data(*args, **kwargs): 
        # st.warning("Dummy get_stock_data called."); 
        return pd.DataFrame({'Date': pd.to_datetime(['2023-01-01', '2023-01-02']), 'AAPL': [150, 152], 'MSFT': [250, 255]}).set_index('Date')
    def add_technical_indicators(df, symbols): 
        # st.warning("Dummy add_technical_indicators called."); 
        return df # Return original df
    def get_mock_macro_features(*args, **kwargs): 
        # st.warning("Dummy get_mock_macro_features called."); 
        return pd.DataFrame()
    def prepare_model(*args, **kwargs): 
        # st.warning("Dummy prepare_model called."); 
        return None # Returns None to indicate no model
    def predict_stocks(model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size):
        # st.warning("Dummy predict_stocks called.");
        # Provide some dummy prediction results to allow the dashboard to render
        dummy_predicted = [155.0, 156.0]
        dummy_actual = [153.0, 154.0]
        return {'AAPL': {'predicted': dummy_predicted, 'actual': dummy_actual}, 'MSFT': {'predicted': [258.0, 260.0], 'actual': [256.0, 257.0]}}, {} # Return dummy results
    def fetch_stock_news(*args, **kwargs): 
        # st.warning("Dummy fetch_stock_news called."); 
        return "No news available (dummy data)."
    def get_advice(*args, **kwargs): 
        # st.warning("Dummy get_advice called."); 
        return "Generic advice (dummy data)."
    def calculate_risk(*args, **kwargs): 
        # st.warning("Dummy calculate_risk called."); 
        return 5.0
    def get_strategy(*args, **kwargs): 
        # st.warning("Dummy get_strategy called."); 
        return "General strategy (dummy data)."


# Configure caching
@st.cache_data(ttl=300, show_spinner=False)
def load_stock_data(symbols: List[str]) -> Optional[pd.DataFrame]:
    """Load stock data with caching and error handling."""
    # This check ensures we use the dummy if logic.py is not loaded
    if not _LOGIC_LOADED: 
        return get_stock_data(symbols) # Calls dummy if _LOGIC_LOADED is False
    
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
    # This check ensures we use the dummy if logic.py is not loaded
    if not _LOGIC_LOADED: 
        # Call dummy predict_stocks, which expects some arguments but can handle None for dummies
        # It's crucial that dummy predict_stocks returns a dictionary with 'predicted' and 'actual' keys
        dummy_results, _ = predict_stocks(None,None,None,None,None,None,None,None) 
        return stock_data, dummy_results
    
    if stock_data is None or stock_data.empty:
        return None, {}
    
    stock_data_copy = stock_data.copy() # Work on a copy to avoid SettingWithCopyWarning
    stock_data_processed = add_technical_indicators(stock_data_copy, symbols)
    macro = get_mock_macro_features(stock_data_processed.index)
    model_result = prepare_model(symbols, stock_data_processed, macro)
    
    if not model_result:
        # st.warning("Model preparation failed. Cannot predict stocks.")
        return stock_data_processed, {} # Return processed data, but empty results if model fails
    
    model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size = model_result
    results, evaluation = predict_stocks(model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size)
    
    return stock_data_processed, results

# State
if "dashboard_run" not in st.session_state:
    st.session_state["dashboard_run"] = False
if "planner_results" not in st.session_state:
    st.session_state["planner_results"] = None # To store the calculated plan for persistence

# Sidebar Navigation
# Sidebar Navigation
tab_options = st.sidebar.radio(
    "🔎 Navigate",
    [
        "🏠 Home",
        "📊 Stock Dashboard",
        "💬 Finance Bot",
        "🎯 Goal Planner",
        "💼 Portfolio Tracker",
        "💸 SIP and Lumpsum Calculator"
    ]
)

# Highlight Active Page
if tab_options == "🏠 Home":
    st.sidebar.success("✅ You are on Home Page")
elif tab_options == "📊 Stock Dashboard":
    st.sidebar.success("📊 Viewing Stock Dashboard")
elif tab_options == "💬 Finance Bot":
    st.sidebar.success("🤖 Chatting with Finance Bot")
elif tab_options == "🎯 Goal Planner":
    st.sidebar.success("🎯 Planning Your Goals")
elif tab_options == "💼 Portfolio Tracker":
    st.sidebar.success("📂 Tracking Your Portfolio")
elif tab_options == "💸 SIP and Lumpsum Calculator":
    st.sidebar.success("💸 Calculating SIP & Lumpsum")



# Home Tab
if tab_options == "🏠 Home":
    st.markdown("## 🏠 Welcome")
    st.markdown("""
    <div style='font-size:18px;'>
        Welcome to the <b>Artha.ai</b>. This tool helps you:
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
            st.rerun()

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
            if stock_data is not None and not stock_data.empty:
                stock_data, results = process_stock_data(stock_data, symbols)
                if results:
                    st.session_state.stock_data = stock_data
                    st.session_state.results = results
                    st.session_state.symbols = symbols
                else:
                    st.warning("⚠️ Prediction failed for the entered symbols. Displaying raw data if available.")
                    st.session_state.stock_data = stock_data # Keep raw data if processing fails
                    st.session_state.results = {}
            else:
                st.error("⚠️ Unable to fetch stock data or data is empty. Please try again or check symbols.")
                st.session_state.stock_data = None
                st.session_state.results = {}

    # --- Display results if available ---
    if st.session_state.stock_data is not None and st.session_state.results:
        tabs = st.tabs([f"📊 {symbol}" for symbol in st.session_state.symbols])

        for idx, symbol in enumerate(st.session_state.symbols):
            with tabs[idx]:
                results = st.session_state.results
                stock_data = st.session_state.stock_data

                # Check for prediction data availability for the specific symbol
                if symbol not in results or not isinstance(results[symbol], dict) or \
                   'predicted' not in results[symbol] or 'actual' not in results[symbol] or \
                   len(results[symbol]['predicted']) == 0 or len(results[symbol]['actual']) == 0:
                    st.error(f"⚠️ No sufficient prediction data available for {symbol}. Raw data may be displayed below.")
                    # Attempt to show raw data if predictions failed for this symbol
                    if symbol in stock_data.columns and not stock_data.empty:
                        st.write(f"Displaying raw price data for {symbol}:")
                        st.line_chart(stock_data[[symbol]])
                    continue # Skip detailed prediction display for this symbol


                predicted = results[symbol]['predicted']
                actual = results[symbol]['actual']

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("### Price Prediction")
                    # Corrected: Remove direct boolean evaluation of 'actual' and 'predicted'
                    if len(actual) > 0 and len(predicted) > 0:
                        change_percentage = ((predicted[-1] - actual[-1]) / actual[-1]) * 100
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("📈 Current Price", f"${actual[-1]:.2f}")
                        with m2:
                            st.metric("🔮 Predicted", f"${predicted[-1]:.2f}", f"{change_percentage:+.2f}%")

                        # Ensure index matches the length of predicted/actual for chart_data
                        # Use a common index that aligns with the results, assuming predicted/actual are last parts of stock_data
                        chart_index_start = max(0, len(stock_data.index) - len(predicted))
                        
                        chart_data = pd.DataFrame({
                            "Predicted": predicted,
                            "Actual": actual
                        }, index=stock_data.index[chart_index_start:chart_index_start + len(predicted)]) # Ensure index slice is correct
                        st.line_chart(chart_data, use_container_width=True)
                    else:
                        st.info("Not enough valid data for price prediction chart for this symbol.")

                with col2:
                    st.markdown("### 📊 Technical Analysis")
                    # Ensure symbol column exists in stock_data before computing RSI
                    if symbol in stock_data.columns and not stock_data.empty:
                        rsi = compute_rsi(stock_data[symbol]) # Assuming compute_rsi returns a Series
                        rsi_value = rsi.dropna().iloc[-1] if not rsi.empty else 50
                    else:
                        rsi = pd.Series([]) # Empty Series if no data
                        rsi_value = 50 # Default RSI
                        st.warning(f"No valid stock data for {symbol} to compute RSI.")

                    
                    # Ensure predicted and actual have values for trend calculation
                    trend = "➡️ Neutral"
                    if len(predicted) > 0 and len(actual) > 0: # Corrected: Use len() check
                        trend = "📈 Uptrend" if predicted[-1] > actual[-1] * 1.01 else "📉 Downtrend" if predicted[-1] < actual[-1] * 0.99 else "➡️ Neutral"
                    
                    rsi_status = "🔥 Overbought" if rsi_value > 70 else "❄️ Oversold" if rsi_value < 30 else "⚖️ Neutral"
                    risk = calculate_risk(symbol, stock_data, results)
                    strategy = get_strategy(get_advice(predicted), risk)

                    st.metric("📊 RSI", f"{rsi_value:.1f}", rsi_status.split()[0]) # Adjusted to show only the status text
                    st.metric("📈 Trend", trend.split()[0]) # Adjusted to show only the status text
                    st.metric("⚠️ Risk", f"{risk:.1f}/10")
                    st.info(strategy)

                with st.expander(f"📊 RSI History - {symbol}"):
                    if not rsi.empty:
                        st.line_chart(rsi)
                    else:
                        st.info("RSI data not available for this symbol.")


                with st.expander(f"🗞️ Latest News - {symbol}"):
                    news_content = fetch_stock_news(symbol)
                    if news_content and news_content != "No news available (dummy data).": # Check against dummy text too
                        st.markdown(news_content)
                    else:
                        st.info("No news available for this symbol.")

                # Ensure chart_data is defined and not empty before allowing download
                # This check ensures 'chart_data' exists from the price prediction section
                if 'chart_data' in locals() and isinstance(chart_data, pd.DataFrame) and not chart_data.empty:
                    # Align RSI data for download by taking the last 'len(predicted)' values
                    rsi_for_download = rsi.iloc[-len(predicted):] if not rsi.empty and len(rsi) >= len(predicted) else [None]*len(predicted)
                    
                    st.download_button(
                        label="📥 Download Data",
                        data=pd.DataFrame({
                            "Date": stock_data.index[chart_index_start:chart_index_start + len(predicted)], # Use correct slice
                            "Predicted": predicted,
                            "Actual": actual,
                            "RSI": rsi_for_download
                        }).to_csv(index=False),
                        file_name=f"{symbol}_prediction.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No data to download yet for this symbol (predictions might be missing).")

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
                                st.session_state[advice_key] = f"Error getting advice: {e}" # Store error message
                        else:
                            st.warning("Please enter a question.")
                            st.session_state[advice_key] = "" # Clear advice if query is empty

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
                st.session_state["advice"] = f"Error getting advice: {e}" # Store error message
        else:
            st.warning("Please enter a query.")
            st.session_state["advice"] = "" # Clear advice if query is empty
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
    current_balance = float(current_savings) # Ensure it's float for calculations
    
    for year_num in range(1, years + 1):
        start_balance_year = current_balance
        contributions_this_year = 0
        interest_this_year = 0
        
        # Simulate month by month for compounding interest and contributions
        for month in range(12):
            monthly_interest = current_balance * monthly_rate
            current_balance += monthly_interest # Interest added
            interest_this_year += monthly_interest
            
            current_balance += monthly_saving # Contribution added
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


# Goal Planner Tab (Dynamic Version)
if tab_options == "🎯 Goal Planner":
    st.markdown("## 🎯 Financial Goal Planner: Your Journey to Financial Freedom")
    st.markdown("Set your financial aspirations and let's calculate a dynamic plan to achieve them! See how your savings grow and explore different scenarios.")

    st.subheader("📝 Define Your Goal")

    col_goal_name, col_current_savings = st.columns(2)
    with col_goal_name:
        goal_name = st.text_input(
            "✨ What are you saving for?", 
            value=st.session_state.get('planner_goal_name', ""), 
            placeholder="e.g., Dream Vacation, Retirement, New Car",
            key="planner_goal_name_input"
        )
    with col_current_savings:
        current_savings = st.number_input(
            "💸 Current Savings (₹)", 
            min_value=0.0, 
            value=None if st.session_state.get('planner_current_savings') is None else st.session_state['planner_current_savings'], 
            format="%.2f",
            placeholder="Example 500.00",
            key="planner_current_savings_input",
            help="The amount you currently have saved towards this goal."
        )

    col_target_amount, col_years, col_return = st.columns(3)

    with col_target_amount:
        target_amount = st.number_input(
            "🎯 Target Amount (₹)", 
            min_value=1000.0, 
            value=None if st.session_state.get('planner_target_amount') is None else st.session_state['planner_target_amount'], 
            placeholder="Example 10000.00",
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

    # --- Inflation Toggle and Slider ---
    inflation_enabled = st.checkbox("Adjust for Inflation?", value=st.session_state.get("planner_inflation_enabled", False), key="planner_inflation_enabled")
    inflation_rate = 0
    if inflation_enabled:
        inflation_rate = st.slider(
            "Inflation Rate (%)",
            min_value=0, max_value=15, value=5,
            key="planner_inflation_rate",
            help="Expected average annual inflation rate."
        )
    
    # Store inputs in session state for persistence
    st.session_state['planner_goal_name'] = goal_name
    st.session_state['planner_current_savings'] = current_savings
    st.session_state['planner_target_amount'] = target_amount
    st.session_state['planner_years'] = years
    st.session_state['planner_annual_return'] = annual_return

    run_calculation = st.button("🚀 Calculate My Plan", type="primary")

    if run_calculation:
        # --- Calculate WITHOUT inflation ---
        real_annual_return_no_inflation = annual_return
        base_monthly_saving_no_inflation = calculate_savings_goal(target_amount, years, real_annual_return_no_inflation).get('monthly_saving', 0.0)
        growth_df_no_inflation = _simulate_goal_growth(current_savings, base_monthly_saving_no_inflation, years, real_annual_return_no_inflation)
        final_balance_no_inflation = growth_df_no_inflation['Ending Balance'].iloc[-1] if not growth_df_no_inflation.empty else current_savings

        # --- Calculate WITH inflation (if enabled) ---
        real_annual_return_with_inflation = annual_return - inflation_rate if inflation_enabled else annual_return
        base_monthly_saving_with_inflation = calculate_savings_goal(target_amount, years, real_annual_return_with_inflation).get('monthly_saving', 0.0)
        growth_df_with_inflation = _simulate_goal_growth(current_savings, base_monthly_saving_with_inflation, years, real_annual_return_with_inflation)
        final_balance_with_inflation = growth_df_with_inflation['Ending Balance'].iloc[-1] if not growth_df_with_inflation.empty else current_savings

        # Save results for display
        st.session_state["planner_results"] = {
            'goal_name': goal_name,
            'current_savings': current_savings,
            'target_amount': target_amount,
            'years': years,
            'annual_return': annual_return,
            'inflation_enabled': inflation_enabled,
            'inflation_rate': inflation_rate,
            'real_annual_return_no_inflation': real_annual_return_no_inflation,
            'real_annual_return_with_inflation': real_annual_return_with_inflation,
            'base_monthly_saving_no_inflation': base_monthly_saving_no_inflation,
            'base_monthly_saving_with_inflation': base_monthly_saving_with_inflation,
            'final_balance_no_inflation': final_balance_no_inflation,
            'final_balance_with_inflation': final_balance_with_inflation,
            'growth_df_no_inflation': growth_df_no_inflation,
            'growth_df_with_inflation': growth_df_with_inflation
        }

    # --- Display Results ---
    if st.session_state["planner_results"] is not None:
        results = st.session_state["planner_results"]
        st.markdown("### 📊 Investment Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Without Inflation")
            st.write(f"**Monthly Saving Needed:** ₹{results['base_monthly_saving_no_inflation']:.2f}")
            st.write(f"**Final Balance:** ₹{results['final_balance_no_inflation']:.2f}")
            st.write(f"**Annual Return Used:** {results['real_annual_return_no_inflation']}%")

            df1 = results['growth_df_no_inflation'].copy().round(2)
            st.table(df1)

        with col2:
            st.markdown("#### With Inflation")
            st.write(f"**Monthly Saving Needed:** ₹{results['base_monthly_saving_with_inflation']:.2f}")
            st.write(f"**Final Balance:** ₹{results['final_balance_with_inflation']:.2f}")
            st.write(f"**Annual Return Used:** {results['real_annual_return_with_inflation']}%")
            st.write(f"**Inflation Rate:** {results['inflation_rate']}%")

            df2 = results['growth_df_with_inflation'].copy().round(2)
            st.table(df2)

        # --- PDF Export ---
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        import io

        # Add these two imports
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        import io

        # Register the font (make sure DejaVuSans.ttf is in the same folder or give full path)
        pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))

        # Define styles using your registered font
        styles = getSampleStyleSheet()
        styles['Normal'].fontName = 'DejaVuSans'
        styles['Heading1'].fontName = 'DejaVuSans'
        styles['Heading2'].fontName = 'DejaVuSans'
        styles['Heading3'].fontName = 'DejaVuSans'


        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        # styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Investment Analysis", styles['Heading1']))

        # Without Inflation
        elements.append(Paragraph("Without Inflation", styles['Heading2']))
        elements.append(Paragraph(f"Monthly Saving Needed: ₹{results['base_monthly_saving_no_inflation']:.2f}", styles['Normal']))
        elements.append(Paragraph(f"Final Balance: ₹{results['final_balance_no_inflation']:.2f}", styles['Normal']))
        elements.append(Paragraph(f"Annual Return Used: {results['real_annual_return_no_inflation']}%", styles['Normal']))
        elements.append(Spacer(1, 12))
        table1 = Table([df1.columns.tolist()] + df1.values.tolist())
        table1.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey)]))
        elements.append(table1)

        elements.append(Spacer(1, 24))

        # With Inflation
        elements.append(Paragraph("With Inflation", styles['Heading2']))
        elements.append(Paragraph(f"Monthly Saving Needed: ₹{results['base_monthly_saving_with_inflation']:.2f}", styles['Normal']))
        elements.append(Paragraph(f"Final Balance: ₹{results['final_balance_with_inflation']:.2f}", styles['Normal']))
        elements.append(Paragraph(f"Annual Return Used: {results['real_annual_return_with_inflation']}%", styles['Normal']))
        elements.append(Paragraph(f"Inflation Rate: {results['inflation_rate']}%", styles['Normal']))
        elements.append(Spacer(1, 12))
        table2 = Table([df2.columns.tolist()] + df2.values.tolist())
        table2.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey)]))
        elements.append(table2)

        doc.build(elements)
        pdf_data = buffer.getvalue()
        buffer.close()

        st.download_button(
            label="📥 Download Investment Analysis (PDF)",
            data=pdf_data,
            file_name="investment_analysis.pdf",
            mime="application/pdf"
        )


# Portfolio Tracking Tab

if tab_options == "💼 Portfolio Tracker":
    st.markdown("## 💼 Portfolio Tracker")

    # Initialize portfolio if not present
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = []

    # Form to add new holdings
    with st.form("add_holding"):
        st.subheader("➕ Add a Stock to Your Portfolio")
        symbol = st.text_input("Symbol (e.g., AAPL, MSFT, HDB)")
        units = st.number_input("Units Owned", min_value=0.0, format="%.2f")
        buy_price = st.number_input("Buy Price per Unit", min_value=0.0)
        buy_date = st.date_input("Buy Date")
        add_clicked = st.form_submit_button("Add to Portfolio")
        if add_clicked and symbol.strip() and units > 0 and buy_price > 0:
            st.session_state.portfolio.append({
                "symbol": symbol.strip().upper(),
                "units": units,
                "buy_price": buy_price,
                "buy_date": str(buy_date)
            })
            st.success(f"{symbol.strip().upper()} added to portfolio!")

    # Display portfolio if any holdings
    if st.session_state.portfolio:
        df = pd.DataFrame(st.session_state.portfolio)
        st.write("Your portfolio holdings:", df)

        # Get symbols from portfolio
        symbols = list(df["symbol"].unique())
        
        try:
            # Fetch current prices
            with st.spinner("Fetching current prices..."):
                prices_df = get_stock_data(symbols)
            
            if prices_df is None or prices_df.empty:
                st.warning("Could not fetch current price data. Please try again later.")
                latest_prices = {sym: 0 for sym in symbols}
            else:
                # Get the latest price for each symbol
                latest_prices = {}
                for symbol in symbols:
                    if symbol in prices_df.columns:
                        latest_prices[symbol] = prices_df[symbol].iloc[-1]
                    else:
                        st.warning(f"Could not fetch price for {symbol}")
                        latest_prices[symbol] = 0

        except Exception as e:
            st.error(f"Error fetching prices: {e}")
            latest_prices = {sym: 0 for sym in symbols}

        # Compute current metrics
        df["Current Price"] = df["symbol"].map(latest_prices)
        df["Current Value"] = df["Current Price"] * df["units"]
        df["Investment Cost"] = df["buy_price"] * df["units"]
        df["Abs Gain/Loss"] = df["Current Value"] - df["Investment Cost"]
        df["% Gain/Loss"] = ((df["Current Price"] - df["buy_price"]) / df["buy_price"]) * 100

        total_value = df["Current Value"].sum()
        total_cost = df["Investment Cost"].sum()
        total_gain_loss = total_value - total_cost
        
        if total_value > 0:
            df["Allocation %"] = (df["Current Value"] / total_value * 100).fillna(0)
        else:
            df["Allocation %"] = 0

        # Display portfolio summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Investment", f"₹{total_cost:,.2f}")
        with col2:
            st.metric("Current Value", f"₹{total_value:,.2f}")
        with col3:
            st.metric("Total Gain/Loss", f"₹{total_gain_loss:,.2f}", 
                     f"{((total_value/total_cost)-1)*100:.2f}%" if total_cost > 0 else "0%")

        # Display detailed portfolio
        st.dataframe(
            df.style.format({
                "buy_price": "₹{:.2f}",
                "Current Price": "₹{:.2f}",
                "Current Value": "₹{:.2f}",
                "Investment Cost": "₹{:.2f}",
                "Abs Gain/Loss": "₹{:.2f}",
                "% Gain/Loss": "{:.2f}%",
                "Allocation %": "{:.2f}%",
            })
        )

        # Allocation Pie Chart
        if total_value > 0:
            fig_alloc = px.pie(df, values="Current Value", names="symbol", 
                              title="Portfolio Allocation by Symbol")
            st.plotly_chart(fig_alloc, use_container_width=True)

        # Download option for portfolio CSV
        csv_data = df.to_csv(index=False)
        st.download_button("📥 Download Portfolio Data", csv_data, "portfolio.csv", "text/csv")

        if st.button("🗑️ Clear Portfolio"):
            st.session_state.portfolio = []
            st.rerun()

    else:
        st.info("➕ Add holdings above to start tracking your portfolio.")
if tab_options=="💸 SIP and Lumpsum Calculator":
    st.markdown("## 📈 SIP & Lumpsum Investment Calculator")
    st.markdown("Compare and visualize your investment outcomes using SIP and Lumpsum options with projected returns.")

    st.subheader("💡 Investment Inputs")

    col1, col2 = st.columns(2)
    with col1:
        investment_type = st.radio("Choose Investment Type:", ["SIP", "Lumpsum"], horizontal=True)

    with col2:
        annual_return = st.slider("Expected Annual Return (%)", min_value=1, max_value=20, value=12)
    
    if investment_type == "SIP":
        col_sip1, col_sip2 = st.columns(2)
        with col_sip1:
            monthly_investment = st.number_input("💰 Monthly Investment (₹)", min_value=500.0, value=5000.0, step=100.0)
        with col_sip2:
            duration_years = st.slider("⏳ Investment Duration (Years)", 1, 40, 10)
    else:
        col_lump1, col_lump2 = st.columns(2)
        with col_lump1:
            lumpsum_amount = st.number_input("💰 Lumpsum Amount (₹)", min_value=500.0, value=100000.0, step=500.0)
        with col_lump2:
            duration_years = st.slider("⏳ Investment Duration (Years)", 1, 40, 10)
    
    # Calculate returns
    calculate_btn = st.button("📊 Calculate Returns")
    if calculate_btn:
        r = annual_return / 100
        n = duration_years

        if investment_type == "SIP":
            fv = monthly_investment * (((1 + r / 12) ** (n * 12) - 1) * (1 + r / 12)) / (r / 12)
            total_invested = monthly_investment * n * 12
        else:
            fv = lumpsum_amount * ((1 + r) ** n)
            total_invested = lumpsum_amount

        interest_earned = fv - total_invested

        # Display metrics
        st.subheader("📈 Investment Summary")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Invested", f"₹{total_invested:,.2f}")
        with col_b:
            st.metric("Total Returns", f"₹{fv:,.2f}")
        with col_c:
            st.metric("Interest Earned", f"₹{interest_earned:,.2f}")

        # Donut chart
        fig = go.Figure(data=[go.Pie(
            labels=["Invested", "Interest Earned"],
            values=[total_invested, interest_earned],
            hole=.5,
            marker=dict(colors=["#10b981", "#6366f1"])
        )])

        fig.update_layout(
            title="📊 Investment Composition",
            showlegend=True,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)


if tab_options=="💸 SIP and Lumpsum Calculator":
    st.markdown("## 📈 SIP & Lumpsum Investment Calculator")
    st.markdown("Compare and visualize your investment outcomes using SIP and Lumpsum options with projected returns.")

    st.subheader("💡 Investment Inputs")

    col1, col2 = st.columns(2)
    with col1:
        investment_type = st.radio("Choose Investment Type:", ["SIP", "Lumpsum"], horizontal=True)

    with col2:
        annual_return = st.slider("Expected Annual Return (%)", min_value=1, max_value=20, value=12)
    
    if investment_type == "SIP":
        col_sip1, col_sip2 = st.columns(2)
        with col_sip1:
            monthly_investment = st.number_input("💰 Monthly Investment (₹)", min_value=500.0, value=5000.0, step=100.0)
        with col_sip2:
            duration_years = st.slider("⏳ Investment Duration (Years)", 1, 40, 10)
    else:
        col_lump1, col_lump2 = st.columns(2)
        with col_lump1:
            lumpsum_amount = st.number_input("💰 Lumpsum Amount (₹)", min_value=500.0, value=100000.0, step=500.0)
        with col_lump2:
            duration_years = st.slider("⏳ Investment Duration (Years)", 1, 40, 10)
    
    # Calculate returns
    calculate_btn = st.button("📊 Calculate Returns")
    if calculate_btn:
        r = annual_return / 100
        n = duration_years

        if investment_type == "SIP":
            fv = monthly_investment * (((1 + r / 12) ** (n * 12) - 1) * (1 + r / 12)) / (r / 12)
            total_invested = monthly_investment * n * 12
        else:
            fv = lumpsum_amount * ((1 + r) ** n)
            total_invested = lumpsum_amount

        interest_earned = fv - total_invested

        # Display metrics
        st.subheader("📈 Investment Summary")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Invested", f"₹{total_invested:,.2f}")
        with col_b:
            st.metric("Total Returns", f"₹{fv:,.2f}")
        with col_c:
            st.metric("Interest Earned", f"₹{interest_earned:,.2f}")

        # Donut chart
        fig = go.Figure(data=[go.Pie(
            labels=["Invested", "Interest Earned"],
            values=[total_invested, interest_earned],
            hole=.5,
            marker=dict(colors=["#10b981", "#6366f1"])
        )])

        fig.update_layout(
            title="📊 Investment Composition",
            showlegend=True,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)



# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.02); border-radius: 10px; margin-top: 30px;">
    <p style="color: #888; margin: 0;">
        📊 <strong>Financial Advisory Bot</strong> | Empowering Your Financial Decisions<br>
        Powered by Deep Learning & Gemini AI<br>
        Developed by Rudrika Sharma & Team
    </p>
    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
        <small style="color: #666;">
            Features: Stock Prediction | Technical Analysis | AI Chatbot | Dynamic Goal Planner
        </small>
    </div>
</div>
""", unsafe_allow_html=True)


import streamlit as st
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
















