import sys
import os
import time
from typing import Dict, Any, Optional, Tuple, List

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page config must be the first Streamlit call
st.set_page_config(
    page_title="📊 Artha.ai",
    page_icon="💼",
    layout="wide"
)

# Import portfolio analytics modules
from data_models import Portfolio, Holding, MarketData
from risk_metrics import RiskCalculator
from portfolio_analytics import PortfolioAnalytics
from optimization import PortfolioOptimizer

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
    st.error("Error: `logic.py` not found or functions missing. Providing dummy functions.")
    _LOGIC_LOADED = False

    # Dummy implementations
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
        return pd.DataFrame({'Date': pd.to_datetime(['2023-01-01', '2023-01-02']), 'AAPL': [150, 152], 'MSFT': [250, 255]}).set_index('Date')
    def add_technical_indicators(df, symbols):
        return df
    def get_mock_macro_features(*args, **kwargs):
        return pd.DataFrame()
    def prepare_model(*args, **kwargs):
        return None
    def predict_stocks(model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size):
        dummy_predicted = [155.0, 156.0]
        dummy_actual = [153.0, 154.0]
        return {'AAPL': {'predicted': dummy_predicted, 'actual': dummy_actual}, 'MSFT': {'predicted': [258.0, 260.0], 'actual': [256.0, 257.0]}}, {}
    def fetch_stock_news(*args, **kwargs):
        return "No news available (dummy data)."
    def get_advice(*args, **kwargs):
        return "Generic advice (dummy data)."
    def calculate_risk(*args, **kwargs):
        return 5.0
    def get_strategy(*args, **kwargs):
        return "General strategy (dummy data)."

# ------------------------------------------------------------
# 1. Login / Signup first
# ------------------------------------------------------------
auth_status = auth_component()

if not auth_status:
    st.warning("Please login to access the app 🚪")
    st.stop()

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

# State
if "dashboard_run" not in st.session_state:
    st.session_state["dashboard_run"] = False
if "planner_results" not in st.session_state:
    st.session_state["planner_results"] = None

# Sidebar Navigation - UNIFIED VERSION
st.sidebar.title("📌 Navigate")
page = st.sidebar.radio("Go to", [
    "Home", 
    "Stock Dashboard", 
    "Portfolio Analytics", 
    "💬 Finance Bot", 
    "Goal Planner"
])

# ------------------------------------------------------------
# PAGES
# ------------------------------------------------------------

# Home
if page == "Home":
    st.title("🏡 Welcome to Your Financial Advisory Bot")
    st.markdown("""
        ### What you can do here:
        - 📊 Explore stock market data with interactive charts  
        - 🤖 Chat with your personal **💬 Finance Bot**  
        - 🎯 Plan and track your financial goals
        - 📈 Analyze your portfolio with advanced risk metrics
        """)

# Portfolio Analytics - YOUR FEATURE
elif page == "Portfolio Analytics":
    st.title("📈 Portfolio Analytics & Risk Management")

    # Initialize session state for portfolio
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = None
    if "market_data" not in st.session_state:
        st.session_state.market_data = None

    # Portfolio input section
    st.header("📊 Portfolio Setup")
    col1, col2 = st.columns([2, 1])

    with col1:
        portfolio_input = st.text_area(
            "Enter your portfolio holdings (one per line: SYMBOL,QUANTITY,PURCHASE_PRICE)",
            value="AAPL,100,150.00\nMSFT,50,250.00\nGOOGL,25,2800.00",
            height=150,
            help="Format: SYMBOL,QUANTITY,PURCHASE_PRICE"
        )

    with col2:
        benchmark_symbol = st.selectbox(
            "Benchmark",
            ["SPY", "QQQ", "IWM", "^GSPC"],
            index=0,
            help="Market index for comparison"
        )
        analyze_portfolio = st.button("🔍 Analyze Portfolio", type="primary")

    if analyze_portfolio:
        try:
            # Parse portfolio input
            holdings = []
            lines = [line.strip() for line in portfolio_input.split('\n') if line.strip()]

            for line in lines:
                if ',' in line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        symbol = parts[0].upper()
                        quantity = float(parts[1])
                        purchase_price = float(parts[2])
                        holdings.append(Holding(
                            symbol=symbol,
                            quantity=quantity,
                            purchase_price=purchase_price,
                            current_price=None
                        ))

            if holdings:
                portfolio = Portfolio("My Portfolio", holdings=holdings, benchmark_symbol=benchmark_symbol)

                # Fetch market data
                with st.spinner("Fetching market data..."):
                    symbols = [h.symbol for h in holdings] + [benchmark_symbol]
                    stock_data = get_stock_data(symbols)

                    if stock_data is not None and not stock_data.empty:
                        returns = stock_data.pct_change().dropna()
                        benchmark_returns = None
                        if benchmark_symbol in stock_data.columns:
                            benchmark_returns = stock_data[[benchmark_symbol]].pct_change().dropna()

                        market_data = MarketData(
                            prices=stock_data,
                            returns=returns,
                            benchmark_returns=benchmark_returns
                        )

                        # Update prices in portfolio
                        for holding in portfolio.holdings:
                            if holding.symbol in stock_data.columns:
                                holding.current_price = stock_data[holding.symbol].iloc[-1]

                        st.session_state.portfolio = portfolio
                        st.session_state.market_data = market_data
                        st.success("✅ Portfolio analyzed successfully!")
                    else:
                        st.error("❌ Failed to fetch market data. Please check symbols.")
            else:
                st.error("❌ No valid holdings found. Please check input format.")

        except Exception as e:
            st.error(f"❌ Error analyzing portfolio: {str(e)}")

    # Display results if portfolio is loaded
    if st.session_state.portfolio and st.session_state.market_data:
        portfolio = st.session_state.portfolio
        market_data = st.session_state.market_data

        # Risk Metrics Dashboard
        st.header("⚠️ Risk Metrics Dashboard")
        try:
            risk_metrics = RiskCalculator.calculate_all_risk_metrics(
                portfolio, market_data, risk_free_rate=0.02
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("VaR (95%)", f"{risk_metrics.var_95:.2%}" if risk_metrics.var_95 else "N/A")
                st.metric("CVaR (95%)", f"{risk_metrics.cvar_95:.2%}" if risk_metrics.cvar_95 else "N/A")
            with col2:
                st.metric("Sharpe Ratio", f"{risk_metrics.sharpe_ratio:.2f}" if risk_metrics.sharpe_ratio else "N/A")
                st.metric("Sortino Ratio", f"{risk_metrics.sortino_ratio:.2f}" if risk_metrics.sortino_ratio else "N/A")
            with col3:
                st.metric("Max Drawdown", f"{risk_metrics.max_drawdown:.2%}" if risk_metrics.max_drawdown else "N/A")
                st.metric("Volatility", f"{risk_metrics.volatility:.2%}" if risk_metrics.volatility else "N/A")
            with col4:
                st.metric("Beta", f"{risk_metrics.beta:.2f}" if risk_metrics.beta else "N/A")
                st.metric("Alpha", f"{risk_metrics.alpha:.2%}" if risk_metrics.alpha else "N/A")

        except Exception as e:
            st.error(f"Error calculating risk metrics: {str(e)}")

        # Portfolio Holdings
        st.header("📋 Portfolio Holdings")
        holdings_data = []
        for holding in portfolio.holdings:
            holdings_data.append({
                "Symbol": holding.symbol,
                "Quantity": holding.quantity,
                "Purchase Price": f"${holding.purchase_price:.2f}",
                "Current Price": f"${holding.current_price:.2f}" if holding.current_price else "N/A",
                "Market Value": f"${holding.market_value:,.2f}" if holding.market_value else "N/A",
                "P&L": f"${holding.unrealized_pnl:,.2f}" if holding.unrealized_pnl else "N/A",
                "P&L %": f"{holding.unrealized_pnl_percent:.2f}%" if holding.unrealized_pnl_percent else "N/A"
            })

        st.dataframe(pd.DataFrame(holdings_data), use_container_width=True)

        # Portfolio Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Value", f"${portfolio.total_value:,.2f}")
        with col2:
            st.metric("Total Invested", f"${portfolio.total_invested:,.2f}")
        with col3:
            st.metric("Total P&L", f"${portfolio.total_unrealized_pnl:,.2f}")

        # Correlation Analysis
        st.header("🔗 Correlation Analysis")
        try:
            corr_matrix = PortfolioAnalytics.calculate_correlation_matrix(market_data)
            if not corr_matrix.matrix.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix.matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, ax=ax)
                ax.set_title('Portfolio Correlation Heatmap')
                st.pyplot(fig)
                st.metric("Diversification Score", f"{corr_matrix.diversification_score:.2f}")
            else:
                st.info("Not enough data for correlation analysis")
        except Exception as e:
            st.error(f"Error in correlation analysis: {str(e)}")

        # Performance Attribution
        st.header("📊 Performance Attribution")
        try:
            attribution = PortfolioAnalytics.calculate_performance_attribution(
                portfolio, market_data, benchmark_symbol
            )
            if attribution.total_return is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Return", f"{attribution.total_return:.2%}")
                with col2:
                    st.metric("Benchmark Return", f"{attribution.benchmark_return:.2%}" if attribution.benchmark_return else "N/A")
                with col3:
                    st.metric("Excess Return", f"{attribution.excess_return:.2%}" if attribution.excess_return else "N/A")
                if attribution.stock_contributions:
                    st.subheader("Stock Contributions")
                    contrib_df = pd.DataFrame(list(attribution.stock_contributions.items()),
                                            columns=["Symbol", "Contribution"])
                    st.dataframe(contrib_df, use_container_width=True)
            else:
                st.info("Not enough data for performance attribution")
        except Exception as e:
            st.error(f"Error in performance attribution: {str(e)}")

        # Portfolio Optimization
        st.header("🎯 Portfolio Optimization")
        try:
            optimizer = PortfolioOptimizer(market_data, risk_free_rate=0.02)
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Maximize Sharpe Ratio"):
                    result = optimizer.optimize_max_sharpe()
                    if result.optimal_weights:
                        st.success("✅ Optimization completed!")
                        st.subheader("Optimal Portfolio Weights")
                        weights_df = pd.DataFrame(list(result.optimal_weights.items()),
                                                columns=["Symbol", "Weight"])
                        st.dataframe(weights_df, use_container_width=True)
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Expected Return", f"{result.expected_return:.2%}")
                        with col_b:
                            st.metric("Expected Volatility", f"{result.expected_volatility:.2%}")
                        with col_c:
                            st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
                    else:
                        st.error("❌ Optimization failed")
            
            with col2:
                if st.button("Minimum Variance"):
                    result = optimizer.optimize_minimum_variance()
                    if result.optimal_weights:
                        st.success("✅ Optimization completed!")
                        st.subheader("Minimum Variance Portfolio")
                        weights_df = pd.DataFrame(list(result.optimal_weights.items()),
                                                columns=["Symbol", "Weight"])
                        st.dataframe(weights_df, use_container_width=True)
                    else:
                        st.error("❌ Optimization failed")
        except Exception as e:
            st.error(f"Error in portfolio optimization: {str(e)}")

# Stock Dashboard
elif page == "Stock Dashboard":
    st.title("📊 Stock Dashboard")
    st.write("Upload stock data (CSV) or use demo data:")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=30),
            "Price": np.random.randint(100, 500, 30),
            "Volume": np.random.randint(1000, 10000, 30)
        })

    st.subheader("Raw Data")
    st.dataframe(df)

    st.subheader("Stock Price Trend")
    fig1 = px.line(df, x="Date", y="Price", title="Stock Price Over Time")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Trading Volume")
    fig2 = px.bar(df, x="Date", y="Volume", title="Daily Trading Volume")
    st.plotly_chart(fig2, use_container_width=True)

# Finance Bot
elif page == "💬 Finance Bot":
    st.title("🤖 Finance Bot")
    st.markdown("💬 Ask me about **investments, savings, or stock basics**:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Type your question:")
    if st.button("Send") and user_input:
        st.session_state.chat_history.append(("🧑 You", user_input))

        if "stock" in user_input.lower():
            bot_response = "📈 Stocks represent ownership in a company. Long-term investing usually reduces risk."
        elif "mutual fund" in user_input.lower():
            bot_response = "💼 A mutual fund pools money from many investors to buy diversified assets."
        elif "savings" in user_input.lower():
            bot_response = "💰 A good practice is to save at least 20% of your income."
        elif "loan" in user_input.lower():
            bot_response = "🏦 Loans should be managed carefully. Avoid EMIs exceeding 40% of your monthly income."
        else:
            bot_response = "🤔 I don't have a detailed answer for that, but always diversify your portfolio."

        st.session_state.chat_history.append(("🤖 Bot", bot_response))

    for sender, msg in st.session_state.chat_history:
        st.markdown(f"**{sender}:** {msg}")

# Goal Planner
elif page == "Goal Planner":
    st.title("🎯 Goal Planner")
    st.write("Plan and track your financial goals with progress bars.")

    if "goals" not in st.session_state:
        st.session_state.goals = []

    with st.form("goal_form", clear_on_submit=True):
        goal = st.text_input("Enter a financial goal:")
        amount = st.number_input("Target Amount (₹)", min_value=1000, step=1000)
        submitted = st.form_submit_button("Add Goal")
        if submitted and goal:
            st.session_state.goals.append({"goal": goal, "amount": amount, "progress": 0})

    if st.session_state.goals:
        for idx, g in enumerate(st.session_state.goals):
            st.subheader(f"🎯 {g['goal']}")
            progress = st.slider(
                f"Progress for {g['goal']}",
                0, g["amount"], g["progress"],
                key=f"progress_{idx}"
            )
            st.session_state.goals[idx]["progress"] = progress
            st.progress(int(progress / g["amount"] * 100))

# Footer
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
            Features: Stock Prediction | Technical Analysis | AI Chatbot | Dynamic Goal Planner | Portfolio Analytics & Risk Management
        </small>
    </div>
</div>
""", unsafe_allow_html=True)