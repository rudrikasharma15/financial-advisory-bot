import sys
import os
import time
from typing import Dict, Any, Optional, Tuple, List

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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


# ------------------------------------------------------------
# 2. Sidebar navigation
# ------------------------------------------------------------
st.sidebar.title("📌 Navigate")
page = st.sidebar.radio("Go to", ["Home", "Stock Dashboard", "Finance Bot", "Goal Planner"])


# ------------------------------------------------------------
# 3. Pages
# ------------------------------------------------------------

# Home
if page == "Home":
    st.title("🏡 Welcome to Your Financial Advisory Bot")
    st.markdown(
        """
        ### What you can do here:
        - 📊 Explore stock market data with interactive charts  
        - 🤖 Chat with your personal **Finance Bot**  
        - 🎯 Plan and track your financial goals  
        """
    )


# Stock Dashboard
elif page == "Stock Dashboard":
    st.title("📊 Stock Dashboard")

    st.write("Upload stock data (CSV) or use demo data:")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Demo dataset
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=30),
            "Price": np.random.randint(100, 500, 30),
            "Volume": np.random.randint(1000, 10000, 30)
        })

    st.subheader("Raw Data")
    st.dataframe(df)

    # Price Trend
    st.subheader("Stock Price Trend")
    fig1 = px.line(df, x="Date", y="Price", title="Stock Price Over Time")
    st.plotly_chart(fig1, use_container_width=True)

    # Volume Bar Chart
    st.subheader("Trading Volume")
    fig2 = px.bar(df, x="Date", y="Volume", title="Daily Trading Volume")
    st.plotly_chart(fig2, use_container_width=True)


# Finance Bot
elif page == "Finance Bot":
    st.title("🤖 Finance Bot")

    st.markdown("💬 Ask me about **investments, savings, or stock basics**:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Type your question:")
    if st.button("Send") and user_input:
        st.session_state.chat_history.append(("🧑 You", user_input))

        # Simple built-in finance Q&A
        if "stock" in user_input.lower():
            bot_response = "📈 Stocks represent ownership in a company. Long-term investing usually reduces risk."
        elif "mutual fund" in user_input.lower():
            bot_response = "💼 A mutual fund pools money from many investors to buy diversified assets."
        elif "savings" in user_input.lower():
            bot_response = "💰 A good practice is to save at least 20% of your income."
        elif "loan" in user_input.lower():
            bot_response = "🏦 Loans should be managed carefully. Avoid EMIs exceeding 40% of your monthly income."
        else:
            bot_response = "🤔 I don’t have a detailed answer for that, but always diversify your portfolio."

        st.session_state.chat_history.append(("🤖 Bot", bot_response))

    # Display chat
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
