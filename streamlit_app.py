import streamlit as st
import pandas as pd
from PIL import Image
import os, json, bcrypt

from logic import (
    fetch_stock_data, compute_rsi, get_general_financial_advice,
    calculate_savings_goal, get_stock_data, add_technical_indicators,
    get_mock_macro_features, prepare_model, predict_stocks, fetch_stock_news,
    get_advice, calculate_risk, get_strategy
)

# -------------------- AUTH FUNCTIONS --------------------
USER_FILE = "users.json"

def _ensure_user_file():
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)

def load_users():
    _ensure_user_file()
    try:
        with open(USER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_users(users: dict):
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f)

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

def create_user(username: str, password: str):
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = hash_password(password)
    save_users(users)
    return True, "Account created successfully."

def validate_user(username: str, password: str) -> bool:
    users = load_users()
    if username not in users:
        return False
    return check_password(password, users[username])

# -------------------- LOGIN / SIGNUP UI --------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
if "show_login" not in st.session_state:
    st.session_state["show_login"] = False  # controls whether to show login after signup

def login_view():
    st.title("🔑 Login")
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
    if submit:
        if validate_user(u, p):
            st.session_state["logged_in"] = True
            st.session_state["username"] = u
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

def signup_view():
    st.title("🆕 Sign Up")
    with st.form("signup_form"):
        u = st.text_input("Choose a username")
        p1 = st.text_input("Choose a password", type="password")
        p2 = st.text_input("Confirm password", type="password")
        submit = st.form_submit_button("Create Account")
    if submit:
        if not u or not p1 or not p2:
            st.error("Please fill in all fields.")
        elif p1 != p2:
            st.error("Passwords do not match.")
        else:
            ok, msg = create_user(u, p1)
            if ok:
                st.success(msg + " Please log in below.")
                st.session_state["show_login"] = True
                st.rerun()
            else:
                st.error(msg)

# -------------------- AUTH FLOW --------------------
if not st.session_state["logged_in"]:
    if st.session_state["show_login"]:
        login_view()
    else:
        signup_view()
    st.stop()

# -------------------- LOGGED IN VIEW --------------------
st.sidebar.success(f"Logged in as **{st.session_state['username']}**")
if st.sidebar.button("Log out"):
    st.session_state.clear()
    st.rerun()

# -------------------- ORIGINAL APP CODE BELOW --------------------
st.set_page_config(page_title="📊 Financial Advisory Bot", page_icon="💼", layout="wide")
st.markdown('<style>.css-1d391kg{padding-top:0rem;}</style>', unsafe_allow_html=True)
st.markdown('<h1 style="text-align :center; color:#2E86C1;">🤖 Financial Chatbot Assistant</h1>', unsafe_allow_html=True)

if "dashboard_run" not in st.session_state:
    st.session_state["dashboard_run"] = False

tab_options = st.sidebar.radio("🔎 Navigate", ["🏠 Home", "📊 Stock Dashboard", "💬 Finance Bot", "🎯 Goal Planner"])

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
