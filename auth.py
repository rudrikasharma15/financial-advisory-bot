# auth.py
import streamlit as st
import json
import os
import random

USER_FILE = "users.json"
USER_DATA_FILE = "user_data.json"

# Load all user-specific data from file
def load_all_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    return {}

# Save all user-specific data to file
def save_all_user_data(data):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(data, f)

# Load data for a specific user
def load_user_data(username):
    all_data = load_all_user_data()
    return all_data.get(username, {})

# Save data for a specific user
def save_user_data(username, user_data):
    all_data = load_all_user_data()
    all_data[username] = user_data
    save_all_user_data(all_data)

# Load users from file
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {"admin": "1234"}  # default user

# Save users to file
def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

# Generate or get captcha question and answer
def get_captcha():
    if "captcha_a" not in st.session_state or "captcha_b" not in st.session_state:
        st.session_state.captcha_a = random.randint(1, 10)
        st.session_state.captcha_b = random.randint(1, 10)
    question = f"{st.session_state.captcha_a} + {st.session_state.captcha_b} = ?"
    answer = st.session_state.captcha_a + st.session_state.captcha_b
    return question, answer

def auth_component():
    # Initialize storage
    if "users" not in st.session_state:
        st.session_state.users = load_users()
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    # Already logged in
    if st.session_state.logged_in:
        st.success(f"✅ Logged in as {st.session_state.current_user}")
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                if key not in ["users"]: # Keep users list
                    del st.session_state[key]
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.rerun()
        return True

    # Tabs for login/signup
    tabs = st.tabs(["🔑 Login", "📝 Sign Up"])

    # --- LOGIN ---
    with tabs[0]:
        st.subheader("Login to your account")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        question, answer = get_captcha()
        captcha_input = st.text_input("Captcha: " + question, key="login_captcha")

        if st.button("Login"):
            if captcha_input.strip() != str(answer):
                st.error("⚠️ Captcha incorrect!")
                # Refresh captcha
                st.session_state.captcha_a = random.randint(1, 10)
                st.session_state.captcha_b = random.randint(1, 10)
            else:
                if username in st.session_state.users and st.session_state.users[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.current_user = username
                    st.success("🎉 Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")

    # --- SIGNUP ---
    with tabs[1]:
        st.subheader("Create a new account")
        new_user = st.text_input("Choose a username", key="signup_user")
        new_pass = st.text_input("Choose a password", type="password", key="signup_pass")

        question, answer = get_captcha()
        captcha_input = st.text_input("Captcha: " + question, key="signup_captcha")

        if st.button("Sign Up"):
            if captcha_input.strip() != str(answer):
                st.error("⚠️ Captcha incorrect!")
                # Refresh captcha
                st.session_state.captcha_a = random.randint(1, 10)
                st.session_state.captcha_b = random.randint(1, 10)
            else:
                if new_user in st.session_state.users:
                    st.error("❌ Username already exists, choose another.")
                elif new_user.strip() == "" or new_pass.strip() == "":
                    st.error("⚠️ Username and password cannot be empty.")
                else:
                    st.session_state.users[new_user] = new_pass
                    save_users(st.session_state.users)
                    st.success("🎉 Account created! You can now login.")
    return False
