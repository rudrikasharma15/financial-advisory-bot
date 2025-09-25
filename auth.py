# auth.py
import streamlit as st
import json
import os
import random

USER_FILE = "users.json"

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
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.rerun()
        return True

    # Hero/Background
    st.markdown(
        """
        <style>
          .auth-hero {
            background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(16,185,129,0.12));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 16px;
          }
          .auth-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 18px 20px;
            backdrop-filter: blur(8px);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="auth-hero">
          <h3 style="margin:0;">Welcome to Artha.ai</h3>
          <p style="margin:6px 0 0 0; opacity:.9;">Securely login or create your account to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs for login/signup
    tabs = st.tabs(["🔑 Login", "📝 Sign Up"])

    # --- LOGIN ---
    with tabs[0]:
        st.subheader("Login to your account")
        with st.container():
            st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")

        question, answer = get_captcha()
        captcha_input = st.text_input("Captcha: " + question, key="login_captcha")

        if st.button("Login", key="login_btn"):
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
            st.markdown("</div>", unsafe_allow_html=True)

    # --- SIGNUP ---
    with tabs[1]:
        st.subheader("Create a new account")
        st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
        new_user = st.text_input("Choose a username", key="signup_user")
        new_pass = st.text_input("Choose a password", type="password", key="signup_pass")

        question, answer = get_captcha()
        captcha_input = st.text_input("Captcha: " + question, key="signup_captcha")

        if st.button("Sign Up", key="signup_btn"):
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
        st.markdown("</div>", unsafe_allow_html=True)
    return False
