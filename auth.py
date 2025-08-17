# auth.py
import streamlit as st
import json
import os

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

    # Tabs for login/signup
    tabs = st.tabs(["🔑 Login", "📝 Sign Up"])

    # --- LOGIN ---
    with tabs[0]:
        st.subheader("Login to your account")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
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

        if st.button("Sign Up"):
            if new_user in st.session_state.users:
                st.error("❌ Username already exists, choose another.")
            elif new_user.strip() == "" or new_pass.strip() == "":
                st.error("⚠️ Username and password cannot be empty.")
            else:
                st.session_state.users[new_user] = new_pass
                save_users(st.session_state.users)
                st.success("🎉 Account created! You can now login.")

    return False
