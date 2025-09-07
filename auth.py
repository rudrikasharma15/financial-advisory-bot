import streamlit as st
import json
import os
import random
import secrets
import datetime
import hashlib

USER_FILE = "users.json"
RESET_TOKENS_FILE = "reset_tokens.json"

# Load users from file
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {"admin": {"password": "1234", "email": "admin@example.com"}}

# Save users to file
def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=2)

# Load reset tokens from file
def load_reset_tokens():
    if os.path.exists(RESET_TOKENS_FILE):
        with open(RESET_TOKENS_FILE, "r") as f:
            return json.load(f)
    return {}

# Save reset tokens to file
def save_reset_tokens(tokens):
    with open(RESET_TOKENS_FILE, "w") as f:
        json.dump(tokens, f, indent=2)

# Generate or get captcha question and answer
def get_captcha():
    if "captcha_a" not in st.session_state or "captcha_b" not in st.session_state:
        st.session_state.captcha_a = random.randint(1, 10)
        st.session_state.captcha_b = random.randint(1, 10)
    question = f"{st.session_state.captcha_a} + {st.session_state.captcha_b} = ?"
    answer = st.session_state.captcha_a + st.session_state.captcha_b
    return question, answer

def refresh_captcha():
    """Refresh captcha values"""
    st.session_state.captcha_a = random.randint(1, 10)
    st.session_state.captcha_b = random.randint(1, 10)

def hash_password(password):
    """Hash password for secure storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_reset_token():
    """Generate a secure reset token"""
    return secrets.token_urlsafe(32)

def create_reset_token(email_or_username):
    """Create reset token and return it for in-app use"""
    users = load_users()
    reset_tokens = load_reset_tokens()
    
    # Find user by email or username
    found_user = None
    user_email = None
    
    for username, user_data in users.items():
        if isinstance(user_data, dict):
            if user_data.get('email') == email_or_username or username == email_or_username:
                found_user = username
                user_email = user_data.get('email')
                break
        else:
            # Handle old format (backward compatibility)
            if username == email_or_username:
                found_user = username
                user_email = f"{username}@example.com"
                break
    
    if found_user:
        # Generate reset token
        reset_token = generate_reset_token()
        expiry_time = datetime.datetime.now() + datetime.timedelta(hours=1)
        
        # Store reset token
        reset_tokens[reset_token] = {
            "username": found_user,
            "email": user_email,
            "expiry": expiry_time.isoformat(),
            "used": False
        }
        save_reset_tokens(reset_tokens)
        
        return True, reset_token, found_user
    else:
        return False, None, None

def verify_reset_token(token):
    """Verify if reset token is valid and not expired"""
    reset_tokens = load_reset_tokens()
    
    if token not in reset_tokens:
        return False, "Invalid or expired reset link"
    
    token_data = reset_tokens[token]
    
    if token_data.get("used", False):
        return False, "Reset link already used"
    
    expiry_time = datetime.datetime.fromisoformat(token_data["expiry"])
    if datetime.datetime.now() > expiry_time:
        return False, "Reset link expired (valid for 1 hour)"
    
    return True, token_data["username"]

def reset_password_with_token(token, new_password):
    """Reset password using valid token"""
    is_valid, username = verify_reset_token(token)
    
    if not is_valid:
        return False, username  # username contains error message here
    
    users = load_users()
    reset_tokens = load_reset_tokens()
    
    # Update password
    if isinstance(users[username], dict):
        users[username]["password"] = new_password
    else:
        # Convert old format to new format
        users[username] = {
            "password": new_password,
            "email": f"{username}@example.com"
        }
    
    # Mark token as used
    reset_tokens[token]["used"] = True
    
    # Save changes
    save_users(users)
    save_reset_tokens(reset_tokens)
    
    return True, "Password reset successful"

def cleanup_expired_tokens():
    """Clean up expired reset tokens"""
    reset_tokens = load_reset_tokens()
    current_time = datetime.datetime.now()
    
    expired_tokens = []
    for token, data in reset_tokens.items():
        expiry_time = datetime.datetime.fromisoformat(data["expiry"])
        if current_time > expiry_time:
            expired_tokens.append(token)
    
    for token in expired_tokens:
        del reset_tokens[token]
    
    if expired_tokens:
        save_reset_tokens(reset_tokens)

def migrate_user_format():
    """Migrate old user format to new format with email support"""
    users = load_users()
    updated = False
    
    for username, user_data in users.items():
        if not isinstance(user_data, dict):
            # Old format: username -> password
            users[username] = {
                "password": user_data,
                "email": f"{username}@example.com"
            }
            updated = True
    
    if updated:
        save_users(users)

def auth_component():
    # Cleanup expired tokens periodically
    cleanup_expired_tokens()
    
    # Migrate user format if needed
    migrate_user_format()
    
    # Initialize storage
    if "users" not in st.session_state:
        st.session_state.users = load_users()
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "show_forgot_password" not in st.session_state:
        st.session_state.show_forgot_password = False
    if "reset_token_generated" not in st.session_state:
        st.session_state.reset_token_generated = None
    if "reset_username" not in st.session_state:
        st.session_state.reset_username = None

    # Already logged in
    if st.session_state.logged_in:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✅ Logged in as {st.session_state.current_user}")
        with col2:
            if st.button("Logout", type="secondary"):
                st.session_state.logged_in = False
                st.session_state.current_user = None
                st.session_state.show_forgot_password = False
                st.session_state.reset_token_generated = None
                st.session_state.reset_username = None
                st.rerun()
        return True

    # Show forgot password form if requested
    if st.session_state.show_forgot_password:
        st.subheader("🔐 Reset Password")
        
        # If token is already generated, show the reset link
        if st.session_state.reset_token_generated:
            st.success("✅ Reset link generated successfully!")
            st.info(f"Reset link for user: **{st.session_state.reset_username}**")
            
            # Create the reset link with query parameter
            reset_url = f"?reset_token={st.session_state.reset_token_generated}"
            
            # Display the clickable reset link
            st.markdown(f"""
            ### 🔗 Your Password Reset Link:
            Click the link below to reset your password:
            
            [**🔐 Reset My Password**]({reset_url})
            
            **Reset Code:** `{st.session_state.reset_token_generated}`
            
            ⚠️ **Important:** This link will expire in 1 hour for security reasons.
            """)
            
            # Manual reset token input option
            st.markdown("---")
            st.subheader("Alternative: Enter Reset Code")
            manual_token = st.text_input("Enter your reset code:", key="manual_reset_token")
            if st.button("Use Reset Code"):
                if manual_token.strip() == st.session_state.reset_token_generated:
                    # Redirect to reset page by setting query params
                    st.query_params["reset_token"] = manual_token.strip()
                    st.rerun()
                else:
                    st.error("❌ Invalid reset code")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Generate New Link"):
                    st.session_state.reset_token_generated = None
                    st.session_state.reset_username = None
                    st.rerun()
            with col2:
                if st.button("Back to Login"):
                    st.session_state.show_forgot_password = False
                    st.session_state.reset_token_generated = None
                    st.session_state.reset_username = None
                    st.rerun()
        
        else:
            # Show form to generate reset token
            email_or_username = st.text_input(
                "Enter your username or email address:",
                key="reset_email_username"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Generate Reset Link", type="primary"):
                    if email_or_username.strip():
                        success, token, username = create_reset_token(email_or_username.strip())
                        
                        if success:
                            st.session_state.reset_token_generated = token
                            st.session_state.reset_username = username
                            st.rerun()
                        else:
                            st.error("❌ Username or email not found.")
                    else:
                        st.warning("⚠️ Please enter your username or email address.")
            
            with col2:
                if st.button("Back to Login"):
                    st.session_state.show_forgot_password = False
                    st.rerun()
        
        return False

    # Main login/signup tabs
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])

    # --- LOGIN ---
    with tab1:
        st.subheader("Login to your account")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        
        question, answer = get_captcha()
        captcha_input = st.text_input("Captcha: " + question, key="login_captcha")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login", type="primary"):
                if captcha_input.strip() != str(answer):
                    st.error("⚠️ Captcha incorrect!")
                    refresh_captcha()
                else:
                    # Check both old and new user format
                    user_data = st.session_state.users.get(username)
                    if user_data:
                        stored_password = user_data.get("password") if isinstance(user_data, dict) else user_data
                        if stored_password == password:
                            st.session_state.logged_in = True
                            st.session_state.current_user = username
                            st.session_state.show_forgot_password = False
                            st.session_state.reset_token_generated = None
                            st.session_state.reset_username = None
                            st.success("🎉 Login successful!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
                    else:
                        st.error("❌ Invalid username or password")
        
        with col2:
            if st.button("Forgot Password?"):
                st.session_state.show_forgot_password = True
                st.session_state.reset_token_generated = None
                st.session_state.reset_username = None
                st.rerun()

    # --- SIGNUP ---
    with tab2:
        st.subheader("Create a new account")
        new_user = st.text_input("Choose a username", key="signup_user")
        new_email = st.text_input("Email address", key="signup_email")
        new_pass = st.text_input("Choose a password", type="password", key="signup_pass")
        confirm_pass = st.text_input("Confirm password", type="password", key="signup_confirm")
        
        question, answer = get_captcha()
        captcha_input = st.text_input("Captcha: " + question, key="signup_captcha")
        
        if st.button("Sign Up", type="primary"):
            if captcha_input.strip() != str(answer):
                st.error("⚠️ Captcha incorrect!")
                refresh_captcha()
            elif new_pass != confirm_pass:
                st.error("⚠️ Passwords do not match!")
            elif new_user in st.session_state.users:
                st.error("❌ Username already exists, choose another.")
            elif not new_user.strip() or not new_pass.strip() or not new_email.strip():
                st.error("⚠️ All fields are required.")
            elif "@" not in new_email or "." not in new_email:
                st.error("⚠️ Please enter a valid email address.")
            elif len(new_pass) < 4:
                st.error("⚠️ Password must be at least 4 characters long.")
            else:
                # Create new user with email
                st.session_state.users[new_user] = {
                    "password": new_pass,
                    "email": new_email
                }
                save_users(st.session_state.users)
                st.success("🎉 Account created! You can now login.")

    return False