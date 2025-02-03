import streamlit as st
import json
import time

with open("authentication.json", 'r') as json_file:
    USER_CREDENTIALS = json.load(json_file)

def authenticate(username, password):
    return USER_CREDENTIALS.get(username) == password

def main_app():
    st.title("Welcome to the Main Application")
    st.write("This is the protected content visible only after login.")

def login_page():
    st.title("Login to Access the App")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate(username, password):
            st.success(f"You have successfully logged in!")
            st.session_state["authenticated"] = True
            time.sleep(1)
        else:
            st.error("Invalid username or password")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_page()
else:
    main_app()
