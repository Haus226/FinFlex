import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

@st.cache_data
def firebase_init():
    cred = credentials.Certificate("finflex-1939d-firebase-adminsdk.json")
    firebase_admin.initialize_app(cred, 
                                {"databaseURL":"https://finflex-1939d-default-rtdb.asia-southeast1.firebasedatabase.app/"},
    )

def login_app():
    with st.form("login"):
        user = st.text_input('Username')
        password = st.text_input('Password',type='password')    
        submit = st.form_submit_button("Login")
        if submit and user and password:
            db = firestore.client()
            doc_ref = db.collection("users").document(user)
            doc = doc_ref.get()
            if doc.exists:
                if password == doc.get("PASSWORD"):
                    ex = pd.ExcelFile("UMH24 - FinTech Dataset.xlsx")
                    st.session_state["data"] = pd.read_excel(ex, sheet_name=user)
                    st.session_state["user"] = doc.get("NAME")
                    st.switch_page("pages/main.py")
                else:
                    st.warning("Incorrect password")
            else:
                st.warning("Username does not exists")
        elif submit and (not user):
            st.warning("Please key in your username.")
        elif submit and (not password):
            st.warning("Please key in your password.")

firebase_init()
login_app()