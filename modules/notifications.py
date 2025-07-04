import streamlit as st

def send_notification(message: str):
    st.toast(message)
