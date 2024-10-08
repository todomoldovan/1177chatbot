import streamlit as st
from app_gemini import show_app_page
from contact_form import show_contact_form

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["App Page", "Contact Form"])

# Display the selected page
if page == "App Page":
    show_app_page()  # Call the main page function
elif page == "Contact Form":
    show_contact_form()  # Call the contact form function
