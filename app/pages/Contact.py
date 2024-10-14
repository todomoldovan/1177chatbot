import streamlit as st
import os
import json
from datetime import datetime

# Streamlit configuration
st.set_page_config(page_title="Contact", page_icon=":pill:", initial_sidebar_state="auto", layout="wide")

def load_submissions():
    if os.path.exists("contact_submissions.json"):
        with open("contact_submissions.json", "r") as f:
            return json.load(f)
    return []

def save_submission(name, email, message):
    submissions = load_submissions()
    timestamp = datetime.now().isoformat()
    submission = {
        "name": name,
        "email": email,
        "message": message,
        "timestamp": timestamp
    }
    submissions.append(submission)
    with open("contact_submissions.json", "w") as f:
        json.dump(submissions, f, indent=2)

def contact_form():
    st.title("Contact Form")
    
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")
    
    if st.button("Submit"):
        if name and email and message:
            try:
                save_submission(name, email, message)
                st.success("Thank you for your message! We'll get back to you soon.")
            except Exception as e:
                st.error(f"An error occurred while submitting your message: {str(e)}")
        else:
            st.warning("Please fill out all fields before submitting.")

if __name__ == "__main__":
    st.session_state.current_page = os.path.basename(__file__).replace(".py", "")
    st.session_state.parent_dir = os.path.dirname(os.path.abspath(__file__))
    st.session_state.logo_path = os.path.join(st.session_state.parent_dir, "../images/1177_logo_selfcreated_large.png")
    st.session_state.collapsed_sidebar_logo_path = os.path.join(st.session_state.parent_dir, "../images/1177_logo_selfcreated_whitebackground.png")

    with st.sidebar:
        st.logo(st.session_state.logo_path, size="large", icon_image=st.session_state.collapsed_sidebar_logo_path)

        st.title("Ask1177")
        st.write("Your health assistant powered by AI. This application was trained on symptoms and diseased data from 1177.se. The AI has webpage data from Oktober 2024 to use and reference.")

    contact_form()