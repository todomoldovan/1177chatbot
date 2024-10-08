import streamlit as st
import chromadb
import time
from streamlit_extras.switch_page_button import switch_page
import os
from streamlit_option_menu import option_menu

def contact_form():
    st.title("Contact Form")
    
    # Create input fields
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")
    
    # Create a submit button
    if st.button("Submit"):
        if name and email and message:
            # In a real application, you would process the form data here
            # For now, we'll just display a success message
            st.success("Thank you for your message! We'll get back to you soon.")
            
            # You could also add code here to send an email or store the message
        else:
            st.warning("Please fill out all fields before submitting.")

if __name__ == "__main__":
    st.session_state.current_page = os.path.basename(__file__).replace(".py", "")
    st.session_state.parent_dir = os.path.dirname(os.path.abspath(__file__))
    st.session_state.logo_path = os.path.join(st.session_state.parent_dir, "../images/1177_logo_selfcreated_large.png")
    st.session_state.collapsed_sidebar_logo_path = os.path.join(st.session_state.parent_dir, "../images/1177_logo_selfcreated_whitebackground.png")

    with st.sidebar:
        st.logo(st.session_state.logo_path, size="large", icon_image=st.session_state.collapsed_sidebar_logo_path)
        # default_index = 1 if st.session_state.current_page == "contact_form" else 0

        # selected = option_menu(
        #     menu_title=None,
        #     options=["Chat with Liv", "Contact"],
        #     icons=["chat-dots", "envelope"],
        #     menu_icon="cast",
        #     default_index=0,
        # )

        # if selected == "Chat":
        #     switch_page("app gemini")

        st.title("Ask1177")
        st.write("Your health assistant powered by AI. This application was trained on symptoms and diseased data from 1177.se. The AI has webpage data from Oktober 2024 to use and reference.")

    contact_form()