import streamlit as st
import chromadb
import time
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page

# Initialize ChromaDB client
client = chromadb.Client()

# Define the collection for storing contacts
collection = client.get_or_create_collection(name="contacts")

# Streamlit page setup
st.set_page_config(layout="wide")

def navigation_bar():
    with st.container():
        selected = option_menu(
            menu_title=None,
            options=["Ask 1177 bot", 'Contact'],
            icons=['chat', 'phone'],
            menu_icon="cast",
            orientation="horizontal",
            styles={
                "nav-link": {
                    "text-align": "left",
                    "--hover-color": "#eee",
                }
            }
        )
        if selected == "Contact":
            switch_page("contact_form")
        if selected == "Ask 1177 bot":
            switch_page("app_gemini")
        

navigation_bar()

st.title("Contact Form")

# Initialize session state for the input fields
if 'name' not in st.session_state:
    st.session_state.name = ""
if 'message' not in st.session_state:
    st.session_state.message = ""

# Contact form fields
name = st.text_input("Your Name", value=st.session_state.name, key='name')
message = st.text_area("Your Message", value=st.session_state.message, key='message')

if st.button("Send"):
    if name and message:
        try:
            # Store contact information in ChromaDB
            collection.add(
                documents=[message],
                metadatas=[{"name": name}],
                ids=[name]  # Use name as a unique identifier
            )
            st.success("Message sent successfully!")

            # Clear the form fields by resetting session state
            #st.session_state.name = ""
            #st.session_state.message = ""
            time.sleep(2)
            # Rerun the app to reset the form inputs
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please fill out all required fields.")
