import streamlit as st
import chromadb
import time

st.set_page_config(page_title="Contact Us - Ask1177", page_icon=":phone:", layout="wide")

# Initialize ChromaDB client
client = chromadb.Client()

# Define the collection for storing contacts
collection = client.get_or_create_collection(name="contacts")

st.title("Contact Us 📞")

# Contact form fields
name = st.text_input("Your Name", key='name')
message = st.text_area("Your Message", key='message')

if st.button("Send"):
    if name and message:
        try:
            # Store contact information in ChromaDB
            collection.add(
                documents=[message],
                metadatas=[{"name": name}],
                ids=[f"{name}_{int(time.time())}"]  # Use name and timestamp as a unique identifier
            )
            st.success("Message sent successfully!")
            time.sleep(2)
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please fill out all required fields.")