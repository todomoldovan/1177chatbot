import google.generativeai as genai
import os
import streamlit as st
from datetime import datetime

# Configure the Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-pro',system_instruction ="You are a knowledgeable resource providing general information about medical conditions based on the content from the uploaded files. Explain concepts thoroughly while ensuring clarity for a non-technical audience. When symptoms are provided, please offer a potential diagnosis. Always emphasize that users should consult a healthcare professional for personalized medical advice.")

# Sidebar
page = st.sidebar.selectbox("Navigation", ["Mainpage", "Contact Form"])

def show_app_page():
    st.set_page_config(page_title = "Ask1177")       # sets title shown in brower tab
    #st.image("app\1177_logo.png", width=100)
    st.title("Ask1177")
    st.divider()
    st.caption("*Disclaimer:* This application was trained on data scraped from 1177.se. The chatbot should assist in getting health advice, but always remember, that it can not replace a doctor. It is a student project and not officially hosted by 1177.se.")
    st.divider()

    @st.cache_data
    def upload_documents():
        text = []
        for i in range(50):
            text.append(genai.upload_file(f"scraping\pdf_downloads\child_page_{i+1}.pdf"))
            if (i+1) % 10 == 0:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{timestamp} : Currently reading document {i+1}...")
        print("All pages are loaded")
        return text

    # Call the cached function
    st.session_state.text = upload_documents()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input prompt from user
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} : Before message compute")    
    if prompt := st.chat_input("What symptoms do you have?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        print(f"{ datetime.now().strftime("%Y-%m-%d %H:%M:%S")} : Step 1 completed") 
        with st.chat_message("user"):
            st.markdown(prompt)
        print(f"{ datetime.now().strftime("%Y-%m-%d %H:%M:%S")} : Step 2 completed") 

        # Generate response using Google Gemini
        # old: response = model.generate_content(prompt)
        response = model.generate_content([prompt]+ st.session_state.text)
        print(f"{ datetime.now().strftime("%Y-%m-%d %H:%M:%S")} : Step 3 completed") 
        
        with st.chat_message("assistant"):
            st.markdown(response.text)
        print(f"{ datetime.now().strftime("%Y-%m-%d %H:%M:%S")} : Step 4 completed") 
        
        # Append the assistant's response to the messages
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} : After message compute")
