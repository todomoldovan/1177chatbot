import google.generativeai as genai
import os
import streamlit as st

# Configure the Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-pro')

st.set_page_config(page_title = "1177.se chatbot")       # sets title shown in brower tab
#st.image("app\1177_logo.png", width=100)
st.title("1177 chatbot")
st.divider()
st.caption("*Disclaimer:* This application was trained on data scraped from 1177.se. The chatbot should assist in getting health advice, but always remember, that it can not replace a doctor. It is a student project and not officially hosted by 1177.se.")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input prompt from user
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using Google Gemini
    response = model.generate_content(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response.text)
    
    # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": response.text})
