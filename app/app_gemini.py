import google.generativeai as genai
import os
import streamlit as st
from datetime import datetime
import time
from dotenv import load_dotenv
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import re
from pypdf import PdfReader
import requests
import pandas as pd

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-1.5-pro')

st.set_page_config(page_title="Ask1177")
st.title("Ask1177")
st.divider()
st.caption("*Disclaimer:* This application was trained on data scraped from 1177.se. The chatbot should assist in getting health advice, but always remember, that it can not replace a doctor. It is a student project and not officially hosted by 1177.se.")
st.divider()

number_of_files = 50
number_of_vector_results = 3

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        page_text = re.sub(r'\s+', ' ', page_text)
        text += page_text + " "
    return text.strip()

@st.cache_resource
def create_chroma_db():
    start_time = time.time()
    chroma_client = chromadb.PersistentClient(path="contents")

    existing_collections = chroma_client.list_collections()
    name = "rag_experiment"

    if any(collection.name == name for collection in existing_collections):
        chroma_client.delete_collection(name=name)
        print("------- DELETED DATABASE -------")

    db = chroma_client.create_collection(name="rag_experiment", embedding_function=GeminiEmbeddingFunction(), metadata={"hnsw:space": "cosine"})

    df = pd.read_csv('scraping/links_and_pdfs.csv')  # Replace with your actual CSV filename
    pdf_url_map = dict(zip(df['PDF Filename'], df['URL']))

    def get_title_from_url(url):
        return url.rstrip('/').split('/')[-1].replace('-', ' ').title()

    for i in range(number_of_files):
        file_name = f"child_page_{i+1}.pdf"
        file_path = f"scraping/pdf_downloads/{file_name}"
        if os.path.exists(file_path):
            text = load_pdf(file_path)
            if not text:
                print(f"Document {i+1} is empty, skipping...")  
            else:
                url = pdf_url_map.get(file_name, "https://www.1177.se")
                title = get_title_from_url(url)
                
                db.add(
                    documents=[text],
                    ids=[str(i)],
                    metadatas=[{"title": title, "url": url}]
                )
                print(f"Document {i+1} was loaded. Title: {title}, URL: {url}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} : All documents loaded. Total time: {elapsed_time:.2f} seconds")
    
    return db

# Create or load the ChromaDB
st.session_state.db = create_chroma_db()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input prompt from user
if prompt := st.chat_input("What symptoms do you have?"):
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} : User asked a question")

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        # Retrieve relevant documents from ChromaDB
        results = st.session_state.db.query(query_texts=[prompt], n_results=number_of_vector_results, include=['documents', 'metadatas'])

        context = ""
        references = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            context += f"Title: {metadata['title']}\nContent: {doc}\n\n"
            references.append({
                'title': metadata['title'],
                'url': metadata['url']
        })

        # Generate response using the Gemini API
        response = model.generate_content([
            "You are a knowledgeable resource providing general information about medical conditions based primarily on the content in the context. Explain concepts thoroughly while ensuring clarity for a non-technical audience. When symptoms are provided, please offer a potential diagnosis. Always emphasize that users should consult a healthcare professional for personalized medical advice. At the end of your response, list the titles of the source documents you used, prefixed with [1], [2], etc.",
            f"Context: {context}",
            f"User question: {prompt}"
        ])

        with st.chat_message("assistant"):
            st.markdown(response.text)
        
        st.divider()
        st.markdown("**References:**")
        for i, ref in enumerate(references, 1):
            st.markdown(f"[{i}] [{ref['title']}]({ref['url']})")

        # Append the assistant's response to the messages
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response.text + "\n\n**References:**\n" + "\n".join(f"[{i}] [{ref['title']}]({ref['url']})" for i, ref in enumerate(references, 1))
        })
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 500:
            st.error("An internal server error occurred (500). Please try again later.")
            print(f"500 error: {e}")
        else:
            st.error(f"An error occurred: {e}")
            print(f"Error: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} : Response generated. Total time: {elapsed_time:.2f} seconds")
