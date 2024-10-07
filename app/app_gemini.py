import google.generativeai as genai
import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import re
from pypdf import PdfReader

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
    chroma_client = chromadb.PersistentClient(path="contents")

    existing_collections = chroma_client.list_collections()
    name = "rag_experiment"

    print("all dbs: ", chroma_client.list_collections())

    if any(collection.name == name for collection in existing_collections):
        chroma_client.delete_collection(name=name)
        print("------- DELETED DATABASE -------")


    # if "rag_experiment" in chroma_client.list_collections():
    #     return chroma_client.get_collection("rag_experiment")
    
    
    db = chroma_client.create_collection(name="rag_experiment", embedding_function=GeminiEmbeddingFunction())

    for i in range(50):  # Adjust range as needed
        file_path = f"../scraping/pdf_downloads/child_page_{i+1}.pdf"
        if os.path.exists(file_path):
            text = load_pdf(file_path)
            if not text:
                print(f"Document {i+1} is empty, skipping...")  
            else:
                db.add(documents=[text], ids=[str(i)])
                print(f"documents{i+1} was loaded")
            #db.add(documents=[text], ids=[str(i)])
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
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"{timestamp} : Before message compute")    
if prompt := st.chat_input("What symptoms do you have?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve relevant documents from ChromaDB
    results = st.session_state.db.query(query_texts=[prompt], n_results=3)
    context = "\n".join(results['documents'][0])

    # Generate response using Google Gemini
    response = model.generate_content([
        "You are a knowledgeable resource providing general information about medical conditions based on the content from the uploaded files. Explain concepts thoroughly while ensuring clarity for a non-technical audience. When symptoms are provided, please offer a potential diagnosis. Always emphasize that users should consult a healthcare professional for personalized medical advice.",
        f"Context: {context}",
        f"User question: {prompt}"
    ])
    
    with st.chat_message("assistant"):
        st.markdown(response.text)
    
    # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": response.text})