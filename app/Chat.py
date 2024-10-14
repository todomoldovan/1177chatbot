import os
import time
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from pypdf import PdfReader
import base64

from typing import List, Dict
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict
import streamlit as st
import pandas as pd
from PIL import Image
import google.generativeai as genai
import google.api_core.exceptions
import requests
import re
from langdetect import detect
from googletrans import Translator
import logging
from collections import OrderedDict


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit configuration
st.set_page_config(page_title="Ask1177", page_icon=":pill:", initial_sidebar_state="auto", layout="wide")

# Constants
NUMBER_OF_FILES = 509 #509
NUMBER_OF_VECTOR_RESULTS = 3
CACHE_EXPIRATION = 3600  # 1 hour

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# Set up paths
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(PARENT_DIR, "images/1177_logo_selfcreated_large.png")
COLLAPSED_SIDEBAR_LOGO_PATH = os.path.join(PARENT_DIR, "images/1177_logo_selfcreated_whitebackground.png")

def load_avatar(image_path, size=(40, 40)):
    img = Image.open(image_path)
    img = img.resize(size)
    return img

def load_logo(image_path, width=150):
    img = Image.open(image_path)
    aspect_ratio = img.height / img.width
    height = int(width * aspect_ratio)
    img = img.resize((width, height))
    return img

def typing_effect(text, delay=0.01):
    placeholder = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text, unsafe_allow_html=True)
        time.sleep(delay)
    placeholder.markdown(text)

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
        logger.info("------- DELETED DATABASE -------")

    db = chroma_client.create_collection(name="rag_experiment", embedding_function=GeminiEmbeddingFunction(), metadata={"hnsw:space": "cosine"})

    df = pd.read_csv('scraping/links_and_pdfs.csv')
    pdf_url_map = dict(zip(df['PDF Filename'], df['URL']))

    def get_title_from_url(url):
        return url.rstrip('/').split('/')[-1].replace('-', ' ').title()

    for i in range(NUMBER_OF_FILES):
        file_name = f"child_page_{i+1}.pdf"
        file_path = f"scraping/pdf_downloads/{file_name}"
        if os.path.exists(file_path):
            text = load_pdf(file_path)
            if not text:
                logger.info(f"Document {i+1} is empty, skipping...")
            else:
                url = pdf_url_map.get(file_name, "https://www.1177.se")
                title = get_title_from_url(url)
                
                db.add(
                    documents=[text],
                    ids=[str(i)],
                    metadatas=[{"title": title, "url": url}]
                )
                logger.info(f"Document {i+1} was loaded. Title: {title}, URL: {url}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"{timestamp} : All documents loaded. Total time: {elapsed_time:.2f} seconds")
    
    return db

def enhanced_query(prompt: str, db, model, num_results: int = NUMBER_OF_VECTOR_RESULTS, similarity_threshold: float = 0.3) -> Dict:
    start_time = time.time()

    references = []
    
    prompt_embedding = genai.embed_content(
        model="models/embedding-001",
        content=prompt,
        task_type="retrieval_document",
        title="User query"
    )["embedding"]
    
    results = db.query(
        query_embeddings=[prompt_embedding],
        n_results=num_results * 2,
        include=['documents', 'metadatas', 'distances']
    )

    similarities = [1 - distance for distance in results['distances'][0]]
    
    filtered_results = [
        (doc, meta, sim) 
        for doc, meta, sim in zip(results['documents'][0], results['metadatas'][0], similarities) 
        if sim > similarity_threshold
    ]
    filtered_results.sort(key=lambda x: x[2], reverse=True)
    
    context_with_citations = ""
    for i, (doc, metadata, similarity) in enumerate(filtered_results[:num_results], 1):
        context_with_citations += f"[{i}] Title: {metadata['title']}\nContent: {doc}\n\n"
        references.append({
            'title': metadata['title'],
            'url': metadata['url'],
            'similarity': similarity
        })

    logger.info(f"Number of references: {len(references)}")

    response = model.generate_content([
        "You are a knowledgeable resource providing general information about medical conditions based primarily on the content in the context. Explain concepts thoroughly while ensuring clarity for a non-technical audience. When symptoms are provided, please offer a potential diagnosis. Always emphasize that users should consult a healthcare professional for personalized medical advice.",
        "Important: For each piece of information you provide, you MUST cite your source using the number in square brackets that precedes the relevant information in the context, e.g. [1]. Only use citations that are provided in the context.",
        f"Context: {context_with_citations}",
        f"User question: {prompt}"
    ])

    
    
    citations = re.findall(r'\[(\d+)\]', response.text)
    logger.info(f"Citations found in response: {citations}")
    
    used_references = [references[int(citation) - 1] for citation in set(citations) if int(citation) <= len(references)]
    logger.info(f"Used references: {used_references}")
    
    end_time = time.time()
    logger.info(f"Query processed in {end_time - start_time:.2f} seconds")
    
    return {"response": response.text, "source": "ai", "references": used_references}

def main():
    with st.sidebar:
        st.logo(load_logo(LOGO_PATH), size="large", icon_image=load_logo(COLLAPSED_SIDEBAR_LOGO_PATH))
        st.title("Ask1177")
        st.write("Your health assistant powered by AI. The model has access to data from 1177.se that was fetched in October 2024.")
        st.divider()

        if st.button("Clear Chat History"):
            st.session_state.messages = []

        if st.button("Export Chat History"):
            chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])
            b64 = base64.b64encode(chat_history.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="chat_history.txt">Download Chat History</a>'
            st.markdown(href, unsafe_allow_html=True)

    st.title("Chat with Liv 💬 :pill:")
    st.warning("*Disclaimer:* This application was trained on symptoms and diseases data from 1177.se. The chatbot can assist with getting health advice, but always remember that it cannot replace a doctor. This is a student project and not officially hosted by 1177.se.", icon="⚠️")
    st.divider()

    st.session_state.db = create_chroma_db()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=load_avatar('app/images/user_image.png') if message["role"] == "user" else load_avatar('app/images/liv_chatassistant.png')):
            st.markdown(message["content"])

    if prompt := st.chat_input("What symptoms do you have?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar=load_avatar('app/images/user_image.png')):
            st.markdown(prompt)

        try:
            response_data = enhanced_query(prompt, st.session_state.db, model)

            with st.chat_message("assistant", avatar=load_avatar('app/images/liv_chatassistant.png')):
                typing_effect(response_data["response"])
            
            st.divider()
            if response_data["references"]:
                st.markdown("**References:**")
                for i, ref in enumerate(response_data["references"], 1):
                    similarity_percentage = f"{ref['similarity'] * 100:.2f}%"
                    st.markdown(f"[{i}] [{ref['title']}]({ref['url']})")
            else:
                st.warning("No references found. This response is generated solely by the AI model without specific source citations.", icon="⚠️")

            references_text = "\n\n**References:**\n" + "\n".join(f"[{i}] [{ref['title']}]({ref['url']})" for i, ref in enumerate(response_data["references"], 1)) if response_data["references"] else "\n\n**Note:** No specific references cited for this response."

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_data["response"] + references_text
            })

        except (google.api_core.exceptions.InternalServerError, requests.exceptions.HTTPError) as e:
            error_message = "An error occurred. Please try again later."
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant", avatar=load_avatar('app/images/liv_chatassistant.png')):
                typing_effect(error_message)
            logger.error(f"API Error: {str(e)}")

        except Exception as e:
            error_message = "An unexpected error occurred. Please try again later."
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant", avatar=load_avatar('app/images/liv_chatassistant.png')):
                typing_effect(error_message)
            logger.error(f"Unexpected Error: {str(e)}")

if __name__ == "__main__":
    main()