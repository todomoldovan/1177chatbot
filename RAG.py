import os
import unstructured
import re
from pypdf import PdfReader
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
import json


os.environ["GEMINI_API_KEY"]="yourkeeey"

def load_pdf(file_path):
    # Logic to read pdf
    reader = PdfReader(file_path)
    # Loop over each page and store it in a variable
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    clean_text = re.sub(r'[\n\t]', ' ', text)
    clean_text = re.sub(r'[^a-zA-Z0-9åäöÅÄÖ\s.,;:!?()\"\'-]+', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()  
    return clean_text


#text =[]
#for i in range(20):
#    loaded_pdf = load_pdf(file_path=f"scraping\pdf_downloads\child_page_{i+1}.pdf")
#    pdf_text = clean_text(loaded_pdf)
#    text.append(pdf_text)
    
# with open('document_list.json', 'w') as f:
#    json.dump(text, f)

# Load list from file
with open('document_list.json', 'r') as f:
    text = json.load(f)

# clean imported json

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]

def create_chroma_db(documents, path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    # Delete the collection if it already exists
    existing_collections = chroma_client.list_collections()
    print(existing_collections)
    if any(collection.name == name for collection in existing_collections):
        chroma_client.delete_collection(name=name)
        print(f"Deleted existing collection '{name}'.")
    # Create a new collection
    try:
        db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    except UniqueConstraintError:
        print(f"Collection '{name}' already exists. Deleting it...")
        chroma_client.delete_collection(name=name)
        db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    for i in range(len(documents)):
        if not documents[i]:
          print(f"Document {i+1} is empty, skipping...")  
        else:
            db.add(documents=documents[i], ids=str(i+1))
    return db, name

db,name =create_chroma_db(documents=text, path="contents", name="rag_experiment")

def load_chroma_collection(path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    return db

db=path=load_chroma_collection("contents", name="rag_experiment")
print(db)

def get_relevant_passage(query, db, n_results):
  passage = db.query(query_texts=[query], n_results=n_results)['documents'][0]
  return passage

relevant_text = get_relevant_passage("Cancer",db,3)
print(f"{len(relevant_text)} RELEVANT TEXT")

def make_rag_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'
  ANSWER:
  """).format(query=query, relevant_passage=escaped)
  return prompt
#  If the passage is irrelevant to the answer, you may ignore it.

def generate_response(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

def generate_answer(db,query):
    #retrieve top 3 relevant text chunks
    relevant_text = get_relevant_passage(query,db,n_results=3)
    prompt = make_rag_prompt(query, 
                             relevant_passage="".join(relevant_text)) # joining the relevant chunks to create a single passage
    answer = generate_response(prompt)
    return answer

db=load_chroma_collection(path="contents", #replace with path of your persistent directory
                          name="rag_experiment") #replace with the collection name

answer = generate_answer(db,query="Vad kan man göra om man har cancer?")
print(answer)