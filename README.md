# 1177chatbot
This is the final project for 1RT730 Large Language Models and Societal Consequences of Artificial Intelligence at Uppsala University.

The application is a multilingual 1177 chatbot with a focus on disease-related queries and symptoms. Therefore, the application was trained on symptoms and diseases data from 1177.se. This is a student project and not officially hosted by 1177.se.

# Getting started and runnign the application
1. Get a Gemini API key here: https://ai.google.dev/gemini-api/docs/api-key
2. Create a conda virtual environment with e.g. this Python version (Python 3.9.20) by running ```conda create --name myenv python=3.9.20``` followed by ```conda activate myenv```
3. Run ```pip install -r requirements.txt```
4. Then run ```streamlit run app/Chat.py``` to start the application
5. Wait for approximately 23 minutes to load all 509 documents, you can also decrease the documents used by editing the Chat.py file constants at the top of the file. One of the first document is this webpage about celiaki, so you can ask about any content on the page (suggest to ask about symptoms): https://www.1177.se/Vastra-Gotaland/sjukdomar--besvar/allergier-och-overkanslighet/celiaki/celiaki/
6. If you do not wish to re-embed all documents into the DB at each rerun of the application and instead use the ones that are already embedded locally from your previous runs, then set the constant ```RELOAD_DB``` to False.

Asking and responding to a user query takes approx 5 seconds. 

**We have used the same instructions on a new virual environment and it works.** 
(Reach out to us if you would however somehow happen to get any unexpected problems)

The creation of the database can take a few minutes so please be patient. (Approximately 5 minutes on an M1 Pro machine.)

# Architecture
- Used beautifulsoup4 to scrape HTML content from 1177.se and convert to PDFs
- Used chromadb as a database to store the files
- Used the Gemini-1.5-flash model accessed via API. We previously used 1.5-pro but we constantly exceeded the daily request limits. 
