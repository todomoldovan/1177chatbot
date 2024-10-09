# 1177chatbot
Final project for 1RT730 Large Language Models and Societal Consequences of Artificial Intelligence at Uppsala University

The application was trained on symptoms and diseases data from 1177.se. This is a student project and not officially hosted by 1177.se.

# Getting started
1. Get a Gemini API key here: https://ai.google.dev/gemini-api/docs/api-key
2. Dowload wkhtmltopdf directly from the website: https://wkhtmltopdf.org/downloads.html
3. Create a virtual environment using conda
4. run ``` pip install -r requirements.txt```
5. run ``` streamlit run app/Chat.py```

The creation of the database can take a few minutes so please be patient. (Approximately 5 minutes on an M1 Pro machine.)

# Scraping
Used beautifulsoup4 to scrape HTML content from 1177.se and convert to PDFs.

# Application
Mainly used this tutorial to build the Streamlit interface: https://www.youtube.com/watch?v=dXxQ0LR-3Hg
