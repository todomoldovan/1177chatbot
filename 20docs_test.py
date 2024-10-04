import os
import google.generativeai as genai


os.environ["GEMINI_API_KEY"]="AIzaSyDPiaGSPG-FGbM3bg5V5tpvrp1RPifjEgA"

text =[]
for i in range(20):
    text.append(genai.upload_file(f"scraping\pdf_downloads\child_page_{i+1}.pdf"))

#print(text)

def generate_response(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-pro',system_instruction ="You are a knowledgeable resource providing general information about medical conditions based on the content from the uploaded files. Explain concepts thoroughly while ensuring clarity for a non-technical audience. When symptoms are provided, please offer a potential diagnosis. Always emphasize that users should consult a healthcare professional for personalized medical advice.")
    answer = model.generate_content([prompt]+ text)
    return answer.text


answer = generate_response("Jag upplever följande symptom: magbesvär och trötthet. Vad kan detta vara för sjukdom, och vad kan jag göra nu?")

print(answer)