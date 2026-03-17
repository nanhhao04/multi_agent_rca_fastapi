import yaml
import os
from langchain_google_genai import ChatGoogleGenerativeAI

config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')

def connect_llm():
    """Initialize Gemini LLM from config."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=cfg['GOOGLE_API_KEY'],
        temperature=0.2
    )
    print("LLM initialized: gemini-2.5-flash")
    return llm

llm = connect_llm()