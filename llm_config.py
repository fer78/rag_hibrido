from dotenv import load_dotenv
import os

from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI

# cargar variables de entorno
load_dotenv()

def get_llm(usar_api=False):
    """
    Devuelve un modelo LLM configurado.
    
    Parameters:
    - usar_api (bool): si True usa DeepSeek, si False usa Ollama
    
    Returns:
    - instancia de LLM
    """
    
    if usar_api:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if not api_key:
            raise ValueError("No se encontró la API key en el archivo .env")
        
        return ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
            model="deepseek-chat"
        )
    
    else:
        return OllamaLLM(model="qwen:4b")