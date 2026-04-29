from dotenv import load_dotenv
import os

from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI

# cargar variables de entorno
load_dotenv()

def get_llm(usar_api=False, params=None):
    """
    Devuelve un modelo LLM configurado.
    
    Parameters:
    - usar_api (bool): si True usa DeepSeek, si False usa Ollama
    
    Returns:
    - instancia de LLM
    """
    
    default_params = {
        "temperature": None, # Aleatoriedad del modelo.
        "max_tokens": 50, # Longitud de la respuesta en DeepSeek
        "top_p": None, # diversidad de palabras considerando la probabilidad acumulada (no usar junto con temperature).
        "top_k": None, # Diversidad de palabras considerando la frecuencia 
        "frequency_penalty": None, # Penalizacion de frecuencia - DeepSeek
        "presence_penalty": None, # Penalizacion de presencia - DeepSeek
        "repeat_penalty": None # Penalizacion de repeticion - Ollama

    }

    if params:
        default_params.update(params)
    
    if usar_api:
        # DeepSeek
        return ChatOpenAI(
            model="deepseek-chat", 
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com",
            temperature=default_params["temperature"],
            max_tokens=default_params["max_tokens"],
            top_p=default_params["top_p"],
            frequency_penalty=default_params["frequency_penalty"],
            presence_penalty=default_params["presence_penalty"]
        )

    else:
        # Ollama
        return OllamaLLM(
            model="qwen:4b",
            temperature=default_params["temperature"],
            num_predict=default_params["max_tokens"],
            top_p=default_params["top_p"],
            top_k=default_params["top_k"]
        )