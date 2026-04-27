from dotenv import load_dotenv
import os

from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# Selector de modelo
usar_api = True


# LLM local
llm_local = OllamaLLM(model="qwen:4b")

# LLM API
llm_api = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com",
    model="deepseek-chat"
)

# Elegir Modelo
llm = llm_api if usar_api else llm_local

respuesta = llm.invoke("Explica qué es RAG en 2 líneas")
print(respuesta)
