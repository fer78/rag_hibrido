from langchain_community.embeddings import OllamaEmbeddings

def get_embeddings(provider="oll", model=None):
    """
    Devuelve un modelo de embeddings configurado.
    """

    if provider == "oll":
        return OllamaEmbeddings(
            model=model or "nomic-embed-text"
        )

    else:
        raise ValueError(f"Proveedor de embeddings no soportado: {provider}")