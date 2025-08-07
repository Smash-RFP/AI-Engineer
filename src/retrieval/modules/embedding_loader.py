from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

def load_embedding_model(model_name: str, provider: str = "huggingface"):
    if provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=model_name)
    elif provider == "openai":
        return OpenAIEmbeddings(model=model_name)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
