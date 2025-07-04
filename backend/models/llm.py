from langchain.llms import Ollama


def get_llm(model_name: str = "llama3") -> Ollama:
    """Return Ollama LLM model."""
    return Ollama(model=model_name)
