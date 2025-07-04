import os

from langchain.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain.base_language import BaseLanguageModel


def get_llm(model_name: str = "llama3", provider: str | None = None) -> BaseLanguageModel:
    """Return LLM from Ollama or LM Studio based on provider."""
    provider = provider or os.environ.get("LLM_PROVIDER", "ollama").lower()
    if provider == "lmstudio":
        return ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model=model_name,
        )
    return Ollama(model=model_name)
