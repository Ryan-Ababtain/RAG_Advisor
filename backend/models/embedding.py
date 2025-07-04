from langchain.embeddings import HuggingFaceEmbeddings


def get_embedding() -> HuggingFaceEmbeddings:
    """Return embedding model."""
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
