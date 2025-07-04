from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding() -> HuggingFaceEmbeddings:
    """Return multilingual embedding model."""
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
