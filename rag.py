from typing import List

from langchain.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class DocumentProcessor:
    """Load and index documents for retrieval."""

    def __init__(self, embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", chunk_size: int = 500, chunk_overlap: int = 50):
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.vectorstore = None
        self.docs = []

    def load_document(self, file_path: str) -> None:
        """Load supported document and prepare vectorstore."""
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith((".ppt", ".pptx")):
            loader = UnstructuredPowerPointLoader(file_path)
        else:
            raise ValueError("Unsupported file type")

        documents = loader.load()
        splits = self.text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        self.docs = splits

    def get_retriever(self):
        if not self.vectorstore:
            raise ValueError("No document loaded")
        return self.vectorstore.as_retriever()


class RAGAdvisor:
    """Perform retrieval-augmented tasks using a local LLM."""

    def __init__(self, processor: DocumentProcessor, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", role: str = "General Advisor", hf_token: str | None = None):
        self.processor = processor
        self.role = role
        self.history = []

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", use_auth_token=hf_token)
        generation = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, do_sample=False)
        self.llm = HuggingFacePipeline(pipeline=generation)

        retriever = self.processor.get_retriever()
        self.chain = ConversationalRetrievalChain.from_llm(self.llm, retriever)

    def run(self, query: str) -> str:
        """Run a query through the RAG pipeline."""
        prompt = f"You are acting as a {self.role}. {query}"
        result = self.chain({"question": prompt, "chat_history": self.history})
        self.history.append((query, result["answer"]))
        return result["answer"]
