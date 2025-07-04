from pathlib import Path
from typing import List, Tuple
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from models.embedding import get_embedding
from models.llm import get_llm
from loaders.pdf_loader import PDFLoader
from loaders.ppt_loader import PPTLoader


class RagEngine:
    def __init__(self, docs_dir: Path, index_dir: Path):
        self.docs_dir = Path(docs_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding = get_embedding()
        self.vectorstore: FAISS | None = None

    def _load_documents(self) -> List[Document]:
        documents: List[Document] = []
        for path in self.docs_dir.glob('*'):
            try:
                if path.suffix.lower() == '.pdf':
                    loader = PDFLoader(str(path))
                elif path.suffix.lower() == '.pptx':
                    loader = PPTLoader(str(path))
                else:
                    continue
                documents.extend(loader.load())
            except Exception as e:  # noqa: BLE001
                logging.error("Failed to parse %s: %s", path, e)
        return documents

    def ingest(self) -> None:
        documents = self._load_documents()
        if not documents:
            return
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)
        try:
            self.vectorstore = FAISS.from_documents(docs, self.embedding)
            self.vectorstore.save_local(str(self.index_dir))
        except Exception as e:  # noqa: BLE001
            logging.error("Failed to build vectorstore: %s", e)

    def _load_vectorstore(self) -> FAISS:
        if self.vectorstore:
            return self.vectorstore
        index_file = self.index_dir / 'index.faiss'
        if index_file.exists():
            try:
                self.vectorstore = FAISS.load_local(
                    str(self.index_dir),
                    self.embedding,
                    allow_dangerous_deserialization=True,
                )
                return self.vectorstore
            except Exception as e:  # noqa: BLE001
                logging.error("Failed to load vectorstore: %s", e)
        raise ValueError("Vector store not found. Ingest documents first.")

    def query(self, query: str, model_name: str) -> Tuple[str, List[str]]:
        vs = self._load_vectorstore()
        retriever = vs.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join(d.page_content for d in docs)
        llm = get_llm(model_name)
        prompt = f"Answer the question based on the context:\n{context}\n\nQuestion: {query}"
        try:
            answer = llm.invoke(prompt)
        except OSError as e:
            if "10061" in str(e):
                raise ConnectionError("LLM service is unreachable") from e
            raise
        except Exception as e:  # noqa: BLE001
            logging.error("LLM invocation failed: %s", e)
            raise
        sources = [d.metadata.get('source', '') for d in docs]
        return answer, sources
