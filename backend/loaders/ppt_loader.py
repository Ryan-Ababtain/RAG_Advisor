from unstructured.partition.pptx import partition_pptx
from langchain.schema import Document
from pathlib import Path


class PPTLoader:
    """Loader for PowerPoint files."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self):
        elements = partition_pptx(filename=str(self.file_path))
        text = "\n".join(el.text for el in elements if hasattr(el, "text"))
        return [Document(page_content=text, metadata={"source": str(self.file_path)})]
