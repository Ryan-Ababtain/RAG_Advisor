# RAG Advisor

A simple Retrieval-Augmented Generation (RAG) application that allows you to define an advisor role, upload PDF or PowerPoint documents, and interact with the content using natural language instructions. The app runs entirely with local open-source models.

## Features

- Upload PDF or PPTX files
- Define a role for the AI advisor (e.g., "Legal Consultant")
- Ask instructions or questions about the uploaded document
- Advisor can provide answers using retrieved context from the document
- Built with Streamlit and LangChain
- Runs a local Hugging Face model (Mistral-7B by default)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
streamlit run app.py
```

3. Choose the advisor role and the model name, optionally provide a Hugging Face token if required, then upload a document to start interacting.

## Extensibility

The code is modular. `rag.py` contains the document processing and retrieval logic so you can easily swap out embeddings, vector stores, or LLMs.
