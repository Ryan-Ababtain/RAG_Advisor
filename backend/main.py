from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from rag_engine import RagEngine

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "vectorstore"

rag = RagEngine(DOCS_DIR, INDEX_DIR)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf", ".pptx"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    dest = DOCS_DIR / file.filename
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename}

@app.post("/ingest")
async def ingest():
    rag.ingest()
    return {"status": "ingested"}

@app.get("/ask")
async def ask(query: str, model: str = "llama3"):
    try:
        answer, sources = rag.query(query, model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"answer": answer, "sources": sources}
