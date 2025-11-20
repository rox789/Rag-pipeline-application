from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
from pathlib import Path
from backend.main import run_rag
from pydantic import BaseModel
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a single document (PDF, TXT, CSV) and save it in the 'docs/' directory.

    The uploaded file is stored as-is and will be used by the RAG pipeline.
    """

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    save_path = UPLOAD_DIR / file.filename

    try:
        file_bytes = await file.read()
        save_path.write_bytes(file_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving file: {str(e)}"
        )

    return {
        "success": True,
        "message": "File uploaded successfully",
        "filename": file.filename
    }


class AskRequest(BaseModel):
    query: str


@app.post("/ask")
async def ask_question(req: AskRequest):
    """
    Process a question using the RAG pipeline.

    Input:
        {"query": "What is Artificial Intelligence?"}

    Returns:
        dict: Answer, used file, chunks, metadata.
    """

    docs_path = Path("docs")
    if not docs_path.exists():
        return {"success": False, "answer": "Docs directory not found."}

    files = sorted(
        docs_path.iterdir(),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )

    if not files:
        return {"success": False, "answer": "No files uploaded yet."}

    latest_file = files[0]

    try:
        rag_json = await asyncio.to_thread(run_rag, req.query)
    except Exception as e:
        return {
            "success": False,
            "answer": f"Error while running RAG: {str(e)}",
            "file_used": latest_file.name
        }

    return {
        "success": True,
        "file_used": latest_file.name,
        **rag_json
    }