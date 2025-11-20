Retrieval-Augmented Generation (RAG) Pipeline with FastAPI
Overview

This project implements a complete Retrieval-Augmented Generation (RAG) system, including:

-> Document ingestion and storage
-> File loading and text extraction (.txt, .pdf, .csv)
-> Document chunking
-> Embedding generation and vector search
-> Large Language Model (LLM) response generation
-> REST API endpoints for document upload and question answering

The backend is built using Python, LangChain, and FastAPI.
The application exposes two primary POST endpoints:
/upload: Uploads and stores documents for indexing
/ask: Processes user queries using the RAG pipeline

The project also includes automated tests using pytest and FastAPI’s TestClient.

Features
Document Upload
-> Supports .txt, .pdf, and .csv files.
Documents are stored inside the docs/ directory and become available to the RAG pipeline.

RAG Pipeline
The backend performs the following operations:
⁕ Loads the most recently uploaded file
⁕ Splits the document into overlapping chunks
⁕ Creates text embeddings
⁕ Retrieves the most relevant chunks
⁕ Generates a contextual answer using an LLM

REST API
Two endpoints are provided:
- POST /upload
- POST /ask
Both endpoints return structured JSON responses.

Automated Testing
Tests cover:
File loading logic
Error handling
Document splitting
RAG endpoint behavior
File upload endpoint functionality

Installation
Clone the repository:
git clone <repository_url>
cd RAG

Create a virtual environment:
python -m venv .venv

Activate the virtual environment:
powershell:
.venv\Scripts\Activate.ps1

macOS/Linux:
source .venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Running the Application
Start the FastAPI server:
uvicorn backend.uploadFiles:app --reload

The API will be available at:
http://localhost:8000

Swagger documentation (auto-generated):
http://localhost:8000/docs


Running Tests
Run all tests:
pytest -v

Run a specific test:
pytest tests/test_main.py::test_split_documents_basic_split -v


Known Limitations

1.CSV documents require valid header rows to be parsed correctly
2.Only the latest uploaded file is used; multiple documents are not indexed
3.Vector store persistence behavior depends on runtime environment
4.Error messages returned by /ask are not yet standardized into an exception handler

These are planned improvements for future versions.
