import pytest
from pathlib import Path


def test_upload_file_success(client, tmp_path, monkeypatch):
    """
    Test that the /upload endpoint successfully saves an uploaded file.

    Steps:
    - Replace UPLOAD_DIR in uploadFiles.py with a temporary directory (tmp_path).
    - Send a POST request including a fake text file.
    - Assert:
        * HTTP 200 is returned
        * JSON response contains success=True and correct filename
        * The file is actually saved on disk
    """

    import backend.uploadFiles  # module under test

    # Redirect UPLOAD_DIR to a temporary directory used only during this test
    monkeypatch.setattr(backend.uploadFiles, "UPLOAD_DIR", tmp_path)

    # Send fake file to API
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"Hello world!", "text/plain")},
    )

    # Validate HTTP response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["filename"] == "test.txt"

    # Validate that file was physically saved
    saved_file = tmp_path / "test.txt"
    assert saved_file.exists()
    assert saved_file.read_bytes() == b"Hello world!"


def test_upload_file_no_filename(client):
    """
    Test that /upload returns HTTP 400 when the uploaded file has no filename.

    The request simulates a broken/malformed client request.
    """

    response = client.post(
        "/upload",
        files={"file": ("", b"abc", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "No filename provided"


def test_ask_question_success(client, tmp_path, monkeypatch):
    """
    Test the /ask endpoint when:
    - A document exists in the docs directory
    - The run_rag function is mocked and returns a known fake response

    Steps:
    - Override Path("docs") to point to tmp_path
    - Create a fake file inside tmp_path
    - Mock run_rag to prevent real RAG pipeline execution
    - POST a query and validate API response structure
    """

    import backend.uploadFiles
    from pathlib import Path as RealPath

    # Mock Path("docs") so that it returns tmp_path instead of real disk path
    def fake_path(arg):
        if arg == "docs":
            return tmp_path
        return RealPath(arg)

    monkeypatch.setattr(backend.uploadFiles, "Path", fake_path)

    # Create fake document in our temporary docs folder
    fake_file = tmp_path / "demo.txt"
    fake_file.write_text("sample content")

    # Mock RAG function
    def mock_run_rag(query: str):
        return {
            "answer": "mocked answer",
            "chunks": ["chunk1", "chunk2"],
            "metadata": {"foo": "bar"},
        }

    monkeypatch.setattr(backend.uploadFiles, "run_rag", mock_run_rag)

    # Make the request
    response = client.post("/ask", json={"query": "test question"})
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["file_used"] == "demo.txt"
    assert data["answer"] == "mocked answer"
    assert data["chunks"] == ["chunk1", "chunk2"]
    assert data["metadata"] == {"foo": "bar"}


def test_ask_question_no_files(client, tmp_path, monkeypatch):
    """
    Test that /ask returns an error message when the docs directory exists
    but contains no files.
    """

    import backend.uploadFiles
    from pathlib import Path as RealPath

    # Mock Path("docs") to use empty tmp_path
    def fake_path(arg):
        if arg == "docs":
            return tmp_path
        return RealPath(arg)

    monkeypatch.setattr(backend.uploadFiles, "Path", fake_path)

    # No files inside tmp_path → should return error
    response = client.post("/ask", json={"query": "test"})
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is False
    assert data["answer"] == "No files uploaded yet."


def test_ask_question_rag_error(client, tmp_path, monkeypatch):
    """
    Test that /ask handles RAG pipeline exceptions correctly.

    Steps:
    - Create a fake document
    - Mock run_rag to raise an exception
    - Assert that API returns success=False and includes error message
    """

    import backend.uploadFiles
    from pathlib import Path as RealPath

    # Mock docs path
    def fake_path(arg):
        if arg == "docs":
            return tmp_path
        return RealPath(arg)

    monkeypatch.setattr(backend.uploadFiles, "Path", fake_path)

    # Create fake file
    fake_file = tmp_path / "doc.pdf"
    fake_file.write_bytes(b"fake pdf")

    # Mock failure inside RAG
    def mock_run_rag_fail(query: str):
        raise RuntimeError("RAG pipeline failed")

    monkeypatch.setattr(backend.uploadFiles, "run_rag", mock_run_rag_fail)

    response = client.post("/ask", json={"query": "test"})
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is False
    assert "RAG pipeline failed" in data["answer"]
    assert data["file_used"] == "doc.pdf"