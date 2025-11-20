import time
import pytest
from pathlib import Path
from langchain_core.documents import Document

from backend.main import load_last_file_from_directory
from backend.main import split_documents

def test_load_last_file_success(tmp_path):
    """
    Test that load_last_file_from_directory returns a non-empty list of 
    Document objects when valid supported files exist in the directory.

    Steps:
    - Create .txt, .pdf and .csv files in a temporary directory.
    - Update the CSV file last so it becomes the most recently modified.
    - Call the function and ensure:
        * the result is a list
        * the list is not empty
        * the first element is a Document instance
    """
    
    file1 = tmp_path / "a.txt"
    file2 = tmp_path / "b.pdf"
    file3 = tmp_path / "c.csv"

    file1.write_text("Hello TXT")
    file2.write_bytes(b"%PDF-1.4")  
    file3.write_text("col1,col2,col3\nvalue1,value2,value3")
    time.sleep(0.1)
    file3.write_text("col1,col2,col3,col4\nvalue1,value2,value3,value4") 

    docs = load_last_file_from_directory(str(tmp_path))

    assert isinstance(docs, list)
    assert len(docs) > 0
    assert isinstance(docs[0], Document)


def test_load_last_file_raises_if_directory_not_exist():
    """
    Test that load_last_file_from_directory raises a ValueError 
    when the provided directory does not exist.

    The test passes if:
    - A ValueError is raised during the function call.
    """
    with pytest.raises(ValueError):
        load_last_file_from_directory("unknown_directory_xyz")


def test_load_last_file_raises_if_no_supported_files(tmp_path):
    """
    Test that load_last_file_from_directory raises a ValueError 
    when the directory exists but contains no supported files 
    (.txt, .pdf, .csv).

    Steps:
    - Create an unsupported file (e.g. .png)
    - Call the function and expect a ValueError
    """
    
    (tmp_path / "image.png").write_bytes(b"fakeimage")

    with pytest.raises(ValueError):
        load_last_file_from_directory(str(tmp_path))
        
        
def test_split_documents_basic_split():
    """
    Test that split_documents correctly splits a long document into multiple 
    chunks according to chunk_size and chunk_overlap.

    Steps:
    - Create a document with long text (e.g. 1000 characters)
    - Call split_documents with chunk_size=300 and chunk_overlap=50
    - Ensure:
        * Returned value is a list
        * More than one chunk is generated
        * Each chunk is a Document instance
        * Chunks respect the maximum size
    """
    
    long_text = "A" * 1000  
    doc = Document(page_content=long_text)

    chunks = split_documents([doc], chunk_size=300, chunk_overlap=50)

    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert isinstance(chunks[0], Document)
    assert all(len(chunk.page_content) <= 300 for chunk in chunks)


def test_split_documents_overlap_respected():
    """
    Test that the overlap between two consecutive chunks is equal to 
    the chunk_overlap parameter.

    Steps:
    - Create a long document
    - Split it with chunk_size=100, chunk_overlap=20
    - Verify:
        * Consecutive chunks share the same 20-character overlap section
    """
    
    text = "1234567890" * 20  
    doc = Document(page_content=text)

    chunk_size = 100
    chunk_overlap = 20
    chunks = split_documents([doc], chunk_size, chunk_overlap)

    first_chunk = chunks[0].page_content
    second_chunk = chunks[1].page_content

    assert first_chunk[-chunk_overlap:] == second_chunk[:chunk_overlap]


def test_split_documents_empty_input():
    """
    Test that split_documents returns an empty list when given an empty list 
    of documents.

    The function should not throw errors for empty input.
    """

    chunks = split_documents([])

    assert isinstance(chunks, list)
    assert len(chunks) == 0