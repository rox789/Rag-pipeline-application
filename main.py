from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document
import os

INDEX_DIR = "db/faiss_index"

LLM_INSTANCE = None


def load_last_file_from_directory(directory_path: str) -> List[Document]:
    """
    Load the most recently modified file from a directory.

    Args:
        directory_path (str): Path to the document directory.

    Returns:
        List[Document]: A list of loaded LangChain Document objects.

    Raises:
        ValueError: If directory does not exist or contains no supported files.
    """
    docs_path = Path(directory_path)

    if not docs_path.exists():
        raise ValueError(f"Directory {directory_path} does not exist")
    
    all_files = [
        f for f in docs_path.glob("*")
        if f.suffix in [".txt", ".pdf", ".csv"] and f.is_file()
    ]

    if not all_files:
        raise ValueError(f"No supported documents found in {directory_path}")

    last_file = max(all_files, key=lambda f: f.stat().st_mtime)

    if last_file.suffix == ".txt":
        loader = TextLoader(str(last_file), encoding="utf-8")
    elif last_file.suffix == ".pdf":
        loader = PyPDFLoader(str(last_file))
    elif last_file.suffix == ".csv":
        loader = CSVLoader(str(last_file))

    print(f"Using latest file: {last_file.name}")
    return loader.load()


def split_documents(
    documents: List[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Split documents into smaller overlapping chunks.

    Args:
        documents (List[Document]): Documents to split.
        chunk_size (int): Maximum chunk size.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[Document]: Split document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def load_embeddings(model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> HuggingFaceEmbeddings:
    """
    Load a HuggingFace embedding model.

    Args:
        model_name (str): HuggingFace model ID.
        device (str): Device to run on ("cpu" or "cuda").

    Returns:
        HuggingFaceEmbeddings: Embedding model wrapper.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device}
    )


def create_vector_store(
    splits: List[Document],
    index_dir: str = INDEX_DIR
) -> FAISS:
    """
    Build a FAISS vector store from document chunks and save it.

    Args:
        splits (List[Document]): Document chunks.
        index_dir (str): Directory to save the index.

    Returns:
        FAISS: The created vector store.
    """
    embeddings = load_embeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    Path(index_dir).parent.mkdir(parents=True, exist_ok=True)
    return vector_store

def load_llm(
    model_name: str = "google/flan-t5-base",
    max_new_tokens: int = 512,
    min_length: int = 10,
    temperature: float = 0.3,
    do_sample: bool = False
) -> HuggingFacePipeline:
    """
    Load a lightweight Flan-T5 language model.

    Args:
        model_name (str): HuggingFace model ID.
        max_new_tokens (int): Max tokens for generation.
        min_length (int): Minimum output length.
        temperature (float): Sampling temperature.
        do_sample (bool): Sampling flag.

    Returns:
        HuggingFacePipeline: Wrapped LLM.
    """
    hf_pipeline = pipeline(
        "text2text-generation",
        model=model_name,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        temperature=temperature,
        do_sample=do_sample
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)


def get_llm():
    """
    Retrieve a cached LLM instance, loading it if necessary.

    Returns:
        HuggingFacePipeline: Cached LLM instance.
    """
    global LLM_INSTANCE
    if LLM_INSTANCE is None:
        print("Loading LLM into memory...")
        LLM_INSTANCE = load_llm()
    return LLM_INSTANCE


def run_rag(
    query: str,
    docs_dir: str = "docs",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Execute a Retrieval-Augmented Generation pipeline.

    Steps:
        1. Load document.
        2. Split into chunks.
        3. Use existing FAISS index or build it.
        4. Retrieve context.
        5. Generate answer using LLM.
        6. Return structured JSON.

    Args:
        query (str): User query.
        docs_dir (str): Directory containing documents.
        chunk_size (int): Chunk size.
        chunk_overlap (int): Overlap size.
        top_k (int): Number of chunks to retrieve.

    Returns:
        Dict[str, Any]: RAG output.
    """
    documents = load_last_file_from_directory(docs_dir)

    splits = split_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    vector_store = create_vector_store(splits)

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    relevant_docs: List[Document] = retriever.invoke(query)

    if not relevant_docs:
        return {
            "answer": None,
            "message": "No relevant information found.",
            "chunks_used": [],
            "sources": [],
            "model": "google/flan-t5-base",
            "top_k": top_k
        }

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    llm = get_llm()

    prompt = f"""
    You are a question-answering assistant that must use ONLY the information in the context below.

    Rules:
    - If the answer is not clearly stated or strongly implied by the context, answer exactly: "I cannot answer this based on the uploaded document."
    - Do NOT use outside knowledge.
    - Do NOT invent names, dates, definitions or examples that are not in the context.
    - Do NOT repeat or quote the whole context.
    - Write a short, direct answer (1-3 sentences).

    Context:
    {context}

    Question: {query}
    """

    response_text = llm.invoke(prompt)

    return {
        "answer": response_text,
        "query": query,
        "context_chunks": [doc.page_content for doc in relevant_docs],
        "sources": [doc.metadata for doc in relevant_docs],
        "unique_files": list({doc.metadata.get("source") for doc in relevant_docs}),
        "model": "google/flan-t5-base",
        "top_k": top_k,
        "total_documents_loaded": len(documents),
        "total_chunks_created": len(splits),
    }


if __name__ == "__main__":
    test_query = "How has Artificial Intelligence evolved over time?"

    response_json = run_rag(
        query=test_query,
        docs_dir="docs",
        chunk_size=300,
        chunk_overlap=50,
        top_k=3
    )

    import json
    print(json.dumps(response_json, indent=4, ensure_ascii=False))
    
    
def run_rag_with_memory(
    query: str,
    memory: List[Dict[str, str]],
    docs_dir: str = "docs",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    RAG + Conversational Memory

    Args:
        query: user question
        memory: list of {"role": "user"/"assistant", "content": "..."} messages
        docs_dir: folder with uploaded docs
        top_k: number of retrieved chunks

    Returns:
        dict with answer, context_chunks, and updated memory
    """

    # 1. Load last document
    documents = load_last_file_from_directory(docs_dir)

    # 2. Split into chunks
    splits = split_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # 3. Create FAISS VectorStore
    vector_store = create_vector_store(splits)

    # 4. Retrieve context
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    relevant_docs: List[Document] = retriever.invoke(query)

    context = "\n\n".join([d.page_content for d in relevant_docs])

    # 5. Prepare memory text (last 10 turns)
    memory_text = ""
    for m in memory[-10:]:
        memory_text += f"{m['role'].capitalize()}: {m['content']}\n"

    # 6. LLM
    llm = get_llm()

    prompt = f"""
You are a document-based QA assistant.

### Rules:
- Use ONLY the context below.
- Use conversation history only to keep consistency.
- If answer is not in the document, reply exactly:
  "I cannot answer this based on the uploaded document."
- Keep answers short (1-3 sentences).

### Conversation History:
{memory_text}

### Document Context:
{context}

### Question:
{query}
"""

    response_text = llm.invoke(prompt)

    # 7. Update memory
    updated_memory = memory + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response_text}
    ]

    return {
        "answer": response_text,
        "context_chunks": [d.page_content for d in relevant_docs],
        "memory": updated_memory
    }