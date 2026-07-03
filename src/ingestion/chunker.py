import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.loader import load_pdf


def chunk_documents(docs: list[Document], chunk_size: int = 512, chunk_overlap: int = 75) -> list[Document]:
    """Split page Documents into token-sized chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    pdf_path = PROJECT_ROOT / "data" / "rag-paper-lewis-2020.pdf"
    document_list = load_pdf(pdf_path)

    chunked_docs = chunk_documents(document_list)
    print(f"Number of chunks = {len(chunked_docs)}")
    print(f"Sample chunk: {chunked_docs[0].page_content[:200]}")
    print(f"Metadata: {chunked_docs[0].metadata}")