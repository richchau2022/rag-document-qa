import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_voyageai import VoyageAIEmbeddings

from ingestion.chunker import chunk_documents
from ingestion.enrich import enrich_metadata
from ingestion.loader import load_pdf


def build_vector_store(chunks: list[Document], persist_dir: Path) -> Chroma:
    """Embed chunks with Voyage and persist them in a local ChromaDB collection."""

    # For idempotency. Giving each chunk a stable ID so re-adding overwrites instead of appends
    ids = [f"{c.metadata['filename']}-{c.metadata['chunk_index']}" for c in chunks]

    embeddings = VoyageAIEmbeddings(model="voyage-3.5")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="rag_docs",
        ids=ids
    )

if __name__ == "__main__":
    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8")

    file_name = "rag-paper-lewis-2020.pdf"

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    pdf_path = PROJECT_ROOT / "data" / file_name
    document_list = load_pdf(pdf_path)
    chunked_docs = chunk_documents(document_list)
    enriched_chunks = enrich_metadata(chunked_docs, file_name)

    vector_store = build_vector_store(enriched_chunks, PROJECT_ROOT / "chroma_db")
    print(f"Vector store collection count = {vector_store._collection.count()}")