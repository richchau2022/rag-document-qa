import sys

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

def load_pdf(file_path: Path) -> list[Document]:
    """Load a PDF and return a list of LangChain Document objects (one per page)."""

    loader = PyPDFLoader(
        file_path=file_path,
        mode="page"
    )

    return loader.load()

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    pdf_path = PROJECT_ROOT / "data" / "rag-paper-lewis-2020.pdf"
    document_list = load_pdf(pdf_path)

    print(f"Page count: {len(document_list)}")
    print(document_list[0].metadata)
