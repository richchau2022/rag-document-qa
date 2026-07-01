import re
import sys
from pathlib import Path

from langchain_core.documents import Document

from ingestion.chunker import chunk_documents
from ingestion.loader import load_pdf


def enrich_metadata(chunks: list[Document], filename: str) -> list[Document]:
    """Add filename, a running chunk_index, and a best-effort section header to each chunk."""

    # Regex + carry-forward heuristic strategy (for learning purposes)
    # PDFs encode visual layout, not semantic structure - heading information is lost on extraction
    # Limitations: chunk-level granularity, generic pattern coverage, and extraction dependent
    section_pattern = (
        r"^\s*("  # start of line, optional indent
        r"\d+(?:\.\d+)*\s+[A-Z][\w -]+?"  # numbered:  "2.1 Models"
        r"|[A-Z]\s+[A-Z][\w -]+?"  # lettered:  "A Implementation Details"
        r"|References|Broader Impact|Acknowledgments"  # named literals
        r")\s*$"  # end of line
    )

    current_section = "unknown"

    for index, chunk in enumerate(chunks):
        matches = list(re.finditer(section_pattern, chunk.page_content, re.MULTILINE))

        chunk.metadata["filename"] = filename
        chunk.metadata["chunk_index"] = index

        if matches:
            # Uses the last heading in the chunk since multiple headers can land in a chunk
            current_section = matches[-1].group(0).strip()
            chunk.metadata["section"] = current_section
        else:
            chunk.metadata["section"] = current_section

    return chunks


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    file_name = "rag-paper-lewis-2020.pdf"

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    pdf_path = PROJECT_ROOT / "data" / file_name
    document_list = load_pdf(pdf_path)

    chunked_docs = chunk_documents(document_list)
    enriched_chunks = enrich_metadata(chunked_docs, file_name)

    for enriched_chunk in enriched_chunks:
        print(enriched_chunk.metadata["section"])