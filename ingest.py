from pathlib import Path

from src.notebook_parser import create_documents
from src.chunker import chunk_documents
from src.vector_store import create_vector_store


DATA_DIR = Path("data/probability")

notebook_paths = sorted(
    DATA_DIR.glob("chapter_*/*/*.ipynb")
)

assert notebook_paths, f"No notebooks found in {DATA_DIR}"

documents = []

for path in notebook_paths:
    documents.extend(create_documents(path))

chunks = chunk_documents(documents)

create_vector_store(chunks)

print(f"Notebooks ingested: {len(notebook_paths)}")
print(f"Section documents: {len(documents)}")
print(f"Chunks stored: {len(chunks)}")