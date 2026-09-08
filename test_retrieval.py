from pathlib import Path

from src.notebook_parser import create_documents
from src.chunker import chunk_documents
from src.vector_store import create_vector_store

TEST_FILE = Path(
    "data/probability/chapter_08_expectation/theory/01_Definition.ipynb"
)

assert TEST_FILE.exists(), f"Test file not found: {TEST_FILE}"

documents = create_documents(TEST_FILE)
chunks = chunk_documents(documents)

vector_store = create_vector_store(chunks)

output = vector_store.similarity_search("Why is expectation called the center of gravity?", k=3)

for i, doc in enumerate(output):
    print(f"\n--- Result {i} ---")
    print("Metadata:", doc.metadata)
    print("Length:", len(doc.page_content))
    print("Content:")
    print(doc.page_content)