from pathlib import Path

from src.notebook_parser import create_documents
from src.chunker import chunk_documents
from src.vector_store import create_vector_store
from src.generator import generate_answer


TEST_FILE = Path(
    "data/probability/chapter_08_expectation/theory/01_Definition.ipynb"
)

assert TEST_FILE.exists(), f"Test file not found: {TEST_FILE}"

documents = create_documents(TEST_FILE)
chunks = chunk_documents(documents)
vector_store = create_vector_store(chunks)

question = "Why is expectation called the center of gravity?"

retrieved_docs = vector_store.similarity_search(question, k=3)

answer = generate_answer(question, retrieved_docs)

print("\nQuestion:", question)
print("\nAnswer:", answer)