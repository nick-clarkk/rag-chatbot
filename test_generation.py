from src.notebook_parser import create_documents
from src.chunker import chunk_documents
from src.vector_store import create_vector_store
from src.generator import generate_answer

documents = create_documents("data/probability/01_Definition.ipynb")
chunks = chunk_documents(documents)
vector_store = create_vector_store(chunks)

question = "Why is expectation called the center of gravity?"

retrieved_docs = vector_store.similarity_search(question, k=3)

answer = generate_answer(question, retrieved_docs)

print("\nQuestion:", question)
print("\nAnswer:", answer)