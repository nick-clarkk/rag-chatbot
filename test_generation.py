from src.vector_store import load_vector_store
from src.generator import generate_answer


vector_store = load_vector_store()

question = "Why is expectation called the center of gravity?"

retrieved_docs = vector_store.similarity_search(
    question,
    k=3,
)

answer = generate_answer(
    question,
    retrieved_docs,
)

print("\nQuestion:")
print(question)

print("\nRetrieved Chunks:")
for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Chunk {i} ---")
    print("Metadata:", doc.metadata)
    print("Content:")
    print(doc.page_content)

print("\nAnswer:")
print(answer)