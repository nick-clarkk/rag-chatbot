from src.vector_store import load_vector_store


vector_store = load_vector_store()

query = "Why is expectation called the center of gravity?"

output = vector_store.similarity_search(
    query,
    k=3,
)

print("\nQuery:")
print(query)

for i, doc in enumerate(output):
    print(f"\n--- Result {i} ---")
    print("Metadata:", doc.metadata)
    print("Length:", len(doc.page_content))
    print("Content:")
    print(doc.page_content)