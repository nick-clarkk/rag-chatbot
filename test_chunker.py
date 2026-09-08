from src.notebook_parser import create_documents
from src.chunker import chunk_documents

notebook_paths = [
    "data/probability/01_Definition.ipynb",
    "data/probability/01_Poisson_Distribution.ipynb",
    "data/probability/01_Transitions.ipynb",
]

documents = []

for path in notebook_paths:
    # parse notebook into section-level documents
    docs = create_documents(path)
    documents.extend(docs)

# chunk the documents
chunks = chunk_documents(documents)

print(f"Total documents: {len(documents)}")
print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks[:5]): # print first 10 chunks for brevity
    print(f"\n--- Chunk {i} ---")
    print("Metadata:", chunk.metadata)
    print("Length:", len(chunk.page_content))
    print("Content:")
    print(chunk.page_content)  # print first 500 characters of the chunk