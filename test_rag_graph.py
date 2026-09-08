from pathlib import Path

from src.notebook_parser import create_documents
from src.chunker import chunk_documents
from src.vector_store import create_vector_store
from src.rag_graph import build_rag_graph

DEBUG_RETRIEVAL = True

DATA_DIR = Path("data/probability")

notebook_paths = sorted(DATA_DIR.glob("chapter_*/theory/*.ipynb"))

assert notebook_paths, f"No notebooks found in {DATA_DIR}"

documents = []

for path in notebook_paths:
    documents.extend(create_documents(path))

chunks = chunk_documents(documents)
vector_store = create_vector_store(chunks)
rag_graph = build_rag_graph(vector_store)

questions = [
    "Can you prove by induction what the general Inclusion-Exclusion formula is?",
    "What is a product space?",
    "What is the difference between the multinomial distribution and the binomial distribution",
    "What is E(X + Y) equal to?",
    "What is the difference between a True Positive and a True Negative?"
]

for question in questions:
    result = rag_graph.invoke({
        "question": question
    })

    print("\nQuestion:")
    print(question)

    print("\nSufficient:")
    print(result["sufficient"])

    if DEBUG_RETRIEVAL:
        print("\nRetrieved Chunks:")
        for i, doc in enumerate(result["retrieved_docs"]):
            print(f"\n--- Chunk {i} ---")
            print("Metadata:", doc.metadata)
            print("Content:")
            print(doc.page_content)

    print("\nRetries:")
    print(result.get("retry_count", 0))

    if result.get("retry_count", 0) > 0:
        print("\nRewritten Query:")
        print(result["retrieval_query"])

    print("\nAnswer:")
    print(result["answer"])

    print("\n" + "=" * 80)