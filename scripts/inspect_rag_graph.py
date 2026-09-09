from src.vector_store import load_vector_store
from src.rag_graph import build_rag_graph


DEBUG_RETRIEVAL = False



def main():
    vector_store = load_vector_store()
    rag_graph = build_rag_graph(vector_store)

    questions = [
        "Explain expectation",
        "Give me a practice problem about expectation",
        "Explain expectation and then give me a practice problem",
    ]

    for question in questions:
        result = rag_graph.invoke({
            "question": question
        })

        print("\nRetrieval Intent:")
        print(result.get("retrieval_intent"))

        print("\nTarget Chapter:")
        print(result.get("target_chapter"))

        print("\nRetrieved Content Types:")
        print([
            doc.metadata.get("content_type")
            for doc in result["retrieved_docs"]
        ])

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


if __name__ == "__main__":
    main()