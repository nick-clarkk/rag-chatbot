import json
from pathlib import Path

from src.vector_store import load_vector_store
from src.rag_graph import build_rag_graph


QUESTIONS_FILE = Path("evaluation/questions.json")


def load_test_cases():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate():
    test_cases = load_test_cases()

    vector_store = load_vector_store()
    rag_graph = build_rag_graph(vector_store)

    sufficiency_correct = 0

    chapter_hits = 0
    chapter_total = 0

    subchapter_hits = 0
    subchapter_total = 0

    for case in test_cases:
        question = case["question"]

        result = rag_graph.invoke({
            "question": question
        })

        retrieved_docs = result["retrieved_docs"]

        retrieved_chapters = [
            doc.metadata.get("chapter")
            for doc in retrieved_docs
        ]

        retrieved_subchapters = [
            doc.metadata.get("subchapter")
            for doc in retrieved_docs
        ]

        expected_answerable = case["answerable"]
        actual_sufficient = result["sufficient"]


        # Sufficiency accuracy

        sufficiency_pass = (
            expected_answerable == actual_sufficient
        )

        if sufficiency_pass:
            sufficiency_correct += 1


        # Chapter hit@5

        chapter_pass = None

        if expected_answerable and "expected_chapter" in case:
            chapter_total += 1

            expected_chapter = case["expected_chapter"]

            chapter_pass = (
                expected_chapter in retrieved_chapters
            )

            if chapter_pass:
                chapter_hits += 1


        # Subchapter hit@5

        subchapter_pass = None

        expected_subchapters = case.get("expected_subchapters", [])

        if expected_answerable and expected_subchapters:
            subchapter_total += 1

            subchapter_pass = any(
                subchapter in retrieved_subchapters
                for subchapter in expected_subchapters
            )

            if subchapter_pass:
                subchapter_hits += 1


        # Per-question output

        print("\n" + "=" * 80)

        print("\nQuestion:")
        print(question)

        print("\nExpected answerable:")
        print(expected_answerable)

        print("\nActual sufficient:")
        print(actual_sufficient)

        print("\nSufficiency:")
        print("PASS" if sufficiency_pass else "FAIL")

        print("\nRetries:")
        print(result.get("retry_count", 0))

        if result.get("retry_count", 0) > 0:
            print("\nRewritten query:")
            print(result.get("retrieval_query"))

        print("\nRetrieved chapters:")
        print(retrieved_chapters)

        if chapter_pass is not None:
            print("\nChapter hit@5:")
            print("PASS" if chapter_pass else "FAIL")

        print("\nRetrieved subchapters:")
        print(retrieved_subchapters)

        if subchapter_pass is not None:
            print("\nSubchapter hit@5:")
            print("PASS" if subchapter_pass else "FAIL")

        print("\nAnswer:")
        print(result["answer"])


    # Final summary

    total_cases = len(test_cases)

    sufficiency_accuracy = (
        sufficiency_correct / total_cases
        if total_cases > 0
        else 0
    )

    chapter_accuracy = (
        chapter_hits / chapter_total
        if chapter_total > 0
        else 0
    )

    subchapter_accuracy = (
        subchapter_hits / subchapter_total
        if subchapter_total > 0
        else 0
    )

    print("\n" + "=" * 80)
    print("\nEVALUATION SUMMARY")

    print(
        f"\nSufficiency accuracy: "
        f"{sufficiency_correct}/{total_cases} "
        f"({sufficiency_accuracy:.1%})"
    )

    print(
        f"Chapter hit@5: "
        f"{chapter_hits}/{chapter_total} "
        f"({chapter_accuracy:.1%})"
    )

    print(
        f"Subchapter hit@5: "
        f"{subchapter_hits}/{subchapter_total} "
        f"({subchapter_accuracy:.1%})"
    )


if __name__ == "__main__":
    evaluate()