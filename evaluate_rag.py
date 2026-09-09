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

    # Sufficiency metric
    sufficiency_correct = 0

    # Chapter retrieval metric
    chapter_hits = 0
    chapter_total = 0

    # Subchapter retrieval metric
    subchapter_hits = 0
    subchapter_total = 0

    # Intent classification metric
    intent_correct = 0
    intent_total = 0

    # Target chapter selection metric
    target_chapter_correct = 0
    target_chapter_total = 0

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

        expected_subchapters = case.get(
            "expected_subchapters",
            []
        )

        if expected_answerable and expected_subchapters:
            subchapter_total += 1

            subchapter_pass = any(
                subchapter in retrieved_subchapters
                for subchapter in expected_subchapters
            )

            if subchapter_pass:
                subchapter_hits += 1


        # Intent accuracy

        intent_pass = None

        expected_intent = case.get("expected_intent")

        if expected_intent is not None:
            intent_total += 1

            actual_intent = result.get(
                "retrieval_intent"
            )

            intent_pass = (
                actual_intent == expected_intent
            )

            if intent_pass:
                intent_correct += 1


        # Target chapter accuracy

        target_chapter_pass = None

        expected_target_chapter = case.get(
            "expected_target_chapter"
        )

        if expected_target_chapter is not None:
            target_chapter_total += 1

            actual_target_chapter = result.get(
                "target_chapter"
            )

            target_chapter_pass = (
                actual_target_chapter
                == expected_target_chapter
            )

            if target_chapter_pass:
                target_chapter_correct += 1


        # Per-question output

        case_failed = (
            not sufficiency_pass
            or chapter_pass is False
            or subchapter_pass is False
            or intent_pass is False
            or target_chapter_pass is False
        )
        if case_failed:
            print("\n" + "=" * 80)

            print("\nQuestion:")
            print(question)

            print("\nExpected answerable:")
            print(expected_answerable)

            print("\nActual sufficient:")
            print(actual_sufficient)

            print("\nSufficiency:")
            print(
                "PASS"
                if sufficiency_pass
                else "FAIL"
            )

            print("\nRetries:")
            print(
                result.get(
                    "retry_count",
                    0
                )
            )

            if result.get("retry_count", 0) > 0:
                print("\nRewritten query:")
                print(
                    result.get(
                        "retrieval_query"
                    )
                )

            print("\nRetrieved chapters:")
            print(retrieved_chapters)

            if chapter_pass is not None:
                print("\nChapter hit@5:")
                print(
                    "PASS"
                    if chapter_pass
                    else "FAIL"
                )

            print("\nRetrieved subchapters:")
            print(retrieved_subchapters)

            if subchapter_pass is not None:
                print("\nSubchapter hit@5:")
                print(
                    "PASS"
                    if subchapter_pass
                    else "FAIL"
                )

            if intent_pass is not None:
                print("\nRetrieval intent:")
                print(
                    result.get(
                        "retrieval_intent"
                    )
                )

                print("\nIntent accuracy:")
                print(
                    "PASS"
                    if intent_pass
                    else "FAIL"
                )

            if target_chapter_pass is not None:
                print("\nTarget chapter:")
                print(
                    result.get(
                        "target_chapter"
                    )
                )

                print("\nTarget chapter accuracy:")
                print(
                    "PASS"
                    if target_chapter_pass
                    else "FAIL"
                )

            print("\nAnswer:")
            print(result["answer"])

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

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

    intent_accuracy = (
        intent_correct / intent_total
        if intent_total > 0
        else 0
    )

    target_chapter_accuracy = (
        target_chapter_correct
        / target_chapter_total
        if target_chapter_total > 0
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

    print(
        f"Intent accuracy: "
        f"{intent_correct}/{intent_total} "
        f"({intent_accuracy:.1%})"
    )

    print(
        f"Target chapter accuracy: "
        f"{target_chapter_correct}/"
        f"{target_chapter_total} "
        f"({target_chapter_accuracy:.1%})"
    )


if __name__ == "__main__":
    evaluate()