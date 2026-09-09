from src.tools.course_retrieval import search_course_materials

def print_preview(result: str):
    lines = result.splitlines()

    for line in lines:
        if line.startswith("Result ") or line.startswith("Source:"):
            print(line)


def main():
    examples = [
        {
            "query": "Explain expectation",
            "content_type": "theory",
        },
        {
            "query": "Give me a practice problem about expectation",
            "content_type": "exercises",
        },
        {
            "query": "Explain expectation and give me a practice problem",
            "content_type": "both",
        },

        # Structural retrieval tests
        {
            "query": "Summarize chapter 6",
            "content_type": "theory",
            "chapter": 6,
        },
        {
            "query": "Summarize the binomial distribution",
            "content_type": "theory",
            "subchapter": "6.1",
        },
        {
            "query": "Give me everything from 6.1",
            "content_type": "both",
            "subchapter": "6.1",
        },
    ]

    for example in examples:
        result = search_course_materials.invoke(example)

        print("\n" + "=" * 80)
        print("QUERY:", example["query"])
        print("CONTENT TYPE:", example["content_type"])

        if "chapter" in example:
            print("CHAPTER:", example["chapter"])

        if "subchapter" in example:
            print("SUBCHAPTER:", example["subchapter"])

        print("\nRESULT:")
        if result.startswith("Result "):
            print_preview(result)
        else:
            print(result[:2000])


if __name__ == "__main__":
    main()