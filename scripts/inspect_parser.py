from pathlib import Path
from src.notebook_parser import create_documents

SAMPLE_FILE = Path(
    "data/probability/chapter_08_expectation/theory/01_Definition.ipynb"
)


def main():
    assert SAMPLE_FILE.exists(), f"Test file not found: {SAMPLE_FILE}"

    documents = create_documents(SAMPLE_FILE)

    for i, doc in enumerate(documents):
        print("=" * 80)
        print(f"DOCUMENT {i}")
        print("METADATA:", doc.metadata)
        print()
        print(doc.page_content)
        print()


if __name__ == "__main__":
    main()