from pathlib import Path
from src.notebook_parser import create_documents

TEST_FILE = Path(
    "data/probability/chapter_08_expectation/theory/01_Definition.ipynb"
)

assert TEST_FILE.exists(), f"Test file not found: {TEST_FILE}"

documents = create_documents(TEST_FILE)

for i, doc in enumerate(documents):
    print("=" * 80)
    print(f"DOCUMENT {i}")
    print("METADATA:", doc.metadata)
    print()
    print(doc.page_content)
    print()