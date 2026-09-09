from pathlib import Path

from src.notebook_parser import create_documents
from src.chunker import chunk_documents


SAMPLE_FILES = [
    Path("data/probability/chapter_08_expectation/theory/01_Definition.ipynb"),
    Path("data/probability/chapter_08_expectation/theory/02_Applying_the_Definition.ipynb"),
]


def main():
    documents = []

    for path in SAMPLE_FILES:
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


if __name__ == "__main__":
    main()