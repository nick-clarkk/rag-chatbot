from src.notebook_parser import create_documents

documents = create_documents(
    "data/probability/01_Transitions.ipynb"
)

for i, doc in enumerate(documents):
    print("=" * 80)
    print(f"DOCUMENT {i}")
    print("METADATA:", doc.metadata)
    print()
    print(doc.page_content)
    print()