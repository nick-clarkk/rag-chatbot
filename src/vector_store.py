from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import hashlib

load_dotenv()  # Load environment variables from .env file


def create_vector_store(
    documents: list[Document],
    persist_directory: str = "chroma_db",
):
    """
    Create a Chroma vector store from a list of LangChain Documents.
    """

    # create embeddings for the documents
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # create a Chroma vector store
    vector_store = Chroma(
        collection_name = "course_materials",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    ids = []

    for doc in documents:
        content_hash = hashlib.sha256(
            doc.page_content.encode("utf-8")
        ).hexdigest()[:16]

        doc_id = (
            f'{doc.metadata["chapter"]}:'
            f'{doc.metadata["subchapter"]}:'
            f'{doc.metadata["source"]}:'
            f'{doc.metadata["section"]}:'
            f'{doc.metadata["start_index"]}:'
            f'{content_hash}'
        )

        ids.append(doc_id)

    vector_store.add_documents(
        documents = documents,
        ids = ids,
    )

    return vector_store


def load_vector_store(    
    persist_directory: str = "chroma_db",
):
    """Load existing vector store if available instead of recreating anytime new data becomes available."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return Chroma(
        collection_name="course_materials",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )