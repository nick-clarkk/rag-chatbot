from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

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

    vector_store.add_documents(documents)

    return vector_store