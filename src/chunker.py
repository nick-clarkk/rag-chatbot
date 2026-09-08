from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(documents, chunk_size=1000, chunk_overlap=150, min_chunk_size=300) -> list[Document]:
    """
    Split section-level LangChain Documents into smaller retrieval chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    #splits Documents into chunks of text (smaller documents) with metadata preserved
    chunks = splitter.split_documents(documents)

    merged_chunks= []

    for chunk in chunks:
        # merged chunks that are smaller than the min size
        if (
            merged_chunks
            and len(chunk.page_content) < min_chunk_size
            and chunk.metadata["source"] == merged_chunks[-1].metadata["source"]
            and chunk.metadata["section"] == merged_chunks[-1].metadata["section"]
        ): 
            # merge the current chunk with the last chunk in merged_chunks
            merged_chunks[-1].page_content += "\n\n" + chunk.page_content
        else:
            merged_chunks.append(chunk)

    # assign chunk index to each chunk for reference
    for i, chunk in enumerate(merged_chunks):
        chunk.metadata["chunk_index"] = i

    return merged_chunks