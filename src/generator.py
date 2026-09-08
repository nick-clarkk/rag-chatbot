from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

def generate_answer(
    question: str,
    context_documents: list[Document],
) -> str:
    """
    Generate an answer to a question using the provided context documents.
    """

    # create a ChatOpenAI instance
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.0,
    )

    # prepare the context for the LLM
    context = "\n\n---\n\n".join([doc.page_content for doc in context_documents])

    prompt = f"""
You are an AI education assisant asnwering student questions using course materials.and

Use only the provided context to answer the question. If the context is not sufficient to answer the question, say that the provided course material does not contain enough
information to answer the question confidently.

Question:
{question}

Context:
{context}
"""

    # generate the answer
    response = llm.invoke(prompt)

    return response.content