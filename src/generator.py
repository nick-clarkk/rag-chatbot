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
        temperature=0.0, #math answers are more rigid -> lower temperature for more deterministic answers
        max_tokens=500,
    )

    # prepare the context for the LLM
    context = "\n\n---\n\n".join(doc.page_content for doc in context_documents)

    prompt = f"""
You are an AI education assistant answering student questions using course materials.

Use only the provided context to answer the question.
Answer clearly and concisely.
Do not add information that is not supported by the provided context and course materials.
If the context is not sufficient to answer the question, say that the provided course materials do not contain enough
information to answer the question confidently.

Question:
{question}

Context:
{context}
"""

    # generate the answer
    response = llm.invoke(prompt)

    return response.content