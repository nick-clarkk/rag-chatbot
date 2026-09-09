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
    context_parts = []

    for i, doc in enumerate(context_documents, start=1):
        source_label = (
            f"[Source {i}: "
            f"Chapter {doc.metadata.get('subchapter')} - "
            f"{doc.metadata.get('subchapter_title')}, "
            f"{doc.metadata.get('section')}]"
        )

        context_parts.append(
            f"{source_label}\n{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are an AI education assistant answering student questions using course materials.

Use only the provided context to answer the question.

Answer clearly and concisely.

Do not add information that is not supported by the provided context and course materials.

When you use information from the context, cite the chapter and section information
shown in the source label.

Do not cite sources using only labels such as [Source 1].
Instead, cite them in this format:

[Chapter 7.1 - Poisson Distribution, Poisson Probabilities]

If multiple sources support the same statement, include each citation separately.

If the context is not sufficient to answer the question, say that the provided course materials do not contain enough information to answer the question confidently.

Question:
{question}

Context:
{context}
"""

    # generate the answer
    response = llm.invoke(prompt)

    return response.content