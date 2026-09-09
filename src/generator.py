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
        max_tokens=650,
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
You are an educational assistant answering questions using only the provided course material.

Rules:

- Use only the provided context.
- If the context does not contain enough information to answer the question confidently, say so.
- Cite the relevant chapter and section using the source labels provided in the context.
- Never include "Source 1", "Source 2", or any other source number in the final citation.
- When citing, use a format such as:
  [Chapter 8.1 - Definition, Definition]
  [Chapter 8.6 - Exercises, Exercise 8]

- If the student asks for a practice problem, exercise, quiz question,
  or something to solve, provide an appropriate problem from the retrieved
  exercise material.
- Do NOT provide the solution unless the student explicitly asks for the
  solution, answer, walkthrough, or help solving it.
- If the student asks for both an explanation and a practice problem,
  explain the concept first and then provide the practice problem without
  revealing its solution.
- Do not offer additional help, follow-up questions, or extra exercises unless the student explicitly asks for them.

Question:
{question}

Context:
{context}

Answer:
"""

    # generate the answer
    response = llm.invoke(prompt)

    return response.content