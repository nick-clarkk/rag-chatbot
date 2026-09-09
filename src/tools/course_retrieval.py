from collections import defaultdict

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from src.vector_store import load_vector_store

#If relevant chunk list is too large, often in the case of chapter summaries, summarize subchapters then combine
summary_llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0.0,
    max_tokens=700,
    reasoning_effort="none",
)

MAX_RAW_STRUCTURAL_CHUNKS = 30

def summarize_docs(docs: list[Document], label: str) -> str:
    """
    Summarize an ordered group of course chunks while preserving the
    important definitions, formulas, reasoning, examples, and conclusions.
    """

    content = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Summarize the following material from {label}.

Preserve:
- important definitions
- formulas and notation
- reasoning and conclusions
- important examples
- distinctions or caveats

Do not add information that is not present in the material.

Course material:
{content}
"""

    response = summary_llm.invoke(prompt)

    return response.content.strip()


# Summarize the subchapters
def summarize_chapter_docs(docs: list[Document]) -> str:
    """
    Group ordered chapter chunks by subchapter and summarize each
    subchapter separately.
    """

    groups = defaultdict(list)

    for doc in docs:
        subchapter = doc.metadata.get("subchapter", "Unknown")
        groups[subchapter].append(doc)

    summaries = []

    for subchapter, group_docs in groups.items():
        title = group_docs[0].metadata.get(
            "subchapter_title",
            "",
        )

        label = f"Subchapter {subchapter}: {title}"

        summary = summarize_docs(
            group_docs,
            label,
        )

        summaries.append(
            f"{label}\n{summary}"
        )

    return "\n\n".join(summaries)


vector_store = load_vector_store()


def identify_target_chapter(query: str) -> int:
    """
    Helper function. Given a query, figure out which chapter is most relevant based on theory docs.
    """
    candidate_docs = vector_store.similarity_search(
        query,
        k=5,
        filter={"content_type": "theory"},
    )

    candidate_lines = []

    # Example: Chapter 8: Expectation | 8.2 Properties of Expectation | Section: Linearity
    for doc in candidate_docs:
        candidate_lines.append(
            f"Chapter {doc.metadata['chapter']}: "
            f"{doc.metadata['chapter_title']} | "
            f"{doc.metadata['subchapter']} "
            f"{doc.metadata['subchapter_title']} | "
            f"Section: {doc.metadata['section']}"
        )

    candidates = "\n".join(candidate_lines)

    llm = ChatOpenAI(
        model="gpt-5.6-luna",
        temperature=0.0,
        max_tokens=50,
        reasoning_effort="none",
    )

    prompt = f"""
A student is taking this probability course sequentially.

Choose the chapter that best matches the specific topic the student is asking about.

Use the candidate course locations below.

If the question refers to a more advanced or specialized treatment of a concept,
choose the chapter that specifically covers that treatment.

Student request:
{query}

Candidate course locations:
{candidates}

Return only the chapter number.
"""

    response = llm.invoke(prompt)

    chapter = response.content.strip()

    if not chapter:
        raise ValueError("Chapter selection LLM returned an empty response.")

    return int(chapter)


@tool
def search_course_materials(
    query: str,
    content_type: str,
    chapter: int | None = None, #None implies optional
    subchapter: str | None = None,
) -> str:
    """
    Search the student's course materials.

    Use chapter or subchapter when the user asks about a specific section of the course, such as a chapter summary.

    If no chapter or subchapter is provided, use semantic retrieval.

    content_type must be one of:
    - theory
    - exercises
    - both
    """

    content_type = content_type.lower().strip()

    if content_type not in {"theory", "exercises", "both"}:
        return (
            "Invalid content_type. "
            "Use 'theory', 'exercises', or 'both'."
        )

    # Structural retrieval:
    # Used when the agent knows the requested chapter/subchapter.
    if chapter is not None or subchapter is not None:

        location_filter = (
            {"subchapter": subchapter}
            if subchapter is not None
            else {"chapter": chapter}
        )

        # if content_type is both, give both everything from the location (exericses and theory)
        if content_type == "both": 
            where_filter = location_filter

        else:
            where_filter = {
                "$and": [
                    {"content_type": content_type},
                    location_filter,
                ]
            }

        # get chunks that match these conditions
        raw_results = vector_store.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )

        # Pair content and metadata to rebuild Document objects as get() does not return a list of Langchain Documents
        retrieved_docs = [
            Document(
                page_content=content,
                metadata=metadata,
            )
            for content, metadata in zip(
                raw_results["documents"],
                raw_results["metadatas"],
            )
        ]

        # restore chunk order
        retrieved_docs.sort(
            key=lambda doc: doc.metadata.get("chunk_index", 0)
        )

        # Compress each subchapter separately before returning to the agent.
        if (
            chapter is not None
            and subchapter is None
            and len(retrieved_docs) > MAX_RAW_STRUCTURAL_CHUNKS):
            return summarize_chapter_docs(retrieved_docs)

    # Semantic retrieval:
    # Used when no exact chapter/subchapter is requested.
    elif content_type == "theory":
        retrieved_docs = vector_store.similarity_search(
            query,
            k=5,
            filter={"content_type": "theory"},
        )

    elif content_type == "exercises":
        target_chapter = identify_target_chapter(query)

        retrieved_docs = vector_store.similarity_search(
            query,
            k=5,
            filter={
                "$and": [
                    {"content_type": "exercises"},
                    {"chapter": target_chapter},
                ]
            },
        )

    else:  # both
        target_chapter = identify_target_chapter(query)

        theory_docs = vector_store.similarity_search(
            query,
            k=3,
            filter={
                "$and": [
                    {"content_type": "theory"},
                    {"chapter": target_chapter},
                ]
            },
        )

        exercise_docs = vector_store.similarity_search(
            query,
            k=3,
            filter={
                "$and": [
                    {"content_type": "exercises"},
                    {"chapter": target_chapter},
                ]
            },
        )

        retrieved_docs = theory_docs + exercise_docs

    if not retrieved_docs:
        return "No relevant course material was found."

    results = []

    for i, doc in enumerate(retrieved_docs):
        results.append(
            f"""Result {i + 1}
Source: {doc.metadata}
Content:
{doc.page_content}
"""
        )

    return "\n\n".join(results)