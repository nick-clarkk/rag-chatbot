from typing import TypedDict, NotRequired 
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from src.generator import generate_answer
from langgraph.graph import StateGraph, START, END

class RAGState(TypedDict):
    """
    State passed through the RAG graph. 

    Only the question exists (as provided by the user) at the start. 
    The other fields are populated naturally by the nodes as the graph runs.
    """
    question: str
    retrieval_query: NotRequired[str] # prompt it ends up using for generation
    retrieval_intent: NotRequired[str] # theory, exercise, or both
    retrieved_docs: NotRequired[list[Document]]
    target_chapter: NotRequired[int]
    sufficient: NotRequired[bool]
    answer: NotRequired[str]
    retry_count: NotRequired[int]


def classify_retrieval_intent_node(state: RAGState) -> dict:
    """
    Classify whether the user's request needs theory, exercises, or both
    """

    question = state["question"]

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.0,
        max_tokens=10,
    )

    prompt = f"""
        Classify what kind of course material would best answer the student's request.

        Return only one of:

        THEORY
        EXERCISE
        BOTH

        THEORY:
        The student wants an explanation, definition, formula, proof, concept,
        or other instructional material.

        EXERCISE:
        The student wants a practice problem, exercise, quiz question,
        or something to solve.

        BOTH:
        The student explicitly wants both explanation/theory and practice material.

        Question:
        {question}
        """

    response = llm.invoke(prompt)

    retrieval_intent = response.content.strip().upper()

    return {
        "retrieval_intent": retrieval_intent
    }


def identify_target_chapter_node(state: RAGState, vector_store) -> dict:
    """
    Choose the course chapter that best matches the student's requested topic.
    """

    query = state.get("retrieval_query", state["question"])

    candidate_docs = vector_store.similarity_search(
        query,
        k=5,
        filter={"content_type": "theory"},
    )

    candidate_lines = []

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
        model="gpt-4.1-mini",
        temperature=0.0,
        max_tokens=20,
    )

    prompt = f"""
        A student is taking this probability course sequentially.

        Choose the chapter that best matches the specific topic the student is asking about.

        Use the candidate course locations below.

        Do not automatically choose the earliest chapter.
        If the question refers to a more advanced or specialized treatment of a concept,
        choose the chapter that specifically covers that treatment.

        Student request:
        {state["question"]}

        Candidate course locations:
        {candidates}

        Return only the chapter number.
        """

    response = llm.invoke(prompt)

    return {
        "target_chapter": int(response.content.strip())
    }

def retrieve_node(state: RAGState, vector_store) -> dict:
    query = state.get("retrieval_query", state["question"])
    retrieval_intent = state.get("retrieval_intent", "THEORY")

    if retrieval_intent == "THEORY":
        retrieved_docs = vector_store.similarity_search(
            query,
            k=5,
            filter={"content_type": "theory"},
        )

    elif retrieval_intent == "EXERCISE":
        retrieved_docs = vector_store.similarity_search(
            query,
            k=5,
            filter={
                "$and": [
                    {"content_type": "exercises"},
                    {"chapter": state["target_chapter"]},
                ]
            },
        )

    else:  # BOTH
        theory_docs = vector_store.similarity_search(
            query,
            k=5,
            filter={"content_type": "theory"},
        )

        exercise_docs = vector_store.similarity_search(
            query,
            k=5,
            filter={
                "$and": [
                    {"content_type": "exercises"},
                    {"chapter": state["target_chapter"]},
                ]
            },
        )

        retrieved_docs = theory_docs + exercise_docs

    return {
        "retrieved_docs": retrieved_docs,
    }

def check_sufficiency_node(state: RAGState) -> dict:
    """
    Check if the retrieved documents are sufficient to answer the question.
    """
    question = state["question"]
    retrieved_docs = state["retrieved_docs"]

    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.0, #math answers are more rigid -> lower temperature for more deterministic answers
        max_tokens = 10,
    )

    prompt = f"""
        You are checking whether retrieved course material is sufficient to answer a student's question.

        Return only:
        SUFFICIENT
        or
        INSUFFICIENT

        Rules:

        - Return SUFFICIENT only if the retrieved context contains enough information
        to answer the specific question.
        - The context does not need to use the exact wording of the question.
        A direct statement, clear paraphrase, or relevant example can be sufficient.
        - A merely related topic is not sufficient.
        - If the user asks for a use case, purpose, application, or when something is used,
        a clear statement describing what it is used to model or accomplish is sufficient.
        - If the user asks for a practice problem, exercise, quiz question, or something to solve,
        return SUFFICIENT if the retrieved context contains an appropriate problem
        that matches the requested topic.
        A solution is not required unless the user explicitly asks for the solution.
        - If the user asks for both an explanation and a practice problem,
        return SUFFICIENT if the retrieved context contains enough theory to explain
        the requested topic and also contains an appropriate problem on that topic.
        - If the user asks for an explanation, comparison, derivation, proof, or example,
        the context must contain enough information to perform that specific task.
        - A result being stated as true is not sufficient if the user asks for a proof
        and the proof is not actually contained in the context.
        - If the context explicitly says a proof or derivation is omitted or not provided,
        return INSUFFICIENT.
        - Do not use outside knowledge.

        Question:
        {question}

        Retrieved Context:
        {context}

        Respond with only one word:

        SUFFICIENT

        or 

        INSUFFICIENT
        """

    response = llm.invoke(prompt)
    sufficient = response.content.strip().upper() == "SUFFICIENT"

    return {
        "sufficient": sufficient
    }

def rewrite_query_node(state: RAGState) -> dict:
    question = state["question"]

    llm = ChatOpenAI(
        model = "gpt-4.1-mini",
        temperature=0.0,
        max_tokens=100,
    )

    prompt = f"""
Rewrite the student's question to improve semantic retrieval from course materials.

Keep the original meaning.
Use clearer or more explicit terminology when helpful.
Do not answer the question.

Original question:
{question}

Return only the rewritten question.
"""

    response = llm.invoke(prompt)

    return {
        "retrieval_query": response.content.strip(),
        "retry_count": state.get("retry_count", 0) + 1
    }

def retrieval_intent_router(state: RAGState) -> str:
    if state["retrieval_intent"] == "THEORY":
        return "retrieve"

    return "identify_target_chapter"

def router(state: RAGState) -> str:
    """
    Route the flow based on whether the retrieved documents are sufficient to answer the question.
    """
    if state["sufficient"]:
        return "generate"

    if state.get("retry_count", 0) < 1:
        return "rewrite"
    
    return "insufficient"

def generate_answer_node(state: RAGState) -> dict:
    """
    Generate an answer to the question using the retrieved documents.
    """
    answer = generate_answer(
        state["question"],
        state["retrieved_docs"],
    )

    return {
        "answer": answer
    }

def insufficient_node(state: RAGState) -> dict:
    """
    Handle the case where the retrieved documents are insufficient to answer the question.
    """
    return {
        "answer": "The retrieved course materials do not contain enough information to answer this question confidently."
    }

def build_rag_graph(vector_store):
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node(
        "classify_retrieval_intent",
        classify_retrieval_intent_node
    )

    graph.add_node(
        "identify_target_chapter",
        lambda state: identify_target_chapter_node(state, vector_store)
    )

    graph.add_node(
        "retrieve",
        lambda state: retrieve_node(state, vector_store)
    )

    graph.add_node(
        "check_sufficiency",
        check_sufficiency_node
    )

    graph.add_node(
        "generate",
        generate_answer_node
    )

    graph.add_node(
        "insufficient",
        insufficient_node
    )

    graph.add_node(
        "rewrite",
        rewrite_query_node
    )

    # Start with intent classification
    graph.add_edge(
        START,
        "classify_retrieval_intent"
    )

    # Theory can retrieve immediately.
    # Exercise/Both first need a target chapter.
    graph.add_conditional_edges(
        "classify_retrieval_intent",
        retrieval_intent_router,
        {
            "retrieve": "retrieve",
            "identify_target_chapter": "identify_target_chapter",
        }
    )

    graph.add_edge(
        "identify_target_chapter",
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "check_sufficiency"
    )

    graph.add_conditional_edges(
        "check_sufficiency",
        router,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "insufficient": "insufficient",
        }
    )

    graph.add_edge(
        "rewrite",
        "retrieve"
    )

    graph.add_edge(
        "generate",
        END
    )

    graph.add_edge(
        "insufficient",
        END
    )

    return graph.compile()