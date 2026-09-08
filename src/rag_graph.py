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
    retrieved_docs: NotRequired[list[Document]]
    sufficient: NotRequired[bool]
    answer: NotRequired[str]
    retry_count: NotRequired[int]

def retrieve_node(state: RAGState, vector_store) -> dict:
    """
    Read user question -> search vector store for 3 most similar chunks -> put retrieved chunks back into the graph state
    """
    query = state.get("retrieval_query", state["question"])
    # retrieve relevant documents from the vector store
    retrieved_docs = vector_store.similarity_search(
        query,
        k=3,
    )

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
You are checking whether the retrieved course materials are sufficient to answer the student's question.

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

    graph.add_edge(
        START,
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "check_sufficiency"
    )

    graph.add_edge(
        "rewrite",
        "retrieve"
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
        "generate",
        END
    )

    graph.add_edge(
        "insufficient",
        END
    )

    return graph.compile()