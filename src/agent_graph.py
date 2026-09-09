from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from src.tools.course_retrieval import search_course_materials


tools = [search_course_materials]

llm = ChatOpenAI(
    model="gpt-5.6-terra",
    temperature=0.0,
    max_tokens=1200,
    reasoning_effort="none",
)

llm_with_tools = llm.bind_tools(tools)


SYSTEM_PROMPT = """
You are an educational assistant with access to the student's probability course materials.

Use your own knowledge for general probability questions when no specific
course source is required.

Use the course-material search tool when the user asks for information tied
to the course corpus, including:
- exercises or practice problems from the course
- summaries of chapters or sections
- how the textbook explains or defines something
- course-specific notation, proofs, examples, or terminology
- what topics the course covers
- material relevant to the student's course or exam preparation
- comparisons involving the course material

When using the course-material tool:
- use content_type="theory" for explanations, summaries, proofs, definitions,
  or other instructional material
- use content_type="exercises" for course exercises or practice problems
- use content_type="both" when the request explicitly needs both

Do not search the probability-course corpus for unrelated subjects.
If a question can be answered from general knowledge and is not asking about
the course materials, answer directly.
"""


def agent_node(state: MessagesState):
    # MessagesState - keeps conversation/tool-call history
    
    messages = [
        ("system", SYSTEM_PROMPT),
        *state["messages"],
    ]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


def build_agent_graph():
    graph = StateGraph(MessagesState)

    #Decide which tool to use
    graph.add_node(
        "agent",
        agent_node,
    )

    # executes tool
    graph.add_node(
        "tools",
        ToolNode(tools),
    )

    graph.add_edge(
        START,
        "agent",
    )

    # Routes to tool node if LLM requested a tool
    graph.add_conditional_edges(
        "agent",
        tools_condition,
    )

    # Inspect tool result -> decide next step
    graph.add_edge(
        "tools",
        "agent",
    )

    return graph.compile()