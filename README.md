# Educational RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot built over UC Berkeley's Data 140 textbook created by Ani Adhikari and Jim Pitman. 

Source: https://data140.org/textbook/

The project ingests course content from Jupyter notebooks, converts it into structured LangChain documents, chunks and embeds the material into a vector store, retrieves relevant passages for a user question, and uses LangGraph to control whether the retrieved context is sufficient before generating a final answer.

This project was built primarily to explore the design and implementation of a complete RAG pipeline rather than relying on a single prebuilt document loader or retrieval chain.

## Architecture

```text
Course Materials (.ipynb)
        |
        v
 notebook_parser.py
        |
        v
LangChain Documents
 + metadata
        |
        v
    chunker.py
        |
        v
 vector_store.py
        |
        v
     Chroma
        |
        v
   User Question
        |
        v
   Retrieval Step
        |
        v
 Retrieved Context
        |
        v
LangGraph Sufficiency Check
        |
   +----+----+
   |         |
Enough    Insufficient
   |         |
   |     Retrieve Again
   |         |
   +----+----+
        |
        v
   Final Answer
```

## Current Features

- Parses Jupyter Notebook (`.ipynb`) course materials.
- Converts notebook content into LangChain `Document` objects.
- Preserves useful metadata for downstream retrieval.
- Distinguishes different types of course content using metadata such as `content_type`.
- Splits documents into retrieval-sized chunks.
- Merges undersized chunks to avoid embedding very small fragments.
- Generates vector embeddings and stores them in Chroma.
- Retrieves relevant course passages for student questions.
- Uses LangGraph to structure the RAG control flow.
- Uses an LLM-based sufficiency check to determine whether the initial retrieved context adequately addresses the user's question.
- Performs an additional retrieval step when context is judged insufficient.
- Generates answers grounded in retrieved course materials.

## Project Structure

```text
project/
│
├── notebook_parser.py
│   └── Parses notebook content and creates structured documents.
│
├── chunker.py
│   └── Splits documents into retrieval-sized chunks and handles
│       undersized chunks.
│
├── vector_store.py
│   └── Creates embeddings, manages the Chroma vector store,
│       and provides retrieval functionality.
│
├── ingest.py
│   └── Runs the ingestion pipeline from source documents through
│       chunking and vector-store creation.
│
├── rag_graph.py
│   └── Defines the LangGraph retrieval, sufficiency-checking,
│       and answer-generation workflow.
│
└── README.md
```

## Data Pipeline

### 1. Notebook Parsing

The source material includes Jupyter notebooks rather than plain text files.

Because `.ipynb` files contain structured JSON representing markdown cells, code cells, outputs, and notebook metadata, the project uses a custom parser rather than treating notebooks as raw text.

The parser extracts relevant educational content and converts it into LangChain `Document` objects.

Metadata is attached during parsing so that information about the source and type of content remains available later in the retrieval pipeline.

For example, content such as theory and exercises can share the same parsing pipeline while remaining distinguishable through metadata such as:

```python
content_type = "theory"
```

or

```python
content_type = "exercise"
```

This avoids maintaining separate ingestion systems for structurally similar content while preserving distinctions that may be useful during retrieval or filtering.

## Chunking

After parsing, documents are divided into smaller chunks suitable for embedding and retrieval.

Chunking is necessary because embedding entire notebooks or large sections would produce overly broad representations and make it harder to retrieve the specific passage needed for a question.

The chunking stage also includes logic for handling very small chunks.

Instead of embedding short fragments independently, undersized chunks can be merged with nearby content when appropriate. This reduces the number of low-information vectors stored in the database and helps preserve useful context.

The goal is to balance:

- retrieval specificity,
- semantic coherence,
- sufficient surrounding context,
- and vector-store quality.

## Embeddings and Vector Storage

The processed chunks are converted into vector embeddings and stored using Chroma.

At query time, the user's question is embedded and compared against the stored course-material vectors.

The most relevant chunks are returned as context for the language model.

This separates two responsibilities:

```text
Vector retrieval:
"What parts of the course material are relevant?"

LLM generation:
"How should those passages be used to answer the question?"
```

## LangGraph Workflow

LangGraph is used to structure the retrieval-and-generation process.

The current system is intentionally closer to a controlled workflow than a fully autonomous agent.

A typical query follows this sequence:

```text
User Question
     |
     v
Retrieve Course Material
     |
     v
Evaluate Context Sufficiency
     |
     +-----------------------+
     |                       |
 sufficient              insufficient
     |                       |
     |                 retrieve again
     |                       |
     +-----------+-----------+
                 |
                 v
           Generate Answer
```

The sufficiency step allows the LLM to judge whether the retrieved passages contain enough information to answer the user's question.

If the context is insufficient, the system performs another retrieval pass before generating the final response.

This introduces model-based decision making while keeping the overall control flow constrained and predictable.

## Workflow vs. Agent

The current implementation should be considered a **LangGraph-based RAG workflow**, not a fully autonomous AI agent.

The model can make a limited decision about whether retrieved context is sufficient, but it cannot dynamically choose from a broad collection of actions.

For example, the current system cannot independently decide to:

- search the web,
- run Python,
- create a visualization,
- inspect an image,
- call an external service,
- or choose among multiple unrelated tools.

These limitations are intentional in the current version and provide a clear foundation for future experimentation with more agentic architectures.

## Example Questions

The system is designed for student-style questions grounded in the ingested material, such as:

```text
Explain the Central Limit Theorem.

What are the differences between the multinomial and binomial distributions?

How does backpropagation update weights in earlier layers?

Why would we use a Poisson distribution?

Show me how the Metropolis Algorithm solves the detailed balance equations.
```

Because the current system treats the course corpus as its source of truth, it is best suited for questions whose answers are represented in the ingested course materials.

## Design Decisions

### Custom Notebook Parsing

A custom notebook parser provides more control over what information enters the retrieval pipeline than simply flattening notebook files into raw text.

It allows notebook structure and useful metadata to be preserved while removing or ignoring information that is not useful for semantic retrieval.

### Shared Parsing Pipeline

Different categories of educational material are processed through the same general pipeline when their underlying structure is similar.

Metadata such as `content_type` is used to distinguish them later rather than duplicating parsing logic.

### Minimum Chunk Handling

Very small text fragments often contain too little semantic information to make useful standalone retrieval units.

The chunking pipeline therefore includes logic for preventing or merging undersized chunks.

### Retrieval Sufficiency Check

A single retrieval pass is not always enough.

Instead of immediately answering from whatever context is returned, LangGraph allows the system to evaluate whether the retrieved information appears sufficient and perform another retrieval pass when necessary.

### Controlled Architecture

The current workflow is deliberately constrained.

This makes the behavior easier to understand, debug, and evaluate while building familiarity with RAG and LangGraph fundamentals.

## Technologies

- Python
- LangChain
- LangGraph
- Chroma
- OpenAI API
- Jupyter Notebook

## Limitations

The current version has several intentional limitations:

- Course materials are the primary source of truth.
- The system cannot search the web for missing or current information.
- Retrieval occurs within a predefined workflow rather than being selected dynamically as a tool.
- The model does not have access to Python execution or visualization tools.
- The application does not currently support multimodal input.
- The system does not maintain long-term user memory.
- The model does not learn or update its weights from conversations.
- Tool selection is not currently agent-directed.
- Evaluation is primarily qualitative rather than based on a dedicated benchmark suite.

These provide natural directions for future development.

## Future Directions

Potential extensions include:

- converting course retrieval into an agent-callable tool,
- allowing the model to dynamically select tools based on user intent,
- adding web search for information outside the course corpus,
- adding Python execution for calculations and simulations,
- generating visualizations for mathematical and machine-learning concepts,
- supporting image-based questions,
- building an evaluation dataset for tool routing and answer quality,
- experimenting with prompt optimization,
- and investigating standards such as Model Context Protocol (MCP).

## Project Goal

The goal of this project is not only to build a functioning chatbot, but to understand the components and architectural decisions involved in modern RAG systems.

The project was therefore built incrementally, with individual components for parsing, chunking, vector storage, ingestion, retrieval, graph orchestration, and generation rather than hiding the entire process behind a single high-level abstraction.

This makes it possible to reason about how information moves through the system, identify retrieval and generation failure modes, and evaluate where a deterministic RAG workflow is appropriate versus where greater model agency may be beneficial.