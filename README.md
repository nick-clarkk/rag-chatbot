# Probability Course Agent

A tool-using educational assistant built around UC Berkeley's Data 140 probability textbook by Ani Adhikari and Jim Pitman.

Source: https://data140.org/textbook/

The project started as a conventional retrieval-augmented generation pipeline and was later rebuilt as a bounded LangGraph agent. The current agent can decide whether to answer from model knowledge, retrieve from the course corpus, search the web, or run Python for calculations, simulations, and plots.

The main goal of the project is to understand the pieces of a modern RAG and tool-using agent system rather than hide them behind a single high-level chain.

## Architecture

```text
                         User
                          |
                          v
                    LangGraph Agent
                          |
             +------------+------------+
             |            |            |
             v            v            v
        answer        call tool     call another
        directly          |            tool
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
   Course Retrieval    Web Search       Python
          |                |                |
          +----------------+----------------+
                           |
                           v
                    Tool result(s)
                           |
                           v
                    LangGraph Agent
                           |
                    decide what next
                           |
                           v
                      Final answer
```

The graph itself is intentionally small. The model receives the available tools, decides whether one is needed, inspects any returned tool results, and can either call another tool or answer.

This is different from the earlier RAG workflow in `src/rag_workflow.py`, where retrieval and sufficiency checking followed a fixed sequence. That file is kept as a baseline for comparison.

## What the Agent Can Do

The agent can:

- answer stable conceptual questions directly from model knowledge;
- retrieve explanations, definitions, summaries, and exercises from the course corpus;
- route explicit chapter and subchapter requests through structural retrieval;
- use semantic retrieval when the user asks about a topic without naming a course location;
- distinguish theory from exercises through metadata;
- search the web for current or recent information;
- run Python for nontrivial probability calculations and simulations;
- create Matplotlib plots and return their saved paths;
- combine tools in one request when needed;
- preserve conversation history for follow-up questions;
- avoid revealing exercise solutions or hints unless the student asks for them.

Examples of multi-tool requests include:

```text
Use my textbook to explain the Gamma family from Chapter 18,
then plot Gamma distributions with shape parameters 1, 2, and 5.
```

and:

```text
What are some recent uses of Gamma distributions in machine learning,
and plot a few Gamma densities to show how the shape changes?
```

## Course Data Pipeline

### Notebook parsing

The source material is stored in Jupyter notebooks rather than plain text files.

The custom parser reads notebook structure and converts relevant content into LangChain `Document` objects while preserving metadata such as:

```text
chapter
chapter_title
subchapter
subchapter_title
section
content_type
source
start_index
```

Theory and exercises share the same parsing pipeline. They are distinguished later through `content_type` metadata instead of being handled by separate ingestion systems.

### Chunking

Parsed documents are split with a recursive text splitter.

The chunking step is designed to keep retrieval units specific enough for semantic search while preserving enough surrounding context to make each chunk useful. Very small chunks are merged with nearby content when appropriate rather than embedded as low-information fragments.

### Embeddings and Chroma

Chunks are embedded and stored in Chroma.

The vector store is used for semantic retrieval, but not every request is handled through similarity search. Explicit chapter and subchapter requests use metadata filters instead.

## Course Retrieval

`search_course_materials` supports two retrieval modes.

### Semantic retrieval

Semantic retrieval is used when the student asks about a topic without naming a chapter or subchapter.

For theory questions, the vector store returns semantically relevant theory chunks.

For exercise requests, the retrieval tool first identifies the most relevant chapter from theory candidates, then retrieves exercises from that chapter. This helps avoid retrieving an exercise that matches individual words but comes from an unrelated part of the course.

### Structural retrieval

If the user explicitly names a chapter or subchapter, the tool retrieves that course location directly through metadata.

Examples:

```text
Summarize Chapter 7
```

uses `chapter=7`.

```text
Summarize 7.1
```

uses `subchapter="7.1"`.

Large chapter requests are compressed by subchapter before being returned to the main agent so that chapter summaries do not require sending the entire raw chapter into one prompt.

## Tools

### Course retrieval

The course tool handles textbook-specific questions, chapter and section summaries, course exercises, notation, proofs, and other requests tied to the Data 140 corpus.

The agent chooses among:

```text
content_type="theory"
content_type="exercises"
content_type="both"
```

depending on the request.

### Web search

Tavily is used for questions that depend on current or recent information.

The agent is instructed not to search the web for stable general knowledge just because a question is difficult. When web search is used, the final answer includes a short source list based on returned URLs.

Recent-development answers are kept tied to the retrieved evidence rather than expanding into unsupported lists of applications.

### Python

The Python tool is intended for local numerical work, simulations, and small visualizations.

It includes lightweight AST-based restrictions that block selected functions, imports, and plotting operations. These checks are meant to reduce accidental misuse in a local project; they are not a production security sandbox.

The tool captures ordinary printed output and also handles a trailing expression in a notebook-like way:

```python
prob = ...
prob
```

so the value is returned even when the generated code does not explicitly call `print`.

For plotting, the model creates Matplotlib figures but does not decide where to save them. The tool saves open figures to `artifacts/`, assigns unique filenames, and limits the number of stored plots.

## LangGraph Agent

The main graph uses `MessagesState`, an agent node, and a shared `ToolNode`.

The loop is:

```text
START
  |
  v
Agent
  |
  +---- no tool needed ----> final answer
  |
  +---- tool call ----------> ToolNode
                               |
                               v
                              Agent
                               |
                               +---- answer
                               |
                               +---- another tool
```

Tool selection is model-directed. There is no hard-coded classifier deciding that a certain type of question must always go to a certain node.

The system prompt provides policy for when tools should be used, but the model still chooses the next action at runtime.

This makes the current system a bounded tool-using agent rather than the fixed RAG workflow used in the original version.

## Exercise Behavior

Course exercises are treated differently from ordinary explanatory questions.

When the student explicitly asks for a course exercise, the agent retrieves one from the exercise corpus and presents the problem without exposing the stored solution, hint, setup, or strategy.

If the student asks for a generic practice problem without requesting one from the course, the agent may generate a fresh problem instead.

This avoids repeatedly serving the same finite set of textbook exercises while still allowing students to request authentic course problems when they want them.

## Evaluation

The project includes a deterministic evaluation suite for agent behavior.

`tests/eval_cases.py` contains regression and fresh evaluation cases covering:

- direct answers with no tools;
- course retrieval;
- chapter and subchapter routing;
- theory versus exercise retrieval;
- fresh versus course-sourced practice problems;
- web search;
- Python computation;
- plotting;
- multi-tool requests;
- explicit tool restraint;
- numerical answer checks.

The evaluator checks behavior such as:

```text
required tools used
forbidden tools avoided
important tool arguments correct
chapter/subchapter routing correct
expected plot files created
Sources section present when required
numeric result within tolerance
final answer produced
```

The tests do not require one exact tool-call trajectory. For example, a web question may perform more than one search if the agent decides another search is useful.

This keeps the evaluation focused on required behavior without turning the agent back into a fixed workflow.

### Failure-driven changes

The evaluation suite has also been used to make small prompt and tool changes based on observed failures.

One example was a hypergeometric probability question. The model produced the correct formula but skipped Python and returned the wrong decimal result. Additional numerical boundary cases showed that the Python-routing rule was too vague for long products and large combinatorial calculations.

The prompt was then narrowed to explicitly prefer Python for large combinations, long products, tail sums, and similar arithmetic. The affected cases passed after the change.

A separate manual quality review found that some web answers covered more applications than their retrieved sources strongly supported. The web policy was adjusted to prefer a smaller set of well-supported examples.

## Interactive Runner

`run_agent.py` provides a small terminal interface for normal use.

It preserves conversation history so follow-up questions can refer to earlier turns.

```text
YOU
----------------------------------------------------------------------
What is a Poisson distribution?

TERRA
----------------------------------------------------------------------
...

YOU
----------------------------------------------------------------------
What if the events are not independent?

TERRA
----------------------------------------------------------------------
...
```

Plot artifacts are cleared once when a new terminal session starts. A failed turn does not need to terminate the entire session.

## Project Structure

```text
project/
|
├── src/
│   ├── agent_graph.py
│   │   └── Main LangGraph tool-using agent.
│   │
│   ├── rag_workflow.py
│   │   └── Earlier deterministic RAG baseline.
│   │
│   ├── notebook_parser.py
│   │   └── Parses notebook content into structured Documents.
│   │
│   ├── chunker.py
│   │   └── Splits and merges course content for retrieval.
│   │
│   ├── vector_store.py
│   │   └── Embeddings and Chroma vector-store setup.
│   │
│   └── tools/
│       ├── course_retrieval.py
│       ├── web_search.py
│       └── python_tool.py
│
├── scripts/
│   ├── inspect_agent.py
│   └── inspect_course_retrieval.py
│
├── tests/
│   └── eval_cases.py
│
├── evaluate_agent.py
├── evaluate_rag.py
├── run_agent.py
├── ingest.py
├── artifacts/
├── chroma_db/
└── README.md
```

`artifacts/` and the local Chroma database are runtime data and should not be treated as source code.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file with the required API keys:

```text
OPENAI_API_KEY=...
TAVILY_API_KEY=...
```

Run ingestion when the course corpus or chunking logic changes:

```bash
python ingest.py
```

Start the interactive agent:

```bash
python run_agent.py
```

Run the deterministic agent evaluation suite:

```bash
python evaluate_agent.py
```

The inspection scripts can be used when debugging retrieval or tool routing without running the full evaluation suite.

## Technologies

- Python
- LangChain
- LangGraph
- Chroma
- OpenAI API
- Tavily
- Matplotlib
- Jupyter Notebook

## Design Decisions

### Keep the graph small

The graph does not contain a separate node for every possible task. The model selects from a bounded set of tools and loops through the same agent node after receiving a tool result.

This keeps the control flow easy to inspect while still allowing multi-step behavior.

### Use both semantic and structural retrieval

Similarity search is useful when the student names a concept.

It is less appropriate for:

```text
Summarize Chapter 18
```

where the requested location is already known.

The retrieval tool therefore uses metadata-based structural retrieval for explicit course locations and semantic retrieval otherwise.

### Keep course retrieval as a custom tool

Course retrieval contains project-specific logic around theory, exercises, chapter routing, subchapter routing, and large chapter summaries. Keeping that logic in a dedicated tool makes those decisions visible and testable.

### Keep Python bounded

The Python executor is intentionally limited to local educational calculations, simulations, and visualizations. The lightweight restrictions make sense for this project without presenting the executor as a secure production sandbox.

### Tune prompts from observed failures

Prompt rules were added or changed when testing exposed a repeatable failure mode.

The goal is to keep the system prompt understandable instead of accumulating rules for hypothetical edge cases.

## Limitations

- The ingested corpus currently covers the probability course material included in the local dataset.
- Web search quality depends on the results returned by the search provider.
- The Python executor uses lightweight local restrictions rather than process-level sandboxing.
- The agent does not support image input.
- There is no graphical web interface.
- There is no long-term user memory beyond the current conversation state.
- The evaluation suite is strongest on routing and deterministic behavior; semantic answer quality is still partly reviewed manually.
- The system does not fine-tune or update model weights from conversations.

## Possible Extensions

Possible extensions include:

- Model Context Protocol (MCP) integration for a genuinely external capability such as repository inspection;
- multimodal questions over diagrams or handwritten notes;
- stronger process isolation or timeouts for Python execution;
- an optional LLM-based answer-quality judge;
- tracing and richer observability;
- a graphical interface if the project eventually needs one.

These are optional extensions rather than requirements for the current agent.

## Project Goal

This project was built to understand the architecture behind RAG and tool-using agents well enough to explain and debug each part.

The system was developed incrementally: notebook parsing, chunking, vector storage, retrieval, a deterministic RAG workflow, tool conversion, agent-directed routing, web search, Python execution, evaluation, and prompt refinement.

Keeping the original RAG workflow alongside the current agent also provides a direct comparison between a predefined control flow and a model that chooses among bounded tools at runtime.
