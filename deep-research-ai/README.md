# Multi-Agent AI Deep Researcher

A hackathon-ready AI research assistant built with Python, LangGraph, LangChain, Streamlit, OpenRouter, and ChromaDB.

## Architecture

User -> Streamlit UI -> LangGraph Workflow -> Agents -> OpenRouter LLMs -> ChromaDB -> Final Report

Flow:

START -> planner -> retriever -> analyzer -> insight -> report -> END

## Agents

- Planner Agent: breaks the user query into focused subtopics.
- Retriever Agent: searches web results, arXiv, and the local ChromaDB collection.
- Analyzer Agent: compares sources, detects contradictions, validates evidence, and summarizes findings.
- Insight Agent: generates trends, predictions, hypotheses, and research gaps.
- Report Agent: writes the final markdown report with citations.

## Setup

### Mac/Linux

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Windows

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add your OpenRouter API key to `.env`:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

OpenRouter uses the OpenAI-compatible base URL:

```python
openai_api_base = "https://openrouter.ai/api/v1"
```

## Run

```bash
streamlit run app/streamlit_app.py
```

Then open the Streamlit URL, enter a research question, and click **Start research**.

## Models

Defaults are configured in `.env.example`:

- planner: `deepseek/deepseek-chat-v3`
- retriever: `openai/gpt-4o-mini`
- analyzer: `anthropic/claude-3.7-sonnet`
- insights: `openai/gpt-4o-mini`
- report: `anthropic/claude-3.7-sonnet`

The Streamlit sidebar can override the API key and select a model per agent at runtime.

## Vector Database

ChromaDB persists to `chroma_db/`. Retrieved web and arXiv snippets are stored with metadata:

- title
- source
- URL
- kind

The demo uses deterministic local hash embeddings so it can run without an additional embedding API key. For production, swap `HashEmbeddings` in `vectorstore/chroma_store.py` for an embedding provider such as OpenAI, Voyage, Cohere, or sentence-transformers.

## Folder Structure

```text
deep-research-ai/
├── app/
│   └── streamlit_app.py
├── agents/
│   ├── planner_agent.py
│   ├── retriever_agent.py
│   ├── analyzer_agent.py
│   ├── insight_agent.py
│   └── report_agent.py
├── graph/
│   ├── workflow.py
│   ├── state.py
│   └── nodes.py
├── llm/
│   ├── openrouter_client.py
│   └── model_router.py
├── vectorstore/
│   ├── chroma_store.py
│   └── retrieval.py
├── tools/
│   ├── web_search.py
│   ├── arxiv_tool.py
│   └── citation_tool.py
├── prompts/
│   └── sample_prompts.md
├── outputs/
├── utils/
├── .env.example
├── requirements.txt
└── README.md
```

## Notes

- The app streams LangGraph node updates into live agent logs and status indicators.
- External search tools handle errors gracefully, which keeps the demo running even when a source is temporarily unavailable.
- If no `OPENROUTER_API_KEY` is configured, agents use fallback behavior where possible and show setup errors in the UI.
