from __future__ import annotations

from collections.abc import AsyncIterator

from langgraph.graph import END, START, StateGraph

from graph.nodes import analyzer_node, insight_node, planner_node, report_node, retriever_node
from graph.state import ResearchState
from llm.model_router import default_models


AGENT_ORDER = ["planner", "retriever", "analyzer", "insight", "report"]


def initial_research_state(
    query: str,
    selected_models: dict[str, str] | None = None,
    openrouter_api_key: str | None = None,
) -> ResearchState:
    return {
        "query": query,
        "logs": ["Research started."],
        "errors": [],
        "current_agent": "planner",
        "current_task": "Generating research subtopics",
        "progress": {agent: 0 for agent in AGENT_ORDER},
        "agent_status": {agent: "waiting" for agent in AGENT_ORDER},
        "selected_models": selected_models or default_models(),
        "openrouter_api_key": openrouter_api_key or "",
        "durations": {},
    }


def build_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("insight", insight_node)
    workflow.add_node("report", report_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "analyzer")
    workflow.add_edge("analyzer", "insight")
    workflow.add_edge("insight", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


async def run_research_stream(
    query: str,
    selected_models: dict[str, str] | None = None,
    openrouter_api_key: str | None = None,
) -> AsyncIterator[dict]:
    graph = build_graph()
    initial_state = initial_research_state(query, selected_models=selected_models, openrouter_api_key=openrouter_api_key)
    async for update in graph.astream(initial_state, stream_mode=["custom", "updates"]):
        yield update


async def run_research(
    query: str,
    selected_models: dict[str, str] | None = None,
    openrouter_api_key: str | None = None,
) -> ResearchState:
    graph = build_graph()
    return await graph.ainvoke(initial_research_state(query, selected_models=selected_models, openrouter_api_key=openrouter_api_key))
