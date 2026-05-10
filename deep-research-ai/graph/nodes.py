from __future__ import annotations

from datetime import datetime
from time import perf_counter

from langgraph.config import get_stream_writer

from agents.analyzer_agent import run_analyzer
from agents.insight_agent import run_insight
from agents.planner_agent import run_planner
from agents.report_agent import run_report
from agents.retriever_agent import run_retriever
from graph.state import ResearchState
from tools.citation_tool import build_citations


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _append(state: ResearchState, message: str) -> list[str]:
    return [*state.get("logs", []), f"[{_timestamp()}] {message}"]


def _emit(agent: str, progress: int, task: str, message: str, status: str = "running") -> None:
    """Send fine-grained progress/log events to LangGraph streaming consumers."""
    try:
        writer = get_stream_writer()
        writer(
            {
                "agent": agent,
                "status": status,
                "progress": progress,
                "task": task,
                "log": f"[{_timestamp()}] {AGENT_NAMES.get(agent, agent.title())}: {message}",
            }
        )
    except RuntimeError:
        # The same nodes should still work when invoked without stream_mode="custom".
        return


AGENT_NAMES = {
    "planner": "Planner Agent",
    "retriever": "Retriever Agent",
    "analyzer": "Analyzer Agent",
    "insight": "Insight Agent",
    "report": "Report Agent",
}


TASKS = {
    "planner": "Generating research subtopics",
    "retriever": "Searching web, arXiv, and vector memory",
    "analyzer": "Comparing evidence and contradictions",
    "insight": "Generating trends, predictions, and hypotheses",
    "report": "Writing structured markdown report",
}


def _start_agent(state: ResearchState, agent: str, task: str, progress: int = 8) -> None:
    _emit(agent, progress, task, "Started.")


def _finish_agent(
    state: ResearchState,
    agent: str,
    task: str,
    started_at: float,
    message: str,
    failed: bool = False,
) -> ResearchState:
    status = {**state.get("agent_status", {}), agent: "failed" if failed else "completed"}
    progress = {**state.get("progress", {}), agent: 100}
    durations = {**state.get("durations", {}), agent: round(perf_counter() - started_at, 2)}
    return {
        "current_agent": agent,
        "current_task": task,
        "agent_status": status,
        "progress": progress,
        "durations": durations,
        "logs": _append(state, message),
    }


async def planner_node(state: ResearchState) -> ResearchState:
    started_at = perf_counter()
    _start_agent(state, "planner", TASKS["planner"], 10)
    _emit("planner", 30, TASKS["planner"], "Reading the research question and selecting a planning model.")
    subtopics, error = await run_planner(
        state["query"],
        selected_models=state.get("selected_models"),
        api_key=state.get("openrouter_api_key"),
    )
    _emit("planner", 80, TASKS["planner"], f"Generated {len(subtopics)} candidate research subtopics.")
    errors = state.get("errors", [])
    if error:
        errors = [*errors, error]
        _emit("planner", 90, TASKS["planner"], error, status="failed")
    finished = _finish_agent(
        state,
        "planner",
        "Generating research subtopics",
        started_at,
        f"Planner created {len(subtopics)} subtopics.",
        failed=False,
    )
    return {"subtopics": subtopics, "errors": errors, **finished}


async def retriever_node(state: ResearchState) -> ResearchState:
    started_at = perf_counter()
    subtopic_count = len(state.get("subtopics", []))
    _start_agent(state, "retriever", TASKS["retriever"], 10)
    _emit("retriever", 22, TASKS["retriever"], f"Preparing searches for the main query and {subtopic_count} subtopics.")
    _emit("retriever", 38, TASKS["retriever"], "Dispatching web and arXiv retrieval tasks.")
    documents, context, errors = await run_retriever(state["query"], state.get("subtopics", []))
    _emit("retriever", 68, TASKS["retriever"], f"Retrieved {len(documents)} raw source chunks.")
    _emit("retriever", 84, TASKS["retriever"], f"Stored sources in ChromaDB and selected {len(context)} evidence chunks.")
    citations = build_citations(documents)
    _emit("retriever", 92, TASKS["retriever"], f"Built {len(citations)} citation records.")
    failed = bool(errors) and not documents
    for error in errors:
        _emit("retriever", 95, TASKS["retriever"], error, status="failed" if failed else "running")
    finished = _finish_agent(
        state,
        "retriever",
        "Searching web, arXiv, and vector memory",
        started_at,
        f"Retriever collected {len(documents)} source chunks and selected {len(context)} evidence chunks.",
        failed=failed,
    )
    return {
        "documents": documents,
        "retrieved_context": context,
        "retrieved_sources": documents,
        "citations": citations,
        "errors": [*state.get("errors", []), *errors],
        **finished,
    }


async def analyzer_node(state: ResearchState) -> ResearchState:
    started_at = perf_counter()
    evidence_count = len(state.get("retrieved_context", []))
    _start_agent(state, "analyzer", TASKS["analyzer"], 10)
    _emit("analyzer", 25, TASKS["analyzer"], f"Loading {evidence_count} evidence chunks for comparison.")
    _emit("analyzer", 45, TASKS["analyzer"], "Checking source agreement, caveats, and contradictions.")
    analysis, error = await run_analyzer(
        state["query"],
        state.get("retrieved_context", []),
        selected_models=state.get("selected_models"),
        api_key=state.get("openrouter_api_key"),
    )
    _emit("analyzer", 82, TASKS["analyzer"], f"Extracted {len(analysis.get('findings', []))} findings.")
    errors = state.get("errors", [])
    if error:
        errors = [*errors, error]
        _emit("analyzer", 90, TASKS["analyzer"], error, status="failed")
    finished = _finish_agent(
        state,
        "analyzer",
        "Comparing evidence and contradictions",
        started_at,
        "Analyzer compared evidence and caveats.",
        failed=False,
    )
    return {"analysis": analysis, "errors": errors, **finished}


async def insight_node(state: ResearchState) -> ResearchState:
    started_at = perf_counter()
    _start_agent(state, "insight", TASKS["insight"], 10)
    _emit("insight", 30, TASKS["insight"], "Reading analysis output and evidence confidence.")
    _emit("insight", 55, TASKS["insight"], "Generating trends, predictions, hypotheses, and research gaps.")
    insights, error = await run_insight(
        state["query"],
        state.get("analysis", {}),
        selected_models=state.get("selected_models"),
        api_key=state.get("openrouter_api_key"),
    )
    insight_count = sum(len(insights.get(key, [])) for key in ("trends", "predictions", "hypotheses", "research_gaps"))
    _emit("insight", 84, TASKS["insight"], f"Generated {insight_count} insight items.")
    errors = state.get("errors", [])
    if error:
        errors = [*errors, error]
        _emit("insight", 90, TASKS["insight"], error, status="failed")
    finished = _finish_agent(
        state,
        "insight",
        "Generating trends, predictions, and hypotheses",
        started_at,
        "Insight agent generated trends and hypotheses.",
        failed=False,
    )
    return {"insights": insights, "errors": errors, **finished}


async def report_node(state: ResearchState) -> ResearchState:
    started_at = perf_counter()
    _start_agent(state, "report", TASKS["report"], 10)
    _emit("report", 28, TASKS["report"], "Assembling findings, caveats, insights, and citations.")
    _emit("report", 52, TASKS["report"], f"Preparing markdown report with {len(state.get('citations', []))} citations.")
    report, error = await run_report(
        state["query"],
        state.get("analysis", {}),
        state.get("insights", {}),
        state.get("citations", []),
        selected_models=state.get("selected_models"),
        api_key=state.get("openrouter_api_key"),
    )
    _emit("report", 86, TASKS["report"], f"Drafted report with {len(report.split())} words.")
    errors = state.get("errors", [])
    if error:
        errors = [*errors, error]
        _emit("report", 92, TASKS["report"], error, status="failed")
    finished = _finish_agent(
        state,
        "report",
        "Writing structured markdown report",
        started_at,
        "Report agent produced the final markdown report.",
        failed=False,
    )
    return {"report": report, "errors": errors, **finished}
