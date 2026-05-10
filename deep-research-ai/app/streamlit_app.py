from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.workflow import AGENT_ORDER, run_research_stream  # noqa: E402
from llm.model_router import SUPPORTED_MODELS, default_models  # noqa: E402
from utils.config import settings  # noqa: E402


AGENT_LABELS = {
    "planner": "Planner Agent",
    "retriever": "Retriever Agent",
    "analyzer": "Analyzer Agent",
    "insight": "Insight Agent",
    "report": "Report Agent",
}

TASK_BY_AGENT = {
    "planner": "Generating research subtopics",
    "retriever": "Searching web, arXiv, and vector memory",
    "analyzer": "Comparing evidence and contradictions",
    "insight": "Generating trends, predictions, and hypotheses",
    "report": "Writing structured markdown report",
}

STATUS_LABELS = {
    "waiting": "Waiting",
    "running": "Running",
    "completed": "Completed",
    "failed": "Failed",
}


st.set_page_config(page_title="Deep Research AI", page_icon="🔎", layout="wide")


def init_session() -> None:
    defaults = default_models()
    st.session_state.setdefault("openrouter_api_key", settings.openrouter_api_key or "")
    st.session_state.setdefault("selected_models", defaults)
    st.session_state.setdefault("agent_status", {agent: "waiting" for agent in AGENT_ORDER})
    st.session_state.setdefault("progress", {agent: 0 for agent in AGENT_ORDER})
    st.session_state.setdefault("agent_tasks", {agent: "Waiting" for agent in AGENT_ORDER})
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("errors", [])
    st.session_state.setdefault("report", "")
    st.session_state.setdefault("sources", [])
    st.session_state.setdefault("current_agent", "")
    st.session_state.setdefault("current_task", "")
    st.session_state.setdefault("durations", {})
    st.session_state.setdefault("execution_duration", 0.0)
    st.session_state.setdefault("render_revision", 0)


def reset_run_state() -> None:
    st.session_state.agent_status = {agent: "waiting" for agent in AGENT_ORDER}
    st.session_state.progress = {agent: 0 for agent in AGENT_ORDER}
    st.session_state.agent_tasks = {agent: "Waiting" for agent in AGENT_ORDER}
    st.session_state.logs = []
    st.session_state.errors = []
    st.session_state.report = ""
    st.session_state.sources = []
    st.session_state.current_agent = "planner"
    st.session_state.current_task = TASK_BY_AGENT["planner"]
    st.session_state.durations = {}
    st.session_state.execution_duration = 0.0


def render_sidebar() -> tuple[str, dict[str, str]]:
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input(
        "OpenRouter API key",
        value=st.session_state.openrouter_api_key,
        type="password",
        help="Stored only in Streamlit session state for this browser session.",
    )
    if api_key:
        st.session_state.openrouter_api_key = api_key.strip()
        os.environ["OPENROUTER_API_KEY"] = st.session_state.openrouter_api_key

    st.sidebar.divider()
    st.sidebar.subheader("Models")
    selected: dict[str, str] = {}
    defaults = default_models()
    for agent in AGENT_ORDER:
        current = st.session_state.selected_models.get(agent, defaults[agent])
        index = SUPPORTED_MODELS.index(current) if current in SUPPORTED_MODELS else 0
        selected[agent] = st.sidebar.selectbox(
            AGENT_LABELS[agent],
            SUPPORTED_MODELS,
            index=index,
            key=f"model_{agent}",
        )

    st.session_state.selected_models = selected
    return st.session_state.openrouter_api_key, selected


def set_next_agent_running(completed_agent: str) -> None:
    try:
        next_agent = AGENT_ORDER[AGENT_ORDER.index(completed_agent) + 1]
    except IndexError:
        st.session_state.current_agent = ""
        st.session_state.current_task = "Workflow complete"
        return

    st.session_state.current_agent = next_agent
    st.session_state.current_task = TASK_BY_AGENT[next_agent]
    if st.session_state.agent_status.get(next_agent) == "waiting":
        st.session_state.agent_status[next_agent] = "running"
        st.session_state.progress[next_agent] = 12
        st.session_state.agent_tasks[next_agent] = TASK_BY_AGENT[next_agent]


def merge_logs(existing: list[str], incoming: list[str]) -> list[str]:
    merged = [*existing]
    for line in incoming:
        if line and line not in merged:
            merged.append(line)
    return merged


def render_top_stats(stats_slot) -> None:
    stats_slot.empty()
    with stats_slot.container():
        top_cols = st.columns([1, 1, 1])
        top_cols[0].metric("Agents", len(AGENT_ORDER))
        top_cols[1].metric("Sources", len(st.session_state.sources))
        top_cols[2].metric("Duration", f"{st.session_state.execution_duration:.2f}s")


def render_agent_progress() -> None:
    st.subheader("Agent Progress")
    cols = st.columns(len(AGENT_ORDER))
    for col, agent in zip(cols, AGENT_ORDER, strict=False):
        status = st.session_state.agent_status.get(agent, "waiting")
        percent = st.session_state.progress.get(agent, 0)
        task = st.session_state.agent_tasks.get(agent, "Waiting")
        if status in {"completed", "failed"}:
            task = STATUS_LABELS.get(status, status.title())
        duration = st.session_state.durations.get(agent)
        duration_text = f"{duration:.2f}s" if isinstance(duration, float) else ""

        with col:
            st.markdown(f"**{AGENT_LABELS[agent]}**")
            st.progress(percent, text=f"{percent}%")
            st.caption(f"{STATUS_LABELS.get(status, status.title())} {duration_text}".strip())
            st.caption(f"Current Task: {task}")


def render_progress_slot(progress_slot, active_slot) -> None:
    """Replace the live progress region instead of appending on each stream event."""
    progress_slot.empty()
    with progress_slot.container():
        render_agent_progress()

    active_slot.empty()
    running_agents = [agent for agent in AGENT_ORDER if st.session_state.agent_status.get(agent) == "running"]
    if running_agents:
        active_text = " | ".join(
            f"{AGENT_LABELS[agent]} - {st.session_state.agent_tasks.get(agent, TASK_BY_AGENT[agent])}"
            for agent in running_agents
        )
        active_slot.info(f"Active agent{'s' if len(running_agents) > 1 else ''}: {active_text}")
    elif all(st.session_state.agent_status.get(agent) == "completed" for agent in AGENT_ORDER):
        active_slot.success("Workflow complete")


def render_sources(sources: list[dict]) -> None:
    valid_sources = [
        source
        for source in sources
        if source.get("kind") != "error"
        and source.get("title") not in {"Web search unavailable", "arXiv unavailable"}
        and (source.get("url") or source.get("content"))
    ]

    if not valid_sources:
        st.info("Sources will appear after retrieval completes.")
        return

    for idx, source in enumerate(valid_sources, start=1):
        title = source.get("title") or source.get("metadata", {}).get("title", "Untitled")
        url = source.get("url") or source.get("metadata", {}).get("url", "")
        origin = source.get("source") or source.get("metadata", {}).get("source", "unknown")
        content = source.get("content", "")[:500]
        with st.expander(f"{idx}. {title} ({origin})", expanded=idx <= 3):
            if url:
                st.markdown(f"[Open source]({url})")
            st.write(content or "No snippet available.")


async def execute_research(
    user_query: str,
    api_key: str,
    selected_models: dict[str, str],
    stats_slot,
    progress_slot,
    active_slot,
    tab_slots,
) -> None:
    started_at = time.perf_counter()
    final_state = {}
    st.session_state.agent_status["planner"] = "running"
    st.session_state.progress["planner"] = 12
    st.session_state.agent_tasks["planner"] = TASK_BY_AGENT["planner"]

    render_top_stats(stats_slot)
    render_into_tabs(tab_slots)
    render_progress_slot(progress_slot, active_slot)

    async for stream_item in run_research_stream(user_query, selected_models=selected_models, openrouter_api_key=api_key):
        if isinstance(stream_item, tuple):
            stream_mode, update = stream_item
        else:
            stream_mode, update = "updates", stream_item

        if stream_mode == "custom":
            agent = update.get("agent")
            if agent in AGENT_ORDER:
                st.session_state.agent_status[agent] = update.get("status", "running")
                st.session_state.progress[agent] = max(st.session_state.progress.get(agent, 0), int(update.get("progress", 0)))
                st.session_state.agent_tasks[agent] = update.get("task", st.session_state.agent_tasks.get(agent, "Running"))
                st.session_state.current_agent = agent
                st.session_state.current_task = st.session_state.agent_tasks[agent]
            if update.get("log"):
                st.session_state.logs = merge_logs(st.session_state.logs, [update["log"]])
            st.session_state.execution_duration = round(time.perf_counter() - started_at, 2)
            render_top_stats(stats_slot)
            render_progress_slot(progress_slot, active_slot)
            render_into_tabs(tab_slots)
            continue

        for node_name, node_state in update.items():
            final_state.update(node_state)
            node_errors = node_state.get("errors", st.session_state.errors)

            st.session_state.agent_status.update(node_state.get("agent_status", {}))
            st.session_state.progress.update(node_state.get("progress", {}))
            for agent, status in st.session_state.agent_status.items():
                if status == "completed":
                    st.session_state.agent_tasks[agent] = "Completed"
                elif status == "failed":
                    st.session_state.agent_tasks[agent] = "Failed"
            st.session_state.durations.update(node_state.get("durations", {}))
            st.session_state.logs = merge_logs(st.session_state.logs, node_state.get("logs", []))
            st.session_state.errors = node_errors
            st.session_state.sources = node_state.get("retrieved_sources", st.session_state.sources)
            st.session_state.report = node_state.get("report", st.session_state.report)
            st.session_state.current_agent = node_state.get("current_agent", node_name)
            st.session_state.current_task = node_state.get("current_task", TASK_BY_AGENT.get(node_name, "Running"))

            if node_errors and st.session_state.agent_status.get(node_name) != "completed":
                st.session_state.agent_status[node_name] = "failed"

            set_next_agent_running(node_name)
            st.session_state.execution_duration = round(time.perf_counter() - started_at, 2)

            render_top_stats(stats_slot)
            render_progress_slot(progress_slot, active_slot)
            render_into_tabs(tab_slots)

    st.session_state.report = final_state.get("report", st.session_state.report)
    st.session_state.execution_duration = round(time.perf_counter() - started_at, 2)
    render_top_stats(stats_slot)
    render_progress_slot(progress_slot, active_slot)
    render_into_tabs(tab_slots)


def render_into_tabs(tab_slots) -> None:
    report_slot, logs_slot, sources_slot = tab_slots
    st.session_state.render_revision += 1
    render_revision = st.session_state.render_revision

    report_slot.empty()
    with report_slot.container():
        if st.session_state.report:
            st.markdown(st.session_state.report)
            st.download_button(
                "Download markdown report",
                data=st.session_state.report,
                file_name="deep_research_report.md",
                mime="text/markdown",
                key=f"download_report_{render_revision}",
            )
        else:
            st.info("The final markdown report will appear here.")

    logs_slot.empty()
    with logs_slot.container():
        metrics = st.columns(3)
        metrics[0].metric("Current Agent", AGENT_LABELS.get(st.session_state.current_agent, "Idle"))
        metrics[1].metric("Execution Time", f"{st.session_state.execution_duration:.2f}s")
        metrics[2].metric("Log Lines", len(st.session_state.logs))
        if st.session_state.errors:
            st.warning("\n".join(st.session_state.errors))
        st.code("\n".join(st.session_state.logs) or "No logs yet.", language="text")

    sources_slot.empty()
    with sources_slot.container():
        render_sources(st.session_state.sources)


init_session()
api_key, selected_models = render_sidebar()

st.title("Multi-Agent AI Deep Researcher")
st.caption("LangGraph + LangChain + OpenRouter + ChromaDB")

query = st.text_area(
    "Research question",
    placeholder="Example: What are the latest technical and market trends in agentic AI for enterprise software?",
    height=120,
)

stats_slot = st.empty()
render_top_stats(stats_slot)

start = st.button("Start research", type="primary", disabled=not query.strip())

if not api_key:
    st.warning("Add an OpenRouter API key in the sidebar to enable LLM agents. The app will still show graceful fallback output.")

progress_slot = st.empty()
active_slot = st.empty()
tabs = st.tabs(["Final Report", "Agent Logs", "Sources"])
with tabs[0]:
    report_slot = st.empty()
with tabs[1]:
    logs_slot = st.empty()
with tabs[2]:
    sources_slot = st.empty()
tab_slots = (report_slot, logs_slot, sources_slot)

render_progress_slot(progress_slot, active_slot)
render_into_tabs(tab_slots)

if start:
    reset_run_state()
    try:
        asyncio.run(execute_research(query.strip(), api_key, selected_models, stats_slot, progress_slot, active_slot, tab_slots))
    except Exception as exc:
        st.session_state.errors.append(f"Workflow failed: {exc}")
        if st.session_state.current_agent:
            st.session_state.agent_status[st.session_state.current_agent] = "failed"
        render_top_stats(stats_slot)
        render_progress_slot(progress_slot, active_slot)
        render_into_tabs(tab_slots)
        st.error("The workflow failed safely. Check Agent Logs for details, update the API key or model selection, and retry.")
