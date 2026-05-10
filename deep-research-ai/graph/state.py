from __future__ import annotations

from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    """Typed state passed between LangGraph nodes."""

    query: str
    current_agent: str
    current_task: str
    subtopics: list[str]
    documents: list[dict[str, Any]]
    retrieved_context: list[dict[str, Any]]
    retrieved_sources: list[dict[str, Any]]
    analysis: dict[str, Any]
    insights: dict[str, Any]
    report: str
    citations: list[dict[str, str]]
    logs: list[str]
    errors: list[str]
    progress: dict[str, int]
    agent_status: dict[str, str]
    selected_models: dict[str, str]
    openrouter_api_key: str
    durations: dict[str, float]
