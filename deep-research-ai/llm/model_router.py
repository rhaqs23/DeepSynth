from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from llm.openrouter_client import get_openrouter_chat
from utils.config import settings


SUPPORTED_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3.7-sonnet",
    "deepseek/deepseek-chat-v3",
    "google/gemini-2.5-pro",
]

MODEL_BY_AGENT = {
    "planner": settings.planner_model,
    "retriever": settings.retriever_model,
    "analyzer": settings.analyzer_model,
    "insight": settings.insight_model,
    "report": settings.report_model,
}


def default_models() -> dict[str, str]:
    return MODEL_BY_AGENT.copy()


def model_for(
    agent_name: str,
    temperature: float = 0.2,
    selected_models: dict[str, str] | None = None,
    api_key: str | None = None,
) -> BaseChatModel:
    if agent_name not in MODEL_BY_AGENT:
        raise ValueError(f"Unknown agent '{agent_name}'. Expected one of {sorted(MODEL_BY_AGENT)}.")
    model_name = (selected_models or {}).get(agent_name, MODEL_BY_AGENT[agent_name])
    return get_openrouter_chat(model_name, temperature=temperature, api_key=api_key)
