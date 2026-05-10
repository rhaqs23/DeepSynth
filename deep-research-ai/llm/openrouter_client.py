from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from utils.config import settings


def get_openrouter_chat(model_name: str, temperature: float = 0.2, api_key: str | None = None) -> BaseChatModel:
    """Create a LangChain chat model using OpenRouter's OpenAI-compatible API."""
    resolved_api_key = api_key or settings.openrouter_api_key
    if not resolved_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=resolved_api_key,
        base_url=settings.openrouter_base_url,
        default_headers={
            "HTTP-Referer": settings.site_url,
            "X-Title": settings.app_name,
        },
        timeout=60,
        max_retries=2,
    )
