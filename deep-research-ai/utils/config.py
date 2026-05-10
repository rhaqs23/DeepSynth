from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    site_url: str = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501")
    app_name: str = os.getenv("OPENROUTER_APP_NAME", "Deep Research AI")
    planner_model: str = os.getenv("PLANNER_MODEL", "deepseek/deepseek-chat-v3")
    retriever_model: str = os.getenv("RETRIEVER_MODEL", "openai/gpt-4o-mini")
    analyzer_model: str = os.getenv("ANALYZER_MODEL", "anthropic/claude-3.7-sonnet")
    insight_model: str = os.getenv("INSIGHT_MODEL", "openai/gpt-4o-mini")
    report_model: str = os.getenv("REPORT_MODEL", "anthropic/claude-3.7-sonnet")
    chroma_dir: str = os.getenv("CHROMA_DIR", "chroma_db")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "deep_research_sources")
    max_web_results: int = int(os.getenv("MAX_WEB_RESULTS", "5"))
    max_arxiv_results: int = int(os.getenv("MAX_ARXIV_RESULTS", "5"))


settings = Settings()
