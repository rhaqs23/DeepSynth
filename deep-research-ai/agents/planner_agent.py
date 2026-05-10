from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from llm.model_router import model_for


PLANNER_SYSTEM = """You are a research planning agent.
Break the user's research question into 4-7 concrete subtopics.
Return strict JSON only with this shape:
{"subtopics": ["subtopic 1", "subtopic 2"]}"""


def _fallback_plan(query: str) -> list[str]:
    return [
        f"Background and definitions for {query}",
        f"Current evidence and major sources about {query}",
        f"Contradictions, limitations, and open questions about {query}",
        f"Trends, forecasts, and practical implications of {query}",
    ]


def _parse_subtopics(text: str, query: str) -> list[str]:
    try:
        return json.loads(text)["subtopics"][:7]
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))["subtopics"][:7]
            except Exception:
                pass
    return _fallback_plan(query)


async def run_planner(query: str, selected_models: dict[str, str] | None = None, api_key: str | None = None) -> tuple[list[str], str | None]:
    try:
        llm = model_for("planner", temperature=0.1, selected_models=selected_models, api_key=api_key)
        response = await llm.ainvoke([SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=query)])
        return _parse_subtopics(str(response.content), query), None
    except Exception as exc:
        return _fallback_plan(query), f"Planner fallback used: {exc}"
