from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from llm.model_router import model_for


ANALYZER_SYSTEM = """You are an evidence analysis agent.
Compare sources, identify strongest findings, contradictions, weak evidence, and confidence.
Return strict JSON:
{
  "findings": ["..."],
  "contradictions": ["..."],
  "evidence_quality": ["..."],
  "confidence": "low|medium|high"
}"""


def _compact_context(context: list[dict]) -> str:
    chunks = []
    for idx, item in enumerate(context[:12], start=1):
        metadata = item.get("metadata", {})
        chunks.append(
            f"[{idx}] {metadata.get('title', 'Untitled')} ({metadata.get('source', 'unknown')})\n"
            f"{item.get('content', '')[:1200]}"
        )
    return "\n\n".join(chunks)


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {
        "findings": [text[:1200] or "No analysis generated."],
        "contradictions": [],
        "evidence_quality": ["Analysis returned in unstructured form."],
        "confidence": "medium",
    }


async def run_analyzer(
    query: str,
    context: list[dict],
    selected_models: dict[str, str] | None = None,
    api_key: str | None = None,
) -> tuple[dict, str | None]:
    prompt = f"Research question: {query}\n\nEvidence:\n{_compact_context(context)}"
    try:
        llm = model_for("analyzer", temperature=0.15, selected_models=selected_models, api_key=api_key)
        response = await llm.ainvoke([SystemMessage(content=ANALYZER_SYSTEM), HumanMessage(content=prompt)])
        return _parse_json(str(response.content)), None
    except Exception as exc:
        return {
            "findings": ["Analyzer could not call the LLM. Review retrieved sources in citations."],
            "contradictions": [],
            "evidence_quality": [str(exc)],
            "confidence": "low",
        }, f"Analyzer fallback used: {exc}"
