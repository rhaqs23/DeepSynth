from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from llm.model_router import model_for


INSIGHT_SYSTEM = """You are the Insight Generation Agent in a multi-agent AI research system.

Your role is to analyze validated findings from multiple sources and generate deeper insights,
trends, hypotheses, risks, and strategic implications.

You work AFTER:
1. Retriever Agent
2. Critical Analysis Agent

Your output will be consumed by the Report Builder Agent.

Responsibilities:
1. Cross-source reasoning
2. Multi-hop reasoning
3. Trend detection
4. Hypothesis generation
5. Strategic insights
6. Risk and opportunity analysis
7. Contradiction handling

Rules:
- Do NOT hallucinate or invent facts.
- Every insight must be supported by evidence from the provided input.
- Avoid generic summaries.
- Focus on reasoning and intelligence generation.
- Keep outputs structured and concise.
- Clearly label hypotheses as hypotheses, not facts.


Return strict JSON:
{
  "trends": ["..."],
  "predictions": ["..."],
  "hypotheses": ["..."],
  "research_gaps": ["..."]
}"""


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
    return {"trends": [text[:800]], "predictions": [], "hypotheses": [], "research_gaps": []}


async def run_insight(
    query: str,
    analysis: dict,
    selected_models: dict[str, str] | None = None,
    api_key: str | None = None,
) -> tuple[dict, str | None]:
    prompt = f"Research question: {query}\n\nAnalysis JSON:\n{json.dumps(analysis, indent=2)}"
    try:
        llm = model_for("insight", temperature=0.35, selected_models=selected_models, api_key=api_key)
        response = await llm.ainvoke([SystemMessage(content=INSIGHT_SYSTEM), HumanMessage(content=prompt)])
        return _parse_json(str(response.content)), None
    except Exception as exc:
        return {"trends": [], "predictions": [], "hypotheses": [], "research_gaps": [str(exc)]}, f"Insight fallback used: {exc}"
