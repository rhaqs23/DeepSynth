from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from llm.model_router import model_for


REPORT_SYSTEM = """You are a senior research report writer.
Create a polished markdown report with these sections:
# Research Report
## Executive Summary
## Key Findings
## Contradictions and Caveats
## Insights, Trends, and Hypotheses
## Citations
## Conclusion
Use citation numbers like [1], [2] where relevant."""


def _fallback_report(query: str, analysis: dict, insights: dict, citations: list[dict[str, str]]) -> str:
    citation_lines = "\n".join(
        f"{idx}. [{citation['title']}]({citation['url']}) - {citation['source']}"
        for idx, citation in enumerate(citations, start=1)
    ) or "No citations available."
    findings = "\n".join(f"- {item}" for item in analysis.get("findings", [])) or "- No findings generated."
    contradictions = "\n".join(f"- {item}" for item in analysis.get("contradictions", [])) or "- No contradictions detected."
    trends = "\n".join(f"- {item}" for item in insights.get("trends", [])) or "- No trends generated."
    predictions = "\n".join(f"- {item}" for item in insights.get("predictions", [])) or "- No predictions generated."
    hypotheses = "\n".join(f"- {item}" for item in insights.get("hypotheses", [])) or "- No hypotheses generated."

    return f"""# Research Report

## Executive Summary
This report investigates: **{query}**.

## Key Findings
{findings}

## Contradictions and Caveats
{contradictions}

## Insights, Trends, and Hypotheses
### Trends
{trends}

### Predictions
{predictions}

### Hypotheses
{hypotheses}

## Citations
{citation_lines}

## Conclusion
The available evidence supports a cautious, source-driven interpretation. Confidence: {analysis.get("confidence", "medium")}.
"""


async def run_report(
    query: str,
    analysis: dict,
    insights: dict,
    citations: list[dict[str, str]],
    selected_models: dict[str, str] | None = None,
    api_key: str | None = None,
) -> tuple[str, str | None]:
    prompt = f"""Research question: {query}

Analysis:
{json.dumps(analysis, indent=2)}

Insights:
{json.dumps(insights, indent=2)}

Citations:
{json.dumps(citations, indent=2)}
"""
    try:
        llm = model_for("report", temperature=0.2, selected_models=selected_models, api_key=api_key)
        response = await llm.ainvoke([SystemMessage(content=REPORT_SYSTEM), HumanMessage(content=prompt)])
        return str(response.content), None
    except Exception as exc:
        return _fallback_report(query, analysis, insights, citations), f"Report fallback used: {exc}"
