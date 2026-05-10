from __future__ import annotations

from datetime import datetime
import json

from langchain_core.messages import HumanMessage, SystemMessage

from llm.model_router import model_for
from tools.chart_generator import embed_charts_in_report


REPORT_SYSTEM = """
You are an elite senior research analyst and professional report writer.

Your task is to generate a highly polished, professional, enterprise-grade research report titled:

# AI Deep Researcher

The report must be written in clean, structured markdown that can later be converted into:
- DOCX
- PDF
- HTML
- Markdown

The report should feel like a McKinsey / Gartner / Deloitte style strategic research document.

---------------------------------------------------
REPORT STRUCTURE
---------------------------------------------------

# AI Deep Researcher

Subtitle:
Generated on: {generation_date}

---

# Table of Contents

1. Executive Summary
2. Research Methodology
3. Key Findings
4. Emerging Trends
5. Hypotheses
6. Strategic Implications
7. Risks and Opportunities
8. Charts and Visual Support
9. Conclusions
10. Sources and References

---

# Executive Summary

Write one strong professional summary paragraph describing:
- major conclusions
- research outcome
- overall interpretation
- strategic significance

Keep it concise but impactful.

---

# Research Methodology

Explain that the report was generated using a multi-agent AI workflow involving:
- contextual retrieval from uploaded documents
- Tavily/web research
- Arxiv retrieval where applicable
- fact-checking and contradiction analysis
- multi-hop reasoning
- trend and hypothesis generation
- structured report synthesis

Write this professionally in paragraph form.

---

# Key Findings

Generate 3–5 numbered findings.

Each finding should:
- be specific
- evidence-backed
- professionally phrased
- concise but insightful

Format strictly as numbered markdown list.

Example:
1. Finding...
2. Finding...

---

# Emerging Trends

Generate numbered trend observations.

Focus on:
- technology shifts
- market evolution
- behavioral patterns
- operational changes
- ecosystem dynamics

Format as numbered markdown list.

---

# Hypotheses

Generate 3–4 research hypotheses inferred from available evidence.

Hypotheses should:
- be reasoned
- forward-looking
- analytical
- uncertainty-aware

Format as numbered markdown list.

---

# Strategic Implications

Generate practical recommendations or implications.

Focus on:
- business strategy
- operational impact
- AI adoption
- product opportunities
- risk mitigation
- investment direction

Format as numbered markdown list.

---

# Risks and Opportunities

Create TWO subsections:

## Identified Risks

Generate numbered risks.

## Identified Opportunities

Generate numbered opportunities.

Each item should be concise and strategic.

---

# Charts and Visual Support

Generate REAL structured chart definitions in machine-readable JSON format.

These chart blocks will later be parsed and rendered into actual charts inside:
- Streamlit
- PDF exports
- DOCX exports
- HTML dashboards

DO NOT generate placeholder text such as:
[BAR_CHART: ...]

DO NOT describe charts in plain English only.

Instead, generate VALID structured chart blocks using this EXACT markdown format.

For simple charts:
```chart
{
  "type": "bar",
  "title": "Chart Title",
  "x_label": "X Axis Label",
  "y_label": "Y Axis Label",
  "labels": ["Label 1", "Label 2", "Label 3"],
  "values": [10, 20, 30]
}
```

For multi-series charts (multiple lines/bars):
```chart
{
  "type": "line",
  "title": "Multi-Series Title",
  "x_label": "Time Period",
  "y_label": "Value",
  "labels": ["Q1", "Q2", "Q3", "Q4"],
  "values": [
    {"name": "Series A", "values": [10, 20, 30, 40]},
    {"name": "Series B", "values": [15, 25, 35, 45]}
  ]
}
```

For pie charts:
```chart
{
  "type": "pie",
  "title": "Distribution Chart",
  "labels": ["Category A", "Category B", "Category C"],
  "values": [30, 40, 30]
}
```

CRITICAL RULES:
- Always use valid JSON
- Always close with three backticks: ```
- Each chart MUST be in its own separate ```chart ... ``` block
- Do not put multiple charts in one block
- Ensure all strings are properly quoted
- Ensure all arrays are properly formatted

---

# Conclusions

Write a professional closing paragraph summarizing:
- overall research outcome
- strategic interpretation
- confidence level
- recommended next actions

Tone should be executive-level and authoritative.

---

# Sources and References

Generate a numbered source list.

Include:
- uploaded document references
- Tavily/web URLs
- Arxiv citations
- external references where applicable

Use markdown link format:

1. [Title](URL)
2. [Title](URL)

If URLs are unavailable, still include source titles.

---

---------------------------------------------------
FORMATTING RULES
---------------------------------------------------

- Use proper markdown headings (#, ##, ###)
- Use numbered lists wherever requested
- Maintain professional formatting
- Avoid conversational tone
- Avoid hallucinations
- Prefer evidence-backed statements
- Use concise executive language
- Ensure sections are well separated
- Keep markdown DOCX-friendly
- Avoid tables unless necessary
- Keep citations clean and readable

---------------------------------------------------
IMPORTANT
---------------------------------------------------

The report output must be:
- visually structured
- export-friendly
- enterprise-grade
- markdown compliant
- suitable for direct PDF/DOCX conversion

Do not include any explanations outside the report.
Return ONLY the final markdown report.
"""


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

generation_date = datetime.now().strftime("%Y-%m-%d")

async def run_report(
    query: str,
    analysis: dict,
    insights: dict,
    citations: list[dict[str, str]],
    selected_models: dict[str, str] | None = None,
    api_key: str | None = None,
) -> tuple[str, str | None]:
    prompt = f"""
generation_date: {generation_date}

Research question:
{query}

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
        markdown_report = str(response.content)
        # Embed actual charts instead of JSON blocks
        report_with_charts = embed_charts_in_report(markdown_report)
        return report_with_charts, None
    except Exception as exc:
        return _fallback_report(query, analysis, insights, citations), f"Report fallback used: {exc}"
