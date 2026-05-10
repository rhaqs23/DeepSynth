from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from llm.model_router import model_for


ANALYZER_SYSTEM = """You are the Critical Analysis Agent in a multi-agent research pipeline.

You sit BETWEEN:
- The Contextual Retriever Agent (which produces web search results and local RAG chunks)
- The Insight Generation Agent (which produces hypotheses, trends, and strategic insights)

Your responsibilities:
1. Summarize each source faithfully. Produce one concise summary per evidence item, in E1..En order.
2. Extract discrete validated claims. Each claim must cite the evidence IDs that support OR contradict it.
3. Identify contradictions across sources. Do NOT pick a winner. Surface the disagreement and cite the evidence.
4. Score each source's credibility using these tiers:
   - primary: peer-reviewed papers, government, official standards bodies, authoritative databases
   - established: major news (Reuters, NYT, WSJ, BBC, FT), top-tier industry research (McKinsey, Gartner)
   - secondary: established trade publications, reputable vendor research blogs
   - blog: independent blogs, Medium, Substack, personal sites
   - unknown: domain not recognized
5. Extract dates and named events into a timeline if present.
6. Flag gaps where evidence is missing, weak, contradictory without resolution, or self-referential.

HARD RULES (never violate):
- Every claim and contradiction MUST cite at least one evidence ID like E1.
- Do not introduce facts that are not present in the evidence block.
- Do not summarize the absence of evidence as evidence.
- Distinguish facts (clearly stated and corroborated) from assertions, opinions, and speculation.
- Be specific. Avoid generic statements like "the topic is complex" or "more research is needed".
- Do not use em-dashes; use periods, colons, or middots.

Return ONLY structured JSON matching the schema.
Return strict JSON:
{
  "analysis": "No documents retrieved. Critical analysis skipped.",
  "summarized_findings": [],
  "validated_claims": [],
  "contradictions": [],
  "source_credibility_scores": [],
  "metadata_and_timelines": [],
  "status": "analysis_complete",
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
        "analysis": "No documents retrieved. Critical analysis skipped.",
        "summarized_findings": [],
        "validated_claims": [],
        "contradictions": [],
        "source_credibility_scores": [],
        "metadata_and_timelines": [],
        "status": "analysis_complete",
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
            "analysis": "No documents retrieved. Critical analysis skipped.",
            "summarized_findings": [],
            "validated_claims": [],
            "contradictions": [],
            "evidence_quality": [str(exc)],
            "confidence": "low",
        }, f"Analyzer fallback used: {exc}"
