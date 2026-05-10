from __future__ import annotations

import asyncio

import arxiv

from utils.config import settings


def _search_arxiv_sync(query: str) -> list[dict]:
    max_results = max(1, min(settings.max_arxiv_results, 5))
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()
    docs: list[dict] = []
    for result in client.results(search):
        docs.append(
            {
                "title": result.title,
                "content": result.summary,
                "source": "arxiv",
                "url": result.entry_id,
                "kind": "paper",
                "published": result.published.isoformat() if result.published else "",
            }
        )
    return docs


async def arxiv_search(query: str) -> list[dict]:
    try:
        return await asyncio.to_thread(_search_arxiv_sync, query)
    except Exception as exc:
        message = str(exc)
        if "HTTP 429" in message:
            message = "arXiv rate limit reached. Retrying later usually resolves this."
        return [{"title": "arXiv unavailable", "content": message, "source": "arxiv", "url": "", "kind": "error"}]
