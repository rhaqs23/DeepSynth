from __future__ import annotations

import asyncio

from tools.arxiv_tool import arxiv_search
from tools.web_search import web_search
from vectorstore.retrieval import store_and_retrieve


async def run_retriever(query: str, subtopics: list[str]) -> tuple[list[dict], list[dict], list[str]]:
    errors: list[str] = []
    search_queries = [query, *subtopics[:5]]

    tasks = []
    for item in search_queries:
        tasks.append(web_search(item))
        tasks.append(arxiv_search(item))

    batches = await asyncio.gather(*tasks, return_exceptions=True)
    documents: list[dict] = []
    for batch in batches:
        if isinstance(batch, Exception):
            errors.append(f"Retrieval task failed: {batch}")
            continue
        for item in batch:
            if item.get("kind") == "error":
                errors.append(f"{item.get('source', 'retrieval')} unavailable: {item.get('content', 'Unknown error')}")
                continue
            documents.append(item)

    # Store retrieved chunks and query back the most relevant evidence.
    try:
        context = await asyncio.to_thread(store_and_retrieve, query, documents, 12)
    except Exception as exc:
        errors.append(f"Vector retrieval failed: {exc}")
        context = []

    return documents, context, errors
