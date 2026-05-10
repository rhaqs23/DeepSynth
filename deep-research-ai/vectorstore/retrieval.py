from __future__ import annotations

from vectorstore.chroma_store import ChromaResearchStore


def store_and_retrieve(query: str, documents: list[dict], k: int = 10) -> list[dict]:
    if not documents:
        return []

    store = ChromaResearchStore()
    store.add_documents(documents)
    return store.similarity_search(query, k=k)
