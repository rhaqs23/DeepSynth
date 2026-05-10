from __future__ import annotations


def build_citations(documents: list[dict]) -> list[dict[str, str]]:
    seen: set[str] = set()
    citations: list[dict[str, str]] = []
    for doc in documents:
        url = doc.get("url") or doc.get("metadata", {}).get("url", "")
        title = doc.get("title") or doc.get("metadata", {}).get("title", "Untitled")
        source = doc.get("source") or doc.get("metadata", {}).get("source", "unknown")
        key = url or title
        if not key or key in seen:
            continue
        seen.add(key)
        citations.append({"title": title, "url": url, "source": source})
    return citations
