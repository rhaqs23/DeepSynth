from __future__ import annotations

from html import unescape
import re
from urllib.parse import quote_plus, unquote, urlparse, parse_qs

import certifi
import httpx

from utils.config import settings


def _clean_duckduckgo_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if "duckduckgo.com" in parsed.netloc:
        uddg = parse_qs(parsed.query).get("uddg")
        if uddg:
            return unquote(uddg[0])
    return raw_url


def _parse_lite_results(html: str) -> list[dict]:
    results: list[dict] = []
    pattern = re.compile(
        r"<a[^>]+href=\"(?P<href>.*?)\"[^>]+class='result-link'[^>]*>(?P<title>.*?)</a>"
        r".*?(?:<td class='result-snippet'>\s*(?P<snippet>.*?)\s*</td>)?",
        re.DOTALL,
    )
    for match in pattern.finditer(html):
        href = _clean_duckduckgo_url(unescape(match.group("href")))
        if href.startswith("//"):
            href = f"https:{href}"
        title = re.sub("<.*?>", "", match.group("title"))
        title = unescape(title).strip()
        snippet = re.sub("<.*?>", "", match.group("snippet") or "")
        snippet = unescape(snippet).strip()
        if not title or not href.startswith("http"):
            continue
        results.append({"title": title, "href": href, "body": snippet})
        if len(results) >= settings.max_web_results:
            break
    return results


async def web_search(query: str) -> list[dict]:
    """Search the web and normalize results for downstream agents."""
    try:
        url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        async with httpx.AsyncClient(timeout=15, verify=certifi.where(), follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
        results = _parse_lite_results(response.text)
    except Exception as exc:
        return [
            {
                "title": "Web search unavailable",
                "content": f"Web search failed: {exc}",
                "source": "web search",
                "url": "",
                "kind": "error",
            }
        ]

    normalized: list[dict] = []
    for item in results if isinstance(results, list) else []:
        normalized.append(
            {
                "title": item.get("title", "Web result"),
                "content": item.get("body") or item.get("title", ""),
                "source": "web search",
                "url": item.get("href", ""),
                "kind": "web",
            }
        )
    return normalized
