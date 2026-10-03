import os
from typing import Any

import httpx

from mcp_server.cache import cached
from mcp_server.models import Paper
from mcp_server.rate_limit import RateLimiter, with_retry

GRAPH = "https://api.semanticscholar.org/graph/v1"
RECOMMENDATIONS = "https://api.semanticscholar.org/recommendations/v1"
SOURCE = "semanticscholar"
FIELDS = "title,abstract,year,venue,externalIds,authors.name"

# Keyed access is capped at 1 request/second; unkeyed shares one global pool.
_limiter = RateLimiter(1.0)


def headers() -> dict[str, str]:
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    return {"x-api-key": key} if key else {}


def qualify(identifier: str) -> str:
    if identifier.startswith("10."):
        return f"DOI:{identifier}"
    return f"PMID:{identifier}" if identifier.isdigit() else identifier


def _to_paper(record: dict[str, Any]) -> Paper:
    external = record.get("externalIds") or {}
    return Paper(
        title=(record.get("title") or "").rstrip("."),
        pmid=external.get("PubMed"),
        doi=external.get("DOI"),
        year=record.get("year"),
        journal=record.get("venue") or None,
        authors=[a["name"] for a in record.get("authors", []) if a.get("name")],
        abstract=record.get("abstract"),
        sources=[SOURCE],
    )


async def _get(client: httpx.AsyncClient, url: str, params: dict[str, Any]) -> dict[str, Any]:
    async def fetch() -> dict[str, Any]:
        async def send() -> httpx.Response:
            await _limiter.acquire()
            return await client.get(url, params=params, headers=headers())

        response = await with_retry(send)
        response.raise_for_status()
        return response.json()

    data, _ = await cached(f"{SOURCE}:{url}", params, fetch)
    return data


async def search(
    client: httpx.AsyncClient,
    query: str,
    max_results: int = 20,
    year_from: int | None = None,
) -> list[Paper]:
    params: dict[str, Any] = {"query": query, "limit": max_results, "fields": FIELDS}
    if year_from:
        params["year"] = f"{year_from}-"
    data = await _get(client, f"{GRAPH}/paper/search", params)
    return [_to_paper(r) for r in data.get("data", [])]


async def fetch_one(client: httpx.AsyncClient, identifier: str) -> Paper | None:
    data = await _get(client, f"{GRAPH}/paper/{qualify(identifier)}", {"fields": FIELDS})
    return _to_paper(data) if data.get("title") else None


async def related(
    client: httpx.AsyncClient, identifier: str, max_results: int = 10
) -> list[Paper]:
    data = await _get(
        client,
        f"{RECOMMENDATIONS}/papers/forpaper/{qualify(identifier)}",
        {"fields": FIELDS, "limit": max_results},
    )
    return [_to_paper(r) for r in data.get("recommendedPapers", [])]


async def citations(
    client: httpx.AsyncClient,
    identifier: str,
    direction: str = "cited_by",
    max_results: int = 20,
) -> list[Paper]:
    endpoint, key = (
        ("citations", "citingPaper") if direction == "cited_by" else ("references", "citedPaper")
    )
    data = await _get(
        client,
        f"{GRAPH}/paper/{qualify(identifier)}/{endpoint}",
        {"fields": FIELDS, "limit": max_results},
    )
    return [_to_paper(r[key]) for r in data.get("data", []) if r.get(key)]
