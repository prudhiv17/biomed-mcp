from typing import Any

import httpx

from mcp_server.cache import cached
from mcp_server.models import Paper

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
SOURCE = "europepmc"


def _to_paper(record: dict[str, Any]) -> Paper:
    journal = ((record.get("journalInfo") or {}).get("journal") or {}).get("title")
    authors = (record.get("authorList") or {}).get("author", [])
    year = str(record.get("pubYear", ""))
    return Paper(
        title=(record.get("title") or "").rstrip("."),
        pmid=record.get("pmid"),
        doi=record.get("doi"),
        year=int(year) if year.isdigit() else None,
        journal=journal or record.get("journalTitle"),
        authors=[a["fullName"] for a in authors if a.get("fullName")],
        abstract=record.get("abstractText"),
        sources=[SOURCE],
    )


async def _query(client: httpx.AsyncClient, params: dict[str, Any]) -> list[Paper]:
    async def fetch() -> dict[str, Any]:
        response = await client.get(BASE, params=params)
        response.raise_for_status()
        return response.json()

    data, _ = await cached(SOURCE, params, fetch)
    return [_to_paper(r) for r in data.get("resultList", {}).get("result", [])]


async def search(
    client: httpx.AsyncClient,
    query: str,
    max_results: int = 20,
    year_from: int | None = None,
) -> list[Paper]:
    term = f"{query} AND PUB_YEAR:[{year_from} TO 3000]" if year_from else query
    return await _query(
        client,
        {"query": term, "format": "json", "pageSize": max_results, "resultType": "core"},
    )


async def fetch_one(client: httpx.AsyncClient, identifier: str) -> Paper | None:
    field = "DOI" if identifier.startswith("10.") else "EXT_ID"
    papers = await _query(
        client,
        {
            "query": f'{field}:"{identifier}"',
            "format": "json",
            "pageSize": 1,
            "resultType": "core",
        },
    )
    return papers[0] if papers else None
