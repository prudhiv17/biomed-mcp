import os
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from mcp_server.cache import cached
from mcp_server.models import Paper
from mcp_server.rate_limit import RateLimiter, with_retry

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SOURCE = "pubmed"

_limiter: RateLimiter | None = None


def _get_limiter() -> RateLimiter:
    # NCBI permits 10 requests/second with a key and 3 without.
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(10.0 if os.getenv("NCBI_API_KEY") else 3.0)
    return _limiter


def _identity() -> dict[str, str]:
    params = {"tool": os.getenv("NCBI_TOOL", "biomed-mcp")}
    email = os.getenv("NCBI_EMAIL", "")
    if email and email != "ADD_VALUE_HERE":
        params["email"] = email
    key = os.getenv("NCBI_API_KEY", "")
    if key:
        params["api_key"] = key
    return params


async def _get(client: httpx.AsyncClient, endpoint: str, params: dict[str, Any]) -> httpx.Response:
    async def send() -> httpx.Response:
        await _get_limiter().acquire()
        return await client.get(f"{BASE}/{endpoint}", params={**params, **_identity()})

    response = await with_retry(send)
    response.raise_for_status()
    return response


def _year_of(pubdate: str | None) -> int | None:
    head = (pubdate or "")[:4]
    return int(head) if head.isdigit() else None


def _to_paper(doc: dict[str, Any]) -> Paper:
    ids = {a.get("idtype"): a.get("value") for a in doc.get("articleids", [])}
    return Paper(
        title=(doc.get("title") or "").rstrip("."),
        pmid=str(doc["uid"]) if doc.get("uid") else None,
        doi=ids.get("doi"),
        year=_year_of(doc.get("pubdate")),
        journal=doc.get("fulljournalname") or doc.get("source"),
        authors=[a["name"] for a in doc.get("authors", []) if a.get("name")],
        sources=[SOURCE],
    )


async def search(
    client: httpx.AsyncClient,
    query: str,
    max_results: int = 20,
    year_from: int | None = None,
) -> list[Paper]:
    term = f"{query} AND {year_from}:3000[dp]" if year_from else query
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }

    async def fetch() -> list[str]:
        response = await _get(client, "esearch.fcgi", params)
        return response.json()["esearchresult"].get("idlist", [])

    pmids, _ = await cached(f"{SOURCE}:esearch", params, fetch)
    return await summaries(client, pmids) if pmids else []


async def summaries(client: httpx.AsyncClient, pmids: list[str]) -> list[Paper]:
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"}

    async def fetch() -> dict[str, Any]:
        response = await _get(client, "esummary.fcgi", params)
        return response.json()["result"]

    result, _ = await cached(f"{SOURCE}:esummary", params, fetch)
    return [_to_paper(result[pmid]) for pmid in pmids if pmid in result]


def _parse_abstracts(xml: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for article in ET.fromstring(xml).iter("PubmedArticle"):
        pmid = article.find(".//PMID")
        if pmid is None or not pmid.text:
            continue
        segments = ["".join(node.itertext()).strip() for node in article.iter("AbstractText")]
        text = " ".join(s for s in segments if s)
        if text:
            found[pmid.text] = text
    return found


async def abstracts(client: httpx.AsyncClient, pmids: list[str]) -> dict[str, str]:
    # esummary carries no abstract, and efetch answers only in XML.
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}

    async def fetch() -> str:
        response = await _get(client, "efetch.fcgi", params)
        return response.text

    xml, _ = await cached(f"{SOURCE}:efetch", params, fetch)
    return _parse_abstracts(xml)


async def fetch_one(client: httpx.AsyncClient, pmid: str) -> Paper | None:
    papers = await summaries(client, [pmid])
    if not papers:
        return None
    paper = papers[0]
    paper.abstract = (await abstracts(client, [pmid])).get(pmid)
    return paper
