import asyncio
import contextlib
import re
from collections.abc import Awaitable, Callable
from difflib import SequenceMatcher

import httpx

from mcp_server import europe_pmc, pubmed, semantic_scholar
from mcp_server.models import Paper, SearchResponse

RRF_K = 60
TITLE_SIMILARITY = 0.90
SOURCE_TIMEOUT = 8.0
SNIPPET_CHARS = 300
OVERFETCH_FACTOR = 5
MAX_PER_SOURCE = 50
MIN_TITLE_CHARS = 12
PLACEHOLDER_TITLES = frozenset(
    {"abstract", "abstracts", "editorial", "correction", "erratum", "untitled", "front matter"}
)
USER_AGENT = "biomed-mcp/0.1 (https://github.com/prudhiv17/biomed-mcp)"

Searcher = Callable[[httpx.AsyncClient, str, int, int | None], Awaitable[list[Paper]]]

SEARCHERS: dict[str, Searcher] = {
    "pubmed": pubmed.search,
    "europepmc": europe_pmc.search,
    "semanticscholar": semantic_scholar.search,
}


def open_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=SOURCE_TIMEOUT, headers={"User-Agent": USER_AGENT})


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    text = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi.strip().lower())
    return text.rstrip("./") or None


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", title.lower())).strip()


def usable(paper: Paper) -> bool:
    # Sources return conference front matter and stray "Abstract" rows with no identifier.
    # Nothing downstream can cite or verify those, so they never reach fusion.
    title = paper.title.strip()
    if len(title) < MIN_TITLE_CHARS or title.lower().strip(" .:") in PLACEHOLDER_TITLES:
        return False
    return bool(paper.pmid or paper.doi)


def _absorb(target: Paper, other: Paper) -> None:
    target.sources = sorted(set(target.sources) | set(other.sources))
    target.pmid = target.pmid or other.pmid
    target.doi = target.doi or other.doi
    target.year = target.year or other.year
    target.journal = target.journal or other.journal
    target.abstract = target.abstract or other.abstract
    if len(other.authors) > len(target.authors):
        target.authors = other.authors


def _match(
    paper: Paper, by_doi: dict[str, int], by_pmid: dict[str, int], titles: list[str]
) -> int | None:
    doi = normalize_doi(paper.doi)
    if doi and doi in by_doi:
        return by_doi[doi]
    if paper.pmid and paper.pmid in by_pmid:
        return by_pmid[paper.pmid]

    candidate = normalize_title(paper.title)
    if not candidate:
        return None
    for position, existing in enumerate(titles):
        if SequenceMatcher(None, candidate, existing).ratio() >= TITLE_SIMILARITY:
            return position
    return None


def _index(paper: Paper, position: int, by_doi: dict[str, int], by_pmid: dict[str, int]) -> None:
    doi = normalize_doi(paper.doi)
    if doi:
        by_doi.setdefault(doi, position)
    if paper.pmid:
        by_pmid.setdefault(paper.pmid, position)


def fuse(ranked: dict[str, list[Paper]]) -> list[Paper]:
    # Sources score relevance on scales that cannot be compared, so fusion uses
    # rank position only: score = sum over sources of 1 / (k + rank).
    merged: list[Paper] = []
    scores: list[float] = []
    titles: list[str] = []
    by_doi: dict[str, int] = {}
    by_pmid: dict[str, int] = {}
    counted: set[tuple[int, str]] = set()

    for source in sorted(ranked):
        for rank, paper in enumerate(ranked[source], start=1):
            position = _match(paper, by_doi, by_pmid, titles)
            if position is None:
                merged.append(paper.model_copy(deep=True))
                scores.append(0.0)
                titles.append(normalize_title(paper.title))
                position = len(merged) - 1
            else:
                _absorb(merged[position], paper)

            # A source may list one paper several times (Europe PMC indexes the
            # preprint, the PMC copy and the MED record separately). Ranks arrive
            # ascending, so the first sighting is that source's best rank.
            if (position, source) not in counted:
                counted.add((position, source))
                scores[position] += 1.0 / (RRF_K + rank)
            _index(merged[position], position, by_doi, by_pmid)

    for paper, score in zip(merged, scores, strict=True):
        paper.rrf_score = round(score, 6)
    return sorted(merged, key=lambda p: p.rrf_score, reverse=True)


def _reason(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return f"timeout after {SOURCE_TIMEOUT:.0f}s"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _snippet(text: str | None) -> str | None:
    if not text:
        return None
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rstrip() + "…"


async def _run_source(
    name: str,
    client: httpx.AsyncClient,
    query: str,
    max_results: int,
    year_from: int | None,
) -> tuple[str, list[Paper], str | None]:
    try:
        papers = await asyncio.wait_for(
            SEARCHERS[name](client, query, max_results, year_from), SOURCE_TIMEOUT
        )
        return name, papers, None
    except Exception as exc:
        return name, [], _reason(exc)


async def _backfill_abstracts(client: httpx.AsyncClient, papers: list[Paper]) -> None:
    # PubMed's esummary carries no abstract, so anything it alone contributed arrives
    # empty. One batched efetch fills them; failing that, fetch_paper still can.
    missing = [p for p in papers if p.pmid and not p.abstract]
    if not missing:
        return
    with contextlib.suppress(Exception):
        texts = await pubmed.abstracts(client, [p.pmid for p in missing if p.pmid])
        for paper in missing:
            paper.abstract = texts.get(paper.pmid or "")


async def search_all(
    query: str,
    max_results: int = 20,
    year_from: int | None = None,
    sources: list[str] | None = None,
) -> SearchResponse:
    wanted = [name for name in SEARCHERS if sources is None or name in sources]
    # Over-fetch hard: fusion can only reward agreement it is allowed to see, and the
    # extra rows cost nothing downstream because the list is truncated after fusing.
    per_source = min(max_results * OVERFETCH_FACTOR, MAX_PER_SOURCE)

    async with open_client() as client:
        outcomes = await asyncio.gather(
            *(_run_source(name, client, query, per_source, year_from) for name in wanted)
        )
        ranked = {
            name: [p for p in papers if usable(p)]
            for name, papers, error in outcomes
            if error is None
        }
        failed = {name: error for name, _, error in outcomes if error is not None}

        papers = fuse(ranked)[:max_results]
        await _backfill_abstracts(client, papers)

    for paper in papers:
        paper.abstract = _snippet(paper.abstract)

    return SearchResponse(
        query=query,
        papers=papers,
        sources_ok=sorted(ranked),
        sources_failed=failed,
    )


async def fetch_paper(identifier: str) -> Paper | None:
    lookups: dict[str, Callable[[httpx.AsyncClient, str], Awaitable[Paper | None]]] = {
        "europepmc": europe_pmc.fetch_one,
        "semanticscholar": semantic_scholar.fetch_one,
    }
    if identifier.isdigit():
        lookups["pubmed"] = pubmed.fetch_one

    async with open_client() as client:
        outcomes = await asyncio.gather(
            *(fn(client, identifier) for fn in lookups.values()), return_exceptions=True
        )

    found = {
        name: [result]
        for name, result in zip(lookups, outcomes, strict=True)
        if isinstance(result, Paper)
    }
    fused = fuse(found)
    return fused[0] if fused else None


async def find_related(identifier: str, max_results: int = 10) -> list[Paper]:
    async with open_client() as client:
        return await semantic_scholar.related(client, identifier, max_results)


async def get_citations(
    identifier: str, direction: str = "cited_by", max_results: int = 20
) -> list[Paper]:
    async with open_client() as client:
        return await semantic_scholar.citations(client, identifier, direction, max_results)
