import logging
import sys
from typing import Literal

from mcp.server import MCPServer

from mcp_server import federation, prompts, resources
from mcp_server.models import Paper, SearchResponse
from storage import get_db

# stdout carries the MCP protocol on the stdio transport, so logs must go to stderr.
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

get_db()

server = MCPServer(
    name="biomed-literature",
    version="0.1.0",
    instructions=(
        "Federated biomedical literature search. Cite every claim with the PMID or DOI of a"
        " paper returned by these tools, and read an abstract before relying on it."
    ),
)


@server.tool()
async def search_literature(
    query: str,
    max_results: int = 20,
    year_from: int | None = None,
    sources: list[str] | None = None,
) -> SearchResponse:
    """Search PubMed, Europe PMC and Semantic Scholar together for papers on a topic.

    Use this to discover papers when you do not already have an identifier. Results are
    merged across the three databases, deduplicated, and ranked by agreement between them.
    Abstracts come back shortened; call fetch_paper for the full text of one.

    Args:
        query: Two to six plain words naming the topic, for example "ethanol content of
            cough syrup". No quotation marks, no boolean operators, no strings of
            synonyms. PubMed combines every term with AND, so a long query matches
            fewer papers, not more: nine words routinely returns nothing at all.
        max_results: Maximum papers to return after merging.
        year_from: Restrict to papers published in this year or later.
        sources: Limit the search to named databases; defaults to all three.
    """
    return await federation.search_all(query, max_results, year_from, sources)


@server.tool()
async def fetch_paper(paper_id: str) -> Paper:
    """Retrieve one paper's full metadata and complete abstract by identifier.

    Use this once you have a PMID or DOI and need the whole abstract, typically to check
    what a paper actually supports before citing it.

    Args:
        paper_id: A PubMed ID such as 29617642, or a DOI such as 10.1056/NEJMoa1800256.
    """
    paper = await resources.resolve_paper(paper_id)
    if paper is None:
        raise ValueError(f"No paper found for identifier {paper_id!r}")
    return paper


@server.tool()
async def find_related(paper_id: str, max_results: int = 10) -> list[Paper]:
    """Find papers on similar subject matter to a given paper.

    Use this to widen a search from one relevant paper to others like it. Relatedness is
    by topic, not by citation, so results need not cite or be cited by the paper.

    Args:
        paper_id: PMID or DOI of the paper to find neighbours of.
        max_results: Maximum related papers to return.
    """
    return await federation.find_related(paper_id, max_results)


@server.tool()
async def get_citations(
    paper_id: str,
    direction: Literal["cited_by", "references"] = "cited_by",
    max_results: int = 20,
) -> list[Paper]:
    """List the papers citing a paper, or the papers it cites.

    Use this to trace a citation trail: 'cited_by' for later work building on it,
    'references' for the earlier work it was built on.

    Args:
        paper_id: PMID or DOI of the paper.
        direction: 'cited_by' for papers citing it, 'references' for its bibliography.
        max_results: Maximum papers to return.
    """
    return await federation.get_citations(paper_id, direction, max_results)


resources.register(server)
prompts.register(server)


def main() -> None:
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport not in {"stdio", "streamable-http"}:
        raise SystemExit(f"unknown transport {transport!r}; use stdio or streamable-http")
    server.run(transport=transport)


if __name__ == "__main__":
    main()
