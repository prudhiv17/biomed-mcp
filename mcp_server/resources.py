import json
import time

from mcp.server import MCPServer

from mcp_server import federation
from mcp_server.models import Paper
from storage import get_db


def save_paper(paper: Paper) -> None:
    db = get_db()
    row = None
    if paper.pmid:
        row = db.execute("SELECT id FROM papers WHERE pmid = ?", (paper.pmid,)).fetchone()
    if row is None and paper.doi:
        row = db.execute("SELECT id FROM papers WHERE doi = ?", (paper.doi,)).fetchone()

    values = (
        paper.pmid,
        paper.doi,
        paper.title,
        paper.abstract,
        json.dumps(paper.authors),
        paper.year,
        paper.journal,
        json.dumps(paper.sources),
        int(time.time()),
    )
    if row is None:
        db.execute(
            "INSERT INTO papers (pmid, doi, title, abstract, authors, year, journal, sources,"
            " fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
    else:
        db.execute(
            "UPDATE papers SET pmid = ?, doi = ?, title = ?, abstract = ?, authors = ?,"
            " year = ?, journal = ?, sources = ?, fetched_at = ? WHERE id = ?",
            (*values, row["id"]),
        )
    db.commit()


def load_paper(identifier: str) -> Paper | None:
    row = get_db().execute(
        "SELECT * FROM papers WHERE pmid = ? OR doi = ?", (identifier, identifier)
    ).fetchone()
    if row is None:
        return None
    return Paper(
        title=row["title"],
        pmid=row["pmid"],
        doi=row["doi"],
        year=row["year"],
        journal=row["journal"],
        authors=json.loads(row["authors"] or "[]"),
        abstract=row["abstract"],
        sources=json.loads(row["sources"] or "[]"),
    )


async def resolve_paper(identifier: str) -> Paper | None:
    cached = load_paper(identifier)
    if cached is not None and cached.abstract:
        return cached
    paper = await federation.fetch_paper(identifier)
    if paper is not None:
        save_paper(paper)
    return paper


def evidence_for(session_id: str) -> list[dict[str, str]]:
    rows = get_db().execute(
        "SELECT paper_ref, chunk, added_at FROM evidence WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def register(server: MCPServer) -> None:
    @server.resource("paper://{pmid}", mime_type="application/json")
    async def paper_resource(pmid: str) -> str:
        """Cached metadata and abstract for one PubMed record."""
        paper = await resolve_paper(pmid)
        if paper is None:
            return json.dumps({"error": f"no record found for {pmid}"})
        return paper.model_dump_json(indent=2)

    @server.resource("session://{session_id}/evidence", mime_type="application/json")
    async def evidence_resource(session_id: str) -> str:
        """Evidence gathered so far in one research session."""
        return json.dumps(
            {"session_id": session_id, "evidence": evidence_for(session_id)}, indent=2
        )
