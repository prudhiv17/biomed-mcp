import json
import re
import time

from pydantic import BaseModel, Field

from agent.mcp_client import ToolBridge
from agent.model import ModelClient
from storage import get_db

JUDGE_PROMPT = """\
Decide whether the abstract below supports the claim.

supported   - the abstract states this, or states something it follows from directly
weak        - the abstract points this way but hedges, or reports it in a narrower
              population, smaller effect or different endpoint than the claim implies
unsupported - the abstract does not address this, or contradicts it

Judge only against the text given. Outside knowledge that the claim happens to be true is
not evidence here. Keep the rationale to one sentence.
"""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["supported", "weak", "unsupported"]},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "rationale"],
    "additionalProperties": False,
}


class Claim(BaseModel):
    claim: str
    citation_refs: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    claim: str
    citation_refs: list[str]
    verdict: str
    rationale: str


def normalize_ref(ref: str) -> str:
    cleaned = re.sub(r"^(pmid|doi)\s*[:\s]\s*", "", ref.strip(), flags=re.IGNORECASE)
    return cleaned.strip().rstrip(".,;")


async def _abstract_for(bridge: ToolBridge, ref: str, cache: dict[str, str]) -> str:
    if ref in cache:
        return cache[ref]
    text, failed = await bridge.call("fetch_paper", {"paper_id": ref})
    if failed:
        cache[ref] = ""
        return ""
    try:
        paper = json.loads(text)
    except json.JSONDecodeError:
        cache[ref] = ""
        return ""
    parts = [paper.get("title") or "", paper.get("abstract") or ""]
    cache[ref] = "\n".join(p for p in parts if p)
    return cache[ref]


def save(session_id: str, turn_id: int | None, verdicts: list[Verdict]) -> None:
    db = get_db()
    now = int(time.time())
    for v in verdicts:
        db.execute(
            "INSERT INTO claims (session_id, turn_id, claim, citation_refs, verdict, rationale,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                turn_id,
                v.claim,
                json.dumps(v.citation_refs),
                v.verdict,
                v.rationale,
                now,
            ),
        )
    db.commit()


async def verify(
    session_id: str,
    claims: list[Claim],
    bridge: ToolBridge,
    model: ModelClient,
    turn_id: int | None = None,
) -> list[Verdict]:
    # Fail-forward: a claim that cannot be checked is reported as unsupported rather than
    # blocking the answer.
    cache: dict[str, str] = {}
    verdicts: list[Verdict] = []

    for claim in claims:
        refs = [normalize_ref(r) for r in claim.citation_refs if r.strip()]
        evidence: list[str] = []
        for ref in refs:
            abstract = await _abstract_for(bridge, ref, cache)
            if abstract:
                evidence.append(f"[{ref}]\n{abstract}")

        if not evidence:
            verdicts.append(
                Verdict(
                    claim=claim.claim,
                    citation_refs=refs,
                    verdict="unsupported",
                    rationale="No abstract could be retrieved for the cited reference.",
                )
            )
            continue

        try:
            result = await model.complete(
                [
                    {"role": "system", "content": JUDGE_PROMPT},
                    {
                        "role": "user",
                        "content": f"Claim:\n{claim.claim}\n\nAbstract(s):\n"
                        + "\n\n".join(evidence)[:8000],
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "verdict",
                        "schema": VERDICT_SCHEMA,
                        "strict": True,
                    },
                },
                model=model.fallback,
                session_id=session_id,
            )
            parsed = json.loads(result.content or "{}")
            verdicts.append(
                Verdict(
                    claim=claim.claim,
                    citation_refs=refs,
                    verdict=parsed.get("verdict", "unsupported"),
                    rationale=parsed.get("rationale", ""),
                )
            )
        except Exception as exc:
            verdicts.append(
                Verdict(
                    claim=claim.claim,
                    citation_refs=refs,
                    verdict="unsupported",
                    rationale=f"Verification failed: {type(exc).__name__}",
                )
            )

    save(session_id, turn_id, verdicts)
    return verdicts


def verdicts_for(session_id: str) -> list[dict[str, str]]:
    rows = get_db().execute(
        "SELECT claim, citation_refs, verdict, rationale FROM claims WHERE session_id = ?"
        " ORDER BY id",
        (session_id,),
    ).fetchall()
    return [dict(row) for row in rows]
