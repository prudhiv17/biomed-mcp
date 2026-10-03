import os
import re
import time
from difflib import SequenceMatcher
from typing import Any

from agent import sessions
from agent.model import ModelClient
from storage import get_db

SYSTEM_PROMPT = """\
You answer biomedical research questions from published literature.

Work in this order:
1. search_literature once, with max_results of 10, to find candidates. Give it two to six
   plain words, the way a colleague would say the topic aloud. Never quoted terms, boolean
   operators or piles of synonyms: long queries return fewer papers, not more.
2. Search again only if those results genuinely miss the question, and only with different
   terms. Never repeat a search you have already run.
3. fetch_paper on the two or three papers you intend to cite, to read the full abstract.
4. Answer.

Every factual statement must be traceable to a paper you actually retrieved, identified by
its PMID or DOI. If the retrieved literature does not settle the question, say so rather
than filling the gap. Report the strength of the evidence, not only its direction.
"""

SUMMARISE_PROMPT = """\
Condense this earlier part of a research conversation into a compact briefing.

Keep: the questions asked, the papers found with their PMIDs or DOIs, and the conclusions
reached. Drop: tool mechanics, phrasing, anything already superseded.

Write it as notes, not prose. It replaces the original turns as context for what follows.
"""

FACTS_PROMPT = """\
Name at most two durable facts about this researcher: a subject area they keep returning to,
or a standing preference for how they want evidence presented.

A fact must still be useful weeks from now, must be phrased as one short line, and must not
mention any single question, paper or study. One broad fact beats three narrow restatements
of the same thing. Reply with nothing at all if this session shows nothing durable.

One per line. No numbering, no bullet characters.
"""

MAX_ACTIVE_FACTS = 12
MAX_FACTS_PER_EXTRACTION = 2


def compaction_trigger() -> int:
    return int(os.getenv("COMPACTION_TRIGGER_TOKENS", "8000"))


def session_budget() -> int:
    return int(os.getenv("SESSION_TOKEN_BUDGET", "60000"))


def estimate_tokens(text: str) -> int:
    # Roughly four characters per token; only ever used to decide when to compact.
    return max(1, len(text) // 4)


def active_facts() -> list[str]:
    rows = get_db().execute(
        "SELECT fact FROM facts WHERE active = 1 ORDER BY id DESC LIMIT ?",
        (MAX_ACTIVE_FACTS,),
    ).fetchall()
    return [row["fact"] for row in rows]


def _is_duplicate(candidate: str, known: list[str]) -> bool:
    # Near-duplicates arrive as reworded versions of a fact already stored, so exact
    # matching alone lets the list grow without bound.
    normalised = re.sub(r"[^a-z0-9 ]+", " ", candidate.lower())
    normalised = re.sub(r"\s+", " ", normalised).strip()
    for existing in known:
        other = re.sub(r"[^a-z0-9 ]+", " ", existing.lower())
        other = re.sub(r"\s+", " ", other).strip()
        if SequenceMatcher(None, normalised, other).ratio() >= 0.75:
            return True
    return False


def save_facts(facts: list[str], session_id: str) -> None:
    if not facts:
        return
    db = get_db()
    known = [row["fact"] for row in db.execute("SELECT fact FROM facts WHERE active = 1")]
    now = int(time.time())

    for fact in facts[:MAX_FACTS_PER_EXTRACTION]:
        if _is_duplicate(fact, known):
            continue
        db.execute(
            "INSERT INTO facts (fact, source_session, created_at) VALUES (?, ?, ?)",
            (fact, session_id, now),
        )
        known.append(fact)

    # Retire the oldest rather than letting the injected block grow every session.
    db.execute(
        "UPDATE facts SET active = 0 WHERE id NOT IN"
        " (SELECT id FROM facts WHERE active = 1 ORDER BY id DESC LIMIT ?)",
        (MAX_ACTIVE_FACTS,),
    )
    db.commit()


def _render(turn: sessions.Turn) -> dict[str, str]:
    if turn.role == "tool":
        return {"role": "assistant", "content": f"[{turn.tool_name} returned]\n{turn.content}"}
    return {"role": turn.role, "content": turn.content}


def build_context(session_id: str, question: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    facts = active_facts()
    if facts:
        joined = "\n".join(f"- {f}" for f in facts)
        messages.append({"role": "system", "content": f"About this researcher:\n{joined}"})

    earlier = sessions.summary(session_id)
    if earlier:
        messages.append({"role": "system", "content": f"Earlier in this session:\n{earlier}"})

    messages.extend(_render(turn) for turn in sessions.turns(session_id))
    messages.append({"role": "user", "content": question})
    return messages


def history_tokens(session_id: str) -> int:
    return sum(estimate_tokens(turn.content) for turn in sessions.turns(session_id))


async def compact_if_needed(session_id: str, model: ModelClient, keep_recent: int = 4) -> bool:
    # Summarise rather than drop: the oldest turns are folded into sessions.summary so
    # their content survives at a fraction of the tokens.
    pending = sessions.turns(session_id)
    if len(pending) <= keep_recent or history_tokens(session_id) < compaction_trigger():
        return False

    older = pending[:-keep_recent]
    transcript = "\n\n".join(f"[{t.role}] {t.content}" for t in older)
    previous = sessions.summary(session_id)
    if previous:
        transcript = f"Existing briefing:\n{previous}\n\nNew turns:\n{transcript}"

    result = await model.complete(
        [
            {"role": "system", "content": SUMMARISE_PROMPT},
            {"role": "user", "content": transcript},
        ],
        model=model.fallback,
        session_id=session_id,
    )
    if not result.content:
        return False

    sessions.set_summary(session_id, result.content.strip())
    sessions.mark_compacted([t.id for t in older])
    return True


async def extract_facts(session_id: str, model: ModelClient) -> list[str]:
    history = sessions.turns(session_id, include_compacted=True)
    if not history:
        return []
    transcript = "\n\n".join(f"[{t.role}] {t.content[:1200]}" for t in history)

    result = await model.complete(
        [
            {"role": "system", "content": FACTS_PROMPT},
            {"role": "user", "content": transcript},
        ],
        model=model.fallback,
        session_id=session_id,
    )
    lines = [line.strip("-• ").strip() for line in (result.content or "").splitlines()]
    facts = [line for line in lines if len(line) > 8]
    save_facts(facts, session_id)
    return facts
