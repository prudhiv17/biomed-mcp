import json
import logging
import os
from typing import Any

from pydantic import BaseModel, Field

from agent import mcp_client, memory, sessions, verifier
from agent.model import ModelClient
from agent.verifier import Claim, Verdict

log = logging.getLogger(__name__)

TOOL_RESULT_CHARS = 5000
MAX_SEARCHES = 3
COMPACT_ABSTRACT_CHARS = 220
KEEP_AUTHORS = 3

SYNTHESIS_INSTRUCTION = """\
Now answer the question using only what the tools returned.

Write the answer as several short paragraphs, never a single sentence:

1. Answer the question directly, in the terms it was asked.
2. Set out what the evidence actually shows, carrying across the concrete numbers the
   abstracts report: concentrations, effect sizes, sample sizes, populations, durations.
   Name the papers in the prose, by journal and year.
3. State where the evidence is thin, where studies disagree, and where it does not
   transfer to the population the question is really about.

Then break that answer into individual claims. Each claim carries the PMID or DOI of the
paper it rests on, must be specific enough to check against that abstract on its own, and
must not restate another claim. A statement you cannot attribute to a retrieved paper does
not belong in the claims at all.
"""

SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "citation_refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["claim", "citation_refs"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "claims"],
    "additionalProperties": False,
}


class AgentResult(BaseModel):
    session_id: str
    answer: str
    verdicts: list[Verdict] = Field(default_factory=list)
    steps: int = 0
    tools_used: list[str] = Field(default_factory=list)
    tokens_used: int = 0
    stopped_because: str = "completed"
    compacted: bool = False


def max_steps() -> int:
    return int(os.getenv("MAX_AGENT_STEPS", "8"))


def compact_result(text: str) -> str:
    # Blind truncation cut the JSON mid-structure and handed the model malformed data.
    # Dropping author lists and long abstracts keeps it valid and costs fewer tokens.
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text[:TOOL_RESULT_CHARS]

    papers = payload.get("papers") if isinstance(payload, dict) else None
    if isinstance(papers, list):
        for paper in papers:
            authors = paper.get("authors") or []
            if len(authors) > KEEP_AUTHORS:
                paper["authors"] = [*authors[:KEEP_AUTHORS], "et al."]
            abstract = paper.get("abstract")
            if isinstance(abstract, str) and len(abstract) > COMPACT_ABSTRACT_CHARS:
                paper["abstract"] = abstract[:COMPACT_ABSTRACT_CHARS].rstrip() + "…"

    def encode() -> str:
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    # Drop whole papers off the tail rather than slicing the string, which would hand
    # the model a JSON document that no longer parses.
    compacted = encode()
    while len(compacted) > TOOL_RESULT_CHARS and isinstance(papers, list) and papers:
        papers.pop()
        compacted = encode()
    return compacted[:TOOL_RESULT_CHARS]


def _assistant_message(completion: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": completion.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in completion.tool_calls
        ],
    }


async def run(question: str, session_id: str | None = None) -> AgentResult:
    session_id = sessions.ensure_session(session_id)
    model = ModelClient()
    budget = memory.session_budget()

    compacted = await memory.compact_if_needed(session_id, model)
    messages = memory.build_context(session_id, question)
    sessions.add_turn(session_id, "user", question)

    steps = 0
    searches = 0
    seen_calls: set[str] = set()
    tools_used: list[str] = []
    stopped = "completed"

    async with mcp_client.connect(session_id) as bridge:
        tools = bridge.openai_tools()

        while steps < max_steps():
            if sessions.token_total(session_id) >= budget:
                stopped = "token budget exhausted"
                break

            completion = await model.complete(messages, tools=tools, session_id=session_id)
            steps += 1

            if not completion.tool_calls:
                if completion.content:
                    messages.append({"role": "assistant", "content": completion.content})
                break

            messages.append(_assistant_message(completion))
            for call in completion.tool_calls:
                try:
                    arguments = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                # A failing tool is reported back to the model as a message, never raised:
                # the model can correct its arguments or choose a different tool. The same
                # channel refuses repeated and excessive searches, which is what stopped the
                # loop burning its whole budget re-running near-identical queries.
                signature = f"{call.name}:{json.dumps(arguments, sort_keys=True)}"
                if signature in seen_calls:
                    text, failed = (
                        "You already made this exact call. Use the earlier result, call"
                        " fetch_paper on a specific paper, or answer now.",
                        True,
                    )
                elif call.name == "search_literature" and searches >= MAX_SEARCHES:
                    text, failed = (
                        f"Search limit of {MAX_SEARCHES} reached for this question. Use"
                        " fetch_paper on a specific paper, or answer with what you have.",
                        True,
                    )
                else:
                    text, failed = await bridge.call(call.name, arguments)
                    seen_calls.add(signature)
                    searches += call.name == "search_literature"

                tools_used.append(call.name)
                trimmed = compact_result(text)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": trimmed}
                )
                sessions.add_turn(session_id, "tool", trimmed, tool_name=call.name)
                if failed:
                    log.info("tool %s returned an error to the model", call.name)
        else:
            stopped = "step cap reached"

        synthesis = await model.complete(
            [*messages, {"role": "user", "content": SYNTHESIS_INSTRUCTION}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "synthesis",
                    "schema": SYNTHESIS_SCHEMA,
                    "strict": True,
                },
            },
            session_id=session_id,
        )

        try:
            payload = json.loads(synthesis.content or "{}")
        except json.JSONDecodeError:
            payload = {}

        answer = payload.get("answer", "") or "No answer could be produced."
        claims = [Claim(**item) for item in payload.get("claims", [])]
        turn_id = sessions.add_turn(session_id, "assistant", answer)
        verdicts = await verifier.verify(session_id, claims, bridge, model, turn_id)

    # Every turn was far too often: one question produced a dozen reworded near-duplicates.
    asked = sum(1 for t in sessions.turns(session_id, include_compacted=True) if t.role == "user")
    if asked >= 2 and asked % 2 == 0:
        await memory.extract_facts(session_id, model)

    return AgentResult(
        session_id=session_id,
        answer=answer,
        verdicts=verdicts,
        steps=steps,
        tools_used=tools_used,
        tokens_used=sessions.token_total(session_id),
        stopped_because=stopped,
        compacted=compacted,
    )
