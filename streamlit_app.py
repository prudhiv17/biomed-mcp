import asyncio
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from agent import loop, memory, sessions, verifier

st.set_page_config(page_title="Biomedical Literature Agent", page_icon="🔬", layout="wide")

BADGE = {
    "supported": ("🟢", "#1a7f37"),
    "weak": ("🟠", "#bf8700"),
    "unsupported": ("🔴", "#cf222e"),
}


def when(timestamp: int | None) -> str:
    if not timestamp:
        return "-"
    return datetime.fromtimestamp(timestamp).astimezone().strftime("%d %b %H:%M")


def reference_link(ref: str) -> str:
    ref = verifier.normalize_ref(ref)
    if ref.startswith("10."):
        return f"[{ref}](https://doi.org/{ref})"
    if ref.isdigit():
        return f"[PMID {ref}](https://pubmed.ncbi.nlm.nih.gov/{ref}/)"
    return ref


def sidebar() -> str | None:
    st.sidebar.title("🔬 Literature Agent")

    if st.sidebar.button("New session", width="stretch"):
        st.session_state.session_id = sessions.create_session()
        st.rerun()

    existing = sessions.list_sessions()
    st.sidebar.caption(f"{len(existing)} stored session(s)")

    for row in existing:
        label = sessions.first_question(row["id"])[:44]
        active = row["id"] == st.session_state.get("session_id")
        if st.sidebar.button(
            f"{'▸ ' if active else ''}{label}",
            key=f"sess-{row['id']}",
            width="stretch",
        ):
            st.session_state.session_id = row["id"]
            st.rerun()
        st.sidebar.caption(
            f"{when(row['updated_at'])} · {row['turn_count']} turns · {row['token_total']:,} tok"
        )

    facts = memory.active_facts()
    if facts:
        with st.sidebar.expander(f"Long-term memory ({len(facts)})"):
            for fact in facts:
                st.write(f"- {fact}")

    return st.session_state.get("session_id")


def render_tool_group(group: list[sessions.Turn]) -> None:
    if not group:
        return
    used = list(dict.fromkeys(t.tool_name or "tool" for t in group))
    label = f"🔧 {len(group)} tool call{'s' if len(group) > 1 else ''} · {', '.join(used)}"
    with st.chat_message("assistant"), st.expander(label):
        for turn in group:
            st.caption(turn.tool_name or "tool")
            st.code(turn.content[:1500], language="json")


def render_chat(session_id: str | None) -> None:
    if session_id:
        group: list[sessions.Turn] = []
        for turn in sessions.turns(session_id, include_compacted=True):
            if turn.role == "tool":
                group.append(turn)
                continue
            render_tool_group(group)
            group = []
            if turn.role in {"user", "assistant"}:
                st.chat_message(turn.role).write(turn.content)
        render_tool_group(group)

    question = st.chat_input("Ask a biomedical research question")
    if not question:
        return

    st.chat_message("user").write(question)
    with st.chat_message("assistant"), st.spinner("Searching literature and verifying claims..."):
        try:
            result = asyncio.run(loop.run(question, session_id))
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
            return

    st.session_state.session_id = result.session_id
    st.session_state.last_result = result.model_dump()
    st.rerun()


def render_evidence(session_id: str | None) -> None:
    if not session_id:
        st.info("Ask a question to see per-claim verification.")
        return

    rows = verifier.verdicts_for(session_id)
    if not rows:
        st.info("No verified claims in this session yet.")
        return

    counts = {k: sum(1 for r in rows if r["verdict"] == k) for k in BADGE}
    total = len(rows)
    columns = st.columns(4)
    columns[0].metric("Claims", total)
    for column, name in zip(columns[1:], BADGE, strict=False):
        share = counts[name] / total * 100 if total else 0
        column.metric(name.capitalize(), counts[name], f"{share:.0f}%")

    st.divider()
    for row in rows:
        icon, colour = BADGE.get(row["verdict"], ("⚪", "#666"))
        refs = json.loads(row["citation_refs"] or "[]")
        links = " · ".join(reference_link(r) for r in refs) or "no citation"
        st.markdown(
            f"{icon} **<span style='color:{colour}'>{row['verdict']}</span>** — {row['claim']}",
            unsafe_allow_html=True,
        )
        st.caption(f"{links} · {row['rationale']}")


def render_audit() -> None:
    calls = sessions.recent_calls(80)
    if not calls:
        st.info("No tool or model calls recorded yet.")
        return

    tool_calls = [c for c in calls if not c["tool"].startswith("model:")]
    model_calls = [c for c in calls if c["tool"].startswith("model:")]
    columns = st.columns(4)
    columns[0].metric("Calls logged", len(calls))
    columns[1].metric("Tool calls", len(tool_calls))
    columns[2].metric("Model calls", len(model_calls))
    columns[3].metric("Tokens", f"{sum(c['tokens_in'] + c['tokens_out'] for c in calls):,}")

    st.dataframe(
        [
            {
                "when": when(c["created_at"]),
                "call": c["tool"],
                "status": c["status"],
                "ms": c["latency_ms"],
                "in": c["tokens_in"],
                "out": c["tokens_out"],
            }
            for c in calls
        ],
        width="stretch",
        hide_index=True,
    )


def render_evals() -> None:
    results = Path(__file__).parent / "evaluation" / "results"
    files = sorted(results.glob("*.json")) if results.exists() else []
    if not files:
        st.info("No eval results yet. These appear once the Day 3 eval harness has been run.")
        return
    for path in files:
        with st.expander(path.name):
            st.json(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    session_id = sidebar()
    chat, evidence, audit, evals = st.tabs(["Chat", "Evidence", "Audit", "Evals"])

    with chat:
        last = st.session_state.get("last_result")
        if last:
            columns = st.columns(4)
            columns[0].metric("Steps", last["steps"])
            columns[1].metric("Tokens", f"{last['tokens_used']:,}")
            columns[2].metric("Claims", len(last["verdicts"]))
            columns[3].metric("Stopped", last["stopped_because"])
            if last.get("compacted"):
                st.caption("History was compacted into a summary before this turn.")
        render_chat(session_id)

    with evidence:
        render_evidence(session_id)

    with audit:
        render_audit()

    with evals:
        render_evals()


main()
