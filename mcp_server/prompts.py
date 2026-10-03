from mcp.server import MCPServer

EVIDENCE_REVIEW = """\
Write a structured evidence review on: {topic}

Search the literature first, then fetch the abstracts of the papers you intend to cite.
Base every statement on those abstracts alone.

Structure:
1. Question in one sentence.
2. What the evidence shows, as short claims. Every claim carries the PMID or DOI it rests on.
3. Where studies disagree, and on what.
4. Gaps: what the retrieved literature does not answer.

Rules:
- Cite at most {max_papers} papers, preferring recent work and larger studies.
- A claim you cannot attribute to a retrieved abstract does not belong in the review.
- Report the strength of evidence, not just its direction.
"""

PICO_QUESTION = """\
Reformulate this clinical question in PICO form: {question}

Return:
- Population: who is studied, with the qualifiers that matter
- Intervention: what is being done or measured
- Comparison: the alternative, or "none" if the question is not comparative
- Outcome: the endpoint that answers the question

Then give a literature search query built from those four parts, using the terminology
a biomedical database would index rather than the wording of the original question.
"""


def register(server: MCPServer) -> None:
    @server.prompt()
    def evidence_review(topic: str, max_papers: int = 10) -> str:
        """Structured, citation-bound literature review on a biomedical topic."""
        return EVIDENCE_REVIEW.format(topic=topic, max_papers=max_papers)

    @server.prompt()
    def pico_question(question: str) -> str:
        """Recast a clinical question as Population, Intervention, Comparison, Outcome."""
        return PICO_QUESTION.format(question=question)
