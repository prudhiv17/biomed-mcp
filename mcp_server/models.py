from pydantic import BaseModel, Field


class Paper(BaseModel):
    title: str
    pmid: str | None = None
    doi: str | None = None
    year: int | None = None
    journal: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    sources: list[str] = Field(default_factory=list)
    rrf_score: float = 0.0


class SearchResponse(BaseModel):
    query: str
    papers: list[Paper]
    sources_ok: list[str]
    sources_failed: dict[str, str] = Field(default_factory=dict)


class Citation(BaseModel):
    paper: Paper
    direction: str
