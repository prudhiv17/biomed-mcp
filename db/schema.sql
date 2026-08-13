-- Applied once on first run. Every statement is IF NOT EXISTS so it is safe to re-run.

-- WAL persists on the database file: lets the Streamlit UI read while the harness writes.
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS api_cache (
  cache_key   TEXT PRIMARY KEY,          -- source + sha256(params)
  payload     TEXT NOT NULL,             -- raw JSON
  created_at  INTEGER NOT NULL,
  ttl_seconds INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_created ON api_cache(created_at);

CREATE TABLE IF NOT EXISTS papers (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  pmid       TEXT,
  doi        TEXT,
  title      TEXT NOT NULL,
  abstract   TEXT,
  authors    TEXT,
  year       INTEGER,
  journal    TEXT,
  sources    TEXT,                       -- which APIs returned it
  fetched_at INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_pmid ON papers(pmid) WHERE pmid IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_doi  ON papers(doi)  WHERE doi  IS NOT NULL;

CREATE TABLE IF NOT EXISTS sessions (
  id          TEXT PRIMARY KEY,
  created_at  INTEGER,
  updated_at  INTEGER,
  token_total INTEGER DEFAULT 0,
  summary     TEXT                       -- compacted history block
);

CREATE TABLE IF NOT EXISTS turns (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT REFERENCES sessions(id),
  role       TEXT NOT NULL,              -- user | assistant | tool
  content    TEXT NOT NULL,
  tool_name  TEXT,
  tokens     INTEGER,
  compacted  INTEGER DEFAULT 0,          -- 1 once folded into sessions.summary
  created_at INTEGER
);
-- Context assembly reads exactly this: uncompacted turns for one session, in order.
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, compacted, id);

CREATE TABLE IF NOT EXISTS facts (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  fact           TEXT NOT NULL,          -- stable user preference/topic
  source_session TEXT,
  active         INTEGER DEFAULT 1,
  created_at     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_facts_active ON facts(active);

CREATE TABLE IF NOT EXISTS evidence (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  paper_ref  TEXT,                       -- pmid or doi
  chunk      TEXT,
  added_at   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_evidence_session ON evidence(session_id);

-- Verifier output. Persisted so the grounding eval scores stored verdicts
-- instead of re-running the whole pipeline.
CREATE TABLE IF NOT EXISTS claims (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id    TEXT REFERENCES sessions(id),
  turn_id       INTEGER REFERENCES turns(id),
  claim         TEXT NOT NULL,
  citation_refs TEXT NOT NULL,           -- JSON array of pmid/doi
  verdict       TEXT,                    -- supported | weak | unsupported
  rationale     TEXT,
  created_at    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_claims_session ON claims(session_id);
CREATE INDEX IF NOT EXISTS idx_claims_verdict ON claims(verdict);

CREATE TABLE IF NOT EXISTS audit_log (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  tool       TEXT,
  args_hash  TEXT,
  status     TEXT,                       -- ok | error | cache_hit
  latency_ms INTEGER,
  tokens_in  INTEGER,
  tokens_out INTEGER,
  created_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id, created_at);
