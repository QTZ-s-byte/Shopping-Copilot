# Shopping Copilot

Track 4 submission workspace for TikTok TechJam 2026.  The implementation is
an offline-first, headless shopping Agent: it routes Buying/Browsing intent,
maintains structured multi-turn context, retrieves catalog candidates, ranks
them, and returns the exact interface expected by the public evaluator.

## Current C-module implementation

The integrated project uses one canonical contract. C owns the lifecycle
plumbing while A and B provide implementations directly against that contract:

```text
starter.agent.Agent
  -> ShoppingOrchestrator
       -> IntentRouter       (Member A: agent/intent_router.py)
       -> InMemoryContextMemory
       -> HybridRetriever + RuleRanker (Member B)
       -> TraceSink + fallback/retry policy
```

The shared contracts are in `shopping_copilot/contracts.py`; see
`docs/integration_contract_v1.md` for the integration rules. The orchestrator
enforces the official `top_k=10` contract, rejects a turn beyond 10 without
calling a plugin, filters duplicate/invalid catalog IDs, supports idempotent
repeated requests, and never stores secrets in traces. Member B's BM25/rule
ranking path is primary when its optional dependency is installed. TF-IDF
semantic reranking is opt-in through `SHOPPING_COPILOT_USE_SEMANTIC=1` because
the full 50,000-item matrix increases memory and latency. The offline SQLite
FTS5 path is selected automatically when the BM25 dependency is unavailable or
when `SHOPPING_COPILOT_FORCE_FALLBACK=1` is set for a deterministic smoke run.

## Official participant kit

The public evaluator and public development set are included for local work.
The 50,000-item catalog is intentionally ignored by Git.  Download
`catalog.jsonl.gz` from the official `participant-kit` release, verify the
published SHA256, and decompress it to `data/catalog.jsonl`.

The expected catalog checksum is:

```text
07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8
```

Python 3.10+ is recommended. No third-party Python dependency is required by
the C fallback path; the primary B path uses the packages in
`requirements.txt`.

## Run the public evaluator

From the repository root:

```bash
python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output results.json
```

The equivalent C-module wrapper is:

```bash
python -m evaluation.run \
  --root . \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output results.json
```

The evaluator is the source of truth for scoring.  It calls:

```python
Agent.reset(session_id, user_profile)
Agent.respond(session_id, user_message, turn, 10)
```

Only the first 10 valid, unique `parent_asin` values are scored.  Hits are
exact catalog-ID matches.  The official metrics are Hit Rate@10, MRR, MTTC,
Efficiency, and the recommended TechnicalScore.

## Integrated Members A/B implementations

Member A should implement:

```python
class IntentRouter:
    def classify(self, user_message: str, state: SessionState) -> IntentResult:
        ...
```

Member B should implement:

```python
class Retriever:
    def retrieve(self, query: str, state: SessionState, top_k: int):
        ...

class Ranker:
    def rank(self, query: str, candidates: list[Candidate], state: SessionState):
        ...
```

These implementations are wired directly by `starter/agent.py`. They use the
canonical types in `shopping_copilot/contracts.py`; the retriever returns a
`RetrievalResult` with `total_count` so the C lifecycle can detect broad-query
clarification cases.

## Tests

Use the bundled Python runtime or any Python 3.10+ installation:

```bash
python -m unittest discover -s tests -v
```

The tests cover state accumulation/removal/replacement, bounded history,
idempotency, invalid catalog IDs, broad-query clarification, ranker failure
fallback, the 10-turn boundary, and the official metric formulas.

## Track 4 constraints kept by this code

- maximum 10 turns per session;
- read-only catalog and no fabricated ASINs;
- in-memory session context and lightweight local indexing;
- text-only, single-user evaluation;
- no required UI or hosted model dependency;
- model/API usage can be added behind an offline fallback and must use
  environment variables rather than committed credentials.

## Trace format

`InMemoryTraceSink` is useful for tests and `JSONLTraceSink` can be enabled for
development.  Events include a session/request digest, turn, intent change,
state diff, candidate count, ask attribute, latency, retries, fallback stages,
and errors.  Raw messages and secrets are not written by default.

## Local result captured during C-module smoke testing

With the default offline router and FTS5 fallback on the public 200-session
set, the smoke run completed successfully.  It is a plumbing baseline, not the
team's final model; Members A/B should improve retrieval and state-aware
ranking before submission.
