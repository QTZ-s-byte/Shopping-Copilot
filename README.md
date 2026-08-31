# Shopping Copilot

Track 4 submission workspace for TikTok TechJam 2026.  The implementation is
an offline-first, headless shopping Agent: it routes Buying/Browsing intent,
maintains structured multi-turn context, retrieves catalog candidates, ranks
them, and returns the exact interface expected by the public evaluator.

## Current C-module implementation

The `shopping_copilot` package owns the lifecycle plumbing so the other team
workstreams can be swapped in through dependency injection:

```text
starter.agent.Agent
  -> ShoppingOrchestrator
       -> IntentRouter       (Member A)
       -> InMemoryContextMemory
       -> Retriever/Ranker    (Member B)
       -> TraceSink + fallback/retry policy
```

The shared contracts are in `shopping_copilot/contracts.py`.  The
orchestrator enforces the official `top_k=10` contract, rejects a turn beyond
10 without calling a plugin, filters duplicate/invalid catalog IDs, supports
idempotent repeated requests, and never stores secrets in traces.  The default
catalog retriever is an offline SQLite FTS5 fallback; it can be replaced by a
stronger hybrid implementation without changing `starter/agent.py`.

## Official participant kit

The public evaluator and public development set are included for local work.
The 50,000-item catalog is intentionally ignored by Git.  Download
`catalog.jsonl.gz` from the official `participant-kit` release, verify the
published SHA256, and decompress it to `data/catalog.jsonl`.

The expected catalog checksum is:

```text
07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8
```

Python 3.10+ is recommended.  No third-party Python dependency is required by
the C module.

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

## Injecting Members A/B implementations

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

Then pass those objects to `ShoppingOrchestrator(router=..., retriever=...,
ranker=...)`.  A router may return either `IntentResult` or a mapping using the
aliases documented in `contracts.py`; a retriever may return a list or a
`RetrievalResult` with `total_count` for broad-query clarification.

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

