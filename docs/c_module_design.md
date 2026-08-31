# C Module Handoff: Orchestrator, Memory, and Evaluation

## Objective

The lifecycle layer owns orchestration outside the official evaluator. It does
not force a specific intent model or product-retrieval strategy. Intent,
retrieval, ranking, and lifecycle components integrate through small protocols
so that the core turn lifecycle remains independently testable.

## Integration points

### Intent routing and slot extraction

Implement:

```python
class IntentRouter:
    def classify(self, user_message: str, state: SessionState) -> IntentResult:
        ...
```

`IntentResult` supports `intent`, `hard_constraints`, `soft_preferences`,
`negative_constraints`, `remove_fields`, `replace_fields`, and
`clarification_attribute`. The explicit remove/replace fields are required for
the intent-override scenario.

The router returns this canonical type directly; it never mutates
`SessionState`.

### Retrieval and ranking

Implement:

```python
class Retriever:
    def retrieve(self, query: str, state: SessionState, top_k: int):
        ...

class Ranker:
    def rank(self, query: str, candidates: list[Candidate], state: SessionState):
        ...
```

A retriever may return a plain sequence or a `RetrievalResult`. The latter can
provide an untruncated `total_count`; the orchestrator uses it to ask a
clarifying question when a query is too broad.

## Official boundaries enforced by C

- The exported entry point is `starter/agent.py` with only `reset` and
  `respond` required by the official harness.
- `top_k` must be exactly 10.
- Turn 10 is processed; no internal plugin is called after turn 10.
- Recommendations are restricted to unique, valid `parent_asin` values from
  the frozen catalog.
- `ask_attribute` is restricted to the attributes allowed by the contract.
- Catalog and evaluator source files are read-only at runtime.
- Plugin failures and optional network failures must use an offline-safe
  fallback path.

## C-module test coverage

- Memory: accumulation, deletion, replacement, intent override, bounded
  history, snapshot/restore, and turn boundaries.
- Orchestrator: call sequencing, duplicate-request idempotency, invalid-ID
  filtering, broad-query clarification, router/retriever/ranker failures,
  timeout handling, and the 10-turn boundary.
- Evaluation: Hit Rate@10, MRR, MTTC, Efficiency, TechnicalScore, and scenario
  grouping.
- Trace: secret redaction while preserving token counters.

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

Run the official public evaluator with:

```bash
python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output results/results.json
```
