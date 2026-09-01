# System Architecture

## Objective

Shopping Copilot is a headless, multi-turn search agent that identifies a
hidden target product as early and as highly ranked as possible. The design
separates intent understanding, state transitions, retrieval, ranking, and
evaluation while exposing one stable entry point to the judging harness.

## End-to-end lifecycle

```text
Agent.reset(session_id, user_profile)
  -> create an isolated SessionState

Agent.respond(session_id, user_message, turn, top_k=10)
  -> validate the request and turn boundary
  -> classify buying or browsing intent
  -> extract structured constraints and state operations
  -> atomically update session memory
  -> build a compact canonical query
  -> filter products by hard and negative constraints
  -> retrieve a bounded BM25 candidate pool
  -> optionally apply TF-IDF scores to that pool
  -> calculate deterministic ranking features
  -> optionally rerank the leading candidates with DeepSeek
  -> remove invalid and duplicate catalog IDs
  -> choose a structured clarification attribute
  -> return the official response and token usage
```

## Components

### Official entry point

`starter/agent.py` constructs the configured pipeline and exports only the
required `Agent.reset` and `Agent.respond` methods. Configuration selects the
hybrid path, SQLite fallback, optional TF-IDF, and optional model reranking.

### Intent and constraints

`agent/intent_router.py` distinguishes buying from browsing and produces a
canonical `IntentResult`. `agent/slot_extractor.py` extracts category, brand,
material, color, size, style, budget, feature, use case, keywords, and
negations using catalog-derived vocabulary where available.

State operations are explicit:

- additive values enter hard, soft, or negative constraint maps;
- `remove_fields` deletes stale values;
- `replace_fields` applies an intent override;
- `no_preference_attributes` prevents repeated questions.

The router never mutates session state directly.

### Session memory

`InMemoryContextMemory` is the single state-transition authority. It provides:

- isolation and locking by `session_id`;
- atomic turn advancement;
- bounded recent-message history;
- state diffs for added, removed, and replaced fields;
- snapshot and rollback after an unexpected pipeline error;
- bounded request-response caching for idempotency;
- clarification and completion bookkeeping.

The organizer supplies an anonymized aggregate `user_profile` at reset. It is
stored with the session but is not persisted across unrelated sessions.

### Retrieval

The primary path uses `HybridRetriever`:

1. `HardConstraintFilter` enforces positive, price, and negative constraints.
2. `BM25Retriever` uses an inverted postings index and bounded top-N scoring.
3. `TFIDFSemanticRetriever`, when enabled, scores the bounded candidate pool
   instead of materializing a full-catalog ranking on every turn.
4. `RetrievalResult` returns candidates plus the pre-truncation count used by
   the clarification policy.

`SQLiteCatalogRetriever` is the independent, lightweight FTS5 fallback.

### Ranking

`RuleRanker` combines normalized keyword, category, attribute, semantic, and
popularity evidence. Buying and browsing use different weight profiles, while
hard constraints remain the responsibility of retrieval.

The optional `LLMRanker` sends at most the configured number of calls per
session to DeepSeek, requests a JSON list of existing product IDs, validates
the response, appends omitted candidates in their local order, and reports
prompt/completion token usage. It cannot fabricate catalog products.

### Orchestration and response safety

`ShoppingOrchestrator` enforces the official boundary:

- `top_k` must be exactly 10;
- turn 10 is processed and later turns invoke no pipeline stage;
- only valid, unique catalog IDs can reach the response;
- optional stage timeouts and bounded retries are supported;
- plugin or network failures preserve a valid offline response;
- catastrophic errors roll state back before returning a valid empty result;
- trace failures never affect scoring.

### Evaluation and observability

`evaluator/local_evaluator.py` runs the 200-session public set through the real
Agent interface. `evaluation/metrics.py` provides reusable formulas for Hit
Rate@10, MRR, MTTC, Efficiency, and TechnicalScore.

Trace sinks record hashes and structured state changes rather than raw secrets.
Events include turn, intent transition, candidate count, clarification,
latency, retries, fallback stages, and errors.

## Fallback behavior

```text
Hybrid dependencies/configuration available
  -> HybridRetriever + RuleRanker
  -> optional DeepSeek reranking

Hybrid initialization unavailable or fallback explicitly requested
  -> SQLiteCatalogRetriever + ScoreRanker

DeepSeek request or response failure
  -> preserve RuleRanker order
```

The fallback path uses the same official Agent contract and does not modify the
catalog or evaluator.

## Current engineering trade-offs

- In-memory indexes reduce per-turn latency but increase startup memory.
- Candidate-scoped TF-IDF is bounded, but it remains lexical rather than dense.
- Rule-based extraction is deterministic and cheap, but paraphrase coverage is
  limited.
- One DeepSeek call per session bounds cost but may occur before the most useful
  state transition.
- Session state is robust, but aggregate profile fields are not yet calibrated
  into ranking features.

## Verification

```bash
python -m pytest -q
python -m evaluator.local_evaluator --catalog data/catalog.jsonl --dataset data/public_set.jsonl --output results/results.json
```
