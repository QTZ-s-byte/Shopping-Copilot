# Shopping Copilot — Devpost Project Description

## Project overview

Shopping Copilot is a multi-turn conversational product-search agent for
TikTok TechJam 2026 Track 4. It helps a customer move from a vague or changing
shopping request to a ranked list of real products from a fixed catalog. The
agent distinguishes buying from browsing, extracts structured
constraints, remembers the conversation, asks structured clarification
questions, handles changes of mind, and returns catalog-valid recommendations
through the competition's required Python API.

Public repository:
https://github.com/QTZ-s-byte/Shopping-Copilot

Demo video: **Add the public YouTube URL here before submitting to Devpost.**

## How the solution addresses the problem

The challenge is not a single-turn keyword search. A customer may start with a
broad category, reveal requirements over several turns, reject an attribute,
state that they have no preference, or replace an earlier intent entirely.
Meanwhile, the evaluator rewards both successful retrieval and early,
high-ranked placement of the hidden target.

Shopping Copilot addresses this with a complete conversational retrieval
pipeline:

1. An intent router identifies buying and browsing behavior.
2. A catalog-aware slot extractor converts messages into category, brand,
   material, color, size, style, budget, feature, use-case, keyword, and
   negative constraints.
3. Explicit add, remove, replace, and no-preference operations update an
   isolated session state without letting individual components mutate memory.
4. Hard and negative constraints filter the catalog before ranking.
5. BM25 retrieves a bounded candidate pool; optional TF-IDF adds a second
   lexical relevance signal without rescoring the full 50,000-item catalog on
   every turn.
6. A deterministic ranker combines keyword, category, attribute, semantic, and
   popularity evidence using different weights for buying and browsing.
7. An optional DeepSeek reranker reorders existing candidates using structured
   JSON output. It cannot fabricate product IDs, and failures preserve the
   local ranking.
8. The orchestrator validates the ten-turn protocol, top-10 output, catalog
   membership, uniqueness, token usage, and fallback behavior.

The system is offline-first: its core search path works without an external
model, while SQLite FTS5 remains available as an independent lightweight
fallback.

## Development tools

- Python 3.10+ and the Python command-line toolchain
- Git and GitHub for version control, branch integration, and the public
  repository
- PowerShell and terminal-based scripts for setup, evaluation, and manual
  end-to-end testing
- pytest and unittest for automated verification
- JSONL-based public evaluator and result reports for reproducible experiments

The project is editor-agnostic and can be opened in VS Code, PyCharm, or any
Python-capable development environment.

## APIs used

- [**DeepSeek OpenAI-compatible Chat Completions API**](https://api-docs.deepseek.com/)
  - Model: `deepseek-v4-flash`
  - Purpose: optional candidate reranking
  - Output: constrained JSON containing only IDs from the supplied candidate
    set
  - Credential: `DEEPSEEK_API_KEY` loaded from an ignored local `.env` file
  - Fallback: deterministic local ranking on any network, API, or parsing
    failure

The organizer's evaluator and local catalog search do not require an external
API.

## Libraries and frameworks

- `rank-bm25` for BM25 compatibility and retrieval experimentation
- `scikit-learn` for optional TF-IDF vectorization and cosine similarity
- `python-dotenv` for local environment configuration
- `pytest` and Python `unittest` for automated tests
- Python standard library components including `sqlite3`, `urllib`, `json`,
  `threading`, `concurrent.futures`, and `dataclasses`

No heavyweight web framework, vector database, or model-training stack is
required.

## Datasets and assets

The competition package is derived from the
[**Amazon Reviews 2023**](https://amazon-reviews-2023.github.io/) dataset
published by McAuley Lab at UCSD.

- Category: `Clothing_Shoes_and_Jewelry`
- Product join key: `parent_asin`
- Frozen product catalog: 50,000 products
- Public development set: 200 labeled sessions
- Scenario mix: 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary
  sessions
- Local vocabulary asset: catalog-derived brands, attributes, and category
  aliases in `data/vocab.json`

The project uses text and structured product metadata only. It does not use
direct user identifiers, purchase timestamps, free-text reviews, raw purchase
histories, private holdout sessions, or copyrighted demo media.

## Architecture and reliability

The official evaluator imports `starter.agent.Agent`, which coordinates:

```text
IntentRouter -> Session Memory -> HybridRetriever -> RuleRanker
                                             -> optional DeepSeek reranker
```

Canonical dataclasses define products, candidates, intent results, retrieval
results, session state, and responses. Session transitions are atomic and
thread-safe. Failed turns can roll back to a snapshot, repeated requests return
cached responses, and all recommendation IDs are checked against the frozen
catalog.

Trace events record state changes, candidate counts, latency, retries, and
fallback stages without storing raw credentials. The catalog and evaluator are
read-only at runtime.

## Evaluation results

We evaluated the real Agent interface on all 200 public development sessions.
These are local development results rather than private leaderboard scores.

| Configuration | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| SQLite FTS5 fallback | 0.715 | 0.180373 | 7.065 | 0.3935 | 0.490312 |
| Hybrid BM25 | 0.655 | 0.425200 | 7.290 | 0.3710 | 0.529260 |
| Hybrid BM25 + DeepSeek | 0.655 | 0.437046 | 7.290 | 0.3710 | 0.532814 |

The DeepSeek run consumed 295,624 prompt tokens and 17,577 completion tokens,
for 313,201 tokens total. Based on the published `deepseek-v4-flash`
cache-miss pricing on 1 September 2026, the approximate cost is USD 0.077 at
off-peak rates or USD 0.153 at peak rates. Actual billing can differ because
of cache usage and future provider price changes.

The model improved MRR but not Hit Rate@10. This confirms that reranking can
move a retrieved target higher but cannot recover a product missing from the
candidate pool.

## Challenges

- Representing multi-turn additions, deletions, and replacements without
  letting stale constraints survive an intent override
- Distinguishing genuine negative product constraints from conversational
  phrases such as "ignore my earlier preference"
- Keeping hybrid retrieval bounded on a 50,000-item catalog
- Preserving the exact judging contract across independent modules
- Adding an external model without making credentials or network access a
  requirement for correctness
- Measuring ranking quality separately from candidate recall

## Limitations and future improvements

The current intent and slot layer is deterministic and efficient, but it cannot
cover every paraphrase or implicit shopping preference. BM25 remains the main
candidate-recall bottleneck, and TF-IDF is lexical rather than a true dense
semantic retriever. The aggregate user profile is retained safely but is not
yet converted into calibrated soft ranking features. Clarification is based on
missing attributes and candidate breadth rather than learned question value.
The one-call-per-session DeepSeek budget may also spend the call before a later
clarification or intent override makes reranking more useful.

With more time, we would add stage-level Recall@N diagnostics, catalog-field
inverted indexes, safe profile-derived ranking priors, value-aware
clarification, state-aware model-call scheduling, and a compact dense retriever
benchmarked against CPU, memory, latency, and private-set generalization.

## Team contributions

- [**YUino-t**](https://github.com/YUino-t) — **Tao Junmin**: intent routing,
  catalog-aware slot extraction, vocabulary generation, clarification signals,
  intent override and no-preference behavior, and intent/state testing.
- [**IMMORTALS-TQM**](https://github.com/IMMORTALS-TQM) — **Qi Zihan**: BM25
  and TF-IDF retrieval, hard and negative filtering, candidate construction,
  deterministic ranking, performance improvements, and retrieval/ranking
  tests and benchmarks.
- [**LordRosan**](https://github.com/LordRosan) — **Zhou Moyu**: canonical
  contracts, orchestration, session memory, evaluation and metrics, tracing,
  fallback paths, DeepSeek integration, cross-component integration,
  documentation, and release preparation.

The public Git history contains the corresponding commits and integration
merges.
