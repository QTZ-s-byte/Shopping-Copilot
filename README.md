# Shopping Copilot

Shopping Copilot is a multi-turn conversational product-search agent built for
TikTok TechJam 2026 Track 4. It converts natural-language shopping requests
into structured intent and constraints, maintains conversation state, retrieves
products from a fixed 50,000-item catalog, and returns ranked recommendations
through the evaluator's required Python interface.

The system is offline-first. BM25 retrieval and deterministic ranking run
without network access, SQLite FTS5 provides a lightweight fallback, and an
optional DeepSeek reranker can improve the ordering of an existing candidate
set when credentials and network access are available.

## Key capabilities

- Buying and browsing intent routing
- Catalog-derived slot extraction for category, brand, material, color, size,
  style, budget, feature, and use case
- Explicit hard, soft, and negative constraints
- Intent-override, field-removal, and no-preference handling
- Isolated multi-turn session memory with rollback and idempotent responses
- Bounded BM25 retrieval with optional TF-IDF candidate reranking
- Deterministic rule-based ranking using relevance, attributes, and popularity
- Optional DeepSeek JSON reranking with token accounting and local fallback
- Validation of catalog IDs, recommendation uniqueness, turn count, and
  `top_k=10`
- Public-set evaluation for Hit Rate@10, MRR, MTTC, Efficiency, and
  TechnicalScore

## Architecture

```text
starter.agent.Agent
  -> ShoppingOrchestrator
       -> IntentRouter + SlotExtractor
       -> InMemoryContextMemory
       -> HybridRetriever
            -> HardConstraintFilter
            -> BM25Retriever
            -> optional TFIDFSemanticRetriever
       -> RuleRanker
       -> optional DeepSeek LLMRanker
       -> response validation, tracing, retry, and fallback policies
```

The evaluator sees only `starter.agent.Agent`. Internal components exchange the
canonical types in `shopping_copilot/contracts.py`; the complete lifecycle and
interface rules are documented in [docs/architecture.md](docs/architecture.md)
and [docs/integration_contract_v1.md](docs/integration_contract_v1.md).

## Repository layout

```text
agent/               intent routing, slot extraction, clarification policy
data/                public development sessions and catalog vocabulary
docs/                architecture, contract, specification, submission notes
evaluation/          evaluator adapter and reusable metric functions
evaluator/           public local evaluator
ranking/             deterministic product ranking
retrieval/           filtering, BM25, and optional TF-IDF retrieval
scripts/             vocabulary builder and interactive session runner
shopping_copilot/    canonical contracts, memory, orchestration, trace, LLM
starter/             official Agent entry point
tests/               unit, integration, and retrieval benchmark coverage
```

## Requirements

- Python 3.10 or newer
- The official 50,000-item `Clothing_Shoes_and_Jewelry` catalog
- Optional: a DeepSeek API key for model-based reranking

Python dependencies are declared in `requirements.txt`:

- `rank-bm25`
- `scikit-learn`
- `python-dotenv`
- `pytest`

The fallback retriever uses Python's standard-library SQLite FTS5 support; the
normal repository installation still includes `python-dotenv` for configuration.

## Setup

Clone the public repository and create a virtual environment:

```bash
git clone https://github.com/QTZ-s-byte/Shopping-Copilot.git
cd Shopping-Copilot
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare the official catalog

The large catalog is not committed to Git. Download `catalog.jsonl.gz` from the
organizer-provided participant kit, place it in `data/`, and verify its SHA256:

```text
07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8
```

PowerShell verification and decompression:

```powershell
(Get-FileHash data\catalog.jsonl.gz -Algorithm SHA256).Hash.ToLower()
python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('data/catalog.jsonl.gz','rb'), open('data/catalog.jsonl','wb'))"
```

macOS/Linux verification and decompression:

```bash
sha256sum data/catalog.jsonl.gz
gzip -dc data/catalog.jsonl.gz > data/catalog.jsonl
```

The expected output file is `data/catalog.jsonl` with 50,000 products. Catalog
files, local results, `.env`, and API credentials are ignored by Git.

## Retrieval modes

The agent exposes three intentional execution modes through consistently named
environment variables.

| Mode | Configuration | Purpose |
|---|---|---|
| SQLite fallback | `SHOPPING_COPILOT_RETRIEVAL_MODE=sqlite` | Lightweight deterministic baseline |
| BM25 hybrid path | retrieval mode `hybrid`, BM25 `1`, TF-IDF `0` | Recommended offline path |
| BM25 + TF-IDF | retrieval mode `hybrid`, BM25 `1`, TF-IDF `1` | Higher-memory semantic candidate reranking |

Use the complete variable names shown below:

```dotenv
SHOPPING_COPILOT_RETRIEVAL_MODE=hybrid
SHOPPING_COPILOT_ENABLE_BM25=1
SHOPPING_COPILOT_ENABLE_TFIDF=0
SHOPPING_COPILOT_ENABLE_LLM=0
```

Setting `SHOPPING_COPILOT_ENABLE_BM25=0` or
`SHOPPING_COPILOT_FORCE_FALLBACK=1` selects the SQLite path.

## Run the public evaluator

The evaluator uses the same interface as the judging harness:

```python
Agent.reset(session_id, user_profile)
Agent.respond(session_id, user_message, turn, 10)
```

Run the recommended offline BM25 configuration on all 200 public sessions:

```powershell
$env:SHOPPING_COPILOT_RETRIEVAL_MODE='hybrid'
$env:SHOPPING_COPILOT_ENABLE_BM25='1'
$env:SHOPPING_COPILOT_ENABLE_TFIDF='0'
$env:SHOPPING_COPILOT_ENABLE_LLM='0'
python -m evaluator.local_evaluator `
  --catalog data/catalog.jsonl `
  --dataset data/public_set.jsonl `
  --output results/hybrid-bm25.json
```

Equivalent one-line command for any shell:

```bash
python -m evaluator.local_evaluator --catalog data/catalog.jsonl --dataset data/public_set.jsonl --output results/hybrid-bm25.json
```

Only the first ten valid, unique `parent_asin` values are scored. Evaluation
reports are written under the ignored `results/` directory.

## Optional DeepSeek reranking

The model adapter uses DeepSeek's OpenAI-compatible Chat Completions endpoint.
Copy the environment template and add the key locally:

```powershell
Copy-Item .env.example .env
```

```dotenv
SHOPPING_COPILOT_RETRIEVAL_MODE=hybrid
SHOPPING_COPILOT_ENABLE_BM25=1
SHOPPING_COPILOT_ENABLE_TFIDF=0
SHOPPING_COPILOT_ENABLE_LLM=1

DEEPSEEK_API_KEY=replace-with-your-private-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/chat/completions
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_TIMEOUT_SECONDS=30
DEEPSEEK_MAX_CALLS_PER_SESSION=1
```

Never commit `.env` or print the key in logs. The default call budget permits
at most one attempted model rerank per session. API failures preserve the
deterministic local ranking, and successful response token counts are returned
through the official `usage` field.

Run the same evaluator command after enabling the model, using a distinct
output file such as `results/deepseek-hybrid-200.json`.

## Recorded public-set results

These are local development results on the 200-session public set, not private
leaderboard scores. The public set contains 80 Buying, 80 Browsing, 30 Intent
Override, and 10 Boundary sessions.

| Configuration | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| SQLite FTS5 fallback | 0.715 | 0.180373 | 7.065 | 0.3935 | 0.490312 |
| Hybrid BM25 | 0.655 | 0.425200 | 7.290 | 0.3710 | 0.529260 |
| Hybrid BM25 + DeepSeek | 0.655 | 0.437046 | 7.290 | 0.3710 | 0.532814 |

The DeepSeek run used 295,624 prompt tokens and 17,577 completion tokens
(313,201 total). Using the published `deepseek-v4-flash` cache-miss rates on
1 September 2026, the approximate API cost is USD 0.077 off-peak or USD 0.153
at peak rates. Provider pricing may change; consult the
[official pricing page](https://api-docs.deepseek.com/quick_start/pricing/)
before reproducing the estimate.

## Manual end-to-end session

Run the interactive client against the real agent:

```bash
python scripts/manual_session.py --catalog data/catalog.jsonl
```

Enter one shopping message per line. The script prints the customer-facing
message, structured clarification attribute, and ranked catalog IDs. Press
Enter on an empty line to exit.

Example conversation:

```text
User: I need waterproof hiking shoes under $100.
Agent: returns ranked products and may ask for a missing high-value attribute.
User: Actually, forget the black color; make them blue instead.
Agent: removes the stale value and returns a new valid top-10 ranking.
```

## Tests

```bash
python -m pytest -q
python -m unittest discover -s tests -v
```

The suite covers intent and slot extraction, no-preference and override state
transitions, hard and negative filtering, retrieval and ranking, rollback,
request idempotency, invalid-ID filtering, fallback behavior, trace redaction,
turn boundaries, and official metric formulas.

## Limitations and future improvements

- The rule-based intent and slot extractor cannot cover every paraphrase,
  implicit preference, or ambiguous product phrase.
- BM25 defines the primary candidate-recall ceiling; an LLM reranker cannot
  recover a target that retrieval did not include.
- TF-IDF is lexical rather than a true dense embedding model and consumes more
  memory on the full catalog.
- The anonymized `user_profile` is preserved in session state but is not yet
  converted into calibrated ranking features.
- The one-call-per-session LLM policy may spend its call before an important
  clarification or intent override arrives.
- The public development set is small, so aggressive tuning risks overfitting
  the 200 visible sessions.
- There is no graphical interface; the official Python API and terminal
  walkthrough are the intended demonstration surfaces.

Given more time, we would add stage-level recall diagnostics, catalog-field
inverted indexes, value-aware clarification, safe profile-derived soft
features, state-aware LLM call scheduling, and a dense retriever evaluated
against latency and memory budgets.

## Data and security

The competition data is derived from the
[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
`Clothing_Shoes_and_Jewelry` category. The project uses product metadata and
anonymized aggregate profiles only. It does not use direct identifiers, raw
review text, purchase timestamps, or private evaluation sessions. See
[DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md).

Secrets are loaded only from environment variables. `.env`, catalog files,
result reports, logs, and organizer-only artifacts are excluded by
`.gitignore`.

## Team contributions

- [**YUino-t**](https://github.com/YUino-t) — **Tao Junmin**: intent routing,
  catalog-aware slot extraction, vocabulary generation, clarification signals,
  intent override, no-preference behavior, and intent/state tests.
- [**IMMORTALS-TQM**](https://github.com/IMMORTALS-TQM) — **Qi Zihan**: BM25
  and TF-IDF retrieval, hard and negative filtering, candidate construction,
  deterministic ranking, retrieval performance work, and retrieval/ranking
  tests and benchmarks.
- [**LordRosan**](https://github.com/LordRosan) — **Zhou Moyu**: canonical
  contracts, orchestration, session memory, evaluation and metrics, tracing,
  fallback and configuration paths, DeepSeek integration, cross-component
  integration, repository documentation, and release preparation.

Contribution details can be verified from the repository's commit and merge
history.

## License

Released under the [MIT License](LICENSE).
