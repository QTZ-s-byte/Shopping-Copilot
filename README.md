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
       -> IntentRouter
       -> InMemoryContextMemory
       -> HybridRetriever + RuleRanker
       -> TraceSink + fallback/retry policy
```

The shared contracts are in `shopping_copilot/contracts.py`; see
`docs/integration_contract_v1.md` for the integration rules. The orchestrator
enforces the official `top_k=10` contract, rejects a turn beyond 10 without
calling a plugin, filters duplicate/invalid catalog IDs, supports idempotent
repeated requests, and never stores secrets in traces. The lightweight SQLite
FTS5 path is available with `SHOPPING_COPILOT_RETRIEVAL_MODE=sqlite`.
The in-memory hybrid path is selected with
`SHOPPING_COPILOT_RETRIEVAL_MODE=hybrid` and uses BM25 by default. TF-IDF
reranking is enabled independently with `SHOPPING_COPILOT_ENABLE_TFIDF=1`
because the full 50,000-item matrix increases memory and latency. The two
retrieval switches use the same `SHOPPING_COPILOT_ENABLE_*` naming convention;
`SHOPPING_COPILOT_ENABLE_BM25=0` selects the SQLite path. The optional LLM
ranking boundary is controlled by `SHOPPING_COPILOT_ENABLE_LLM=1` when a
provider adapter and credentials are configured. `SHOPPING_COPILOT_FORCE_FALLBACK=1`
always forces SQLite for a deterministic smoke run.

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
the SQLite path; the hybrid path uses the packages in `requirements.txt`.

## Reproducible Windows setup

Run these commands in PowerShell from the repository root. The bundled runtime
path below is the runtime used in the development environment; replace it with
your own Python 3.10+ executable if needed.

```powershell
$py = 'C:\Users\16349\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py --version
& $py -m pip install -r requirements.txt
```

The official catalog is not committed because it is large. Download
`catalog.jsonl.gz` from the official `participant-kit` release, place it under
`data\`, verify the published SHA256, and decompress it to
`data\catalog.jsonl`:

```powershell
$catalogGz = 'data\catalog.jsonl.gz'
(Get-FileHash $catalogGz -Algorithm SHA256).Hash.ToLower()
& $py -c "import gzip,shutil; src='data/catalog.jsonl.gz'; dst='data/catalog.jsonl'; shutil.copyfileobj(gzip.open(src,'rb'), open(dst,'wb'))"
```

The expected checksum is shown in the participant-kit section above. Do not
commit either catalog file; both are ignored by Git.

## Run the public evaluator

From the repository root:

```powershell
$env:SHOPPING_COPILOT_RETRIEVAL_MODE='hybrid'
$env:SHOPPING_COPILOT_ENABLE_BM25='1'
$env:SHOPPING_COPILOT_ENABLE_TFIDF='0'
& $py -m evaluator.local_evaluator `
  --catalog data/catalog.jsonl `
  --dataset data/public_set.jsonl `
  --output results/results.json
```

On PowerShell, use backticks for line continuation (or put the command on one
line). This invokes the same public `Agent.reset`/`Agent.respond` contract used
by the judge. Evaluation reports are written under `results\` by default; the
directory is ignored by Git so local reports never pollute the repository root.

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

```powershell
& $py -m unittest discover -s tests -v
& $py -m pytest -q
```

The tests cover state accumulation/removal/replacement, bounded history,
idempotency, invalid catalog IDs, broad-query clarification, ranker failure
fallback, the 10-turn boundary, and the official metric formulas.

## Manual user-style session

After the catalog has been decompressed, start an interactive ten-turn session:

```powershell
& $py scripts/manual_session.py --catalog data/catalog.jsonl
```

Type one natural-language shopping request per line. The script prints the
assistant message, any clarification question, and the returned catalog IDs.
Press Enter on an empty line to exit. This uses the same public `Agent` class
as the evaluator and therefore exercises the real orchestrator and memory.

To exercise the indexed hybrid retrieval locally, opt in explicitly:

```powershell
$env:SHOPPING_COPILOT_RETRIEVAL_MODE='hybrid'
$env:SHOPPING_COPILOT_ENABLE_BM25='1'
$env:SHOPPING_COPILOT_ENABLE_TFIDF='0'
& $py scripts/manual_session.py --catalog data/catalog.jsonl
```

Semantic retrieval is intentionally opt-in because it consumes substantially
more memory on the 50,000-item catalog.

## Optional DeepSeek reranking

DeepSeek integration uses the official OpenAI-compatible Chat Completions
endpoint. Copy the committed template to the ignored local `.env` file:

```powershell
Copy-Item .env.example .env
```

Then edit only the following line in `.env` and keep the key local:

```dotenv
DEEPSEEK_API_KEY=replace-with-your-private-key
```

The template is otherwise ready for the 200-session run. It enables hybrid
BM25 retrieval, disables TF-IDF, enables DeepSeek reranking, and uses:

```dotenv
DEEPSEEK_BASE_URL=https://api.deepseek.com/chat/completions
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_TIMEOUT_SECONDS=30
DEEPSEEK_MAX_CALLS_PER_SESSION=1
```

The one-call-per-session limit caps the public evaluation at 200 attempted API
calls. DeepSeek token usage is copied into the official response `usage`
field. API failures preserve the deterministic local ranking order. The `.env`
file is ignored by Git and must never be committed or pasted into logs.

Clear shell-level overrides when finished:

```powershell
Remove-Item Env:SHOPPING_COPILOT_RETRIEVAL_MODE -ErrorAction SilentlyContinue
Remove-Item Env:SHOPPING_COPILOT_ENABLE_BM25 -ErrorAction SilentlyContinue
Remove-Item Env:SHOPPING_COPILOT_ENABLE_TFIDF -ErrorAction SilentlyContinue
Remove-Item Env:SHOPPING_COPILOT_ENABLE_LLM -ErrorAction SilentlyContinue
```

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

## Local result captured during the offline smoke testing

With the SQLite mode on the public 200-session set, the smoke run completed
successfully. The hybrid BM25 mode also completes after indexed retrieval
optimization and produces a stronger ranking baseline. TF-IDF and LLM ranking
remain opt-in because their resource use depends on the catalog and provider.
