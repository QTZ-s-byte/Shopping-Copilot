# Internal Integration Contract

This document defines the single internal contract used by the integrated
agent. The official evaluator imports only `starter.agent.Agent`; all internal
components use the canonical types in `shopping_copilot/contracts.py`.

## Ownership and mutation rules

- Intent routing returns structured observations and requested state changes.
- Retrieval reads state and returns catalog-backed candidates.
- Ranking reads state and candidates and returns a new ordered sequence.
- The memory layer is the only component allowed to mutate `SessionState`.
- No component may modify the frozen catalog, evaluator, or public labels.

## Canonical session state

`SessionState` contains:

- session ID and anonymized aggregate profile;
- current buying/browsing intent;
- hard, soft, and negative constraints;
- turn count and bounded recent messages;
- generated state summary;
- last candidate IDs;
- asked and no-preference attributes;
- completion and version metadata.

`InMemoryContextMemory` applies all transitions and returns a `StateDiff`.
Snapshots allow a failed turn to roll back without corrupting the next turn.

## Intent result

```python
IntentResult(
    intent="buying" or "browsing",
    confidence=0.0,
    hard_constraints={},
    soft_preferences={},
    negative_constraints={},
    remove_fields=(),
    replace_fields={},
    clarification_attribute=None,
    override=False,
    no_preference=False,
    no_preference_attributes=(),
    raw=user_message,
)
```

Additive constraint maps never mutate state directly. `remove_fields` deletes
all stale copies of a field before additions are applied. `replace_fields`
represents explicit changes of mind. `override` is explanatory metadata; the
actual transition remains auditable through remove/replace operations.

## Product and candidate

`Product` is the only catalog product model. `Candidate` carries the product
payload through retrieval and ranking:

```python
Candidate(
    parent_asin=product.parent_asin,
    product=product,
    score=final_score,
    source_scores={
        "bm25": keyword_score,
        "tfidf": semantic_score,
        "category": category_score,
        "attribute": attribute_score,
        "popularity": popularity_score,
    },
    reasons=(...),
)
```

Only a `parent_asin` found in the frozen catalog may enter the official
response.

## Retriever protocol

```python
class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        state: SessionState,
        top_k: int,
    ) -> RetrievalResult:
        ...
```

`RetrievalResult.candidates` contains a bounded candidate sequence.
`total_count` represents the number of products surviving the constraint
filter before candidate truncation, when known. `exhausted` indicates whether
the returned candidates cover the available set.

Rules:

- hard constraints filter before ranking;
- negative constraints exclude matching products;
- soft preferences affect scoring rather than eligibility;
- internal candidate pools may exceed the official top 10 but remain bounded;
- retrieval must be deterministic for equal inputs.

## Ranker protocol

```python
class Ranker(Protocol):
    def rank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        state: SessionState,
    ) -> Sequence[Candidate]:
        ...
```

The ranker may update scores by returning new candidates or consistently
updating the supplied candidates. It may not introduce an ID outside the input
candidate set. An optional external reranker must preserve the local order if
the request fails or returns invalid data.

## Lifecycle

```text
Agent.respond
  -> validate session, turn, and top_k
  -> read SessionState
  -> IntentRouter.classify
  -> Memory.apply_turn
  -> build_query
  -> Retriever.retrieve
  -> Ranker.rank
  -> optional external rerank
  -> validate unique catalog IDs
  -> choose clarification
  -> cache response and emit trace
```

No pipeline operation may execute after the ten-turn boundary. Duplicate
requests with the same session, turn, and message return the cached response.

## Official response

```python
{
    "message": "Here are the closest matches I found.",
    "ask_attribute": None,
    "recommendations": [
        {"parent_asin": "B000...", "score": 0.91}
    ],
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
    },
}
```

The first ten valid, unique recommendation IDs are scored. `score` is useful
for debugging but ignored by the official metric calculation.
