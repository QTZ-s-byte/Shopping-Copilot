# Integrated Pipeline Contract v1

This document defines the single internal contract shared by the intent,
retrieval/ranking, and orchestration workstreams. The official evaluator only
sees `starter.agent.Agent`; all internal components use the types below.

## Component boundaries

- The intent component returns classification, slot extraction, and
  intent-level clarification signals.
- The retrieval/ranking component returns catalog candidates and scores.
- The lifecycle component owns session state mutation, fallback, tracing, and
  evaluation integration.
- Only the lifecycle component may mutate `SessionState`; other components
  return structured values and must not mutate the state or catalog.

## Canonical state

`shopping_copilot.contracts.SessionState` is the only session state model. It
contains the current intent, hard/soft/negative constraints, anonymized profile,
recent messages, clarification bookkeeping, candidate IDs, turn metadata, and
completion state.

`shopping_copilot.memory.InMemoryContextMemory` is the only state transition
implementation. It applies `IntentResult`, records a `StateDiff`, and provides
session isolation, rollback, and request idempotency.

## Canonical intent result

The intent component returns `shopping_copilot.contracts.IntentResult`:

```python
IntentResult(
    intent="buying" or "browsing",
    confidence=0.0,
    hard_constraints={...},
    soft_preferences={...},
    negative_constraints={...},
    remove_fields=(...),
    replace_fields={...},
    clarification_attribute=None,
    no_preference=False,
    raw=user_message,
)
```

`override=True` is an explanation flag. The actual state operation is
represented by `remove_fields` and `replace_fields`; C normalizes an A result
before applying it.

## Canonical catalog and candidate

`shopping_copilot.contracts.Product` is the only product record. A candidate
keeps that Product attached so ranking never needs a second Product model:

```python
Candidate(
    parent_asin=product.parent_asin,
    product=product,
    score=final_score,
    source_scores={"keyword": ..., "semantic": ...},
    reasons=(...),
)
```

Only `parent_asin` values from the frozen catalog may reach the official
response.

## Retriever and ranker interfaces

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

`top_k=10` is the official response limit. The implementation may retrieve a
larger internal candidate pool before ranking, but it must return an accurate
`RetrievalResult.total_count` when possible so C can detect broad queries.

Hard constraints must be enforced before ranking. Negative constraints must
exclude matching products rather than merely add negative words to a text
query. Soft preferences affect ranking only unless explicitly promoted by the
intent policy.

## Lifecycle

```text
Agent.respond
  -> read SessionState
  -> A IntentRouter.classify
  -> C Memory.apply_turn
  -> build canonical query
  -> B Retriever.retrieve
  -> B Ranker.rank
  -> validate unique catalog IDs
  -> choose clarification
  -> emit trace and cache response
```

No component may call the official evaluator, modify the catalog, fabricate an
ASIN, or make a plugin call after turn 10.
