# Track 4 Problem and Evaluation Specification

This file summarizes the organizer-provided Track 4 requirements used to build
and validate this repository. The official competition document remains the
source of truth if any wording differs.

## Objective

Build a multi-turn shopping agent that finds a hidden target product as early
as possible and ranks it as highly as possible. The target corresponds to a
real product record, while customer messages are generated from a hidden intent
card derived from product metadata.

## Allowed solution scope

Relevant techniques include:

- buying/browsing intent routing;
- query rewriting and structured constraints;
- keyword, dense, or hybrid retrieval;
- semantic or model-based reranking;
- conversation-state management and intent override;
- adaptive clarification;
- safe use of the anonymized aggregate profile;
- failure detection, strategy switching, and offline fallback;
- transparent recommendation messages.

The challenge does not require a graphical interface, real transactions,
catalog modification, private-label reconstruction, full-model training,
multimodal processing, or infrastructure-heavy vector databases.

## Data

The fixed `Clothing_Shoes_and_Jewelry` catalog contains 50,000 products. Visible
fields include `parent_asin`, `title`, `features`, `description`, `price`,
`categories`, `details`, `average_rating`, `rating_number`, and `store`. Only
`parent_asin` is scored.

The public development set contains 200 sessions; the private evaluation set
contains 800 sessions. Hidden intent cards, simulator state, and private target
labels are not sent to the agent.

| Scenario | Share | Public sessions |
|---|---:|---:|
| Buying | 40% | 80 |
| Browsing | 40% | 80 |
| Intent Override | 15% | 30 |
| Boundary | 5% | 10 |

The agent receives only a safe aggregate `user_profile`; direct identifiers,
purchase timestamps, raw histories, and free-text reviews are removed.

## Session protocol

1. The evaluator creates a random `session_id` and calls
   `reset(session_id, user_profile)`.
2. The simulated customer sends the first scenario-dependent message.
3. The agent returns a natural-language message, a structured clarification
   attribute, and ranked recommendations.
4. The evaluator normalizes the first ten valid, unique catalog IDs.
5. A target hit records its rank and turn; otherwise the simulator generates
   the next reply.
6. Intent Override sessions cannot convert before the replacement intent is
   disclosed.
7. A session ends after a valid hit or turn 10.

## Required interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "B000..."}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        }
```

Response rules:

- `message` is a customer-facing string;
- `ask_attribute` is an allowed attribute or `null`;
- recommendations are ordered best to worst;
- invalid and duplicate IDs are removed;
- optional recommendation scores are ignored by the evaluator;
- usage values are non-negative integers when a model is used;
- exceptions, invalid outputs, and timeouts may count as misses.

Allowed clarification attributes are `category`, `material`, `color`, `size`,
`style`, `brand`, `budget`, `feature`, `use_case`, and `other`.

## Metrics

```text
HitRate@10 = successful sessions / N
MRR = sum(1 / target rank, with misses equal to 0) / N
MTTC = sum(first-hit turn, with misses assigned 11) / N
Efficiency = clip((11 - MTTC) / 10, 0, 1)
TechnicalScore = 0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency
```

Metrics are also grouped by Buying, Browsing, Intent Override, and Boundary.
Token use and latency demonstrate feasibility but do not change the core score.

## Model and API requirements

Teams manage their own legally accessible model credentials. API keys must be
provided through environment variables and never committed. The submission
must disclose the model, approximate cost, token usage, latency expectations,
and fallback behavior. The agent should remain valid if network access is
unavailable.

## Required competition deliverables

- a written project description submitted through Devpost;
- a public source repository with setup and reproduction instructions;
- a short public YouTube demo linked from Devpost;
- a brief report covering architecture, tools, APIs, datasets, results,
  limitations, and contributions;
- one demonstrated end-to-end multi-turn session.
