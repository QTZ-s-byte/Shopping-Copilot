# Demo Video Script

Run the demo with:

```text
python scripts/demo_video.py
```

The script runs four turns against the real `Agent` API (hybrid BM25
retrieval), printing the detected intent, extracted hard constraints, the
assistant reply, and the ranked recommendations.

## Turn sequence

```text
1. I need running shoes                         -> buying, category=running shoes
2. I want it from Nike                          -> accumulates brand=nike
3. Actually, I want a leather bag instead       -> intent override -> bag + leather
4. Show me some summer outfit ideas             -> browsing + structured clarification
```

## Voiceover

**Opening**

Hi, this is Shopping Copilot — a multi-turn conversational shopping agent built
for Track 4, Conversational Search. In at most ten turns, it finds the product
a customer actually wants, using intent routing, hybrid retrieval, and ranking.

**Setup / environment**

Here we load the official catalog of fifty thousand Clothing, Shoes, and
Jewelry products, all indexed in memory. Retrieval uses a hybrid approach —
BM25 keyword recall plus rule-based ranking — with no external vector database,
so it runs fully offline.

**Turn 1 — Buying intent + recommendations**

In the first turn, the customer says "I need running shoes." Our intent router
classifies this as buying, and extracts the hard constraint category equals
running shoes. The system returns ten candidate shoes, and at the same time
proactively asks about budget to narrow things down.

**Turn 2 — Information accumulation**

In turn two, the customer adds "I want it from Nike." The conversation state
machine accumulates brand equals Nike on top of the existing constraints, and
the results now converge to Nike running shoes. This shows multi-turn
information accumulation.

**Turn 3 — Intent override**

In turn three, the customer says "Actually, I want a leather bag instead." This
is an intent override — the system detects that the goal has switched from
shoes to a bag, clears the old constraints, rewrites them as category bags plus
material leather, and asks about size.

**Turn 4 — Browsing + clarification**

Turn four is open-ended browsing: "Show me some summer outfit ideas." It is
classified as browsing, and the system guides the customer with a structured
clarification — ask attribute equals color — instead of dumping a huge list of
products.

**Closing — Evaluation results**

Finally, here are our local evaluation results on the two hundred public
development sessions: HitRate@10 is ___, MRR is ___, MTTC is ___, and the
combined TechnicalScore is ___. The full code is in our public GitHub
repository and is reproducible with a single command.
