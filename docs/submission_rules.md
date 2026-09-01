# Submission and Reproduction Checklist

This checklist maps the competition deliverables to repository artifacts.

## Devpost project description

Use the content in [`PROJECT_DESCRIPTION.md`](../PROJECT_DESCRIPTION.md) as the
written submission. Before publishing, confirm that it includes:

- how the solution addresses the problem;
- development tools;
- APIs;
- libraries and frameworks;
- datasets and assets;
- architecture and evaluation results;
- limitations and planned improvements;
- the public repository URL;
- the public YouTube demo URL when available.

## Public repository

Repository URL:

```text
https://github.com/QTZ-s-byte/Shopping-Copilot
```

Required public contents:

- complete, commented source code;
- `README.md` with overview, setup, reproduction, limitations, and team
  contributions;
- `requirements.txt`;
- public development sessions and small vocabulary assets;
- official `starter.agent.Agent` entry point;
- tests and evaluator instructions;
- dataset attribution and MIT license.

Do not commit:

- `.env` or any API key;
- the large catalog archive or decompressed catalog;
- local evaluation result files and logs;
- organizer-only code or private evaluation data;
- credentials, direct user identifiers, or raw purchase histories.

## Required interface verification

```python
from starter.agent import Agent

agent = Agent("data/catalog.jsonl")
agent.reset(session_id, user_profile)
response = agent.respond(session_id, user_message, turn, 10)
```

Verify that every response has:

- a string `message`;
- a valid `ask_attribute` or `None`;
- ordered recommendations containing catalog-valid `parent_asin` values;
- no duplicate IDs in the scored top 10;
- non-negative integer usage counters.

## Reproduction commands

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m evaluator.local_evaluator --catalog data/catalog.jsonl --dataset data/public_set.jsonl --output results/results.json
python scripts/manual_session.py --catalog data/catalog.jsonl
```

The catalog must first be obtained from the official participant kit and
decompressed to `data/catalog.jsonl` as described in the README.

## External model disclosure

The optional external provider is DeepSeek:

```text
API: OpenAI-compatible Chat Completions
Model: deepseek-v4-flash
Credential: DEEPSEEK_API_KEY in ignored local .env
Default call budget: one attempted rerank per session
Fallback: preserve deterministic RuleRanker order
```

The recorded 200-session run used 313,201 total tokens. Consult the current
DeepSeek pricing page when finalizing the cost estimate because provider rates
may change.

## Demo video

The video must:

- demonstrate the real agent end to end;
- show setup or environment configuration without exposing credentials;
- include at least one multi-turn request and one intent change or
  clarification;
- show ranked catalog IDs and/or evaluator results;
- be uploaded to YouTube with public visibility;
- avoid unlicensed third-party trademarks and copyrighted media;
- be linked from the Devpost description.

A terminal walkthrough is acceptable because this is a backend/NLP track.

## Final pre-submission checks

- [ ] Repository visibility is public.
- [ ] `main` contains the final integrated code.
- [ ] Tests pass from a clean environment.
- [ ] Catalog setup and checksum instructions are correct.
- [ ] Evaluation command writes under `results/`.
- [ ] No `.env`, API key, catalog, or result file is tracked.
- [ ] README limitations and contributions are present.
- [ ] Devpost description contains the repository link.
- [ ] Public YouTube link is added to Devpost.
- [ ] Devpost submission is completed before the deadline.
