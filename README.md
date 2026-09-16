# MAS Starter Skeleton

This is the base multi-agent system, already wired end-to-end with
placeholder ("stub") versions of every research module. It runs right
now, with no API key, using a mock LLM client — that's on purpose, so
everyone can build and test against a working pipeline from day one
instead of waiting for the real modules to exist.

## Quick start

```bash
pip install -r requirements.txt
python pipeline.py              # runs one end-to-end pass, prints the result
pytest tests/ -v                # runs the smoke test suite
```

Both should work immediately, out of the box, with zero setup.

## How this is organized

```
schemas/models.py          <- THE CONTRACT. Every module reads/writes these shapes.
llm_client.py               <- MockLLMClient (no API key needed) + where the real Grok client goes.
agents/agents.py            <- Planner, Evidence, Critical agents.
validation/                 <- OWNER: Person A  (sequential handoff validator)
peer_consistency/           <- OWNER: Person B  (the headline mechanism)
fault_injection/            <- OWNER: Person C  (AutoTransform/AutoInject adaptation)
guard/decision.py           <- Real implementation already (simple, deterministic rule)
guard/recovery.py           <- OWNER: Person D  (regenerate + independently verify)
evaluation/logger.py        <- Shared. Every message and run result gets logged here.
pipeline.py                 <- Wires everything together. Changes rarely after this point.
tests/test_pipeline_smoke.py <- Run this after every merge.
```

## The rule that makes this work

**Nobody builds their module against "what they think the pipeline looks
like."** Everybody builds against the shapes in `schemas/models.py` and
the stub function signatures already in their folder. If your module
takes the same input and returns the same shape as the stub it's
replacing, it will plug in without anyone else changing a line of code.

If you think you need a new field in a schema or a different function
signature, raise it with the team first — that file is the one place
where a solo change breaks everyone else.

## Your task, if you own a module

1. Open your file. It has a docstring explaining exactly what a real
   implementation needs to do, and what it must never do (e.g., the
   validator and peer-checker must never use a generative LLM to make
   the actual pass/fail decision — see the docstrings for why).
2. Replace the stub logic, keep the function signature and return type
   identical.
3. Run `pytest tests/ -v` — if it still passes, you haven't broken the
   pipeline for anyone else.
4. Add your own more detailed tests for your module's actual logic
   (the smoke test only checks that the pipeline runs, not that your
   logic is correct).
5. Merge at least once a week, even if unfinished — integrate early and
   often rather than all at once at the end.

## Conditions (for the three-way experiment)

`pipeline.run_pipeline(question, condition=...)` supports:

- `"A_baseline"` — no validation, no peer check, no intervention.
- `"B_sequential_only"` — sequential validator active, peer check off.
- `"C_proposed"` — both signals active (the actual contribution).

## Switching from mock to real LLM calls

Everything above runs on `MockLLMClient` — deterministic, free, instant.
When you're ready to test with the real model, fill in `RealGrokClient`
in `llm_client.py`, then pass `llm_client=RealGrokClient(api_key=...)`
into `run_pipeline(...)`. Nothing else needs to change.
