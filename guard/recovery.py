"""
OWNER: Person D

Runs when the Guard decides RECOVER. Regenerates the flagged output and
verifies it independently -- IMPORTANT: verification must use a DIFFERENT
mechanism than whatever flagged the fault in the first place (never let a
module verify its own detection -- that's circular, and was flagged
explicitly as a design risk earlier in the project).

  - If the sequential validator flagged it, verify using the peer-
    consistency check (or against the original source documents).
  - If the peer check flagged it, verify using the sequential validator
    (or against the original source documents).

CONTRACT: recover(...) returns (final_output, recovered: bool, verified: bool, abstained: bool).
Retry budget is fixed and small -- don't let this loop indefinitely.
"""

from __future__ import annotations

MAX_RETRIES = 2


def recover(agent_output, regenerate_fn, verify_fn, triggered_by: list[str]):
    """
    STUB.

    agent_output: the flagged output to fix
    regenerate_fn: callable that asks the originating agent to redo its output
    verify_fn: callable that independently checks the new output
               (must use a different signal than `triggered_by`)
    triggered_by: which signal(s) flagged this, so verify_fn can pick a
                  different one
    """
    for attempt in range(MAX_RETRIES):
        new_output = regenerate_fn(agent_output)
        if verify_fn(new_output):
            return new_output, True, True, False
    # Retries exhausted, correctness still not established
    return agent_output, True, False, True
