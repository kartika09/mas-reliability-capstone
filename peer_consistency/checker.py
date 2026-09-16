"""
OWNER: Person B

This is the headline novel mechanism -- worth extra care and extra tests.

Compares Evidence's and Critical's INDEPENDENTLY produced claims (they
must not have seen each other's output yet -- check pipeline.py to confirm
this ordering hasn't been broken) and scores disagreement between them.

CONTRACT: check(...) must always return a PeerConsistencyResult.

Suggested real implementation:
  1. For each claim in evidence_output.claims, find the most similar claim
     in critical_output.claims via embedding similarity (only compare
     claims that are plausibly about the same thing -- don't NLI-compare
     unrelated claims, it's wasted compute and noisy).
  2. For each matched pair above a similarity threshold, run NLI.
  3. disagreement_score = max contradiction confidence across matched pairs
     (or proportion flagged "contradiction" -- try both, report whichever
     is more stable in your experiments).
"""

from __future__ import annotations
from schemas.models import PeerConsistencyResult, EvidenceOutput, CriticalOutput


def check(evidence_output: EvidenceOutput, critical_output: CriticalOutput) -> PeerConsistencyResult:
    """STUB -- replace with real embedding-match + NLI-contradiction scoring."""
    return PeerConsistencyResult(disagreement_score=0.0, flagged_claim_pairs=[])
