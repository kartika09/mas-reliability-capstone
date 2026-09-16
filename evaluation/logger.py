"""
Shared logging utility. Every message and every run result gets appended
here -- this is what the evaluation harness (Section 8-9 of the
implementation guide) reads later. Log early, log often; it's much
cheaper to log now than to regenerate baseline runs later.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone

LOG_DIR = "logs"


def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def log_message(run_id: str, sender: str, receiver: str, content: dict, round_num: int = 0):
    _ensure_log_dir()
    entry = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sender": sender,
        "receiver": receiver,
        "round": round_num,
        "content": content,
    }
    with open(os.path.join(LOG_DIR, "messages.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_run_result(run_result_dict: dict):
    _ensure_log_dir()
    with open(os.path.join(LOG_DIR, "runs.jsonl"), "a") as f:
        f.write(json.dumps(run_result_dict) + "\n")
