"""
Tests that evaluation.harness._make_llm_factory("groq") correctly loads
GROQ_API_KEY from a .env file at the repo root via python-dotenv -- this is
what lets every teammate clone the repo and use the shared key without it
ever being committed to git (.env is gitignored; only .env.example is
committed). Loading is anchored to the repo root explicitly (see
_make_llm_factory), not to the working directory the script happens to be
run from, so these tests write/remove a real .env next to the repo root.

Run with: pytest tests/test_env_loading.py -v
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from evaluation.harness import _make_llm_factory

_REPO_ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"


_GROQ_ENV_VARS = ("GROQ_API_KEY", "GROQ_MODEL", "GROQ_REASONING_EFFORT")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # IMPORTANT: GroqClient calls python-dotenv's load_dotenv(), which writes
    # directly into the real os.environ -- that bypasses monkeypatch's
    # tracking entirely (monkeypatch only auto-reverts changes IT made), so a
    # value loaded from a test .env file here would otherwise leak into every
    # later test in the whole suite, including test_groq_client.py, no matter
    # which file pytest happens to run first. Clean up explicitly, before AND
    # after, rather than relying on monkeypatch alone.
    for var in _GROQ_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    original = _REPO_ROOT_ENV.read_text() if _REPO_ROOT_ENV.exists() else None
    if _REPO_ROOT_ENV.exists():
        _REPO_ROOT_ENV.unlink()
    yield
    if _REPO_ROOT_ENV.exists():
        _REPO_ROOT_ENV.unlink()
    if original is not None:
        _REPO_ROOT_ENV.write_text(original)
    for var in _GROQ_ENV_VARS:  # undo whatever load_dotenv() wrote directly
        os.environ.pop(var, None)


def test_missing_key_raises_a_clear_error():
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        _make_llm_factory("groq")


def test_env_file_at_repo_root_is_picked_up_automatically():
    _REPO_ROOT_ENV.write_text("GROQ_API_KEY=key-from-dotenv\nGROQ_MODEL=llama-3.1-8b-instant\n")

    factory = _make_llm_factory("groq")
    client = factory(seed=0)

    assert client.client.api_key == "key-from-dotenv"
    assert client.model == "llama-3.1-8b-instant"


def test_reasoning_effort_defaults_to_low_and_is_overridable_via_dotenv():
    _REPO_ROOT_ENV.write_text("GROQ_API_KEY=key-from-dotenv\n")
    client = _make_llm_factory("groq")(seed=0)
    assert client.reasoning_effort == "low"

    _REPO_ROOT_ENV.write_text("GROQ_API_KEY=key-from-dotenv\nGROQ_REASONING_EFFORT=high\n")
    client = _make_llm_factory("groq")(seed=0)
    assert client.reasoning_effort == "high"


def test_real_env_var_works_without_a_dotenv_file(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "key-from-shell")

    factory = _make_llm_factory("groq")
    client = factory(seed=0)

    assert client.client.api_key == "key-from-shell"
    assert client.model == "openai/gpt-oss-120b"  # default, since GROQ_MODEL unset


def test_shell_env_var_wins_over_dotenv_file(monkeypatch):
    # python-dotenv's default load_dotenv() does not override existing
    # environment variables -- confirms a teammate's own shell export takes
    # priority over a stale .env file.
    _REPO_ROOT_ENV.write_text("GROQ_API_KEY=key-from-dotenv\n")
    monkeypatch.setenv("GROQ_API_KEY", "key-from-shell")

    factory = _make_llm_factory("groq")
    client = factory(seed=0)

    assert client.client.api_key == "key-from-shell"


def test_works_regardless_of_current_working_directory(tmp_path, monkeypatch):
    # The whole point of anchoring to the repo root: this must still find
    # .env even when invoked from some unrelated directory.
    monkeypatch.chdir(tmp_path)
    _REPO_ROOT_ENV.write_text("GROQ_API_KEY=key-from-dotenv\n")

    factory = _make_llm_factory("groq")
    client = factory(seed=0)

    assert client.client.api_key == "key-from-dotenv"