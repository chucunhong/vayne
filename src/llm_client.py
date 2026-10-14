"""
OpenRouter client with p_yes parsing, parse retries, and an on-disk cache.

Every query is identified by (model, temperature, max_tokens, system, user,
repeat index). Completed queries are stored in a SQLite cache so that an
interrupted run can resume without re-issuing them, and so that figures and
tables can be recomputed without any API call.

Set VAYNE_MOCK=1 to replace the API with a deterministic pseudo-scorer for
offline smoke tests; mock scores are never written to the cache.
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from typing import Optional

from dotenv import load_dotenv

from src.config import CACHE_PATH, MAX_OUTPUT_TOKENS, PARSE_RETRIES

load_dotenv()
logger = logging.getLogger(__name__)

MOCK = os.environ.get("VAYNE_MOCK") == "1"

_client = None
_client_lock = threading.Lock()


def get_client():
    global _client
    with _client_lock:
        if _client is None:
            from openai import OpenAI
            key = os.environ.get("OPENROUTER_API_KEY", "")
            if not key:
                raise EnvironmentError("OPENROUTER_API_KEY not set in environment or .env")
            _client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    return _client


# ── Cache ───────────────────────────────────────────────────────────────────

class QueryCache:
    def __init__(self, path=CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS q (key TEXT PRIMARY KEY, model TEXT, "
                "text TEXT, p_yes REAL, attempts INTEGER, ts REAL)"
            )
            self._conn.commit()

    @staticmethod
    def key(model, temperature, max_tokens, system, user, rep) -> str:
        blob = json.dumps([model, temperature, max_tokens, system, user, rep])
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str):
        with self._lock:
            row = self._conn.execute("SELECT p_yes, attempts FROM q WHERE key=?",
                                     (key,)).fetchone()
        return row  # (p_yes or None, attempts) or None

    def put(self, key, model, text, p_yes, attempts):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO q VALUES (?,?,?,?,?,?)",
                               (key, model, text, p_yes, attempts, time.time()))
            self._conn.commit()


_cache: Optional[QueryCache] = None


def get_cache() -> QueryCache:
    global _cache
    if _cache is None:
        _cache = QueryCache()
    return _cache


# ── API call ────────────────────────────────────────────────────────────────

def _call(system: str, user: str, model: str, temperature: float, max_tokens: int,
          retries: int = 5, retry_delay: float = 5.0) -> str:
    """One chat completion, retrying transport errors with exponential backoff."""
    client = get_client()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=temperature,
                # Disables chain-of-thought on models that support toggling it;
                # ignored elsewhere.
                extra_body={"reasoning": {"enabled": False, "exclude": True}},
            )
            content = resp.choices[0].message.content or ""
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            wait = retry_delay * (2 ** attempt) + (10 if status == 429 else 0)
            logger.warning("Attempt %d/%d failed for %s (wait %.0fs): %s",
                           attempt + 1, retries, model, wait, exc)
            if attempt == retries - 1:
                raise
            time.sleep(wait)
    return ""


_JSON_RE = re.compile(r'\{\s*"p_yes"\s*:\s*(-?\d+(?:\.\d+)?)\s*\}')


def extract_p_yes(text: str) -> Optional[float]:
    """Parse {"p_yes": x} from the reply; None if no valid JSON object is present."""
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    return float(min(1.0, max(0.0, float(m.group(1)))))


def _mock_score(system: str, user: str, rep: int) -> float:
    h = hashlib.sha256(f"{system}\n{user}\n{rep}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32


def query_p_yes(system: str, user: str, model: str, temperature: float,
                rep: int = 0, max_tokens: int = MAX_OUTPUT_TOKENS) -> Optional[float]:
    """
    Score one prompt (repeat index `rep`). A reply that cannot be parsed is
    retried up to PARSE_RETRIES times; None is returned if it still fails, and
    the caller applies the fallback described in the paper.
    """
    if MOCK:
        return _mock_score(system, user, rep)

    cache = get_cache()
    key = QueryCache.key(model, temperature, max_tokens, system, user, rep)
    hit = cache.get(key)
    if hit is not None:
        return hit[0]

    text, p = "", None
    attempts = 0
    for attempts in range(1, PARSE_RETRIES + 2):   # first try + PARSE_RETRIES retries
        text = _call(system, user, model, temperature, max_tokens)
        p = extract_p_yes(text)
        if p is not None:
            break
    if p is None:
        logger.debug("Unparseable reply after %d attempts: %s", attempts, text[:200])
    cache.put(key, model, text, p, attempts)
    return p
