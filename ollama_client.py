"""
Thin, thread-safe HTTP client for one or more Ollama servers.

The pipeline runs one server per GPU. Every worker thread is assigned an
endpoint the first time it makes a call and keeps using it, so the load is
spread evenly across the GPUs without any per-request coordination.
"""

import itertools
import threading
import time

import requests

from config import (
    MAX_RETRIES,
    MAX_TOKENS,
    MODEL_NAME,
    OLLAMA_BASE_PORT,
    OLLAMA_HOST,
    OLLAMA_KEEP_ALIVE,
    REQUEST_TIMEOUT,
    TEMPERATURE,
)

# Populated by configure_endpoints(); falls back to a single local server.
_ENDPOINTS: list[str] = [f"http://{OLLAMA_HOST}:{OLLAMA_BASE_PORT}"]
_counter = itertools.count()
_thread_local = threading.local()
_session_lock = threading.Lock()
_sessions: dict[int, requests.Session] = {}


def configure_endpoints(base_urls: list[str]) -> None:
    """Register the Ollama servers the client should spread requests over."""
    global _ENDPOINTS
    if base_urls:
        _ENDPOINTS = list(base_urls)


def get_endpoints() -> list[str]:
    return list(_ENDPOINTS)


def _endpoint_for_thread() -> str:
    """Sticky round-robin: each thread keeps the endpoint it was first given."""
    endpoint = getattr(_thread_local, "endpoint", None)
    if endpoint is None:
        endpoint = _ENDPOINTS[next(_counter) % len(_ENDPOINTS)]
        _thread_local.endpoint = endpoint
    return endpoint


def _session() -> requests.Session:
    """One connection-pooled Session per thread."""
    key = threading.get_ident()
    with _session_lock:
        session = _sessions.get(key)
        if session is None:
            session = requests.Session()
            _sessions[key] = session
        return session


def check_ollama_ready(base_url: str | None = None, model: str = MODEL_NAME) -> bool:
    """
    True if the server responds and has `model` available.

    With no argument, every configured endpoint must pass.
    """
    targets = [base_url] if base_url else _ENDPOINTS
    for url in targets:
        try:
            response = requests.get(f"{url}/api/tags", timeout=10)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
        except (requests.exceptions.RequestException, ValueError) as exc:
            print(f"   ❌ {url} unreachable: {exc}")
            return False

        # Ollama reports "llama3.2:3b"; accept a bare "llama3.2" too.
        if not any(n == model or n.split(":")[0] == model.split(":")[0] for n in names):
            print(f"   ❌ {url} is up but does not have '{model}' (has: {names})")
            return False
    return True


def generate(
    prompt: str,
    system_prompt: str | None = None,
    seed: int | None = 42,
    timeout: int = REQUEST_TIMEOUT,
    base_url: str | None = None,
    **kwargs,
) -> str:
    """
    Generate a completion, retrying with backoff on transient failures.

    Args:
        prompt: User prompt.
        system_prompt: Optional system prompt.
        seed: Fixed seed for deterministic decoding; None for stochastic sampling.
        timeout: Per-request timeout in seconds.
        base_url: Override the endpoint this thread would otherwise use.
        **kwargs: `temperature` and `max_tokens` are forwarded to Ollama.

    Raises:
        TimeoutError: if every retry fails.
    """
    url = f"{base_url or _endpoint_for_thread()}/api/generate"

    options = {
        "temperature": kwargs.get("temperature", TEMPERATURE),
        "num_predict": kwargs.get("max_tokens", MAX_TOKENS),
    }
    # Omit the seed entirely when sampling stochastically — sending seed=None
    # would make the uncertainty samples degenerate.
    if seed is not None:
        options["seed"] = seed

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": options,
    }
    if system_prompt:
        payload["system"] = system_prompt

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = _session().post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json().get("response", "")
        except (requests.exceptions.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                backoff = 5 * (attempt + 1)
                print(
                    f"\n[⚠️  {url}] attempt {attempt + 1}/{MAX_RETRIES} failed: {exc}. "
                    f"Retrying in {backoff}s..."
                )
                time.sleep(backoff)

    raise TimeoutError(f"Ollama request to {url} failed after {MAX_RETRIES} attempts: {last_error}")
