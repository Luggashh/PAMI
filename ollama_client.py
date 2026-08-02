"""
Thin wrapper around the Ollama REST API.
Handles both deterministic (greedy) and stochastic (sampling) generation.
"""

import requests
import json
from config import OLLAMA_BASE_URL, MODEL_NAME, MAX_TOKENS

def generate(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    seed: int | None = None,
    model: str = MODEL_NAME,
) -> str:
    """
    Send a generation request to the local Ollama server.

    Args:
        prompt: The user/content prompt.
        system_prompt: System-level instruction.
        temperature: Sampling temperature (0.0 = greedy).
        seed: Random seed for reproducibility (None = stochastic).
        model: Ollama model tag.

    Returns:
        The generated text response.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": MAX_TOKENS,
        },
    }

    if seed is not None:
        payload["options"]["seed"] = seed

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama. Is it running? Start with: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama request timed out after 120s.")

def check_ollama_ready() -> bool:
    """Verify Ollama is running and the model is available."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        # Check for exact match or match without tag
        model_base = MODEL_NAME.split(":")[0]
        available = any(model_base in m for m in models)
        if not available:
            print(f"⚠  Model '{MODEL_NAME}' not found. Available: {models}")
            print(f"   Run: ollama pull {MODEL_NAME}")
            return False
        return True
    except Exception as e:
        print(f"⚠  Ollama not reachable at {OLLAMA_BASE_URL}: {e}")
        return False