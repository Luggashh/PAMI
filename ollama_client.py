import requests
import time

def generate(prompt, system_prompt=None, seed=42, timeout=300, **kwargs):
    """
    Extrem robuste Generierungsfunktion für den 100+ Fragen Großrechenlauf.
    Verhindert Abstürze durch automatisches Retry bei Timeouts.
    """
    url = "http://localhost:11434/api/generate"
    
    # Optionen dynamisch zusammenbauen, damit die Audit-Schleife nicht abstürzt
    options = {
        "seed": seed,
        "temperature": kwargs.get("temperature", 0.7)
    }
    
    # Mappt Parameter aus Ihrer config.py / audit_loop.py ab
    if "max_tokens" in kwargs:
        options["num_predict"] = kwargs["max_tokens"]
    if "temperature_greedy" in kwargs and kwargs.get("temperature_greedy") == 0.0:
        options["temperature"] = 0.0

    payload = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False,
        "options": options
    }
    if system_prompt:
        payload["system"] = system_prompt

    # Bis zu 5 Wiederholungsversuche bei Auslastungs-Timeouts
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json().get("response", "")
        except (requests.exceptions.RequestException, Exception) as e:
            if attempt < max_retries - 1:
                # Exponentiell ansteigende Wartezeit: 5s, 10s, 15s, 20s...
                wartezeit = 5 * (attempt + 1)
                print(f"\n[⚠️ GPU Auslastung/Timeout] Versuch {attempt+1}/{max_retries} verzögert: {e}. Warte {wartezeit}s...")
                time.sleep(wartezeit)
            else:
                print("\n❌ Ollama-Server antwortet permanent nicht. Versuche den Prozess im Skript zu retten...")
                raise TimeoutError(f"Ollama-Anfrage permanent fehlgeschlagen nach {max_retries} Versuchen: {e}")
