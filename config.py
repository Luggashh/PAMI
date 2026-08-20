"""Configuration for the Iterative Audit Loop pipeline."""

import os

# ── Model ────────────────────────────────────────────────────────
MODEL_NAME = "llama3.2:3b"

TEMPERATURE = 0.7
TEMPERATURE_GREEDY = 0.0
MAX_TOKENS = 1024
SEED_MAIN = 42

# ── Audit loop ───────────────────────────────────────────────────
MAX_STEPS = 4
UNCERTAINTY_SAMPLES = 5
AGREEMENT_THRESHOLD = 4

# ── Dataset ──────────────────────────────────────────────────────
GSM8K_SPLIT = "test"
NUM_EXAMPLES = 800

OUTPUT_DIR = "results"
SAVE_COT = True

# ── Ollama server / GPU fan-out ──────────────────────────────────
# One Ollama server is launched per GPU (llama3.2:3b is ~2 GB, so it fits
# comfortably on a single 40 GB card and replicating it across cards gives
# near-linear throughput). Set NUM_SERVERS to override auto-detection.
OLLAMA_HOST = "127.0.0.1"
OLLAMA_BASE_PORT = int(os.environ.get("PAMI_BASE_PORT", 11434))
NUM_SERVERS = int(os.environ.get("PAMI_NUM_SERVERS", 0))  # 0 = one per detected GPU

# Concurrent requests each Ollama server handles (OLLAMA_NUM_PARALLEL).
NUM_PARALLEL_PER_SERVER = int(os.environ.get("PAMI_PARALLEL_PER_SERVER", 4))

# Keep the model resident so it is not unloaded between examples.
OLLAMA_KEEP_ALIVE = "24h"

# ── No-sudo Ollama install ───────────────────────────────────────
# Used only if `ollama` is not already on PATH. Everything lands inside the
# user's home directory, so no root privileges are required.
OLLAMA_INSTALL_DIR = os.path.expanduser("~/.local/ollama")
OLLAMA_MODELS_DIR = os.environ.get(
    "OLLAMA_MODELS", os.path.expanduser("~/.ollama/models")
)
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download/ollama-linux-amd64.tgz"

# ── Request handling ─────────────────────────────────────────────
REQUEST_TIMEOUT = 600
MAX_RETRIES = 5
