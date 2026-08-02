"""
Configuration for the Iterative Audit Loop pipeline.
"""

# ── Ollama Settings ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b"

# ── Generation Parameters ────────────────────────────────────────
TEMPERATURE = 0.7
TEMPERATURE_GREEDY = 0.0
MAX_TOKENS = 1024           # ← Increased to allow full CoT responses
SEED_MAIN = 42

# ── Audit Loop Settings ─────────────────────────────────────────
MAX_STEPS = 4
UNCERTAINTY_SAMPLES = 5
AGREEMENT_THRESHOLD = 4

# ── Dataset Settings ─────────────────────────────────────────────
GSM8K_SPLIT = "test"
NUM_EXAMPLES = 50

# ── Output ───────────────────────────────────────────────────────
OUTPUT_DIR = "results"
SAVE_COT = True              # ← Save full chain-of-thought traces