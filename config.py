"""Configuration for the Iterative Audit Loop pipeline."""

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b"

TEMPERATURE = 0.7
TEMPERATURE_GREEDY = 0.0
MAX_TOKENS = 1024
SEED_MAIN = 42

MAX_STEPS = 4
UNCERTAINTY_SAMPLES = 5
AGREEMENT_THRESHOLD = 4

GSM8K_SPLIT = "test"
NUM_EXAMPLES = 800           # Updated to 800

OUTPUT_DIR = "results"
SAVE_COT = True