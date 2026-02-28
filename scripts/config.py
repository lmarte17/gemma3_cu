import os

# --- Model Settings ---
MODEL_ID = "google/gemma-3-4b-it"

# --- Dataset Settings ---
DATASET_ID_RL = "GUI-Libra/GUI-Libra-81K-RL"
DATASET_ID_SFT = "GUI-Libra/GUI-Libra-81K-SFT"
# Default to RL for testing, but can be switched to SFT for Phase 1
DEFAULT_DATASET = DATASET_ID_RL
# For local testing, we might want to save processed data here
LOCAL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

# --- ASFT (Action-aware Supervised Fine-Tuning) Settings ---
# Tokens to look for in the labels to apply the loss multiplier
ACTION_KEYWORDS = ["Action:", "click(", "write(", "scroll(", "hover(", "long_press(", "swipe("]
# The multiplier weight applied to action and coordinate tokens (paper suggests 2.0 to 5.0)
ACTION_WEIGHT_MULTIPLIER = 3.0

# --- Prompt Settings ---
# Max length for the tokenizer to prevent OOM
MAX_SEQ_LENGTH = 2048
