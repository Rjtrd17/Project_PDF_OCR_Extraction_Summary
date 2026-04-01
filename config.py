"""
config.py — Load and validate all settings from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (same dir as this file)
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

# ─── LLM Provider ────────────────────────────────────────────────────────────
# Global ON/OFF switches — edit .env: ACTIVE_LLM=anthropic|openai|gemini|ollama
ACTIVE_LLM: str = os.getenv("ACTIVE_LLM", "anthropic").lower().strip()

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI    = "openai"
PROVIDER_GEMINI    = "gemini"
PROVIDER_OLLAMA    = "ollama"

SUPPORTED_PROVIDERS = {PROVIDER_ANTHROPIC, PROVIDER_OPENAI, PROVIDER_GEMINI, PROVIDER_OLLAMA}

if ACTIVE_LLM not in SUPPORTED_PROVIDERS:
    raise ValueError(
        f"ACTIVE_LLM='{ACTIVE_LLM}' is not supported. "
        f"Choose from: {sorted(SUPPORTED_PROVIDERS)}"
    )

# ─── API Keys ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY:    str = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY:    str = os.getenv("GEMINI_API_KEY", "")

# ─── Model Names ─────────────────────────────────────────────────────────────
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")
OPENAI_MODEL:    str = os.getenv("OPENAI_MODEL",    "gpt-4o")
GEMINI_MODEL:    str = os.getenv("GEMINI_MODEL",    "gemini-1.5-pro")
OLLAMA_MODEL:    str = os.getenv("OLLAMA_MODEL",    "llama3.1")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ─── OCR ─────────────────────────────────────────────────────────────────────
OCR_LANGUAGE:       str = os.getenv("OCR_LANGUAGE", "eng")
OCR_DPI:            int = int(os.getenv("OCR_DPI", "300"))
OCR_THREADS:        int = int(os.getenv("OCR_THREADS", "4"))
TESSERACT_CMD:      str = os.getenv("TESSERACT_CMD", "")
SCANNED_THRESHOLD:  int = int(os.getenv("SCANNED_THRESHOLD", "50"))

# ─── Pipeline ────────────────────────────────────────────────────────────────
OCR_OUTPUT_DIR:       str = os.getenv("OCR_OUTPUT_DIR", "./ocr_output")
MIN_WORDS_PER_SECTION: int = int(os.getenv("MIN_WORDS_PER_SECTION", "100"))
LLM_MAX_TOKENS:       int = int(os.getenv("LLM_MAX_TOKENS", "8000"))
RETRY_DELAY:          int = int(os.getenv("RETRY_DELAY", "8"))
RETRY_ATTEMPTS:       int = int(os.getenv("RETRY_ATTEMPTS", "3"))

# ─── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE:    int = int(os.getenv("CHUNK_SIZE", "6000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "400"))

# Text longer than this uses Map-Reduce; shorter uses a single direct call
DIRECT_CALL_CHAR_LIMIT: int = 75_000

# ─── File Paths ──────────────────────────────────────────────────────────────
CHECKPOINT_FILE: str = os.getenv("CHECKPOINT_FILE", "pipeline_checkpoint.json")
LOG_FILE:        str = os.getenv("LOG_FILE",        "pipeline_log.txt")
OUTPUT_EXCEL:    str = os.getenv("OUTPUT_EXCEL",    "policy_summaries.xlsx")

# ─── Framework Sections (fixed — do not reorder) ────────────────────────────
FRAMEWORK_SECTIONS = [
    {
        "key":  "introduction",
        "name": "Introduction / Preamble",
        "desc": "Opening context, purpose, and scope of the policy",
    },
    {
        "key":  "vision",
        "name": "Vision / Objectives / Goals",
        "desc": "What the policy aims to achieve; strategic intent",
    },
    {
        "key":  "key_provisions",
        "name": "Key Policy Provisions / Measures",
        "desc": "Core policy decisions, rules, directives, and regulatory measures",
    },
    {
        "key":  "implementation",
        "name": "Implementation Plan / Action Points",
        "desc": "How the policy will be executed; roles, responsibilities, mechanisms",
    },
    {
        "key":  "target_groups",
        "name": "Target Groups / Beneficiaries",
        "desc": "Who is covered, served, or affected by this policy",
    },
    {
        "key":  "governance",
        "name": "Institutional Framework / Governance",
        "desc": "Bodies, committees, and agencies responsible for the policy",
    },
    {
        "key":  "monitoring",
        "name": "Monitoring, Evaluation & Review",
        "desc": "How progress will be tracked, measured, and reviewed",
    },
    {
        "key":  "financial",
        "name": "Financial Provisions / Resource Allocation",
        "desc": "Funding, budget, financial support, and economic provisions",
    },
]

SECTION_KEYS = [s["key"] for s in FRAMEWORK_SECTIONS]


def validate():
    """Call at startup to catch missing API keys early."""
    checks = {
        PROVIDER_ANTHROPIC: ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
        PROVIDER_OPENAI:    ("OPENAI_API_KEY",    OPENAI_API_KEY),
        PROVIDER_GEMINI:    ("GEMINI_API_KEY",    GEMINI_API_KEY),
        PROVIDER_OLLAMA:    (None, "ok"),   # Ollama needs no API key
    }
    var_name, value = checks[ACTIVE_LLM]
    if var_name and not value:
        raise EnvironmentError(
            f"ACTIVE_LLM='{ACTIVE_LLM}' but {var_name} is not set in .env"
        )
