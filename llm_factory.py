"""
llm_factory.py
==============
Multi-provider LLM factory.

Controlled entirely by ACTIVE_LLM in .env:
    anthropic  →  Claude  (ANTHROPIC_API_KEY required)
    openai     →  GPT     (OPENAI_API_KEY required)
    gemini     →  Gemini  (GEMINI_API_KEY required)
    ollama     →  Local   (Ollama daemon must be running)

Returns a LangChain BaseChatModel compatible object so the rest of
the pipeline works identically regardless of which provider is active.
"""

from langchain_core.language_models.chat_models import BaseChatModel
import config as cfg


def get_llm() -> BaseChatModel:
    """Build and return the LangChain LLM for the active provider."""

    provider = cfg.ACTIVE_LLM

    # ── 1. ANTHROPIC CLAUDE ──────────────────────────────────────────────────
    if provider == cfg.PROVIDER_ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=cfg.ANTHROPIC_MODEL,
            temperature=0.1,
            max_tokens=cfg.LLM_MAX_TOKENS,
            api_key=cfg.ANTHROPIC_API_KEY,
        )

    # ── 2. OPENAI GPT ────────────────────────────────────────────────────────
    elif provider == cfg.PROVIDER_OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=cfg.OPENAI_MODEL,
            temperature=0.1,
            max_tokens=cfg.LLM_MAX_TOKENS,
            api_key=cfg.OPENAI_API_KEY,
        )

    # ── 3. GOOGLE GEMINI ─────────────────────────────────────────────────────
    elif provider == cfg.PROVIDER_GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=cfg.GEMINI_MODEL,
            temperature=0.1,
            max_output_tokens=cfg.LLM_MAX_TOKENS,
            google_api_key=cfg.GEMINI_API_KEY,
        )

    # ── 4. OLLAMA (local) ────────────────────────────────────────────────────
    elif provider == cfg.PROVIDER_OLLAMA:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=cfg.OLLAMA_MODEL,
            base_url=cfg.OLLAMA_BASE_URL,
            temperature=0.1,
            num_predict=cfg.LLM_MAX_TOKENS,
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def provider_info() -> str:
    """Return a short human-readable description of the active provider."""
    p = cfg.ACTIVE_LLM
    mapping = {
        cfg.PROVIDER_ANTHROPIC: f"Anthropic Claude  [{cfg.ANTHROPIC_MODEL}]",
        cfg.PROVIDER_OPENAI:    f"OpenAI GPT        [{cfg.OPENAI_MODEL}]",
        cfg.PROVIDER_GEMINI:    f"Google Gemini     [{cfg.GEMINI_MODEL}]",
        cfg.PROVIDER_OLLAMA:    f"Ollama (local)    [{cfg.OLLAMA_MODEL} @ {cfg.OLLAMA_BASE_URL}]",
    }
    return mapping.get(p, p)
