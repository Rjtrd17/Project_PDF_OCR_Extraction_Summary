"""
summarizer.py
=============
Build 8 policy framework summaries STRICTLY from OCR-extracted text chunks.
No document metadata, no external knowledge — summaries derive solely from
what the OCR engine extracted.

Strategy:
  Short doc (≤ DIRECT_CALL_CHAR_LIMIT) → single LLM call  (DIRECT mode)
  Long  doc  (> DIRECT_CALL_CHAR_LIMIT) → Map-Reduce       (CHUNK mode)
    MAP    : extract section notes from every chunk independently
    REDUCE : synthesise chunk notes into final 8-section summaries
"""

import re
import json
import time
import logging
from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

import config as cfg

logger = logging.getLogger("pdf_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a senior policy analyst. You produce accurate, thorough, structured "
    "summaries of government, regulatory, and institutional policy documents. "
    "You analyse ONLY the text provided — you do not use outside knowledge."
)

_DIRECT_PROMPT = """\
Analyse the following OCR-extracted policy document text and produce a structured summary.

STRICT RULE: Base your entire response ONLY on information present in the text below.
Do NOT use external knowledge or assumptions.

Return ONLY a valid JSON object — no markdown fences, no preamble, no trailing text.
Each JSON value MUST contain AT LEAST {min_words} words.
If the text has insufficient information for a section, write your best inference
from the available text, labelled "(Inferred from context)".

Required JSON structure:
{{
  "introduction":   "<≥{min_words} words — opening context, purpose, and scope>",
  "vision":         "<≥{min_words} words — strategic objectives and goals>",
  "key_provisions": "<≥{min_words} words — core policy rules, directives, measures>",
  "implementation": "<≥{min_words} words — execution plan, roles, responsibilities>",
  "target_groups":  "<≥{min_words} words — who is covered or affected>",
  "governance":     "<≥{min_words} words — responsible bodies, committees, agencies>",
  "monitoring":     "<≥{min_words} words — tracking, evaluation, review mechanisms>",
  "financial":      "<≥{min_words} words — budget, funding, financial provisions>"
}}

OCR text:
\"\"\"
{text}
\"\"\"\
"""

_CHUNK_EXTRACT_PROMPT = """\
You are reading a CHUNK of a larger OCR-extracted policy document.
Extract ALL relevant information from this chunk for each of the 8 policy sections.

STRICT RULE: Use ONLY information present in this chunk. No external knowledge.

Return ONLY a valid JSON object with exactly these 8 keys.
Write concise bullet-point notes. If the chunk contains nothing for a section, write "N/A".

{{
  "introduction":   "...",
  "vision":         "...",
  "key_provisions": "...",
  "implementation": "...",
  "target_groups":  "...",
  "governance":     "...",
  "monitoring":     "...",
  "financial":      "..."
}}

Chunk text (from OCR):
\"\"\"
{chunk}
\"\"\"\
"""

_REDUCE_PROMPT = """\
You have been given bullet-point notes extracted from ALL chunks of an OCR-scanned policy document.
Synthesise these notes into final comprehensive summaries for each of the 8 policy framework sections.

STRICT RULE: Use ONLY information present in the notes below — no external knowledge.

Return ONLY a valid JSON object — no markdown fences, no preamble, no trailing text.
Each value MUST contain AT LEAST {min_words} words.
Where notes are sparse, expand analytically using only what the notes contain.
Label any inference as "(Inferred from available text)".

Required JSON structure:
{{
  "introduction":   "<≥{min_words} words>",
  "vision":         "<≥{min_words} words>",
  "key_provisions": "<≥{min_words} words>",
  "implementation": "<≥{min_words} words>",
  "target_groups":  "<≥{min_words} words>",
  "governance":     "<≥{min_words} words>",
  "monitoring":     "<≥{min_words} words>",
  "financial":      "<≥{min_words} words>"
}}

Aggregated section notes from all OCR chunks:
\"\"\"
{notes}
\"\"\"\
"""

_EXPAND_PROMPT = """\
The following policy section summary is too short (under {min_words} words).
Expand it using ONLY the information available in the existing text below.
Do NOT add external knowledge. Return ONLY the expanded paragraph text, no JSON.

Section: {section_name}
Current text:
{current_text}
"""


# ─────────────────────────────────────────────────────────────────────────────
# JSON Parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(response_text: str) -> Optional[Dict[str, str]]:
    """Extract and validate the JSON dict from an LLM response."""
    text = re.sub(r"```(?:json)?", "", response_text).strip().rstrip("`").strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        if all(k in data for k in cfg.SECTION_KEYS):
            return {k: str(data[k]).strip() for k in cfg.SECTION_KEYS}
    except json.JSONDecodeError:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM Call Helper (with retry + exponential back-off)
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(llm: BaseChatModel, messages: list, label: str = "") -> str:
    for attempt in range(1, cfg.RETRY_ATTEMPTS + 1):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            wait = cfg.RETRY_DELAY * attempt
            logger.warning(f"  [{label}] LLM error attempt {attempt}: {e} — retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"LLM call '{label}' failed after {cfg.RETRY_ATTEMPTS} attempts")


# ─────────────────────────────────────────────────────────────────────────────
# Direct Summarisation (short docs)
# ─────────────────────────────────────────────────────────────────────────────

def _summarise_direct(ocr_text: str, llm: BaseChatModel) -> Dict[str, str]:
    prompt = _DIRECT_PROMPT.format(
        text=ocr_text[: cfg.DIRECT_CALL_CHAR_LIMIT],
        min_words=cfg.MIN_WORDS_PER_SECTION,
    )
    for attempt in range(1, cfg.RETRY_ATTEMPTS + 1):
        raw = _call_llm(
            llm,
            [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
            label="direct",
        )
        sections = _parse_json(raw)
        if sections:
            return sections
        logger.warning(f"  [direct] JSON parse failed attempt {attempt}, retrying …")
        time.sleep(cfg.RETRY_DELAY)
    raise RuntimeError("Direct summarisation: JSON parse failed after all retries")


# ─────────────────────────────────────────────────────────────────────────────
# Map-Reduce Summarisation (long docs)
# ─────────────────────────────────────────────────────────────────────────────

def _map_chunks(chunks: List[str], llm: BaseChatModel) -> List[Dict[str, str]]:
    """MAP phase: extract section notes from each chunk independently."""
    all_notes = []
    for i, chunk in enumerate(chunks):
        logger.info(f"  [MAP] chunk {i+1}/{len(chunks)} …")
        prompt = _CHUNK_EXTRACT_PROMPT.format(chunk=chunk)
        for attempt in range(1, cfg.RETRY_ATTEMPTS + 1):
            raw   = _call_llm(llm, [HumanMessage(content=prompt)], label=f"map-{i}")
            notes = _parse_json(raw)
            if notes:
                all_notes.append(notes)
                break
            logger.warning(f"  [MAP] chunk {i+1} parse fail attempt {attempt}")
            time.sleep(cfg.RETRY_DELAY)
        else:
            logger.warning(f"  [MAP] chunk {i+1} skipped after all retries")
    return all_notes


def _reduce_notes(chunk_notes: List[Dict[str, str]], llm: BaseChatModel) -> Dict[str, str]:
    """REDUCE phase: synthesise all chunk notes into final 8-section summary."""
    # Aggregate notes per section
    aggregated = {k: [] for k in cfg.SECTION_KEYS}
    for note_dict in chunk_notes:
        for k in cfg.SECTION_KEYS:
            v = note_dict.get(k, "").strip()
            if v and v.lower() not in ("n/a", "none", ""):
                aggregated[k].append(v)

    combined_notes = "\n\n".join(
        f"=== {k.upper()} ===\n" + "\n".join(lines)
        for k, lines in aggregated.items()
        if lines
    )

    prompt = _REDUCE_PROMPT.format(
        notes=combined_notes[:90_000],
        min_words=cfg.MIN_WORDS_PER_SECTION,
    )

    for attempt in range(1, cfg.RETRY_ATTEMPTS + 1):
        raw      = _call_llm(
            llm,
            [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
            label="reduce",
        )
        sections = _parse_json(raw)
        if sections:
            return sections
        logger.warning(f"  [REDUCE] JSON parse fail attempt {attempt}, retrying …")
        time.sleep(cfg.RETRY_DELAY)
    raise RuntimeError("Map-Reduce REDUCE phase: JSON parse failed after all retries")


def _summarise_mapreduce(ocr_text: str, llm: BaseChatModel) -> Dict[str, str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
    )
    chunks      = splitter.split_text(ocr_text)
    logger.info(f"  [MAP-REDUCE] {len(chunks)} chunks")
    chunk_notes = _map_chunks(chunks, llm)
    if not chunk_notes:
        raise RuntimeError("Map-Reduce: all chunks failed extraction")
    return _reduce_notes(chunk_notes, llm)


# ─────────────────────────────────────────────────────────────────────────────
# Word Count Enforcement
# ─────────────────────────────────────────────────────────────────────────────

def _enforce_word_counts(sections: Dict[str, str], llm: BaseChatModel) -> Dict[str, str]:
    """Expand any section that falls below MIN_WORDS_PER_SECTION."""
    min_w  = cfg.MIN_WORDS_PER_SECTION
    shorts = [(k, v) for k, v in sections.items() if len(v.split()) < min_w]
    if not shorts:
        return sections

    logger.info(f"  Expanding {len(shorts)} short section(s) …")
    section_name_map = {s["key"]: s["name"] for s in cfg.FRAMEWORK_SECTIONS}

    for key, current_text in shorts:
        prompt = _EXPAND_PROMPT.format(
            min_words=min_w,
            section_name=section_name_map[key],
            current_text=current_text,
        )
        for attempt in range(1, cfg.RETRY_ATTEMPTS + 1):
            raw = _call_llm(llm, [HumanMessage(content=prompt)], label=f"expand-{key}")
            if len(raw.split()) >= min_w:
                sections[key] = raw.strip()
                logger.info(f"    {key}: expanded to {len(raw.split())} words")
                break
            logger.warning(f"  expand {key} attempt {attempt} still short")
            time.sleep(cfg.RETRY_DELAY)

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_summaries(ocr_text: str, llm: BaseChatModel) -> Dict[str, str]:
    """
    Generate 8-section policy summaries STRICTLY from the OCR text.

    Automatically chooses DIRECT or MAP-REDUCE based on text length.
    Enforces minimum word count per section.

    Args:
        ocr_text: The full OCR-extracted text for one PDF.
        llm:      A LangChain BaseChatModel instance.

    Returns:
        Dict mapping section key → summary string (≥ MIN_WORDS_PER_SECTION words each).
    """
    char_count = len(ocr_text)

    if char_count <= cfg.DIRECT_CALL_CHAR_LIMIT:
        logger.info(f"  Strategy: DIRECT  ({char_count:,} chars)")
        sections = _summarise_direct(ocr_text, llm)
    else:
        logger.info(f"  Strategy: MAP-REDUCE  ({char_count:,} chars)")
        sections = _summarise_mapreduce(ocr_text, llm)

    sections = _enforce_word_counts(sections, llm)

    # Log word counts
    for sec in cfg.FRAMEWORK_SECTIONS:
        wc = len(sections.get(sec["key"], "").split())
        status = "✓" if wc >= cfg.MIN_WORDS_PER_SECTION else "⚠"
        logger.debug(f"    {status} {sec['name']}: {wc} words")

    return sections
