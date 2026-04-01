"""
pipeline.py
===========
LangGraph DAG pipeline for PDF → OCR → Summarise → Excel.

                    ┌─────────────────────────────────────────────────────┐
                    │                  LANGGRAPH DAG                      │
                    │                                                     │
                    │  START ──► ocr_node ──► save_ocr_node              │
                    │                              │                      │
                    │                        summarize_node               │
                    │                              │                      │
                    │                        write_row_node ──► END       │
                    │                                                     │
                    │  Each PDF is a separate graph invocation.          │
                    │  Error edges route directly to END (errors          │
                    │  are recorded in checkpoint; pipeline continues).  │
                    └─────────────────────────────────────────────────────┘

Do we need AI Agents for a DAG?
───────────────────────────────
SHORT ANSWER: No — not for a fixed-step deterministic pipeline.

DETAIL:
  • An "AI Agent" dynamically decides its own next action (tool calling,
    multi-step reasoning loops, self-correction).
  • This pipeline has a FIXED DAG: every PDF follows the same linear steps.
  • LangGraph provides the DAG scaffolding; the LLM is used as a
    "smart function" inside summarize_node — not as an autonomous agent.
  • If you later want adaptive behaviour (e.g. the LLM decides whether to
    re-OCR, fetch external references, or request human review), THEN you
    would introduce an agent node with a ReAct / tool-calling loop.
    A global flag ENABLE_AGENT_NODE in .env controls this (currently off).

Checkpoint / Resume:
  Checkpoint JSON is updated after every successfully processed PDF.
  Restarting the script automatically skips already-completed titles.
"""

import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TypedDict

from langgraph.graph import StateGraph, END, START

import config as cfg
import ocr_engine
import summarizer
import excel_writer
from llm_factory import get_llm

logger = logging.getLogger("pdf_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph State Schema
# ─────────────────────────────────────────────────────────────────────────────

class PDFState(TypedDict):
    """State passed between nodes for one PDF document."""
    # Input
    pdf_title:  str
    pdf_path:   str

    # OCR
    ocr_text:   str
    ocr_method: str       # "pdfplumber" | "scanned_ocr" | "full_ocr_upgrade" | ...
    ocr_file:   str       # Path to saved OCR .txt file

    # Summaries
    summaries:  Dict[str, str]   # {section_key → text}

    # Control flow
    error:      Optional[str]    # Non-None → skip remaining nodes
    status:     str              # "init" | "ocr_done" | "saved" | "summarized" | "done"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Manager
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """Persist progress to JSON so the pipeline can resume after interruption."""

    def __init__(self, path: str):
        self.path = path
        self._data: Dict[str, Any] = {
            "version":      "2.0",
            "started_at":   datetime.now().isoformat(),
            "last_updated": None,
            "completed":    {},   # {title → {section_key → text}}
            "failed":       {},   # {title → error_message}
            "skipped":      [],   # [title, ...]  — no text extractable
        }
        self._load()

    def _load(self):
        p = Path(self.path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self._data.update(saved)
                n_done  = len(self._data["completed"])
                n_fail  = len(self._data["failed"])
                n_skip  = len(self._data["skipped"])
                logger.info(
                    f"Checkpoint loaded: {n_done} done | {n_fail} failed | {n_skip} skipped"
                )
            except Exception as e:
                logger.warning(f"Checkpoint load failed (starting fresh): {e}")

    def _save(self):
        self._data["last_updated"] = datetime.now().isoformat()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ── Read ──────────────────────────────────────────────────────────────
    def is_done(self, title: str) -> bool:
        return title in self._data["completed"]

    @property
    def completed(self) -> Dict[str, Dict[str, str]]:
        return self._data["completed"]

    @property
    def failed(self) -> Dict[str, str]:
        return self._data["failed"]

    @property
    def skipped(self) -> List[str]:
        return self._data["skipped"]

    @property
    def metadata(self) -> Dict[str, str]:
        return {k: self._data[k] for k in ("started_at", "last_updated")}

    # ── Write ─────────────────────────────────────────────────────────────
    def mark_complete(self, title: str, sections: Dict[str, str]):
        self._data["completed"][title] = sections
        self._save()

    def mark_failed(self, title: str, error: str):
        self._data["failed"][title] = error
        self._save()

    def mark_skipped(self, title: str):
        if title not in self._data["skipped"]:
            self._data["skipped"].append(title)
        self._save()

    def delete(self):
        p = Path(self.path)
        if p.exists():
            p.unlink()
            logger.info(f"Checkpoint deleted: {self.path}")


# ─────────────────────────────────────────────────────────────────────────────
# DAG Node Definitions
# ─────────────────────────────────────────────────────────────────────────────

def ocr_node(state: PDFState) -> PDFState:
    """
    NODE 1: Run OCR on the PDF.
    Extracts text using pdfplumber (native) or pdf2image+Tesseract (scanned).
    """
    if state.get("error"):
        return state

    title = state["pdf_title"]
    path  = state["pdf_path"]

    logger.info(f"[OCR NODE] {title}")
    try:
        text, method = ocr_engine.extract_text(path)
        word_count   = len(text.split())
        logger.info(f"  OCR complete | method={method} | words={word_count:,}")

        if word_count < 30:
            return {**state, "error": f"Insufficient text extracted ({word_count} words)",
                    "status": "skipped"}

        return {**state, "ocr_text": text, "ocr_method": method, "status": "ocr_done"}

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"  [OCR NODE] ERROR: {e}\n{tb}")
        return {**state, "error": f"OCR failed: {e}", "status": "error"}


def save_ocr_node(state: PDFState) -> PDFState:
    """
    NODE 2: Save OCR text to OCR_OUTPUT_DIR/<pdf_title>.txt
    This creates a permanent record of what the OCR extracted.
    """
    if state.get("error"):
        return state

    title = state["pdf_title"]
    text  = state.get("ocr_text", "")

    logger.info(f"[SAVE OCR NODE] {title}")
    try:
        file_path = ocr_engine.save_ocr_text(title, text)
        return {**state, "ocr_file": file_path, "status": "saved"}
    except Exception as e:
        logger.error(f"  [SAVE OCR NODE] ERROR: {e}")
        # Non-fatal: continue even if save fails
        return {**state, "ocr_file": "", "status": "saved"}


def summarize_node(state: PDFState) -> PDFState:
    """
    NODE 3: Build 8-section summaries STRICTLY from OCR text chunks.
    Uses the active LLM provider (controlled by ACTIVE_LLM in .env).
    """
    if state.get("error"):
        return state

    title    = state["pdf_title"]
    ocr_text = state.get("ocr_text", "")

    logger.info(f"[SUMMARIZE NODE] {title}")
    try:
        llm      = get_llm()
        sections = summarizer.build_summaries(ocr_text, llm)
        return {**state, "summaries": sections, "status": "summarized"}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"  [SUMMARIZE NODE] ERROR: {e}\n{tb}")
        return {**state, "error": f"Summarization failed: {e}", "status": "error"}


def write_row_node(state: PDFState) -> PDFState:
    """
    NODE 4: Mark this PDF as complete in the checkpoint.
    The full Excel is written once at the end of the entire batch.
    (Row-level writes avoid partial/corrupt Excel files mid-run.)
    """
    if state.get("error"):
        return state

    logger.info(f"[WRITE ROW NODE] {state['pdf_title']} → checkpoint updated")
    return {**state, "status": "done"}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Router
# ─────────────────────────────────────────────────────────────────────────────

def _route(state: PDFState) -> str:
    """Route to END immediately if an error occurred in any node."""
    return "end" if state.get("error") else "continue"


# ─────────────────────────────────────────────────────────────────────────────
# Build the LangGraph DAG
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Construct and compile the LangGraph StateGraph.

    DAG structure:
        START → ocr_node → [route] → save_ocr_node → summarize_node → write_row_node → END
                                ↓ (error)
                               END
    """
    graph = StateGraph(PDFState)

    # Add nodes
    graph.add_node("ocr_node",       ocr_node)
    graph.add_node("save_ocr_node",  save_ocr_node)
    graph.add_node("summarize_node", summarize_node)
    graph.add_node("write_row_node", write_row_node)

    # Entry edge
    graph.add_edge(START, "ocr_node")

    # Conditional routing after OCR (bail out on error/skip)
    graph.add_conditional_edges(
        "ocr_node",
        _route,
        {"continue": "save_ocr_node", "end": END},
    )

    # Linear edges for the rest of the happy path
    graph.add_edge("save_ocr_node",  "summarize_node")
    graph.add_edge("summarize_node", "write_row_node")
    graph.add_edge("write_row_node", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Batch Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    all_items:  List[Dict[str, str]],
    checkpoint: CheckpointManager,
    output_excel: str,
) -> None:
    """
    Process every PDF in all_items through the LangGraph DAG.

    Skips PDFs already present in the checkpoint (resume support).
    Writes the final Excel after the entire batch completes.
    Also writes the Excel on KeyboardInterrupt so partial progress is saved.
    """
    graph = build_graph()

    # Filter already-done items
    pending = [it for it in all_items if not checkpoint.is_done(it["title"])]
    logger.info(
        f"Batch: {len(all_items)} total | "
        f"{len(all_items)-len(pending)} already done | "
        f"{len(pending)} to process"
    )

    try:
        for i, item in enumerate(pending, start=1):
            title = item["title"]
            path  = item["path"]
            logger.info(
                f"\n{'═'*62}\n"
                f"  [{i}/{len(pending)}] {title}\n"
                f"{'═'*62}"
            )
            t0 = time.time()

            # Initialise state
            initial_state: PDFState = {
                "pdf_title":  title,
                "pdf_path":   path,
                "ocr_text":   "",
                "ocr_method": "",
                "ocr_file":   "",
                "summaries":  {},
                "error":      None,
                "status":     "init",
            }

            # Run the DAG
            try:
                final_state = graph.invoke(initial_state)
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"  Graph invocation failed: {e}\n{tb}")
                checkpoint.mark_failed(title, f"Graph error: {e}")
                continue

            # Persist result
            if final_state.get("error"):
                err = final_state["error"]
                if final_state.get("status") == "skipped":
                    logger.warning(f"  SKIPPED: {err}")
                    checkpoint.mark_skipped(title)
                else:
                    logger.error(f"  FAILED: {err}")
                    checkpoint.mark_failed(title, err)
            else:
                checkpoint.mark_complete(title, final_state["summaries"])
                elapsed = time.time() - t0
                logger.info(f"  ✓ Done in {elapsed:.1f}s | OCR→{final_state['ocr_method']}")

    except KeyboardInterrupt:
        logger.warning("\n[INTERRUPTED] Saving partial progress …")

    # Always write Excel (even on interrupt)
    _flush_excel(all_items, checkpoint, output_excel)


def _flush_excel(
    all_items:  List[Dict[str, str]],
    checkpoint: CheckpointManager,
    output_path: str,
) -> None:
    """Write the final Excel using all completed checkpoint data."""
    excel_writer.write_excel(
        output_path=output_path,
        all_items=all_items,
        completed=checkpoint.completed,
        failed=checkpoint.failed,
        skipped=checkpoint.skipped,
        run_metadata=checkpoint.metadata,
    )
