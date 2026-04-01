"""
main.py — Entry point for the PDF Policy Summarization Pipeline
================================================================

Usage examples:

  # Scan all PDFs in a directory
  python main.py --pdf-dir ./pdfs

  # Use an input Excel with PDF paths
  python main.py --input pdfs_list.xlsx

  # Specify output file
  python main.py --pdf-dir ./pdfs --output my_results.xlsx

  # Resume after interruption (auto-detects checkpoint)
  python main.py --pdf-dir ./pdfs

  # Fresh start (delete existing checkpoint)
  python main.py --pdf-dir ./pdfs --reset

  # Override active LLM without editing .env
  python main.py --pdf-dir ./pdfs --llm openai

Input Excel format (see input_template.xlsx):
  Column A: PDF_Title   — display name used as row label in output
  Column B: PDF_Path    — full or relative path to the PDF file
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Dict

# ── Suppress LangChain pydantic-v1 shim warning on Python ≥ 3.14 ─────────────
# This is a cosmetic warning from LangChain internals — not an error.
# Remove this block once langchain-core ≥ 0.4 drops the v1 shim.
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible",
    category=UserWarning,
)

import pandas as pd

import config as cfg
from pipeline import CheckpointManager, run_batch
from llm_factory import provider_info


# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("pdf_pipeline")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # File handler — full debug log
    fh = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    # Console handler — info+
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Input Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf_list(
    input_excel: str | None,
    pdf_dir:     str | None,
    logger:      logging.Logger,
) -> List[Dict[str, str]]:
    """
    Return ordered list of {title, path} dicts.

    Priority:
      1. input_excel — read PDF_Title and PDF_Path columns.
      2. pdf_dir     — auto-scan for *.pdf files.
    """
    items: List[Dict[str, str]] = []

    # ── Option 1: Excel input ─────────────────────────────────────────────
    if input_excel and Path(input_excel).exists():
        logger.info(f"Loading PDF list from: {input_excel}")
        df = pd.read_excel(input_excel)
        df.columns = [str(c).strip() for c in df.columns]

        # Auto-detect title column
        title_col = next(
            (c for c in df.columns if "title" in c.lower() or "name" in c.lower()),
            df.columns[0]
        )
        # Auto-detect path column
        path_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ("path", "file", "pdf"))),
            df.columns[1] if len(df.columns) >= 2 else None,
        )

        for _, row in df.iterrows():
            title = str(row[title_col]).strip()
            if not title or title.lower() in ("nan", "pdf_title"):
                continue   # skip header-like rows
            path = str(row[path_col]).strip() if path_col else ""
            # Resolve relative paths using pdf_dir if provided
            if path and not Path(path).is_absolute() and pdf_dir:
                path = str(Path(pdf_dir) / path)
            if path and Path(path).exists():
                items.append({"title": title, "path": path})
            else:
                logger.warning(f"  PDF not found: {path!r}  (title={title!r})")

    # ── Option 2: Directory scan ──────────────────────────────────────────
    elif pdf_dir and Path(pdf_dir).is_dir():
        logger.info(f"Auto-scanning directory: {pdf_dir}")
        for pdf_file in sorted(Path(pdf_dir).rglob("*.pdf")):
            items.append({"title": pdf_file.stem, "path": str(pdf_file)})

    else:
        logger.error("No valid --input Excel or --pdf-dir provided.")
        sys.exit(1)

    logger.info(f"PDFs found: {len(items)}")
    return items


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python main.py",
        description="PDF Policy Summarization Pipeline (LangGraph DAG)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",      "-i", help="Input Excel with PDF_Title / PDF_Path columns")
    p.add_argument("--pdf-dir",    "-d", help="Directory to auto-scan for PDFs")
    p.add_argument("--output",     "-o", help="Output Excel path (default from .env)")
    p.add_argument("--checkpoint", "-c", help="Checkpoint JSON path (default from .env)")
    p.add_argument("--log-file",   "-l", help="Log file path (default from .env)")
    p.add_argument("--llm",              help="Override ACTIVE_LLM (anthropic|openai|gemini|ollama)")
    p.add_argument("--reset",      action="store_true",
                   help="Delete checkpoint and start fresh")
    p.add_argument("--dry-run",    action="store_true",
                   help="List PDFs that would be processed without running OCR/LLM")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Apply CLI overrides ───────────────────────────────────────────────
    if args.llm:
        os.environ["ACTIVE_LLM"] = args.llm
        # Reload module-level variable
        import importlib
        importlib.reload(cfg)

    output_excel    = args.output     or cfg.OUTPUT_EXCEL
    checkpoint_file = args.checkpoint or cfg.CHECKPOINT_FILE
    log_file        = args.log_file   or cfg.LOG_FILE

    # ── Logging ───────────────────────────────────────────────────────────
    logger = setup_logging(log_file)

    # ── Banner ────────────────────────────────────────────────────────────
    logger.info("=" * 62)
    logger.info("  PDF Policy Summarization Pipeline  v2.0")
    logger.info(f"  LLM Provider  : {provider_info()}")
    logger.info(f"  OCR Language  : {cfg.OCR_LANGUAGE}  |  DPI: {cfg.OCR_DPI}")
    logger.info(f"  OCR Output Dir: {cfg.OCR_OUTPUT_DIR}")
    logger.info(f"  Output Excel  : {output_excel}")
    logger.info(f"  Checkpoint    : {checkpoint_file}")
    logger.info(f"  Log File      : {log_file}")
    logger.info("=" * 62)

    # ── Validate API key ──────────────────────────────────────────────────
    try:
        cfg.validate()
    except EnvironmentError as e:
        logger.error(str(e))
        sys.exit(1)

    # ── Checkpoint ────────────────────────────────────────────────────────
    checkpoint = CheckpointManager(checkpoint_file)
    if args.reset:
        checkpoint.delete()
        checkpoint = CheckpointManager(checkpoint_file)

    # ── Load PDF list ─────────────────────────────────────────────────────
    all_items = load_pdf_list(args.input, args.pdf_dir, logger)
    if not all_items:
        logger.error("No valid PDFs found. Exiting.")
        sys.exit(1)

    # ── Dry run ───────────────────────────────────────────────────────────
    if args.dry_run:
        pending = [it for it in all_items if not checkpoint.is_done(it["title"])]
        print(f"\n{'─'*55}")
        print(f"DRY RUN — {len(all_items)} PDFs ({len(pending)} pending)")
        print(f"{'─'*55}")
        for it in all_items:
            status = "✓ done" if checkpoint.is_done(it["title"]) else "○ pending"
            print(f"  [{status}]  {it['title']}")
            print(f"           {it['path']}")
        print(f"{'─'*55}\n")
        return

    # ── Run the DAG pipeline ─────────────────────────────────────────────
    run_batch(all_items, checkpoint, output_excel)

    # ── Final summary ─────────────────────────────────────────────────────
    done    = len(checkpoint.completed)
    failed  = len(checkpoint.failed)
    skipped = len(checkpoint.skipped)
    pending = len(all_items) - done - failed - skipped

    logger.info("\n" + "=" * 62)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Total    : {len(all_items)}")
    logger.info(f"  Done     : {done}")
    logger.info(f"  Failed   : {failed}")
    logger.info(f"  Skipped  : {skipped}")
    logger.info(f"  Pending  : {pending}")
    logger.info(f"  Output   : {output_excel}")
    logger.info(f"  OCR dir  : {cfg.OCR_OUTPUT_DIR}")
    logger.info(f"  Logs     : {log_file}")
    logger.info("=" * 62)


if __name__ == "__main__":
    main()
