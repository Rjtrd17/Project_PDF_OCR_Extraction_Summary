"""
ocr_engine.py
=============
Smart OCR pipeline:
  1. Try pdfplumber for native text extraction
  2. Detect if the PDF is scanned (avg chars/page < threshold)
  3. Scanned → pdf2image → OpenCV preprocessing → Tesseract OCR
  4. Text-based → pdfplumber primary; sample OCR QA to decide if full OCR is better
  5. Save extracted text to OCR_OUTPUT_DIR/<pdf_title>.txt
"""

import os
import re
import logging
from pathlib import Path
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytesseract
from PIL import Image
import pdfplumber
from pypdf import PdfReader

import config as cfg

logger = logging.getLogger("pdf_pipeline")

# ── OpenCV import (Windows / Linux / macOS fix) ───────────────────────────────
# Windows / macOS  →  pip install opencv-python
# Linux headless   →  pip install opencv-python-headless
# Both expose the same cv2 module; only the pip package name differs.
try:
    import cv2
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "\n\n"
        "  cv2 (OpenCV) not found. Install the correct package for your OS:\n"
        "\n"
        "    Windows / macOS  →  pip install opencv-python\n"
        "    Linux headless   →  pip install opencv-python-headless\n"
        "\n"
        "  Then re-run:  python main.py ...\n"
    )

# ── Tesseract binary path (Windows: set TESSERACT_CMD in .env) ────────────────
if cfg.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = cfg.TESSERACT_CMD


# ─────────────────────────────────────────────────────────────────────────────
# OpenCV Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_for_ocr(pil_image: Image.Image) -> np.ndarray:
    """
    Apply OpenCV preprocessing pipeline for maximum Tesseract accuracy:
      • Grayscale conversion
      • Upscale if resolution is too low
      • Fast non-local means denoising
      • Adaptive Gaussian thresholding (binarise)
      • Deskew via minAreaRect
      • Morphological close to repair character gaps
    """
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img

    # Upscale very low-res images
    h, w = gray.shape
    if h < 1000:
        scale = 1000 / h
        gray = cv2.resize(gray, (int(w * scale), 1000), interpolation=cv2.INTER_CUBIC)

    denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)

    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=11,
    )

    binary = _deskew(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cleaned


def _deskew(binary: np.ndarray) -> np.ndarray:
    """Correct document skew using minAreaRect on dark pixel coordinates."""
    try:
        coords = np.column_stack(np.where(binary < 128))
        if len(coords) < 10:
            return binary
        angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.5:
            return binary
        h, w = binary.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(binary, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return binary


# ─────────────────────────────────────────────────────────────────────────────
# Tesseract OCR (parallel per image)
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_single_image(args: Tuple[int, Image.Image]) -> Tuple[int, str]:
    idx, pil_img = args
    try:
        processed = _preprocess_for_ocr(pil_img)
        pil_proc  = Image.fromarray(processed)
        # OEM 3 = LSTM+Legacy fallback; PSM 3 = fully automatic page segmentation
        cfg_str = r"--oem 3 --psm 3"
        text = pytesseract.image_to_string(pil_proc, lang=cfg.OCR_LANGUAGE, config=cfg_str)
        return idx, text
    except Exception as e:
        logger.warning(f"  [OCR] Page {idx} error: {e}")
        return idx, ""


def _run_tesseract_on_images(images: List[Image.Image]) -> str:
    """Run Tesseract in parallel across a list of PIL images."""
    args    = [(i, img) for i, img in enumerate(images)]
    results = [""] * len(images)
    with ThreadPoolExecutor(max_workers=cfg.OCR_THREADS) as pool:
        futures = {pool.submit(_ocr_single_image, a): a[0] for a in args}
        for future in as_completed(futures):
            idx, text = future.result()
            results[idx] = text
    return "\n\n".join(results)


# ─────────────────────────────────────────────────────────────────────────────
# PDF Text Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_native_text(pdf_path: str) -> str:
    """Extract text from a native PDF using pdfplumber."""
    parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                parts.append(t)
    except Exception as e:
        logger.debug(f"  pdfplumber error: {e}")
    return "\n\n".join(parts)


def _page_count(pdf_path: str) -> int:
    try:
        return len(PdfReader(pdf_path).pages)
    except Exception:
        return 1


def extract_text(pdf_path: str) -> Tuple[str, str]:
    """
    Main OCR dispatcher.

    Returns (extracted_text, method_description).

    Decision logic:
      native_text → avg chars/page < threshold → SCANNED path
                                               → TEXT path
    """
    from pdf2image import convert_from_path

    native_text = _extract_native_text(pdf_path)
    n_pages     = _page_count(pdf_path)
    avg_chars   = len(native_text.strip()) / max(n_pages, 1)
    is_scanned  = avg_chars < cfg.SCANNED_THRESHOLD

    logger.info(f"  avg_chars/page={avg_chars:.0f} | scanned={is_scanned}")

    if is_scanned:
        # ── SCANNED: pdf2image → OpenCV → Tesseract ──────────────────────────
        logger.info("  [SCANNED PATH] pdf2image → OpenCV → Tesseract")
        try:
            images   = convert_from_path(pdf_path, dpi=cfg.OCR_DPI)
            ocr_text = _run_tesseract_on_images(images)
            result   = ocr_text if len(ocr_text) > len(native_text) else native_text
            return result, "scanned_ocr"
        except Exception as e:
            logger.warning(f"  pdf2image failed ({e}), falling back to pdfplumber")
            return native_text, "pdfplumber_fallback"

    else:
        # ── TEXT: pdfplumber primary; sample Tesseract for QA ────────────────
        logger.info("  [TEXT PATH] pdfplumber primary + sample OCR QA")
        try:
            sample_n      = min(3, n_pages)
            sample_images = convert_from_path(pdf_path, dpi=cfg.OCR_DPI, last_page=sample_n)
            sample_ocr    = _run_tesseract_on_images(sample_images)
            # If sample OCR yields >20% more text than native (proportionally), run full OCR
            sample_native_len = len(native_text) * (sample_n / n_pages)
            if len(sample_ocr) > sample_native_len * 1.2:
                logger.info("  OCR sample superior — running full page OCR")
                all_images = convert_from_path(pdf_path, dpi=cfg.OCR_DPI)
                full_ocr   = _run_tesseract_on_images(all_images)
                return full_ocr, "full_ocr_upgrade"
        except Exception as e:
            logger.debug(f"  Sample OCR skipped: {e}")

        return native_text, "pdfplumber"


# ─────────────────────────────────────────────────────────────────────────────
# Save OCR Output
# ─────────────────────────────────────────────────────────────────────────────

def save_ocr_text(pdf_title: str, text: str) -> str:
    """
    Save OCR text to OCR_OUTPUT_DIR/<pdf_title>.txt
    Returns the saved file path.
    """
    out_dir = Path(cfg.OCR_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename: replace path-unsafe characters
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", pdf_title)
    out_path  = out_dir / f"{safe_name}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    logger.info(f"  OCR saved → {out_path}")
    return str(out_path)


def load_ocr_text(pdf_title: str) -> str:
    """Load previously saved OCR text. Returns empty string if not found."""
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", pdf_title)
    out_path  = Path(cfg.OCR_OUTPUT_DIR) / f"{safe_name}.txt"
    if out_path.exists():
        return out_path.read_text(encoding="utf-8")
    return ""
