# PDF Policy Summarization Pipeline v2.0

> **100+ PDFs → OCR → 8-Section Summaries → Excel**
> LangGraph DAG · Multi-LLM Provider · Resume-able · Full Logging

---

## Architecture Overview

```
main.py
  └─► pipeline.py  (LangGraph DAG orchestration)
        ├─► ocr_engine.py   (OpenCV + Tesseract OCR)
        ├─► summarizer.py   (LangChain Map-Reduce)
        └─► excel_writer.py (openpyxl formatted output)

config.py          ← reads .env, exposes all settings
llm_factory.py     ← returns active LLM (Anthropic / OpenAI / Gemini / Ollama)
```

### LangGraph DAG (per PDF)

```
START
  │
  ▼
[ocr_node]          Extract text: pdfplumber + OpenCV + Tesseract
  │  ↓ error → END
  ▼
[save_ocr_node]     Save OCR text to ocr_output/<title>.txt
  │
  ▼
[summarize_node]    Build 8-section summaries from OCR chunks ONLY
  │
  ▼
[write_row_node]    Persist result to checkpoint JSON
  │
  ▼
END
```

> **Do I need AI Agents for this?**
> No — this pipeline uses a **fixed DAG** (same steps for every PDF).
> An AI Agent is only needed if you want dynamic decision-making
> (e.g. the LLM decides to re-OCR, fetch external sources, or route
> differently per document). The LLM here is a "smart function" inside
> `summarize_node`, not an autonomous agent.

---

## Quick Start

### Step 1 — Install system tools

**Ubuntu / Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

**Windows:**
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Poppler: https://github.com/oschwartz10612/poppler-windows/releases/

### Step 2 — Install Python packages
```bash
pip install -r requirements.txt
```

### Step 3 — Configure `.env`
```bash
cp .env.example .env
# Edit .env with your chosen LLM provider and API key
```

Minimal `.env` for Anthropic (default):
```
ACTIVE_LLM=anthropic
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Step 4 — Prepare your input Excel
```bash
python create_input_template.py
# Opens input_template.xlsx — add your PDF titles and file paths
```

### Step 5 — Run
```bash
# Using input Excel
python main.py --input input_template.xlsx

# Auto-scan a directory
python main.py --pdf-dir ./my_pdfs

# Specify output file
python main.py --input input_template.xlsx --output results.xlsx
```

---

## Multi-LLM Provider Switching

Edit `ACTIVE_LLM` in `.env` — **no code change needed**:

| `.env` setting       | Provider       | Key required         |
|---------------------|----------------|----------------------|
| `ACTIVE_LLM=anthropic` | Claude (Anthropic) | `ANTHROPIC_API_KEY` |
| `ACTIVE_LLM=openai`    | GPT-4o (OpenAI)    | `OPENAI_API_KEY`    |
| `ACTIVE_LLM=gemini`    | Gemini (Google)    | `GEMINI_API_KEY`    |
| `ACTIVE_LLM=ollama`    | Local via Ollama   | *(none)*            |

**Override via CLI** (no .env edit):
```bash
python main.py --pdf-dir ./pdfs --llm openai
```

**Ollama setup** (local, free):
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1
# Set in .env: ACTIVE_LLM=ollama
```

---

## Input Excel Format

The pipeline reads `PDF_Title` and `PDF_Path` columns from your input Excel:

| Column | Name | Required | Notes |
|--------|------|----------|-------|
| A | `PDF_Title` | ✅ | Row label in output Excel |
| B | `PDF_Path` | ✅ | Full or relative path to PDF |

**Relative paths** are resolved from the working directory or `--pdf-dir`.

---

## Output Excel Format

| Column | Section |
|--------|---------|
| A | PDF_Title |
| B | Introduction / Preamble |
| C | Vision / Objectives / Goals |
| D | Key Policy Provisions / Measures |
| E | Implementation Plan / Action Points |
| F | Target Groups / Beneficiaries |
| G | Institutional Framework / Governance |
| H | Monitoring, Evaluation & Review |
| I | Financial Provisions / Resource Allocation |

Every cell contains **≥ 100 words** derived strictly from OCR text.

---

## OCR Pipeline (per PDF)

```
pdfplumber → native text → avg chars/page
                                │
              < 50 chars/pg ────┼──── ≥ 50 chars/pg
              (SCANNED)         │     (TEXT-BASED)
                 │              │          │
           pdf2image        pdfplumber     │
             (300 DPI)      primary        │
                 │              │     sample 3 pages
           OpenCV preprocess    │     OCR quality check
           • Grayscale          │          │
           • Denoise            │    OCR > 120% native?
           • Adaptive thresh    │     YES ─► full OCR
           • Deskew             │     NO  ─► keep pdfplumber
           • Morph close        │
                 │              │
           Tesseract OCR        │
           (parallel threads)   │
                 └──────────────┘
                        │
              save to ocr_output/<title>.txt
```

---

## Summarization Strategy

```
OCR text length ≤ 75,000 chars  →  DIRECT   (single LLM call)
OCR text length  > 75,000 chars  →  MAP-REDUCE

MAP-REDUCE:
  MAP    : Split into 6,000-char chunks
           Extract section notes from each chunk independently
  REDUCE : Aggregate notes per section
           Final synthesis call → 8 full paragraphs
  EXPAND : Any section < 100 words gets a targeted expansion call

STRICT RULE: All summaries are built ONLY from OCR-extracted text.
No external knowledge injected at any step.
```

---

## Resume / Checkpoint

Progress is saved to `pipeline_checkpoint.json` after every PDF.
Re-running the same command automatically skips completed PDFs.

```bash
# Resume (auto-detected)
python main.py --input input_template.xlsx

# Start completely fresh
python main.py --input input_template.xlsx --reset

# Preview what would run without executing
python main.py --input input_template.xlsx --dry-run
```

---

## OCR Output Files

Every PDF's extracted text is saved individually:
```
ocr_output/
├── National Health Policy 2017.txt
├── Digital India Programme.txt
├── National Education Policy 2020.txt
└── ...
```

Filename = `PDF_Title` (with path-unsafe characters replaced by `_`).

---

## All CLI Options

```
python main.py [options]

  -i, --input       Input Excel with PDF_Title / PDF_Path columns
  -d, --pdf-dir     Directory to auto-scan for *.pdf files
  -o, --output      Output Excel path (default: policy_summaries.xlsx)
  -c, --checkpoint  Checkpoint JSON path
  -l, --log-file    Log file path
      --llm         Override ACTIVE_LLM (anthropic|openai|gemini|ollama)
      --reset       Delete checkpoint and start fresh
      --dry-run     List PDFs to be processed without running anything
```

---

## All `.env` Settings

```ini
# LLM provider (anthropic | openai | gemini | ollama)
ACTIVE_LLM=anthropic

# API keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# Model names (optional — shown are defaults)
ANTHROPIC_MODEL=claude-opus-4-5
OPENAI_MODEL=gpt-4o
GEMINI_MODEL=gemini-1.5-pro
OLLAMA_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434

# OCR
OCR_LANGUAGE=eng            # e.g. eng+hin for multi-language
OCR_DPI=300
OCR_THREADS=4               # parallel threads for page OCR
TESSERACT_CMD=              # full path if tesseract not on PATH
SCANNED_THRESHOLD=50        # avg chars/page to detect scanned PDFs

# Pipeline
OCR_OUTPUT_DIR=./ocr_output
MIN_WORDS_PER_SECTION=100
LLM_MAX_TOKENS=8000
RETRY_ATTEMPTS=3
RETRY_DELAY=8               # seconds between retries

# Chunking
CHUNK_SIZE=6000
CHUNK_OVERLAP=400

# Files
CHECKPOINT_FILE=pipeline_checkpoint.json
LOG_FILE=pipeline_log.txt
OUTPUT_EXCEL=policy_summaries.xlsx
```

---

## File Structure

```
pdf_pipeline_v2/
├── main.py                    ← Entry point (CLI)
├── pipeline.py                ← LangGraph DAG orchestration
├── ocr_engine.py              ← OCR: OpenCV + Tesseract
├── summarizer.py              ← LLM summarization (Map-Reduce)
├── excel_writer.py            ← Formatted Excel output
├── llm_factory.py             ← Multi-provider LLM factory
├── config.py                  ← .env loader & settings
│
├── .env.example               ← Copy to .env and fill in keys
├── requirements.txt           ← Python dependencies
├── create_input_template.py   ← Generate input Excel template
├── README.md                  ← This file
│
├── input_template.xlsx        ← [generated] Add your PDFs here
├── pipeline_checkpoint.json   ← [auto-created] Resume state
├── pipeline_log.txt           ← [auto-created] Full debug log
├── policy_summaries.xlsx      ← [auto-created] Final output
│
└── ocr_output/                ← [auto-created] OCR text files
    ├── Document_A.txt
    ├── Document_B.txt
    └── ...
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `tesseract not found` | Install `tesseract-ocr` or set `TESSERACT_CMD=/path/to/tesseract` in `.env` |
| `poppler not found` | Install `poppler-utils` (apt) or `poppler` (brew) |
| API key error | Check `.env` has the correct key for `ACTIVE_LLM` |
| Low OCR quality on scanned docs | Increase `OCR_DPI=400` in `.env` |
| Rate limit errors | Reduce `OCR_THREADS`; pipeline already retries with backoff |
| Corrupted / password-protected PDF | PDF will be skipped with `[SKIPPED]` in output |
| JSON parse failure from LLM | Pipeline retries up to `RETRY_ATTEMPTS` times |
| Ollama not responding | Run `ollama serve` and confirm `OLLAMA_BASE_URL` in `.env` |
