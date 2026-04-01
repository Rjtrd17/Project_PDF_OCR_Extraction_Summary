@echo off
REM ═══════════════════════════════════════════════════════════════
REM  install_windows.bat
REM  Run this once inside your virtual environment to install all
REM  Python dependencies correctly for Windows.
REM
REM  Usage:
REM    1. Activate your venv first:
REM         .venv\Scripts\activate
REM    2. Then run:
REM         install_windows.bat
REM ═══════════════════════════════════════════════════════════════

echo.
echo ─────────────────────────────────────────────────────────────
echo  PDF Policy Pipeline — Windows Dependency Installer
echo ─────────────────────────────────────────────────────────────
echo.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Core pipeline packages
pip install python-dotenv ^
    langchain langchain-core langchain-text-splitters langgraph ^
    langchain-anthropic anthropic ^
    langchain-openai openai ^
    langchain-google-genai google-generativeai ^
    langchain-ollama ^
    pypdf pdfplumber pdf2image ^
    pytesseract ^
    Pillow numpy ^
    openpyxl pandas tqdm

REM ── opencv-python (NOT opencv-python-headless on Windows) ─────
echo.
echo Installing opencv-python for Windows...
pip install opencv-python

echo.
echo ─────────────────────────────────────────────────────────────
echo  Python packages installed.
echo.
echo  NEXT: Install these system tools if you haven't already:
echo.
echo  1. Tesseract OCR
echo     Download: https://github.com/UB-Mannheim/tesseract/wiki
echo     Default path: C:\Program Files\Tesseract-OCR\tesseract.exe
echo     Then set in .env:
echo       TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
echo.
echo  2. Poppler (for pdf2image)
echo     Download: https://github.com/oschwartz10612/poppler-windows/releases/
echo     Extract and add the \Library\bin folder to Windows PATH
echo.
echo  3. Copy .env.example to .env and fill in your API key:
echo     copy .env.example .env
echo     notepad .env
echo.
echo ─────────────────────────────────────────────────────────────
pause
