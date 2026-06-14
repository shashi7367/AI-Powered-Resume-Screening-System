"""
pdf_extractor.py — Extracts raw text from PDF files.

Accepts file paths, raw bytes, or in-memory file objects (e.g. Streamlit's
UploadedFile).  Uses pdfplumber as the primary engine and falls back to
PyPDF2 when pdfplumber fails or returns empty text.

Raises PDFExtractionError (a custom exception defined in this module) with
a descriptive message when both engines fail — never crashes silently.
"""

import io
import re
import logging
from pathlib import Path
from typing import Union, BinaryIO

import pdfplumber
import PyPDF2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class PDFExtractionError(Exception):
    """Raised when text cannot be extracted from a PDF by any available engine."""
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_input(pdf_file: Union[str, Path, bytes, BinaryIO]) -> io.BytesIO:
    """
    Converts whatever the caller passes (path, raw bytes, file-like object)
    into a seekable ``BytesIO`` stream so both engines can consume it.
    """
    if isinstance(pdf_file, (str, Path)):
        path = Path(pdf_file)
        if not path.exists():
            raise PDFExtractionError(f"PDF file not found: {path}")
        if not path.is_file():
            raise PDFExtractionError(f"Path is not a file: {path}")
        try:
            return io.BytesIO(path.read_bytes())
        except PermissionError as exc:
            raise PDFExtractionError(f"Permission denied reading {path}: {exc}") from exc
        except OSError as exc:
            raise PDFExtractionError(f"OS error reading {path}: {exc}") from exc

    if isinstance(pdf_file, bytes):
        if not pdf_file:
            raise PDFExtractionError("Received empty bytes — nothing to extract.")
        return io.BytesIO(pdf_file)

    # File-like object (Streamlit UploadedFile, open file handle, etc.)
    try:
        content = pdf_file.read()
        # Reset the original stream's cursor so it can be reused elsewhere
        if hasattr(pdf_file, "seek"):
            try:
                pdf_file.seek(0)
            except Exception:
                pass
        if not content:
            raise PDFExtractionError("File-like object was empty — nothing to extract.")
        return io.BytesIO(content)
    except PDFExtractionError:
        raise
    except Exception as exc:
        raise PDFExtractionError(
            f"Could not read from the provided file-like object: {exc}"
        ) from exc


def _clean_text(text: str) -> str:
    """
    Strips excessive whitespace while preserving meaningful structure:
    • Bullet points and section line-breaks stay intact.
    • Runs of spaces/tabs on the same line are collapsed to a single space.
    • At most one consecutive blank line is kept (avoids huge vertical gaps).
    """
    if not text:
        return ""

    # Replace non-breaking spaces and other exotic whitespace characters
    text = text.replace("\xa0", " ")

    # Collapse horizontal whitespace (spaces and tabs) within each line
    text = re.sub(r"[ \t]+", " ", text)

    # Strip each line individually
    lines = [line.strip() for line in text.splitlines()]

    # Keep at most one consecutive blank line so sections stay visually
    # separated without producing walls of empty space.
    cleaned: list[str] = []
    prev_blank = False
    for line in lines:
        if line == "":
            if not prev_blank:
                cleaned.append(line)
            prev_blank = True
        else:
            prev_blank = False
            cleaned.append(line)

    return "\n".join(cleaned).strip()


def _extract_with_pdfplumber(stream: io.BytesIO) -> str:
    """
    Attempts page-by-page text extraction using pdfplumber.
    Returns the concatenated text, or an empty string on failure.
    """
    stream.seek(0)
    try:
        with pdfplumber.open(stream) as pdf:
            pages: list[str] = []
            for idx, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)
                except Exception as page_err:
                    logger.warning(
                        "pdfplumber: page %d extraction failed — %s", idx + 1, page_err
                    )
            return "\n".join(pages)
    except Exception as exc:
        logger.warning("pdfplumber failed entirely — %s", exc)
        return ""


def _extract_with_pypdf2(stream: io.BytesIO) -> str:
    """
    Attempts page-by-page text extraction using PyPDF2.
    Returns the concatenated text, or an empty string on failure.
    """
    stream.seek(0)
    try:
        reader = PyPDF2.PdfReader(stream)
        pages: list[str] = []
        for idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            except Exception as page_err:
                logger.warning(
                    "PyPDF2: page %d extraction failed — %s", idx + 1, page_err
                )
        return "\n".join(pages)
    except Exception as exc:
        logger.warning("PyPDF2 failed entirely — %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_pdf(
    pdf_file: Union[str, Path, bytes, BinaryIO],
) -> str:
    """
    Extracts and returns cleaned text from a PDF.

    Args:
        pdf_file: One of —
            • A file-system path (``str`` or ``pathlib.Path``).
            • Raw ``bytes`` of a PDF document.
            • A readable file-like / binary-IO object (e.g. Streamlit's
              ``UploadedFile``).

    Returns:
        str — the extracted text with normalised whitespace.  Bullet points
        and section breaks are preserved so downstream NLP can still
        distinguish resume sections.

    Raises:
        PDFExtractionError: If neither pdfplumber nor PyPDF2 can extract
            any usable text from the document.
    """
    # 1. Normalise whatever the caller gave us into a seekable BytesIO
    stream = _normalise_input(pdf_file)

    # 2. Try pdfplumber (primary)
    text = _extract_with_pdfplumber(stream)
    if text.strip():
        logger.info("Successfully extracted text using pdfplumber.")
        return _clean_text(text)

    # 3. Fallback to PyPDF2
    logger.info("pdfplumber returned no text — falling back to PyPDF2.")
    text = _extract_with_pypdf2(stream)
    if text.strip():
        logger.info("Successfully extracted text using PyPDF2 fallback.")
        return _clean_text(text)

    # 4. Both engines failed — raise a clear, descriptive error
    raise PDFExtractionError(
        "Could not extract any text from the provided PDF. "
        "Both pdfplumber and PyPDF2 returned empty results. "
        "The file may be scanned/image-only, corrupt, or password-protected."
    )
