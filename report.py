#!/usr/bin/env python3
"""
Opticolumn OCR searchability report generator.

For every PDF in A/ (original), finds its identically-named counterpart in
B/ (OCR-processed), counts "true" searchable words in each using a
four-stage filter, and writes one CSV row per file pair — plus a trailing
TOTAL row summarizing the whole batch.
"""

import csv
import datetime
import logging
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
from spellchecker import SpellChecker

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_DIR  = "A"        # Original PDFs
OUTPUT_DIR = "B"        # OCR-processed PDFs
REPORT_DIR = "D"        # CSV output destination

TOOL_NAME    = "Opticolumn"
TOOL_VERSION = "2026"

# Minimum length for an *unrecognized* capitalized token to still be
# accepted as a plausible proper noun (Stage 4). Below this, short unknown
# capitalized fragments — "Th", "St", "Bn" — are far more likely to be OCR
# noise than real names, so they're now discarded instead of auto-counted.
MIN_PROPER_NOUN_LEN = 3

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── CSV fields ─────────────────────────────────────────────────────────────────
# One row per matched A/B file pair, plus a trailing TOTAL row for the batch.
CSV_FIELDS = [
    "tool_name",
    "tool_version",
    "report_datetime",
    "file_name",
    "word_count_A",
    "word_count_B",
    "percent_searchability",
]

# ── Compiled patterns ──────────────────────────────────────────────────────────

# Strip leading/trailing non-word characters
_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")

# Stage 2a — clean word token: letters only with optional apostrophe/hyphen.
# Explicitly excludes digits, ~, :, ;, |, \, !, etc.
_WORD_RE = re.compile(r"^[A-Za-z]+(?:['\u2019\-][A-Za-z]+)*$")

# Stage 2b — clean numeric/date/reference token.
#   Alt 1 (digits-first):  "1900", "15th", "22nd", "Aug-16" suffix case
#   Alt 2 (alpha-prefix):  "Aug-16", "No18", "l5th" (OCR l→1 substitution)
# Alphabetic portion capped at 4-char prefix and 2-char suffix to prevent
# long garbled strings like "3ecrP-tC.ry" from matching via the digit rule.
_NUM_RE = re.compile(
    r"^\d[\d\-./]*(?:[A-Za-z]{1,2})?$"                      # digits-first
    r"|^[A-Za-z]{1,4}[-.]?\d[\d\-./]*(?:[A-Za-z]{1,2})?$"  # alpha-prefix
)

# Spell checker — loaded once (dictionary load is expensive)
_spell = SpellChecker()


# ── Token helpers ──────────────────────────────────────────────────────────────

def _clean_token(token: str) -> str:
    """Strip leading/trailing punctuation, preserving internal structure."""
    return _EDGE_PUNCT.sub("", token)


def _has_normal_case(token: str) -> bool:
    """
    Return True if the token's letter-case pattern is 'normal':
      • all-lowercase  ("meeting", "arrested")
      • ALL-UPPERCASE  ("HILL", "SULLIVAN")
      • Title-case     ("Idaho", "Wardner", "Seattle")

    Tokens with erratic internal capitals are rejected as OCR noise:
      "soMe"  "IJetters"  "reGardin"  "caref'Ul"
    """
    alpha = [c for c in token if c.isalpha()]
    if not alpha:
        return True  # no alphabetic chars to evaluate
    all_lower  = all(c.islower() for c in alpha)
    all_upper  = all(c.isupper() for c in alpha)
    title_case = alpha[0].isupper() and all(c.islower() for c in alpha[1:])
    return all_lower or all_upper or title_case


def is_searchable(raw_token: str) -> bool:
    """
    Return True if the token should be counted as a keyword-searchable word.
    See module docstring for the full four-stage filter description.
    """
    # Stage 1 — edge cleaning and minimum length
    token = _clean_token(raw_token)
    if len(token) <= 1:
        return False

    is_word = bool(_WORD_RE.match(token))
    is_num  = bool(_NUM_RE.match(token))

    # Stage 2 — structural pre-filter
    if not is_word and not is_num:
        return False

    # Numeric tokens pass immediately — no further checks needed
    if is_num:
        return True

    # Stage 3 — case normality (word tokens only)
    if not _has_normal_case(token):
        return False

    # Stage 4 — dictionary / proper-noun rules
    if token.lower() in _spell:
        return True          # known English word

    if token[0].isupper() and len(token) >= MIN_PROPER_NOUN_LEN:
        return True          # Title-case or ALL-CAPS, long enough → plausible proper noun

    return False             # unknown all-lowercase, or too short to trust → discard


# ── PDF word count ─────────────────────────────────────────────────────────────

def count_words(pdf_path: Path) -> int:
    """
    Extract all text from a PDF's existing text layer, apply the four-stage
    searchability filter, and return the count of valid words.

    Works for born-digital PDFs, PDFs with an invisible OCR text layer
    (render_mode=3 / searchable PDF), and files with NO text layer at all —
    the latter simply return 0 rather than erroring or being skipped, which
    is what makes an un-OCR'd file in A/ still comparable to its OCR'd
    counterpart in B/ (see main()).
    """
    try:
        with fitz.open(str(pdf_path)) as doc:
            full_text = " ".join(page.get_text() for page in doc)
    except Exception as e:
        logger.error(f"Could not read text layer from '{pdf_path.name}': {e}")
        return 0

    return sum(1 for tok in full_text.split() if is_searchable(tok))


def percent_searchability(words_a: int, words_b: int) -> float:
    """
    Percent change in searchable word count from A to B, used identically
    for both individual file rows and the batch TOTAL row.
    """
    if words_a > 0:
        return ((words_b - words_a) / words_a) * 100
    # A had no searchable text at all (e.g. image-only, never OCR'd) —
    # B's entire count is new coverage, not a "percent increase" of zero.
    return 100.0 if words_b > 0 else 0.0


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    a_folder = Path(INPUT_DIR)
    b_folder = Path(OUTPUT_DIR)
    d_folder = Path(REPORT_DIR)

    # ── Sanity checks ──────────────────────────────────────────────────────────
    for folder in (a_folder, b_folder):
        if not folder.exists():
            logger.error(f"Required folder '{folder}' not found.")
            sys.exit(1)

    d_folder.mkdir(parents=True, exist_ok=True)

    a_pdfs = sorted(a_folder.glob("*.pdf"))
    if not a_pdfs:
        logger.error(f"No PDF files found in '{INPUT_DIR}'.")
        sys.exit(1)

    # ── Compare each matched A/B pair, one CSV row per file ────────────────────
    now = datetime.datetime.now()
    report_datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")

    report_rows      = []
    total_words_a    = 0
    total_words_b    = 0
    files_processed  = 0

    for a_pdf in a_pdfs:
        # Matched purely by filename — this is what guarantees a file with
        # NO existing OCR/text layer in A/ still gets compared against its
        # OCR'd version in B/, rather than being treated as "no data."
        b_pdf = b_folder / a_pdf.name

        if not b_pdf.exists():
            logger.warning(
                f"No processed counterpart for '{a_pdf.name}' in B/ — skipping."
            )
            continue

        words_a = count_words(a_pdf)   # 0 if A/ has no text layer at all
        words_b = count_words(b_pdf)
        file_pct = percent_searchability(words_a, words_b)

        logger.info(
            f"{a_pdf.name}: A={words_a:,} words  B={words_b:,} words  ({file_pct:+.2f}%)"
        )

        report_rows.append({
            "tool_name":             TOOL_NAME,
            "tool_version":          TOOL_VERSION,
            "report_datetime":       report_datetime_str,
            "file_name":             a_pdf.name,
            "word_count_A":          words_a,
            "word_count_B":          words_b,
            "percent_searchability": f"{file_pct:.2f}%",
        })

        total_words_a += words_a
        total_words_b += words_b
        files_processed += 1

    if files_processed == 0:
        logger.error(
            "No matched PDF pairs found between A/ and B/. "
            "Ensure B/ contains files with the same names as A/."
        )
        sys.exit(1)

    # ── Batch TOTAL row ─────────────────────────────────────────────────────────
    pct = percent_searchability(total_words_a, total_words_b)

    report_rows.append({
        "tool_name":             TOOL_NAME,
        "tool_version":          TOOL_VERSION,
        "report_datetime":       report_datetime_str,
        "file_name":             f"TOTAL ({files_processed} files)",
        "word_count_A":          total_words_a,
        "word_count_B":          total_words_b,
        "percent_searchability": f"{pct:.2f}%",
    })

    # ── Write CSV ──────────────────────────────────────────────────────────────
    csv_path = d_folder / f"opticolumn_report_{now.strftime('%Y%m%d_%H%M%S')}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)

    # ── Final summary ──────────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Report written       → {csv_path}")
    logger.info(f"Files processed      : {files_processed}")
    logger.info(f"Total words  A       : {total_words_a:,}")
    logger.info(f"Total words  B       : {total_words_b:,}")
    logger.info(f"Searchability change : {pct:.2f}%")


if __name__ == "__main__":
    main()