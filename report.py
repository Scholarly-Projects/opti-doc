#!/usr/bin/env python3
"""
report.py — Opticolumn OCR Quality Report Generator
=====================================================
Compares original PDFs in A/ against OCR-processed PDFs in B/ and writes
a timestamped CSV summary to D/.

Per-file metrics (plus a combined totals row):
  • tool_name / tool_version
  • processed_datetime          — from B PDF metadata (or file mtime)
  • word_count_original_layer   — words already in A's text layer (0 for raw scans)
  • word_count_ocr_layer        — words in B's invisible OCR text layer
  • trocr_word_count_original   — TrOCR word count running on A's rendered pages
  • trocr_word_count_ocr        — TrOCR word count running on B's rendered pages
  • avg_confidence_original     — mean per-token TrOCR confidence for A
  • avg_confidence_ocr          — mean per-token TrOCR confidence for B

Note on A vs B confidence:
  Because B's invisible OCR text layer (render_mode=3) is not visible when
  the page is rasterised, TrOCR sees the same image in both A and B.
  The confidence figures therefore reflect OCR quality at the given DPI and
  preprocessing settings rather than differences in visual content between
  the two folders.  The key comparative metric is usually
  trocr_word_count vs word_count_ocr_layer (OCR coverage / recall).
"""

import sys
import csv
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import torch
from PIL import Image, ImageFilter, ImageOps
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ── Configuration (mirrors main script) ───────────────────────────────────────
INPUT_DIR  = "A"       # Original PDFs
OUTPUT_DIR = "B"     
REPORT_DIR = "D"       # CSV output destination
MODELS_DIR = "mlmodels"

DPI = 200

TROCR_MODELS = {
    "handwritten":       "microsoft/trocr-base-handwritten",
    "printed":           "microsoft/trocr-base-printed",
    "large_handwritten": "microsoft/trocr-large-handwritten",
    "large_printed":     "microsoft/trocr-large-printed",
}
TROCR_MODEL_NAME = TROCR_MODELS["large_handwritten"]

CONFIDENCE_THRESHOLD              = 0.25
SINGLE_CHAR_CONFIDENCE_THRESHOLD  = 0.50
MIN_SEGMENT_HEIGHT                = 10
ENABLE_PREPROCESSING              = True

TOOL_NAME    = "Opticolumn"
TOOL_VERSION = "2026"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ── Model loading ──────────────────────────────────────────────────────────────
def load_models() -> Tuple[TorchVGSLModel, TrOCRProcessor, VisionEncoderDecoderModel]:
    seg_model_path = Path(MODELS_DIR) / "blla.mlmodel"
    logger.info(f"Loading segmentation model: {seg_model_path}")
    seg_model = TorchVGSLModel.load_model(str(seg_model_path))
    seg_model.eval()

    logger.info(f"Loading TrOCR model: {TROCR_MODEL_NAME}")
    processor   = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trocr_model.to(device)
    logger.info(f"Using device: {device}")

    return seg_model, processor, trocr_model


# ── Image helpers ──────────────────────────────────────────────────────────────
def page_to_pil(page: fitz.Page, dpi: int = DPI) -> Image.Image:
    """Rasterise a PDF page to an RGB PIL Image."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    """Return a preprocessed copy suitable for Kraken + TrOCR."""
    if not ENABLE_PREPROCESSING:
        return pil_image.copy()
    gray = pil_image.convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=2)
    proc = gray.convert("RGB")
    proc = proc.filter(ImageFilter.SHARPEN)
    return proc


# ── PDF text-layer extraction (no model required) ─────────────────────────────
def extract_text_layer_words(pdf_path: Path) -> int:
    """
    Extract and count words from the PDF's existing text layer using PyMuPDF.
    Returns 0 for raw scans that have no text layer.
    """
    try:
        with fitz.open(str(pdf_path)) as doc:
            full_text = " ".join(page.get_text() for page in doc)
        return len(full_text.split())
    except Exception as e:
        logger.error(f"Text-layer extraction failed for {pdf_path.name}: {e}")
        return 0


# ── TrOCR full-document scan ───────────────────────────────────────────────────
def run_trocr_on_pdf(
    pdf_path: Path,
    seg_model: TorchVGSLModel,
    processor: TrOCRProcessor,
    trocr_model: VisionEncoderDecoderModel,
) -> Tuple[int, float]:
    """
    Render every page, segment with Kraken blla, recognise each line with
    TrOCR, and return (total_word_count, average_per_token_confidence).

    Only lines whose confidence meets CONFIDENCE_THRESHOLD are counted —
    matching the filter used in the main OCR pipeline.
    """
    total_words: int      = 0
    confidences: List[float] = []
    device = next(trocr_model.parameters()).device

    try:
        with fitz.open(str(pdf_path)) as doc:
            n_pages = len(doc)
            for page_num, page in enumerate(doc):
                logger.info(
                    f"  [{pdf_path.name}] TrOCR page {page_num + 1}/{n_pages}"
                )
                pil_img = page_to_pil(page, dpi=DPI)
                ocr_img = preprocess_for_ocr(pil_img)

                # ── Segmentation ──────────────────────────────────────────────
                try:
                    seg = blla.segment(ocr_img, model=seg_model)
                except Exception as e:
                    logger.error(
                        f"Segmentation error on {pdf_path.name} "
                        f"page {page_num + 1}: {e}"
                    )
                    continue

                lines = getattr(seg, "lines", [])
                logger.info(f"    {len(lines)} lines detected.")

                for line in lines:
                    try:
                        # ── Bounding box ──────────────────────────────────────
                        if hasattr(line, "boundary") and len(line.boundary) >= 3:
                            xs = [p[0] for p in line.boundary]
                            ys = [p[1] for p in line.boundary]
                            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                        elif hasattr(line, "bbox"):
                            x0, y0, x1, y1 = line.bbox
                        else:
                            continue

                        seg_h, seg_w = y1 - y0, x1 - x0
                        if seg_h < MIN_SEGMENT_HEIGHT or seg_w < 5:
                            continue

                        crop = ocr_img.crop((x0, y0, x1, y1))

                        # ── TrOCR inference ───────────────────────────────────
                        pixel_values = processor(
                            crop, return_tensors="pt"
                        ).pixel_values.to(device)

                        with torch.no_grad():
                            out = trocr_model.generate(
                                pixel_values,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )

                        line_text = processor.batch_decode(
                            out.sequences, skip_special_tokens=True
                        )[0].strip()

                        # ── Per-token confidence ──────────────────────────────
                        if out.scores:
                            probs     = [torch.softmax(s, dim=-1) for s in out.scores]
                            max_probs = [torch.max(p).item() for p in probs]
                            conf      = sum(max_probs) / len(max_probs)
                        else:
                            conf = 0.0

                        # Apply the same threshold as the main pipeline
                        if line_text and conf >= CONFIDENCE_THRESHOLD:
                            total_words += len(line_text.split())
                            confidences.append(conf)

                    except Exception as e:
                        logger.error(f"Line processing error: {e}")

    except Exception as e:
        logger.error(f"Failed to open/process {pdf_path.name}: {e}")

    avg_conf = (
        sum(confidences) / len(confidences) if confidences else 0.0
    )
    logger.info(
        f"  [{pdf_path.name}] TrOCR complete — "
        f"{total_words} words, avg confidence {avg_conf:.4f}"
    )
    return total_words, avg_conf


# ── PDF metadata helpers ───────────────────────────────────────────────────────
def _parse_pdf_date(raw: str) -> Optional[datetime.datetime]:
    """Parse a PDF date string (D:YYYYMMDDHHmmSS...) into a datetime."""
    try:
        s = raw.lstrip("D:").strip()
        return datetime.datetime(
            int(s[0:4]),  int(s[4:6]),  int(s[6:8]),
            int(s[8:10]), int(s[10:12]), int(s[12:14]),
        )
    except Exception:
        return None


def get_processed_datetime(pdf_path: Path) -> str:
    """
    Return the creation timestamp of the processed PDF as a human-readable
    string.  Falls back to file modification time if metadata is absent.
    """
    try:
        with fitz.open(str(pdf_path)) as doc:
            cd = doc.metadata.get("creationDate", "")
            if cd:
                dt = _parse_pdf_date(cd)
                if dt:
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    try:
        mtime = pdf_path.stat().st_mtime
        return datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "unknown"


# ── Report generation ──────────────────────────────────────────────────────────
CSV_FIELDS = [
    "filename",
    "tool_name",
    "tool_version",
    "processed_datetime",
    "word_count_original_layer",   # PyMuPDF text extraction — A folder
    "word_count_ocr_layer",        # PyMuPDF text extraction — B folder
    "trocr_word_count_original",   # TrOCR recognised words  — A folder
    "trocr_word_count_ocr",        # TrOCR recognised words  — B folder
    "avg_confidence_original",     # mean per-token confidence — A folder
    "avg_confidence_ocr",          # mean per-token confidence — B folder
]


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

    # ── Load models once ───────────────────────────────────────────────────────
    logger.info("Loading models…")
    try:
        seg_model, processor, trocr_model = load_models()
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        sys.exit(1)

    # ── Per-file processing ────────────────────────────────────────────────────
    rows:  List[Dict] = []
    totals = {
        "word_count_original_layer": 0,
        "word_count_ocr_layer":      0,
        "trocr_word_count_original": 0,
        "trocr_word_count_ocr":      0,
        "conf_original_vals":        [],   # collect per-file averages for macro avg
        "conf_ocr_vals":             [],
    }

    for a_pdf in a_pdfs:
        stem  = a_pdf.stem
        b_pdf = b_folder / f"{stem}.pdf"

        if not b_pdf.exists():
            logger.warning(
                f"No processed counterpart found for '{a_pdf.name}' "
                f"(expected '{b_pdf.name}') — skipping."
            )
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Pair: {a_pdf.name}  ↔  {b_pdf.name}")

        # ── 1. Cheap text-layer word counts (no model) ─────────────────────
        words_a_layer = extract_text_layer_words(a_pdf)
        words_b_layer = extract_text_layer_words(b_pdf)
        logger.info(
            f"Text-layer words — A (original): {words_a_layer}  "
            f"B (OCR layer): {words_b_layer}"
        )

        # ── 2. TrOCR word counts + per-token confidence ────────────────────
        logger.info("Running TrOCR on A (original)…")
        trocr_words_a, conf_a = run_trocr_on_pdf(
            a_pdf, seg_model, processor, trocr_model
        )

        logger.info("Running TrOCR on B (OCR-processed)…")
        trocr_words_b, conf_b = run_trocr_on_pdf(
            b_pdf, seg_model, processor, trocr_model
        )

        # ── 3. Metadata timestamp ──────────────────────────────────────────
        proc_dt = get_processed_datetime(b_pdf)

        # ── 4. Accumulate totals ───────────────────────────────────────────
        totals["word_count_original_layer"] += words_a_layer
        totals["word_count_ocr_layer"]      += words_b_layer
        totals["trocr_word_count_original"] += trocr_words_a
        totals["trocr_word_count_ocr"]      += trocr_words_b
        if conf_a > 0:
            totals["conf_original_vals"].append(conf_a)
        if conf_b > 0:
            totals["conf_ocr_vals"].append(conf_b)

        rows.append({
            "filename":                  a_pdf.name,
            "tool_name":                 TOOL_NAME,
            "tool_version":              TOOL_VERSION,
            "processed_datetime":        proc_dt,
            "word_count_original_layer": words_a_layer,
            "word_count_ocr_layer":      words_b_layer,
            "trocr_word_count_original": trocr_words_a,
            "trocr_word_count_ocr":      trocr_words_b,
            "avg_confidence_original":   f"{conf_a:.4f}",
            "avg_confidence_ocr":        f"{conf_b:.4f}",
        })

        logger.info(
            f"File summary — "
            f"layer words A/B: {words_a_layer}/{words_b_layer} | "
            f"TrOCR words A/B: {trocr_words_a}/{trocr_words_b} | "
            f"confidence A/B: {conf_a:.4f}/{conf_b:.4f}"
        )

    if not rows:
        logger.error(
            "No matching file pairs found between A/ and B/. "
            "Ensure B/ contains files named {stem}.pdf."
        )
        sys.exit(1)

    # ── Combined totals row ────────────────────────────────────────────────────
    macro_conf_a = (
        sum(totals["conf_original_vals"]) / len(totals["conf_original_vals"])
        if totals["conf_original_vals"] else 0.0
    )
    macro_conf_b = (
        sum(totals["conf_ocr_vals"]) / len(totals["conf_ocr_vals"])
        if totals["conf_ocr_vals"] else 0.0
    )
    rows.append({
        "filename":                  "COMBINED TOTAL",
        "tool_name":                 TOOL_NAME,
        "tool_version":              TOOL_VERSION,
        "processed_datetime":        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "word_count_original_layer": totals["word_count_original_layer"],
        "word_count_ocr_layer":      totals["word_count_ocr_layer"],
        "trocr_word_count_original": totals["trocr_word_count_original"],
        "trocr_word_count_ocr":      totals["trocr_word_count_ocr"],
        "avg_confidence_original":   f"{macro_conf_a:.4f}",
        "avg_confidence_ocr":        f"{macro_conf_b:.4f}",
    })

    # ── Write CSV ──────────────────────────────────────────────────────────────
    report_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = d_folder / f"opticolumn_report_{report_ts}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # ── Final summary ──────────────────────────────────────────────────────────
    n_files = len(rows) - 1   # exclude the totals row
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Report written → {csv_path}")
    logger.info(f"Files processed          : {n_files}")
    logger.info(f"Combined OCR words (B)   : {totals['word_count_ocr_layer']}")
    logger.info(f"Combined TrOCR words (A) : {totals['trocr_word_count_original']}")
    logger.info(f"Combined TrOCR words (B) : {totals['trocr_word_count_ocr']}")
    logger.info(
        f"Macro avg confidence A/B : {macro_conf_a:.4f} / {macro_conf_b:.4f}"
    )


if __name__ == "__main__":
    main()