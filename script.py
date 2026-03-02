#!/usr/bin/env python3
import sys
import os
import tempfile
from pathlib import Path
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from kraken import blla
from kraken.lib.vgsl import TorchVGSLModel
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import logging
from typing import List, Tuple
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
import platform
import datetime
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
import math
import pikepdf

# ---------------- Configuration ----------------
INPUT_DIR = "A"
OUTPUT_DIR = "B"
MODELS_DIR = "mlmodels"
POPPLER_PATH = None
DPI = 200 
TROCR_MODELS = {
    "handwritten": "microsoft/trocr-base-handwritten",
    "printed": "microsoft/trocr-base-printed",
    "large_handwritten": "microsoft/trocr-large-handwritten",
    "large_printed": "microsoft/trocr-large-printed"
}
TROCR_MODEL_NAME = TROCR_MODELS["large_handwritten"]
ENABLE_PREPROCESSING = True
CONFIDENCE_THRESHOLD = 0.25
SINGLE_CHAR_CONFIDENCE_THRESHOLD = 0.5
MIN_SEGMENT_HEIGHT = 10
FONT_NAME = "FreeSans"
FONT_PATH = "fonts/FreeSans.ttf"
SRGB_ICC_PATH = "srgb.icc"
DEBUG_OCR_LAYER = False
DEBUG_TEXT_POSITIONS = False
DEBUG_SAVE_INTERMEDIATE = False
DEBUG_PDFA = False
COMPRESSION_LEVEL = 88  
AGGRESSIVE_COMPRESSION = False  

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Helper function to format date for PDF
def get_pdf_date_string(dt=None):
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("D:%Y%m%d%H%M%S")

# Helper function to format date for XMP
def get_xmp_date_string(dt=None):
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

# ---------------- Font and ICC Profile Setup ----------------
def setup_pdfa_resources():
    try:
        font_dir = Path("fonts")
        font_dir.mkdir(exist_ok=True)
        font_path = Path(FONT_PATH)
        if not font_path.exists():
            logger.info("Downloading FreeSans font for embedding...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://github.com/opensourcedesign/fonts/raw/master/gnu-freefont_freesans/FreeSans.ttf",
                str(font_path)
            )
        srgb_path = Path(SRGB_ICC_PATH)
        if not srgb_path.exists():
            logger.info("Downloading sRGB ICC profile...")
            try:
                if platform.system() == "Darwin":
                    system_profile = "/System/Library/ColorSync/Profiles/sRGB Profile.icc"
                elif platform.system() == "Windows":
                    system_profile = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'),
                                               'System32', 'spool', 'drivers', 'color', 'sRGB Color Space Profile.icm')
                elif platform.system() == "Linux":
                    system_profile = "/usr/share/color/icc/sRGB.icc"
                else:
                    system_profile = None
                if system_profile and Path(system_profile).exists():
                    shutil.copy2(system_profile, str(srgb_path))
                else:
                    urllib.request.urlretrieve(
                        "https://www.color.org/srgb.xalter",
                        str(srgb_path)
                    )
            except Exception as e:
                logger.warning(f"Could not get sRGB ICC profile: {e}")
        return True
    except Exception as e:
        logger.error(f"Failed to setup PDF/A resources: {e}")
        return False

# ---------------- XMP Metadata Creation ----------------
def create_xmp_metadata(title, author, subject, creator, producer, creation_date, modify_date):
    try:
        xmp_packet = f"""<?xpacket begin="ï»¿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 5.6-c140 79.164452, 2017/09/07-01:11:22        ">
   <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description rdf:about="" xmlns:pdf="http://ns.adobe.com/pdf/1.3/">
         <pdf:Producer>{producer}</pdf:Producer>
      </rdf:Description>
      <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">
         <dc:title><rdf:Alt><rdf:li xml:lang="x-default">{title}</rdf:li></rdf:Alt></dc:title>
         <dc:creator><rdf:Seq><rdf:li>{author}</rdf:li></rdf:Seq></dc:creator>
         <dc:description><rdf:Alt><rdf:li xml:lang="x-default">{subject}</rdf:li></rdf:Alt></dc:description>
         <dc:language><rdf:Bag><rdf:li>en-US</rdf:li></rdf:Bag></dc:language>
      </rdf:Description>
      <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
         <xmp:CreatorTool>{creator}</xmp:CreatorTool>
         <xmp:CreateDate>{creation_date}</xmp:CreateDate>
         <xmp:ModifyDate>{modify_date}</xmp:ModifyDate>
         <xmp:Language>en-US</xmp:Language>
      </rdf:Description>
      <rdf:Description rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">
         <pdfaid:part>1</pdfaid:part>
         <pdfaid:conformance>B</pdfaid:conformance>
      </rdf:Description>
      <!-- Custom metadata for Opticolumn -->
      <rdf:Description rdf:about="" xmlns:opt="http://github.com/Scholarly-Projects/opticolumn/">
         <opt:ToolName>Opticolumn</opt:ToolName>
         <opt:Version>2026</opt:Version>
      </rdf:Description>
   </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
        return xmp_packet
    except Exception as e:
        logger.error(f"Failed to create XMP metadata: {e}")
        return None

# ---------------- Model Loading ----------------
def load_models():
    try:
        if not setup_pdfa_resources():
            logger.warning("PDF/A resources setup failed. PDF/A compliance may be affected.")
        seg_model_path = Path(MODELS_DIR) / "blla.mlmodel"
        logger.info(f"Loading segmentation model: {seg_model_path}")
        seg_model = TorchVGSLModel.load_model(str(seg_model_path))
        seg_model.eval()
        logger.info(f"Loading TrOCR model: {TROCR_MODEL_NAME}")
        processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trocr_model.to(device)
        logger.info(f"Using device: {device}")
        return seg_model, processor, trocr_model
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

try:
    seg_model, processor, trocr_model = load_models()
except Exception as e:
    logger.error("Model loading failed. Exiting.")
    sys.exit(1)

# ---------------- Image Preprocessing ----------------
def preprocess_image(pil_image: Image.Image) -> Image.Image:
    if not ENABLE_PREPROCESSING:
        return pil_image
    try:
        # Convert to grayscale for better text detection
        gray_image = pil_image.convert('L')
        # Apply mild contrast enhancement
        from PIL import ImageOps
        gray_image = ImageOps.autocontrast(gray_image, cutoff=2)
        # Convert back to RGB for consistency
        processed_image = gray_image.convert('RGB')
        # Apply mild sharpening to enhance text edges
        processed_image = processed_image.filter(ImageFilter.SHARPEN)
        return processed_image
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return pil_image

# ---------------- PDF Utilities ----------------
def flatten_pdf_to_images(input_path: str, temp_pdf_path: str) -> bool:
    """
    Create a flattened, image-only PDF for OCR processing.

    Each page is rendered via get_pixmap(), which correctly resolves all
    /Rotate entries and EXIF orientation before returning pixels. The
    rendered bitmap is saved as lossless PNG instead of the previous JPEG,
    avoiding quality degradation on already-raster pages.

    IMPORTANT — page dimensions:
    PyMuPDF's new_page(width, height) takes values in PDF *points*, not
    pixels.  We pass the original page point dimensions so that the
    flattened PDF retains standard page geometry.  This means the pixel
    coordinates returned by the segmentation model must later be scaled
    to point space before being written back (see process_single_pdf_ocr).
    """
    try:
        logger.debug(f"Flattening PDF: {input_path}")
        with fitz.open(input_path) as doc, fitz.open() as output_pdf:
            for page_num, page in enumerate(doc):
                # Render at target DPI — get_pixmap handles /Rotate and EXIF.
                pix = page.get_pixmap(dpi=DPI)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_buffer = BytesIO()
                img.save(img_buffer, format="PNG", compress_level=6)  # lossless
                img_buffer.seek(0)
                # Use the ORIGINAL page point dimensions, not pixel dimensions.
                # This keeps the page the same size as the source document and
                # means convert_from_path at DPI will produce images at exactly
                # (page_points * DPI/72) pixels — a known, predictable ratio.
                img_page = output_pdf.new_page(
                    width=page.rect.width,
                    height=page.rect.height,
                )
                img_page.insert_image(img_page.rect, stream=img_buffer.read())
            output_pdf.save(temp_pdf_path, deflate=True, garbage=3, clean=True)
        return True
    except Exception as e:
        logger.error(f"Error flattening PDF: {e}")
        return False

# ---------------- Text Recognition with TrOCR ----------------
def recognize_text_with_trocr(image: Image.Image, processor, model) -> tuple[str, float]:
    try:
        pixel_values = processor(image, return_tensors="pt").pixel_values
        device = next(model.parameters()).device
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
            generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
            scores = generated_ids.scores
            if scores:
                probs = [torch.softmax(score, dim=-1) for score in scores]
                max_probs = [torch.max(prob).item() for prob in probs]
                confidence = sum(max_probs) / len(max_probs)
            else:
                confidence = 0.0
        return generated_text.strip(), confidence
    except Exception as e:
        logger.error(f"Error recognizing text with TrOCR: {e}")
        return "", 0.0

# ---------------- Noise Detection ----------------
def is_likely_noise(text: str, confidence: float, segment_height: int, segment_width: int) -> bool:
    if not text:
        return True
    if segment_height < MIN_SEGMENT_HEIGHT:
        return True
    if segment_width < 15:
        return True
    aspect_ratio = segment_width / segment_height
    if aspect_ratio < 0.1 or aspect_ratio > 100:
        return True
    text_clean = text.strip()
    text_length = len(text_clean)
    if text_length == 1:
        if confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD:
            return True
        return False
    if confidence < CONFIDENCE_THRESHOLD:
        return True
    if len(set(text_clean)) == 1 and text_length > 2:
        return True
    noise_patterns = [
        r'^[oOlI\.\|]+$',
        r'^[0-9\.\,]+$',
        r'^[^a-zA-Z0-9\s]+$',
    ]
    for pattern in noise_patterns:
        if re.match(pattern, text_clean):
            if confidence < SINGLE_CHAR_CONFIDENCE_THRESHOLD:
                return True
    if text_length > 3 and not any(char.lower() in 'aeiou' for char in text_clean):
        if confidence < 0.7:
            return True
    return False

# ---------------- Improved Column Detection and Sorting ----------------
def improved_column_sort(lines: List) -> List:
    """
    Improved column detection and sorting algorithm that:
    1. More accurately detects columns using a combination of horizontal projection and clustering
    2. Sorts lines within columns top-to-bottom
    3. Sorts columns left-to-right
    4. Handles irregular column layouts better
    """
    if len(lines) <= 1:
        return lines
    
    bboxes = []
    for line in lines:
        if hasattr(line, 'boundary') and len(line.boundary) >= 3:
            x_coords = [p[0] for p in line.boundary]
            y_coords = [p[1] for p in line.boundary]
            x0, y0 = min(x_coords), min(y_coords)
            x1, y1 = max(x_coords), max(y_coords)
        elif hasattr(line, 'bbox'):
            x0, y0, x1, y1 = line.bbox
        else:
            continue
        bboxes.append((x0, y0, x1, y1, line))
    
    if not bboxes:
        return lines
    
    page_width = max(box[2] for box in bboxes)
    page_height = max(box[3] for box in bboxes)
    
    resolution = 10
    hist_width = int(page_width / resolution) + 1
    hist = [0] * hist_width
    
    for x0, y0, x1, y1, _ in bboxes:
        start_bin = int(x0 / resolution)
        end_bin = int(x1 / resolution)
        for bin_idx in range(start_bin, min(end_bin + 1, hist_width)):
            hist[bin_idx] += (y1 - y0)
    
    smoothed_hist = hist.copy()
    for i in range(1, len(hist) - 1):
        smoothed_hist[i] = (hist[i-1] + 2*hist[i] + hist[i+1]) / 4
    
    valleys = []
    for i in range(1, len(smoothed_hist) - 1):
        if smoothed_hist[i] < smoothed_hist[i-1] and smoothed_hist[i] < smoothed_hist[i+1]:
            neighborhood_max = max(smoothed_hist[i-1], smoothed_hist[i+1])
            if neighborhood_max > 0 and smoothed_hist[i] / neighborhood_max < 0.3:
                valleys.append(i * resolution)
    
    if not valleys:
        widths = [x1 - x0 for x0, y0, x1, y1, _ in bboxes]
        avg_width = sum(widths) / len(widths) if widths else 100
        estimated_col_count = max(1, int(page_width / (avg_width * 1.5)))
        estimated_col_count = min(estimated_col_count, 5)
        col_width = page_width / estimated_col_count
        valleys = [int((i + 1) * col_width) for i in range(estimated_col_count - 1)]
    
    valleys = sorted([v for v in valleys if 0 < v < page_width])
    columns = [[] for _ in range(len(valleys) + 1)]
    
    for box in bboxes:
        x0, y0, x1, y1, line = box
        center_x = (x0 + x1) / 2
        col_idx = 0
        for valley in valleys:
            if center_x > valley:
                col_idx += 1
            else:
                break
        columns[col_idx].append(box)
    
    sorted_lines = []
    for column in columns:
        for bbox in sorted(column, key=lambda box: box[1]):
            sorted_lines.append(bbox[4])
    
    return sorted_lines

# ---------------- OCR Text Element Extraction ----------------
def create_ocr_text_elements(images: List[Image.Image], filename: str) -> List[List[dict]]:
    """
    Run segmentation and TrOCR on each page image.

    Returns a list of pages; each page is a list of dicts with keys:
        x0, y_baseline, text, font_size
    All coordinates and font sizes are in IMAGE PIXEL SPACE.

    The caller is responsible for scaling these values into PDF point space
    using the ratio (pdf_page_points / image_pixels) before calling
    page.insert_text().  See process_single_pdf_ocr for that scaling step.
    """
    font_path = Path(FONT_PATH)
    if not font_path.exists():
        logger.error(f"CRITICAL: Font file {font_path} not found.")
        raise FileNotFoundError(f"Required font {font_path} is missing.")

    all_pages: List[List[dict]] = []
    total_text_elements = 0

    for img_idx, pil_image in enumerate(images):
        page_num = img_idx + 1
        logger.info(f"Processing page {page_num}/{len(images)} of {filename}")
        pdf_width, pdf_height = pil_image.size
        page_elements: List[dict] = []

        try:
            processed_image = preprocess_image(pil_image)
            segmentation = blla.segment(processed_image, model=seg_model)
            logger.info(f"Found {len(segmentation.lines)} text lines on page {page_num}")

            if len(segmentation.lines) == 0:
                logger.warning("No text lines detected. Saving debug image...")
                debug_dir = Path("debug_images")
                debug_dir.mkdir(exist_ok=True)
                processed_image.save(debug_dir / f"{filename}_page{page_num}_preprocessed.png")

            sorted_lines = improved_column_sort(segmentation.lines)
            logger.info(f"Sorted {len(sorted_lines)} lines into column-based reading order.")
            filtered_lines = 0

            for i, line in enumerate(sorted_lines):
                try:
                    if hasattr(line, 'boundary') and len(line.boundary) >= 3:
                        x_coords = [p[0] for p in line.boundary]
                        y_coords = [p[1] for p in line.boundary]
                        x0, y0 = min(x_coords), min(y_coords)
                        x1, y1 = max(x_coords), max(y_coords)
                    elif hasattr(line, 'bbox'):
                        x0, y0, x1, y1 = line.bbox
                    else:
                        continue

                    segment_height = y1 - y0
                    segment_width  = x1 - x0
                    if segment_height < 5 or segment_width < 5:
                        filtered_lines += 1
                        continue

                    line_image = processed_image.crop((x0, y0, x1, y1))
                    text, confidence = recognize_text_with_trocr(line_image, processor, trocr_model)

                    if is_likely_noise(text, confidence, segment_height, segment_width):
                        filtered_lines += 1
                        continue

                    # All values stored in image pixel space.
                    # font_size is derived from segment_height (pixels); it will
                    # be multiplied by the same y-scale factor as y_baseline so
                    # the rendered text height stays proportional to the line box.
                    page_elements.append({
                        "x0":         x0,
                        "y_baseline": y1,          # bottom of bounding box = baseline
                        "font_size":  max(6, min(segment_height * 0.9, 72)),
                        "text":       text,
                    })

                except Exception as e:
                    logger.error(f"Error processing text line {i+1}: {e}")
                    continue

            logger.info(f"Page {page_num}: {len(page_elements)} elements, {filtered_lines} filtered.")
            total_text_elements += len(page_elements)

        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")

        all_pages.append(page_elements)

    logger.info(f"OCR extraction complete: {total_text_elements} total text elements across {len(images)} pages")
    return all_pages


# ---------------- PDF/A Compliance Setup ----------------
def setup_pdfa_compliance(pdf_path: str):
    try:
        # Load the ICC profile path
        srgb_path = Path(SRGB_ICC_PATH)
        
        # Use pikepdf to add OutputIntent and embed ICC profile
        with pikepdf.open(pdf_path) as pdf:
            if "OutputIntents" not in pdf.Root:
                pdf.Root["OutputIntents"] = pikepdf.Array()
            
            # Create an OutputIntent dictionary
            output_intent_dict = {
                "/Type": pikepdf.Name.OutputIntent,
                "/S": pikepdf.Name.GTS_PDFX,
                "/Info": "sRGB",
                "/DestOutputProfile": pdf.make_indirect_reference(pdf.add_file(srgb_path))
            }
            
            # Append the OutputIntent to the Root dictionary
            pdf.Root["OutputIntents"].append(pikepdf.Dictionary(output_intent_dict))
            
            # Save the changes
            pdf.save(pdf_path)
        logger.info("PDF/A compliance setup complete.")
    except Exception as e:
        logger.error(f"Failed to set up PDF/A compliance: {e}")

# ---------------- PDF Processing (OCR) ----------------
def process_single_pdf_ocr(input_path: str, output_path: str) -> bool:
    filename = os.path.basename(input_path)
    logger.info(f"Starting OCR for: {filename}")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf_path = temp_pdf.name

    if not flatten_pdf_to_images(input_path, temp_pdf_path):
        logger.error(f"Failed to flatten {filename}")
        return False

    try:
        original_max_pixels = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None

        pil_images = convert_from_path(temp_pdf_path, dpi=DPI, poppler_path=POPPLER_PATH)
        logger.info(f"Converted to {len(pil_images)} images @ {DPI} DPI")

        ocr_pages = create_ocr_text_elements(pil_images, filename)

        logger.info("Writing OCR text directly into base PDF pages...")
        font_path = str(Path(FONT_PATH))

        with fitz.open(temp_pdf_path) as base_pdf:
            creation_date = get_pdf_date_string()
            modify_date   = creation_date
            metadata = {
                "title":        filename,
                "author":       "Opticolumn",
                "subject":      "OCR processed document",
                "creator":      "Opticolumn 2026",
                "producer":     "PyMuPDF",
                "creationDate": creation_date,
                "modDate":      modify_date,
            }
            
            xmp_metadata = create_xmp_metadata(
                title=metadata["title"],
                author=metadata["author"],
                subject=metadata["subject"],
                creator=metadata["creator"],
                producer=metadata["producer"],
                creation_date=get_xmp_date_string(),
                modify_date=get_xmp_date_string(),
            )
            
            if xmp_metadata:
                base_pdf.set_xml_metadata(xmp_metadata)
            else:
                logger.warning("Failed to create XMP metadata")

            page_count = min(len(base_pdf), len(ocr_pages))
            logger.info(f"Inserting text into {page_count} pages...")

            for page_num in range(page_count):
                page     = base_pdf[page_num]
                elements = ocr_pages[page_num]
                pil_img  = pil_images[page_num]

                # ── Coordinate scaling ────────────────────────────────────────
                # The segmentation model operated on pil_img (pixel space).
                # The PDF page uses point space.  flatten_pdf_to_images created
                # each page with its original point dimensions, so:
                #
                #   pixel_coord * (page_points / image_pixels) = point_coord
                #
                # We compute one scale factor per axis and apply it to every
                # coordinate and to the font size.
                img_w, img_h = pil_img.size          # pixels
                page_w = page.rect.width              # points
                page_h = page.rect.height             # points
                sx = page_w / img_w                   # points per pixel, x-axis
                sy = page_h / img_h                   # points per pixel, y-axis

                logger.debug(
                    f"Page {page_num+1}: image {img_w}×{img_h}px → "
                    f"page {page_w:.1f}×{page_h:.1f}pt  (sx={sx:.4f}, sy={sy:.4f})"
                )

                inserted = 0
                for elem in elements:
                    try:
                        page.insert_text(
                            fitz.Point(elem["x0"] * sx, elem["y_baseline"] * sy),
                            elem["text"],
                            fontsize=max(4, elem["font_size"] * sy),
                            fontname=FONT_NAME,
                            fontfile=font_path,
                            render_mode=3,   # invisible text (PDF spec §9.3.6)
                            color=(0, 0, 0),
                        )
                        inserted += 1
                    except Exception as e:
                        logger.error(f"Failed to insert text element on page {page_num+1}: {e}")
                        continue

                logger.info(f"Page {page_num+1}: inserted {inserted}/{len(elements)} text elements")

            logger.info("Applying PDF/A compliance fixes...")
            srgb_path = Path(SRGB_ICC_PATH)
            if not srgb_path.exists():
                logger.error("CRITICAL: sRGB ICC profile not found. PDF/A-1B compliance is impossible.")
            else:
                setup_pdfa_compliance(output_path)

            base_pdf.save(
                output_path,
                deflate=True,
                garbage=4,
                clean=True,
                encryption=fitz.PDF_ENCRYPT_KEEP,
            )
            logger.info(f"OCR-enhanced PDF saved: {output_path}")

        logger.info("Verifying OCR layer in final output...")
        try:
            with fitz.open(output_path) as final_pdf:
                total_text_length = 0
                for i in range(len(final_pdf)):
                    page = final_pdf[i]
                    text = page.get_text()
                    page_text_length = len(text.strip())
                    total_text_length += page_text_length
                    logger.info(f"Final PDF page {i+1} extractable text length: {page_text_length}")
                if total_text_length > 0:
                    logger.info(f"SUCCESS: Final PDF contains {total_text_length} characters of searchable text")
                else:
                    logger.error("PROBLEM: Final PDF has no extractable text!")
        except Exception as e:
            logger.error(f"Failed to verify final output: {e}")

        Image.MAX_IMAGE_PIXELS = original_max_pixels
        return True

    except Exception as e:
        Image.MAX_IMAGE_PIXELS = original_max_pixels
        logger.error(f"OCR processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

# ---------------- Enhanced Compression (Size Targeting) ----------------
def enhanced_compress_to_target_size(input_pdf: Path, output_pdf: Path, original_size: int) -> Path:
    """
    Enhanced compression function that tries to get the processed file as close as possible 
    to the original file size, allowing up to a 15% increase for the added OCR layer.
    """
    max_target = int(original_size * 1.15)
    
    logger.info(f"Targeting maximum size: {max_target//1024} KB (15% increase from original)")
    
    current_size = input_pdf.stat().st_size
    logger.info(f"OCR file size before compression: {current_size//1024} KB")
    
    if current_size <= max_target:
        shutil.copy2(input_pdf, output_pdf)
        logger.info(f"OCR file already within target size. No compression needed.")
        return output_pdf
    
    compression_options = [
        {"deflate": True, "garbage": 4, "clean": True, "deflate_images": True, "pretty": False},
        {"deflate": True, "garbage": 3, "clean": True, "deflate_images": True, "pretty": False},
        {"deflate": True, "garbage": 2, "clean": True, "deflate_images": False, "pretty": False},
    ]
    
    for i, options in enumerate(compression_options):
        temp_output = output_pdf.with_suffix(f".temp_{i}.pdf")
        try:
            with fitz.open(str(input_pdf)) as doc:
                doc.save(str(temp_output), **options, encryption=fitz.PDF_ENCRYPT_KEEP)
            
            compressed_size = temp_output.stat().st_size
            size_increase_pct = (compressed_size - original_size) / original_size * 100
            logger.info(f"Compression option {i+1}: {compressed_size//1024} KB ({size_increase_pct:+.1f}% from original)")
            
            if compressed_size <= max_target:
                shutil.move(str(temp_output), str(output_pdf))
                logger.info(f"Found suitable compression with option {i+1}")
                try:
                    with fitz.open(str(output_pdf)) as final_pdf:
                        total_chars = sum(len(page.get_text().strip()) for page in final_pdf)
                        if total_chars > 0:
                            logger.info(f"OCR preserved: {total_chars} characters found.")
                            return output_pdf
                        else:
                            logger.error("OCR LOST after compression!")
                            shutil.copy2(input_pdf, output_pdf)
                            return output_pdf
                except Exception as e:
                    logger.warning(f"Could not verify OCR after compression: {e}")
                    shutil.copy2(input_pdf, output_pdf)
                    return output_pdf
            else:
                temp_output.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Compression option {i+1} failed: {e}")
            if temp_output.exists():
                temp_output.unlink(missing_ok=True)
    
    logger.info("Standard compression options insufficient. Trying image recompression...")
    
    try:
        temp_output = output_pdf.with_suffix(".temp_recompress.pdf")
        with fitz.open(str(input_pdf)) as doc:
            for page in doc:
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    pil_image = Image.open(BytesIO(base_image["image"]))
                    img_buffer = BytesIO()
                    pil_image.save(img_buffer, format="JPEG", quality=40, optimize=True, progressive=True)
                    doc.update_image(xref, img_buffer.getvalue())
            doc.save(str(temp_output), deflate=True, garbage=4, clean=True,
                     deflate_images=True, pretty=False, encryption=fitz.PDF_ENCRYPT_KEEP)
        
        compressed_size = temp_output.stat().st_size
        size_increase_pct = (compressed_size - original_size) / original_size * 100
        logger.info(f"Image recompression: {compressed_size//1024} KB ({size_increase_pct:+.1f}% from original)")
        shutil.move(str(temp_output), str(output_pdf))
        
        try:
            with fitz.open(str(output_pdf)) as final_pdf:
                total_chars = sum(len(page.get_text().strip()) for page in final_pdf)
                if total_chars > 0:
                    logger.info(f"OCR preserved: {total_chars} characters found.")
                    return output_pdf
                else:
                    logger.error("OCR LOST after image recompression!")
                    shutil.copy2(input_pdf, output_pdf)
                    return output_pdf
        except Exception as e:
            logger.warning(f"Could not verify OCR after image recompression: {e}")
            shutil.copy2(input_pdf, output_pdf)
            return output_pdf
    except Exception as e:
        logger.error(f"Image recompression failed: {e}")
        shutil.copy2(input_pdf, output_pdf)
        return output_pdf

# ---------------- Main ----------------
def main():
    input_folder = Path(INPUT_DIR)
    output_folder = Path(OUTPUT_DIR)
    if not input_folder.exists():
        logger.error(f"Input folder '{INPUT_DIR}' not found.")
        sys.exit(1)
    output_folder.mkdir(exist_ok=True)
    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files in '{INPUT_DIR}'")
        sys.exit(1)
    logger.info(f"Processing {len(pdf_files)} files with TrOCR: {TROCR_MODEL_NAME}")
    logger.info(f"Target: Final size as close as possible to original size (max 15% increase for OCR)")

    for pdf_path in pdf_files:
        original_size = pdf_path.stat().st_size
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {pdf_path.name} | Original: {original_size//1024} KB")

        ocr_temp_path = output_folder / f"{pdf_path.stem}_ocr_temp.pdf"
        if not process_single_pdf_ocr(str(pdf_path), str(ocr_temp_path)):
            logger.error(f"Skipping {pdf_path.name} due to OCR failure.")
            continue

        final_path = output_folder / f"{pdf_path.stem}_final.pdf"
        result_path = enhanced_compress_to_target_size(ocr_temp_path, final_path, original_size)

        if result_path.exists():
            final_size = result_path.stat().st_size
            size_increase = (final_size - original_size) / original_size * 100
            logger.info(f"SUCCESS: {result_path.name} | {final_size//1024} KB ({size_increase:+.1f}% increase from original)")
        else:
            logger.error(f"Failed to generate final output for {pdf_path.name}")

        try:
            ocr_temp_path.unlink()
        except Exception as e:
            logger.warning(f"Could not delete temp file: {e}")

    logger.info(f"\nAll done! Output files in '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()
