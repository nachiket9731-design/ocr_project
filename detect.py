"""
detect.py - Detection logic for all field types.

Four strategies:
  1a. Header text fields (Date, Khoji No, City, Name) → Multi-pass EasyOCR
                                                         (CLAHE + sharpen preprocessing;
                                                          dual EN / HI+EN readers with
                                                          confidence-based selection)
  1b. Free-text field (Q14)                           → EasyOCR (Hindi + English)
  2.  Yes/No checkboxes (P1 Q1, Q2)                  → dark pixel density per box
  3.  Circled numbers (all range questions)           → template subtraction + contour detection
"""

import ssl
import os
import cv2
import numpy as np
import base64
import json
import urllib.request
import urllib.error

# Ensure EasyOCR can download models even if Windows root certificates are missing
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import easyocr
from PIL import Image
import config as C
from config import NUM_SLOTS

# These are measured from the blank template at 3x zoom.
# The 10 Devanagari numbers start at ~17% from the left edge of the crop
# and end at ~93% — they are NOT spread edge-to-edge.
# Measured actual pixel offsets from the left of the row crop:
#   Number 1 centre ≈ 216px, Number 10 centre ≈ 1188px (row width = 1275)
# These ratios are stable because all rows have the same x1/x2 bounds.
NUM_START_RATIO = 216 / 1275   # ≈ 0.169  – x-fraction where number-1 sits
NUM_END_RATIO   = 1188 / 1275  # ≈ 0.932  – x-fraction where number-10 sits

# ---------------------------------------------------------------------------
# EasyOCR readers (loaded once, lazy-initialised)
#   _reader_en    : English-only  → used for header fields (Date, Khoji, City, Name)
#                   Avoids Hindi character substitutions on Roman handwriting.
#   _reader_hi_en : Hindi+English → used for Q14 free-text which may be in Hindi.
# ---------------------------------------------------------------------------
_reader_en    = None
_reader_hi_en = None

def get_reader_en() -> easyocr.Reader:
    """Lazy-load the English-only EasyOCR reader."""
    global _reader_en
    if _reader_en is None:
        print("  [INFO] Loading EasyOCR model (English only)...")
        _reader_en = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("  [INFO] EasyOCR (English) ready.")
    return _reader_en

def get_reader_hi_en() -> easyocr.Reader:
    """Lazy-load the Hindi+English EasyOCR reader."""
    global _reader_hi_en
    if _reader_hi_en is None:
        print("  [INFO] Loading EasyOCR model (Hindi + English)...")
        _reader_hi_en = easyocr.Reader(['hi', 'en'], gpu=False, verbose=False)
        print("  [INFO] EasyOCR (Hindi+English) ready.")
    return _reader_hi_en

# Backwards-compat alias used by read_text() for Q14
def get_reader() -> easyocr.Reader:
    return get_reader_hi_en()


# ---------------------------------------------------------------------------
# Helper: remove horizontal printed lines (underlines) from a grayscale crop
# ---------------------------------------------------------------------------
def _remove_horizontal_lines(gray: np.ndarray) -> np.ndarray:
    """
    Suppress long horizontal printed lines (form underlines) so they don't
    confuse OCR.  Uses morphological opening with a wide horizontal kernel.
    """
    # Detect horizontal lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    hlines   = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_h)
    # Subtract them from inverted image (dark-on-white), then re-invert
    inv       = 255 - gray
    inv_clean = cv2.subtract(inv, hlines)
    return 255 - inv_clean


# ---------------------------------------------------------------------------
# 1a. Header text field OCR  — Multi-pass EasyOCR with smart preprocessing
#
# Strategy:
#   Apply 3 preprocessing variants on the crop:
#     A) CLAHE contrast enhancement + unsharp sharpening
#     B) Otsu global binarisation (clean printed underlines already removed)
#     C) Adaptive Gaussian threshold  (handles uneven lighting)
#   For each variant, run BOTH:
#     - English-only reader  (better for Roman-script handwriting)
#     - Hindi+English reader (required for Devanagari handwriting)
#   Pick the (variant, reader) pair with the highest average EasyOCR confidence.
#   Falls back to the raw Hindi+English result if nothing above threshold.
# ---------------------------------------------------------------------------

def _pad_image(gray: np.ndarray, pad: int = 12) -> np.ndarray:
    """Add white padding around a grayscale image so OCR engines see full strokes."""
    return cv2.copyMakeBorder(gray, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=255)


def _remove_horizontal_lines_adaptive(gray: np.ndarray) -> np.ndarray:
    """
    Remove horizontal printed underlines using a kernel width proportional
    to the image width (avoids removing letter strokes after upscaling).
    """
    w = gray.shape[1]
    # Kernel must span at least 40% of image width to only catch full lines
    kw = max(60, int(w * 0.4))
    kernel_h  = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    hlines    = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_h)
    inv       = 255 - gray
    inv_clean = cv2.subtract(inv, hlines)
    return 255 - inv_clean


def _remove_all_lines(gray: np.ndarray) -> np.ndarray:
    """
    Remove both horizontal underlines AND vertical form borders.
    Horizontal: wide kernel catches full-width lines.
    Vertical:   tall kernel catches full-height border lines.
    """
    h, w = gray.shape

    # Horizontal lines
    kw       = max(60, int(w * 0.4))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    hlines   = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_h)

    # Vertical lines (form borders at crop edges)
    kh       = max(30, int(h * 0.4))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    vlines   = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_v)

    # Combine and subtract from inverted image
    all_lines = cv2.add(hlines, vlines)
    inv       = 255 - gray
    inv_clean = cv2.subtract(inv, all_lines)
    return 255 - inv_clean


def _enhance_crop(gray: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE contrast enhancement followed by unsharp masking.
    Makes faint pen strokes darker and crisper for better OCR.
    """
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl      = clahe.apply(gray)
    blurred = cv2.GaussianBlur(cl, (0, 0), sigmaX=3)
    sharp   = cv2.addWeighted(cl, 1.5, blurred, -0.5, 0)
    return sharp


def _build_preprocessing_variants(gray: np.ndarray) -> list:
    """
    Return a list of preprocessed images to try with EasyOCR.

    Variant A: Raw (no line removal).
    Variant B: CLAHE + sharpen.
    Variant C: Otsu binarisation on CLAHE.
    Variant D: Adaptive Gaussian threshold.
    Variant E: All lines (H+V) removed + CLAHE.  ← kills form borders
    """
    # A: Raw
    variant_a = gray

    # B: CLAHE + sharpen
    variant_b = _enhance_crop(gray)

    # C: Otsu on enhanced
    _, variant_c = cv2.threshold(
        variant_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # D: Adaptive Gaussian threshold
    variant_d = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25, C=10
    )

    # E: H+V line removal + CLAHE  (eliminates form border artifacts)
    no_lines  = _remove_all_lines(gray)
    variant_e = _enhance_crop(no_lines)

    return [variant_a, variant_b, variant_c, variant_d, variant_e]


def _clean_ocr_text(text: str) -> str:
    """
    Strip form-border artifacts and normalise whitespace from OCR output.
    Characters like <, |, >, ^, ~ appear when EasyOCR reads form lines.
    """
    import re
    # Remove known form-border characters
    text = re.sub(r'[<>|\\^~`]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _easyocr_with_confidence(reader: easyocr.Reader,
                              img: np.ndarray,
                              min_conf: float = 0.15) -> tuple:
    """
    Run EasyOCR and return (text, avg_confidence).
    Tokens with confidence >= min_conf are included.
    Returns ALL detected text (even low-conf) with its avg confidence.
    """
    try:
        results = reader.readtext(
            img, detail=1, paragraph=False,
            contrast_ths=0.1, adjust_contrast=0.7
        )
        if not results:
            return "", 0.0
        # Include tokens above min_conf threshold
        kept = [(r[1], r[2]) for r in results if r[2] >= min_conf]
        if not kept:
            # Fall back: return best single token even if below threshold
            best = max(results, key=lambda r: r[2])
            return best[1].strip(), best[2]
        text     = " ".join(t for t, _ in kept).strip()
        avg_conf = sum(c for _, c in kept) / len(kept)
        return text, avg_conf
    except Exception:
        return "", 0.0





def read_text_header(image: np.ndarray, bbox: tuple,
                     field_type: str = "default") -> str:
    """
    OCR for header fields: Date, Khoji No, City, Name.

    Field-type-aware strategy:
      - date / khoji : English-only reader (numbers + Latin text).
                       HI+EN reader is excluded — it wins confidence on these
                       fields but gives wrong Hindi-misread results.
      - city / name  : Both EN and HI+EN readers across all variants.
                       (These fields may be in Hindi or English.)

    Args:
        image:      Full-page BGR image (raw, pre-alignment).
        bbox:       (x1, y1, x2, y2) crop region.
        field_type: 'date', 'khoji', 'city', 'name', or 'default'.

    Returns:
        Extracted text string (stripped), or "" if nothing found.
    """
    x1, y1, x2, y2 = bbox

    # Expand crop downward by 30px so descenders are captured
    img_h = image.shape[0]
    y2_exp = min(img_h, y2 + 30)
    crop = image[y1:y2_exp, x1:x2]
    if crop.size == 0:
        return ""

    # --- Step 2: Fallback to local optimized EasyOCR pipeline ---

    # --- Base: grayscale + aggressive upscale ---
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Upscale to at least 200px tall — source crops are only ~50px
    if h < 200:
        scale = max(2, 200 // h)
        gray  = cv2.resize(gray, (w * scale, h * scale),
                           interpolation=cv2.INTER_CUBIC)

    # Trim left/right margins (3%) to exclude vertical form border lines
    marg = max(6, int(gray.shape[1] * 0.03))
    gray = gray[:, marg:-marg] if gray.shape[1] > marg * 3 else gray

    # Add padding so EasyOCR sees complete strokes at borders
    gray = _pad_image(gray, pad=16)

    # --- Field-type-aware reader selection ---
    # date/khoji: EN only (HI+EN gives wrong high-confidence misreads)
    # city/name:  Both EN and HI+EN (may be written in Hindi)
    if field_type in ("date", "khoji"):
        active_readers = [("en", get_reader_en())]
    else:
        active_readers = [
            ("en",    get_reader_en()),
            ("hi_en", get_reader_hi_en()),
        ]

    # --- Build variants and run all (variant × reader) combos ---
    variants = _build_preprocessing_variants(gray)

    best_text = ""
    best_conf = 0.0

    for img_variant in variants:
        for _, reader in active_readers:
            text, conf = _easyocr_with_confidence(reader, img_variant)
            if conf > best_conf:
                best_conf = conf
                best_text = text

    return _clean_ocr_text(best_text) if best_text else ""


# ---------------------------------------------------------------------------
# 1b. Free-text field OCR  (Hindi+English reader — used for Q14)
# ---------------------------------------------------------------------------
def read_text(image: np.ndarray, bbox: tuple) -> str:
    """
    OCR for free-text fields (Q14).  Supports Hindi Devanagari + English.

    Args:
        image: Full-page BGR image.
        bbox:  (x1, y1, x2, y2) crop region.

    Returns:
        Extracted text string (stripped), or "" if nothing found.
    """
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return ""

    # --- Step 1: Local EasyOCR Pipeline ---
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Upscale small crops for better OCR accuracy
    h, w = gray.shape
    if h < 80:
        scale = max(2, 80 // h)
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Binarize using Otsu's method for clean black-on-white text
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    reader = get_reader_hi_en()
    results = reader.readtext(binary, detail=0, paragraph=True)
    text = " ".join(results).strip()
    return text


# ---------------------------------------------------------------------------
# 2. Yes/No checkbox detection (Template subtraction approach)
# ---------------------------------------------------------------------------
def read_checkbox(filled_img: np.ndarray,
                  tmpl_img: np.ndarray,
                  han_bbox: tuple,
                  na_bbox: tuple) -> str:
    """
    Determine which checkbox (हाँ=Yes / ना=No) has a tick mark.
    If both are ticked/marked, returns "Yes, No".

    Uses template subtraction on the inner box area to eliminate the pre-printed '□'
    square outlines, isolating only the user's pen stroke (tick / check / cross / fill).

    Args:
        filled_img: Aligned full-page BGR image of the filled form.
        tmpl_img:   Aligned full-page BGR image of the blank template.
        han_bbox:   (x1,y1,x2,y2) for the हाँ (Yes) checkbox.
        na_bbox:    (x1,y1,x2,y2) for the ना (No) checkbox.

    Returns:
        "Yes", "No", "Yes, No" (if double-marked), or "" if neither is clearly ticked.
    """
    def get_clean_ink(bbox):
        x1, y1, x2, y2 = bbox
        f_crop = filled_img[y1+6:y2-6, x1+6:x2-6]
        if f_crop.size == 0:
            return 0

        f_gray = cv2.cvtColor(f_crop, cv2.COLOR_BGR2GRAY)
        if tmpl_img is not None:
            t_crop = tmpl_img[y1+6:y2-6, x1+6:x2-6]
            if t_crop.size > 0:
                t_gray = cv2.cvtColor(t_crop, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(f_gray, t_gray)
                _, diff_bin = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                diff_clean = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, kernel)
                return int(np.sum(diff_clean > 0))

        _, binary = cv2.threshold(f_gray, 150, 255, cv2.THRESH_BINARY_INV)
        return int(np.sum(binary > 0))

    han_score = get_clean_ink(han_bbox)
    na_score  = get_clean_ink(na_bbox)

    if han_score < 150 and na_score < 150:
        return ""  # neither checkbox ticked

    # Genuine double mark: both boxes have heavy deliberate ink with minimal disparity
    if han_score >= 380 and na_score >= 380 and abs(han_score - na_score) < 40:
        return "Yes, No"

    if han_score > na_score:
        return "Yes"
    return "No"


# ---------------------------------------------------------------------------
# 3. Circled number detection  (template subtraction approach)
# ---------------------------------------------------------------------------
def read_circled_number(filled_img: np.ndarray,
                        template_img: np.ndarray,
                        row_bbox: tuple) -> str:
    """
    Detect which number (1–10) is circled or marked in a number row.
    If multiple numbers are marked/circled, returns comma-separated list (e.g. "4, 9").

    Uses template ink differencing + morphological contour analysis to isolate hand-drawn
    user pen marks from pre-printed numbers and varying paper background illumination.

    Args:
        filled_img:   Aligned full-page BGR image of the filled form.
        template_img: Aligned full-page BGR image of the blank template.
        row_bbox:     (x1, y1, x2, y2) bounding box of the 10-number row.

    Returns:
        String "1"–"10", comma-separated string (e.g. "4, 9") if multiple, or "" if none.
    """
    x1, y1, x2, y2 = row_bbox

    filled_crop = filled_img[y1:y2, x1:x2]
    if filled_crop.size == 0:
        return ""

    # Convert to grayscale
    gray_filled = cv2.cvtColor(filled_crop, cv2.COLOR_BGR2GRAY)
    row_h, row_w = gray_filled.shape

    # 1. Local background illumination estimation & normalization (flat-field correction)
    bg = cv2.medianBlur(gray_filled, 31)
    norm = cv2.divide(gray_filled, bg, scale=255.0).astype(np.uint8)

    # 2. Extract dark ink strokes via Black-Hat morphology
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(norm, cv2.MORPH_BLACKHAT, kernel_bh)

    # 3. Threshold to isolate pen ink from background paper texture
    _, ink_mask = cv2.threshold(blackhat, 25, 255, cv2.THRESH_BINARY)

    # 4. Dilate mask to bridge hand-drawn circle stroke gaps
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(ink_mask, kernel_d, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    start_px = row_w * NUM_START_RATIO
    end_px   = row_w * NUM_END_RATIO
    span     = end_px - start_px

    if span <= 0 or not contours:
        return ""

    def _get_slot(cnt):
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        rel = (cx - start_px) / span
        if rel < -0.06 or rel > 1.06:
            return None
        slot = int(round(rel * (NUM_SLOTS - 1)))
        return max(0, min(NUM_SLOTS - 1, slot)) + 1

    # Template new pen ink check to verify slot 10 ink
    tmpl_crop = template_img[y1:y2, x1:x2] if template_img is not None else None
    if tmpl_crop is not None and tmpl_crop.shape == filled_crop.shape:
        gray_tmpl = cv2.cvtColor(tmpl_crop, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_tmpl, gray_filled)
        darker_mask = (gray_filled < gray_tmpl).astype(np.uint8)
        new_ink_mask = cv2.threshold(diff, 55, 255, cv2.THRESH_BINARY)[1] * darker_mask
    else:
        new_ink_mask = None

    cnts_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    slots = []

    for cnt in cnts_sorted:
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)

        # Upper area limit to reject massive lighting blobs
        if area > row_w * row_h * 0.35:
            continue

        s = _get_slot(cnt)
        if not s:
            continue

        # Slot-dependent minimum area threshold:
        # Pre-printed single digits 1..9 have area ~500-680
        # Pre-printed double digit 10 ('१०') has area ~850-1430 because '1' and '0' merge when dilated.
        # Hand-drawn mark on 1..9 has area >= 1000.
        # Hand-drawn mark on 10 has area >= 1400.
        # Pre-printed '१०' has area 488-1024; real circles produce area 2000+.
        # New-ink mask is unreliable for slot 10 (alignment jitter at crop edge).
        if s == 10:
            if area < 1400:
                continue
        else:
            if area < 750:
                continue
            if area < 1000 or bw < 35 or bh < 30:
                # Borderline area — verify with new ink mask
                if new_ink_mask is not None:
                    step = span / (NUM_SLOTS - 1)
                    cx = int(start_px + (s - 1) * step)
                    slot_w = int(step * 0.6)
                    x_left = max(0, cx - slot_w // 2)
                    x_right = min(row_w, cx + slot_w // 2)
                    slot_ink = np.sum(new_ink_mask[:, x_left:x_right] > 0)
                    if slot_ink < 120:
                        continue
                else:
                    continue

        if s not in slots:
            slots.append(s)

    if not slots:
        return ""

    slots.sort()
    return ", ".join(str(s) for s in slots)


def _detect_circle_fallback(gray: np.ndarray, row_w: int, row_h: int) -> str:
    """
    Fallback circle detection using contour analysis on the filled crop alone.
    Used when template subtraction finds nothing.
    """
    # Adaptive threshold to handle varying brightness
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21, C=10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    start_px = row_w * NUM_START_RATIO
    end_px   = row_w * NUM_END_RATIO
    span     = end_px - start_px
    if span <= 0:
        return ""

    min_area = row_h * row_h * 0.25
    max_area = row_h * row_h * 6.0

    detected_slots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        convexity = area / hull_area

        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0

        score = convexity * aspect
        if score >= 0.3:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                relative = (cx - start_px) / span
                slot = int(round(relative * (NUM_SLOTS - 1)))
                slot = max(0, min(NUM_SLOTS - 1, slot))
                num = slot + 1
                # Enforce higher area requirement for slot 10
                if num == 10 and area < 1600:
                    continue
                if num not in detected_slots:
                    detected_slots.append(num)

    if not detected_slots:
        return ""

    detected_slots.sort()
    return ", ".join(str(n) for n in detected_slots)


# ---------------------------------------------------------------------------
# 4. Single-Call Google Cloud Vision API Optimization (1 Credit per Form)
#    Supports both Service Account JSON key (Recommended) and REST API Key
# ---------------------------------------------------------------------------
_vision_client = None

def _get_vision_client():
    """Lazy-load the Google Cloud Vision SDK client from Service Account JSON."""
    global _vision_client
    if _vision_client is not None:
        return _vision_client

    sa_path = getattr(C, "GOOGLE_SERVICE_ACCOUNT_PATH", "service_account.json")
    if not os.path.isabs(sa_path):
        sa_path = os.path.join(os.path.dirname(__file__), sa_path)

    if os.path.exists(sa_path) and os.path.getsize(sa_path) > 10:
        try:
            from google.cloud import vision
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(sa_path)
            _vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            return _vision_client
        except Exception as e:
            print(f"  [WARN] Failed to load Service Account credentials from '{sa_path}': {e}")

    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ and os.path.exists(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]):
        try:
            from google.cloud import vision
            _vision_client = vision.ImageAnnotatorClient()
            return _vision_client
        except Exception as e:
            print(f"  [WARN] Failed to load default Google credentials: {e}")

    return None


def _call_google_vision_single_image(composite_img: np.ndarray, api_key: str = "") -> list:
    """
    Sends a single composite image to Google Cloud Vision API (1 credit).
    Tries Service Account JSON first, then REST API Key fallback.
    Returns list of text annotations.
    """
    success, buffer = cv2.imencode('.jpg', composite_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        return []

    # Priority 1: Service Account JSON (official SDK)
    client = _get_vision_client()
    if client is not None:
        try:
            from google.cloud import vision
            image = vision.Image(content=buffer.tobytes())
            image_context = vision.ImageContext(language_hints=["hi", "en"])
            response = client.document_text_detection(image=image, image_context=image_context)

            if response.error.message:
                print(f"  [ERROR] Google Vision SDK error: {response.error.message}")
                return []

            raw_annotations = []
            for ann in response.text_annotations:
                vertices = [{"x": v.x, "y": v.y} for v in ann.bounding_poly.vertices]
                raw_annotations.append({
                    "description": ann.description,
                    "boundingPoly": {"vertices": vertices}
                })
            return raw_annotations
        except Exception as e:
            print(f"  [ERROR] Google Vision SDK request failed: {e}")

    # Priority 2: REST API Key
    if api_key and api_key.strip():
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        payload = json.dumps({
            "requests": [
                {
                    "image": {"content": img_b64},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                    "imageContext": {
                        "languageHints": ["hi", "en"]
                    }
                }
            ]
        }).encode('utf-8')

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                responses = data.get("responses", [])
                if responses and "textAnnotations" in responses[0]:
                    return responses[0]["textAnnotations"]
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode('utf-8', errors='replace')
            print(f"  [ERROR] Google Vision API HTTP {e.code}: {err_msg}")
        except Exception as e:
            print(f"  [ERROR] Google Vision API request failed: {e}")

    return []


def _parse_vision_annotations(annotations: list, slot_bounds: list) -> dict:
    """
    Maps Google Cloud Vision word-level detections to each field slot based on
    vertical (y) coordinate and orders words left-to-right / top-to-bottom.
    """
    field_words = {name: [] for name, _, _ in slot_bounds}

    for ann in annotations[1:]:
        text = ann.get("description", "").strip()
        if not text:
            continue
        vertices = ann.get("boundingPoly", {}).get("vertices", [])
        if not vertices:
            continue

        y_coords = [v.get("y", 0) for v in vertices if "y" in v]
        x_coords = [v.get("x", 0) for v in vertices if "x" in v]
        if not y_coords or not x_coords:
            continue

        y_center = sum(y_coords) / len(y_coords)
        x_min    = min(x_coords)
        y_min    = min(y_coords)

        # Match to the corresponding slot by vertical position
        for name, y_start, y_end in slot_bounds:
            if (y_start <= y_center <= y_end) or (y_start <= y_min <= y_end):
                field_words[name].append((y_min, x_min, text))
                break

    result = {}
    for name, _, _ in slot_bounds:
        words = field_words[name]
        if not words:
            result[name] = ""
            continue

        # Sort words top-to-bottom, left-to-right (line grouping with 15px tolerance)
        words_sorted_y = sorted(words, key=lambda w: w[0])
        lines = []
        curr_line = []
        curr_y = None
        for y, x, t in words_sorted_y:
            if curr_y is None or abs(y - curr_y) < 15:
                curr_line.append((x, t))
                curr_y = y
            else:
                lines.append(curr_line)
                curr_line = [(x, t)]
                curr_y = y
        if curr_line:
            lines.append(curr_line)

        line_texts = []
        for line in lines:
            line_sorted = sorted(line, key=lambda item: item[0])
            line_texts.append(" ".join(t for _, t in line_sorted))

        full_field_text = " ".join(line_texts).strip()
        result[name] = _clean_ocr_text(full_field_text) if name != "P2_Q14" else full_field_text

    return result


def extract_all_text_fields_single_call(
    p1_aligned: np.ndarray,
    p2_aligned: np.ndarray,
    api_key: str = None
) -> dict:
    """
    Extracts all 5 text fields (Date, Khoji_No, City, Name, P2_Q14) in ONE single
    Google Cloud Vision API call by stitching them into a composite canvas.

    Cost: Exactly 1 Google Cloud Vision unit / credit per form (80% cost reduction).
    Fallback: Automatically falls back to local EasyOCR if API credentials are not set or fail.
    """
    if api_key is None:
        api_key = getattr(C, "GOOGLE_VISION_API_KEY", "")

    def _get_crop(img, bbox):
        x1, y1, x2, y2 = bbox
        return img[y1:y2, x1:x2]

    # Collect crops from aligned pages
    date_crop  = _get_crop(p1_aligned, C.PAGE1_DATE)
    khoji_crop = _get_crop(p1_aligned, C.PAGE1_KHOJI)
    city_crop  = _get_crop(p1_aligned, C.PAGE1_CITY)
    name_crop  = _get_crop(p1_aligned, C.PAGE1_NAME)
    q14_crop   = _get_crop(p2_aligned, C.PAGE2_Q14_TEXT)

    crops = [
        # ("Date", date_crop),
        # ("Khoji_No", khoji_crop),
        # ("City", city_crop),
        # ("Name", name_crop),
        ("P2_Q14", q14_crop)
    ]

    has_service_account = _get_vision_client() is not None
    has_api_key = bool(api_key and api_key.strip())

    # If Google Cloud Vision credentials are available, make 1-credit single call
    if has_service_account or has_api_key:
        valid_crops = [c for _, c in crops if c is not None and c.size > 0]
        if valid_crops:
            max_w = max(c.shape[1] for c in valid_crops)
            gap = 30
            slot_bounds = []
            current_y = 0
            canvas_parts = []

            for name, crop in crops:
                if crop is None or crop.size == 0:
                    continue
                h, w = crop.shape[:2]
                padded = np.full((h, max_w, 3), 255, dtype=np.uint8)
                padded[:, :w] = crop

                canvas_parts.append(padded)
                slot_bounds.append((name, current_y, current_y + h))
                current_y += h

                sep = np.full((gap, max_w, 3), 255, dtype=np.uint8)
                canvas_parts.append(sep)
                current_y += gap

            composite = np.vstack(canvas_parts)
            # Save debug composite image for user inspection
            try:
                cv2.imwrite("debug_composite_form.png", composite)
            except Exception:
                pass

            print("  [Google Vision] Sending 1 combined image for all 5 text fields (1 Credit)...")
            annotations = _call_google_vision_single_image(composite, api_key.strip() if has_api_key else "")

            if annotations:
                parsed = _parse_vision_annotations(annotations, slot_bounds)
                print(f"  [Google Vision] Date:     {parsed.get('Date', '')!r}")
                print(f"  [Google Vision] Khoji_No: {parsed.get('Khoji_No', '')!r}")
                print(f"  [Google Vision] City:     {parsed.get('City', '')!r}")
                print(f"  [Google Vision] Name:     {parsed.get('Name', '')!r}")
                print(f"  [Google Vision] P2_Q14:   {parsed.get('P2_Q14', '')!r}")
                return parsed
            else:
                print("  [WARN] Google Vision returned no annotations. Falling back to local EasyOCR...")

    # Fallback to local EasyOCR readers
    print("  [INFO] Using local EasyOCR pipeline...")
    # date_val  = read_text_header(p1_aligned, C.PAGE1_DATE,  field_type="date")
    # khoji_val = read_text_header(p1_aligned, C.PAGE1_KHOJI, field_type="khoji")
    # city_val  = read_text_header(p1_aligned, C.PAGE1_CITY,  field_type="city")
    # name_val  = read_text_header(p1_aligned, C.PAGE1_NAME,  field_type="name")
    
    date_val = ""
    khoji_val = ""
    city_val = ""
    name_val = ""

    # Fallback to Page 2 headers if Page 1 was blank
    # if not date_val:
    #     date_val  = read_text_header(p2_aligned, C.PAGE2_DATE,  field_type="date")
    # if not khoji_val:
    #     khoji_val = read_text_header(p2_aligned, C.PAGE2_KHOJI, field_type="khoji")
    # if not city_val:
    #     city_val  = read_text_header(p2_aligned, C.PAGE2_CITY,  field_type="city")
    # if not name_val:
    #     name_val  = read_text_header(p2_aligned, C.PAGE2_NAME,  field_type="name")

    q14_val   = read_text(p2_aligned, C.PAGE2_Q14_TEXT)

    return {
        "Date": date_val,
        "Khoji_No": khoji_val,
        "City": city_val,
        "Name": name_val,
        "P2_Q14": q14_val
    }


def read_q14_google_vision(image: np.ndarray, bbox: tuple = None, api_key: str = None) -> str:
    """
    Extract Question 14 (free-text answer on Page 2) using Google Cloud Vision API.
    Supports Service Account JSON and REST API Key.
    Falls back gracefully to local EasyOCR if API is unavailable or unconfigured.

    Args:
        image: Full-page BGR image of Page 2 (aligned).
        bbox:  (x1, y1, x2, y2) crop region. Defaults to C.PAGE2_Q14_TEXT.
        api_key: Optional Google Vision API key string.

    Returns:
        Extracted text string (stripped).
    """
    if bbox is None:
        bbox = getattr(C, "PAGE2_Q14_TEXT", (140, 2225, 1650, 2390))

    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return ""

    if api_key is None:
        api_key = getattr(C, "GOOGLE_VISION_API_KEY", "")

    # Encode crop to JPEG
    success, buffer = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        return read_text(image, bbox)

    # 1. Try Service Account JSON (official Google Cloud Vision SDK)
    client = _get_vision_client()
    if client is not None:
        try:
            from google.cloud import vision
            vision_img = vision.Image(content=buffer.tobytes())
            image_context = vision.ImageContext(language_hints=["hi", "en"])
            response = client.document_text_detection(image=vision_img, image_context=image_context)

            if response.error.message:
                print(f"  [ERROR] Google Vision SDK error for Q14: {response.error.message}")
            else:
                if response.full_text_annotation and response.full_text_annotation.text:
                    extracted = response.full_text_annotation.text.strip()
                    print(f"  [Google Vision] Q14 Extracted: {extracted!r}")
                    return extracted
                elif response.text_annotations:
                    extracted = response.text_annotations[0].description.strip()
                    print(f"  [Google Vision] Q14 Extracted: {extracted!r}")
                    return extracted
                else:
                    return ""
        except Exception as e:
            print(f"  [WARN] Google Vision SDK request failed for Q14: {e}")

    # 2. Try REST API Key fallback
    if api_key and api_key.strip():
        try:
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key.strip()}"
            payload = json.dumps({
                "requests": [
                    {
                        "image": {"content": img_b64},
                        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                        "imageContext": {
                            "languageHints": ["hi", "en"]
                        }
                    }
                ]
            }).encode('utf-8')

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                responses = data.get("responses", [])
                if responses:
                    res0 = responses[0]
                    if "fullTextAnnotation" in res0 and res0["fullTextAnnotation"].get("text"):
                        extracted = res0["fullTextAnnotation"]["text"].strip()
                        print(f"  [Google Vision REST] Q14 Extracted: {extracted!r}")
                        return extracted
                    elif "textAnnotations" in res0 and res0["textAnnotations"]:
                        extracted = res0["textAnnotations"][0].get("description", "").strip()
                        print(f"  [Google Vision REST] Q14 Extracted: {extracted!r}")
                        return extracted
        except Exception as e:
            print(f"  [WARN] Google Vision REST API failed for Q14: {e}")

    # Fallback to local EasyOCR
    print("  [INFO] Google Vision credentials not active. Falling back to local EasyOCR for Q14...")
    return read_text(image, bbox)



