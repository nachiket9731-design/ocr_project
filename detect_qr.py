"""
detect_qr.py - QR code detection for form identification.

Scans the form image for a pasted QR code (located in the top-right corner)
and extracts the encoded user ID (UUID).

The QR sticker on these forms encodes a string like:
    user_ids=0de859c7-d798-4b0b-9828-5bba22ac56a6

extract_qr_user_id() returns just the UUID part:
    "0de859c7-d798-4b0b-9828-5bba22ac56a6"

Detection strategy:
  1. Crop the top-right corner of the page (where the QR sticker sits).
  2. Primary  : cv2.QRCodeDetector  (built into OpenCV, no extra install)
  3. Fallback : pyzbar              (optional; pip install pyzbar)
  Both detectors are tried on several preprocessed variants for robustness.

Usage (standalone test):
    python detect_qr.py aa/prathamesh_pendke.pdf

Usage (from code):
    from detect_qr import extract_qr_user_id
    user_id = extract_qr_user_id(page_image)   # BGR np.ndarray
"""

import cv2
import numpy as np
import re
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Region of Interest: where on the page is the QR sticker?
# Measured from the rendered page (1785 x 2526 px at 3x zoom).
# The sticker sits in the top-right corner.
# ---------------------------------------------------------------------------
QR_ROI_TOP    = 0.00   # from top of page
QR_ROI_BOTTOM = 0.25   # bottom edge of search region (25% of height)
QR_ROI_LEFT   = 0.50   # left edge of search region (50% of width to capture stickers placed further left)
QR_ROI_RIGHT  = 1.00   # right edge of page


# ---------------------------------------------------------------------------
# Primary Decoders: zxing-cpp, pyzbar, OpenCV built-in
# ---------------------------------------------------------------------------

def _detect_with_zxing(image: np.ndarray) -> Optional[str]:
    """
    Try to decode a QR code using zxing-cpp.
    Returns the raw decoded string, or None.
    """
    try:
        import zxingcpp
        results = zxingcpp.read_barcodes(image, try_rotate=True, try_downscale=True, try_invert=True)
        for r in results:
            if r.text:
                return r.text.strip()
    except Exception:
        pass
    return None


def _detect_with_opencv(image: np.ndarray) -> Optional[str]:
    """
    Try to decode a QR code using OpenCV built-in QRCodeDetector.
    Returns the raw decoded string, or None.
    """
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(image)
    if data:
        return data.strip()
    return None


# ---------------------------------------------------------------------------
# Fallback: pyzbar
# ---------------------------------------------------------------------------

def _detect_with_pyzbar(image: np.ndarray) -> Optional[str]:
    """
    Try to decode a QR code using pyzbar (optional dependency).
    Install: pip install pyzbar
    Returns the raw decoded string, or None.
    """
    try:
        from pyzbar import pyzbar  # type: ignore
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        barcodes = pyzbar.decode(gray)
        for barcode in barcodes:
            if barcode.type == "QRCODE":
                return barcode.data.decode("utf-8").strip()
    except ImportError:
        pass
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Preprocessing: multiple variants to maximise detection success
# ---------------------------------------------------------------------------

def _preprocess_for_qr(image: np.ndarray) -> list:
    """
    Return preprocessed variants of the image for QR detection.
    Handles faint prints, uneven lighting, tilted stickers, and small QR codes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Add generous white padding around crop so tilted stickers aren't clipped
    padded = cv2.copyMakeBorder(gray, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    ph, pw = padded.shape[:2]
    center = (pw // 2, ph // 2)

    variants = []

    # Rotation angles to try for tilted/slanted QR stickers
    angles = [0, -5, -3, -2, 2, 3, 5, -8, 8, -12, 12]

    for angle in angles:
        if angle != 0:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rot = cv2.warpAffine(padded, M, (pw, ph), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        else:
            rot = padded

        variants.append(rot)

        # Variant: CLAHE contrast enhancement
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(rot)
        variants.append(enhanced)

        # Variant: Otsu binarisation
        _, binary = cv2.threshold(rot, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(binary)

    # 2x upscale variant for small/low-res QR codes
    h, w = gray.shape
    upscaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    variants.append(upscaled)

    return variants


# ---------------------------------------------------------------------------
# Parse the raw QR string to extract just the UUID
# ---------------------------------------------------------------------------

def _parse_user_id(raw: str) -> str:
    """
    Extract the user ID from the raw QR string.

    Handles formats:
      "user_ids=0de859c7-d798-4b0b-9828-5bba22ac56a6"  -> UUID
      "0de859c7-d798-4b0b-9828-5bba22ac56a6"            -> UUID (plain UUID)
      Any other string                                   -> returned as-is
    """
    kv_match = re.match(r'^[^=]+=(.+)$', raw.strip())
    if kv_match:
        return kv_match.group(1).strip()
    return raw.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_qr_raw(image: np.ndarray, debug: bool = False) -> Optional[str]:
    """
    Scan the top-right corner of the form for a QR code.
    Returns the raw decoded string exactly as encoded, or None.
    """
    h, w = image.shape[:2]

    # Crop to the QR search region (top-right corner)
    y1 = int(h * QR_ROI_TOP)
    y2 = int(h * QR_ROI_BOTTOM)
    x1 = int(w * QR_ROI_LEFT)
    x2 = int(w * QR_ROI_RIGHT)
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    variants = _preprocess_for_qr(roi)

    for idx, variant in enumerate(variants):
        # 1. Try zxing-cpp (most robust for scanned QR codes)
        result = _detect_with_zxing(variant)
        if result:
            if debug:
                print(f"  [QR] zxing-cpp detected on variant {idx}: '{result}'")
            return result

        # 2. Try pyzbar
        result = _detect_with_pyzbar(variant)
        if result:
            if debug:
                print(f"  [QR] pyzbar detected on variant {idx}: '{result}'")
            return result

        # 3. Try OpenCV built-in
        result = _detect_with_opencv(variant)
        if result:
            if debug:
                print(f"  [QR] OpenCV detected on variant {idx}: '{result}'")
            return result

    if debug:
        print("  [QR] No QR code found in the top-right region.")
    return None


def extract_qr_user_id(image: np.ndarray, debug: bool = False) -> Optional[str]:
    """
    Scan the form for a QR code and return the parsed user/form ID (UUID).

    Args:
        image: Full-page BGR numpy array (from PDF renderer or cv2.imread).
        debug: If True, print detection details.

    Returns:
        The user ID string (UUID) or sticker serial number (e.g. F-0002), or None if not found.
    """
    raw = extract_qr_raw(image, debug=debug)
    if raw is None:
        # Fallback for defective/damaged QR stickers: OCR the printed sticker ID (e.g., F-0002)
        try:
            import easyocr
            import re
            h, w = image.shape[:2]
            roi = image[int(h * QR_ROI_TOP):int(h * QR_ROI_BOTTOM), int(w * QR_ROI_LEFT):int(w * QR_ROI_RIGHT)]
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            res = reader.readtext(roi, detail=0)
            text = " ".join(res)
            match = re.search(r'\b([A-Z]-\d{4})\b', text)
            if match:
                sticker_id = match.group(1)
                if debug:
                    print(f"  [QR Fallback OCR] Detected sticker code: '{sticker_id}' from damaged QR sticker.")
                return f"QR_DAMAGED (Found: {sticker_id})"
        except Exception as e:
            if debug:
                print(f"  [QR Fallback OCR Error] {e}")
        return None

    parsed = _parse_user_id(raw)
    if debug:
        print(f"  [QR] Parsed user ID: '{parsed}'")
    return parsed


def get_qr_bbox(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the QR code and return its bounding box on the FULL page image.

    Returns:
        (x1, y1, x2, y2) in full-page pixel coordinates, or None.
    """
    h, w = image.shape[:2]
    y1_roi = int(h * QR_ROI_TOP)
    y2_roi = int(h * QR_ROI_BOTTOM)
    x1_roi = int(w * QR_ROI_LEFT)
    roi     = image[y1_roi:y2_roi, x1_roi:]

    detector = cv2.QRCodeDetector()
    variants = _preprocess_for_qr(roi)
    for variant in variants:
        data, points, _ = detector.detectAndDecode(variant)
        if data and points is not None:
            pts = points[0].astype(int)
            rx1 = int(pts[:, 0].min()) + x1_roi
            ry1 = int(pts[:, 1].min()) + y1_roi
            rx2 = int(pts[:, 0].max()) + x1_roi
            ry2 = int(pts[:, 1].max()) + y1_roi
            return (rx1, ry1, rx2, ry2)
    return None


# ---------------------------------------------------------------------------
# Standalone test  -  python detect_qr.py <image_or_pdf_path>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python detect_qr.py <path_to_form_image_or_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    ext  = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        doc  = fitz.open(path)
        page = doc[0]
        mat  = fitz.Matrix(3, 3)
        pix  = page.get_pixmap(matrix=mat)
        img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(f"Loaded PDF first page: {img.shape[1]}x{img.shape[0]} px")
    else:
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] Could not load image: {path}")
            sys.exit(1)

    print("\n--- QR Detection ---")
    raw     = extract_qr_raw(img, debug=True)
    user_id = extract_qr_user_id(img, debug=True)

    if user_id:
        print(f"\n[OK] QR Code detected!")
        print(f"     Raw value : {raw}")
        print(f"     User ID   : {user_id}")

        # Draw bounding box on a debug image and save it
        bbox = get_qr_bbox(img)
        debug_img = img.copy()
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            cv2.putText(debug_img, f"User ID: {user_id}", (x1, max(30, y1 - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        out_path = path.rsplit('.', 1)[0] + "_qr_debug.png"
        cv2.imwrite(out_path, debug_img)
        print(f"     Debug image output saved at: {out_path}")
    else:
        print("\n[FAIL] No QR code detected.")
