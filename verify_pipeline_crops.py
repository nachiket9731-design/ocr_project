"""
verify_pipeline_crops.py - Pipeline Manual Inspection & Verification Tool.

Updated with:
1. QR Code extraction & sticker visualization (extract_qr_user_id from detect_qr).
2. Backend API profile lookup (Name, City) via fetch_user_profile.
3. Updated Page 2 Q14 crop coordinates (140, 2225, 1650, 2390) for full 2-line capture.
4. Field flagging (Double marks, unanswered questions, QR status).
5. Robust path resolution to project root from unused_archive/dev_scripts/.
6. Comprehensive self-contained HTML verification report.

Usage:
    python verify_pipeline_crops.py
    python verify_pipeline_crops.py <path_to_filled.pdf>
"""

import sys
import os

# Reconfigure stdout/stderr to UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import cv2
import numpy as np
import base64
import glob
import json
import ssl
import urllib.request
import urllib.error

# Ensure project root is in sys.path regardless of where script is executed from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(SCRIPT_DIR, "Nisarg form_Nov 2025.pdf")):
    PROJECT_ROOT = SCRIPT_DIR
else:
    candidate = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    PROJECT_ROOT = candidate if os.path.exists(os.path.join(candidate, "Nisarg form_Nov 2025.pdf")) else SCRIPT_DIR

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from align import pdf_to_images, align_page
import config as C
from detect_qr import extract_qr_user_id
from detect import (
    read_checkbox,
    read_circled_number,
    read_text_header,
    read_text,
    NUM_START_RATIO,
    NUM_END_RATIO,
    NUM_SLOTS
)

BLANK_PDF = os.path.join(PROJECT_ROOT, "Nisarg form_Nov 2025.pdf")


def fetch_user_profile(user_id: str) -> dict:
    """Fetch user name and city from the backend API using the UUID."""
    if not user_id or user_id.startswith("QR_DAMAGED") or user_id in ("Not Found", "None"):
        return {}
    
    url = f"https://staging.api.happythoughts.in/iam/public/user-profile/?pid={user_id}"
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"  [API ERROR] Failed to fetch profile for {user_id}: {e}")
        return {}


def draw_bboxes_on_page(img: np.ndarray, field_bboxes: list) -> np.ndarray:
    """
    Draw colored bounding boxes with field names on a full-page image.
    field_bboxes: list of (field_name, (x1, y1, x2, y2), color_bgr)
    """
    annotated = img.copy()
    for name, (x1, y1, x2, y2), color in field_bboxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1 - 8, th + 5)
        cv2.rectangle(annotated, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), color, -1)
        cv2.putText(annotated, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def generate_ink_debug_mask(filled_crop: np.ndarray) -> np.ndarray:
    """Generate a side-by-side visual image showing raw crop and detected ink mask with slots."""
    if filled_crop is None or filled_crop.size == 0:
        return None
    
    gray = cv2.cvtColor(filled_crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Local background illumination normalization
    bg = cv2.medianBlur(gray, 31)
    norm = cv2.divide(gray, bg, scale=255.0).astype(np.uint8)

    # Black-Hat morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(norm, cv2.MORPH_BLACKHAT, kernel)
    _, ink_mask = cv2.threshold(blackhat, 25, 255, cv2.THRESH_BINARY)

    # Dilate
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(ink_mask, kernel_d, iterations=2)

    # Colorize mask for debug display
    mask_bgr = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    start_px = w * NUM_START_RATIO
    end_px   = w * NUM_END_RATIO
    span     = end_px - start_px

    # Draw vertical slot dividers in blue
    for s in range(NUM_SLOTS):
        x = int(start_px + (s / (NUM_SLOTS - 1)) * span) if NUM_SLOTS > 1 else int(start_px)
        cv2.line(mask_bgr, (x, 0), (x, h), (255, 150, 0), 1)

    cnts_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in cnts_sorted:
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else bx + bw // 2
        rel = (cx - start_px) / span if span > 0 else 0
        slot = int(round(rel * (NUM_SLOTS - 1))) + 1 if M["m00"] > 0 else None

        is_valid = False
        if area <= w * h * 0.35:
            if slot == 10:
                if area >= 1600:
                    is_valid = True
            else:
                if area >= 1000 and bw >= 35 and bh >= 30:
                    is_valid = True

        if is_valid:
            # Highlight valid circle contour in GREEN
            cv2.drawContours(mask_bgr, [cnt], -1, (0, 255, 0), 2)
            cv2.rectangle(mask_bgr, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        else:
            # Draw rejected noise contours in RED
            cv2.drawContours(mask_bgr, [cnt], -1, (0, 0, 255), 1)

    combined = np.hstack([filled_crop, mask_bgr])
    return combined


def verify_pdf(pdf_path: str):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.join(PROJECT_ROOT, "pipeline_verification", pdf_name)
    crops_dir = os.path.join(out_dir, "crops")
    masks_dir = os.path.join(out_dir, "debug_masks")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" PIPELINE VERIFICATION RUN: {os.path.basename(pdf_path)}")
    print(f" Output Folder: {out_dir}")
    print(f"{'='*70}\n")

    print("[Step 1/6] Rendering template and filled PDF pages...")
    tmpl_imgs = pdf_to_images(BLANK_PDF)
    filled_imgs = pdf_to_images(pdf_path)

    if len(filled_imgs) < 2:
        print("[ERROR] Filled PDF must have at least 2 pages.")
        return

    p1_raw = filled_imgs[0]
    p2_raw = filled_imgs[1]

    # -----------------------------------------------------------------------
    # Step 2: QR Code Extraction & API Lookup
    # -----------------------------------------------------------------------
    print("[Step 2/6] Extracting QR Code and querying Profile API...")
    user_id = extract_qr_user_id(p1_raw, debug=False)
    user_id_str = user_id if user_id else "Not Found"
    print(f"  User_ID: {user_id_str}")

    api_profile = fetch_user_profile(user_id) if user_id and not user_id.startswith("QR_DAMAGED") else {}
    api_name = f"{api_profile.get('first_name', '')} {api_profile.get('last_name', '')}".strip() if api_profile else ""
    api_city = api_profile.get("city", "") if api_profile else ""
    if api_name or api_city:
        print(f"  API Profile -> Name: {api_name!r}, City: {api_city!r}")

    # Crop QR sticker region (top-right 25% height, right 50% width) for inspection
    h_p1, w_p1 = p1_raw.shape[:2]
    qr_crop = p1_raw[0:int(h_p1 * 0.25), int(w_p1 * 0.50):w_p1]
    qr_crop_rel_path = "crops/QR_Sticker_ROI.png"
    cv2.imwrite(os.path.join(crops_dir, "QR_Sticker_ROI.png"), qr_crop)

    # -----------------------------------------------------------------------
    # Step 3: Page Alignment
    # -----------------------------------------------------------------------
    print("[Step 3/6] Aligning pages against template...")
    p1 = align_page(filled_imgs[0], tmpl_imgs[0])
    p2 = align_page(filled_imgs[1], tmpl_imgs[1])
    p3 = align_page(filled_imgs[2], tmpl_imgs[2]) if len(filled_imgs) >= 3 else None

    # Save aligned page images
    cv2.imwrite(os.path.join(out_dir, "Page_1_aligned.png"), p1)
    cv2.imwrite(os.path.join(out_dir, "Page_2_aligned.png"), p2)
    if p3 is not None:
        cv2.imwrite(os.path.join(out_dir, "Page_3_aligned.png"), p3)

    # Field list for Page 1
    p1_fields = [
        ("Date", C.PAGE1_DATE, (0, 165, 255)),
        ("Khoji_No", C.PAGE1_KHOJI, (0, 165, 255)),
        ("City", C.PAGE1_CITY, (0, 165, 255)),
        ("Name", C.PAGE1_NAME, (0, 165, 255)),
        ("Q1_Han", C.PAGE1_Q1_HAN_BOX, (255, 0, 0)),
        ("Q1_Na", C.PAGE1_Q1_NA_BOX, (255, 0, 0)),
        ("Q2_Han", C.PAGE1_Q2_HAN_BOX, (255, 0, 0)),
        ("Q2_Na", C.PAGE1_Q2_NA_BOX, (255, 0, 0)),
        ("P1_Q3", C.PAGE1_Q3_ROW, (0, 200, 0)),
        ("P1_Q4", C.PAGE1_Q4_ROW, (0, 200, 0)),
        ("P1_Q5", C.PAGE1_Q5_ROW, (0, 200, 0)),
        ("P1_Q6", C.PAGE1_Q6_ROW, (0, 200, 0)),
        ("P1_Q7", C.PAGE1_Q7_ROW, (0, 200, 0)),
        ("P1_Q8", C.PAGE1_Q8_ROW, (0, 200, 0)),
        ("P1_Q9", C.PAGE1_Q9_ROW, (0, 200, 0)),
        ("P1_Q10", C.PAGE1_Q10_ROW, (0, 200, 0)),
        ("P1_Q11", C.PAGE1_Q11_ROW, (0, 200, 0)),
        ("P1_Q12", C.PAGE1_Q12_ROW, (0, 200, 0)),
    ]

    p2_fields = [
        ("P2_Q1", C.PAGE2_Q1_ROW, (0, 200, 0)),
        ("P2_Q2", C.PAGE2_Q2_ROW, (0, 200, 0)),
        ("P2_Q3", C.PAGE2_Q3_ROW, (0, 200, 0)),
        ("P2_Q4", C.PAGE2_Q4_ROW, (0, 200, 0)),
        ("P2_Q5", C.PAGE2_Q5_ROW, (0, 200, 0)),
        ("P2_Q6", C.PAGE2_Q6_ROW, (0, 200, 0)),
        ("P2_Q7", C.PAGE2_Q7_ROW, (0, 200, 0)),
        ("P2_Q8", C.PAGE2_Q8_ROW, (0, 200, 0)),
        ("P2_Q9", C.PAGE2_Q9_ROW, (0, 200, 0)),
        ("P2_Q10", C.PAGE2_Q10_ROW, (0, 200, 0)),
        ("P2_Q11", C.PAGE2_Q11_ROW, (0, 200, 0)),
        ("P2_Q12", C.PAGE2_Q12_ROW, (0, 200, 0)),
        ("P2_Q13", C.PAGE2_Q13_ROW, (0, 200, 0)),
        ("P2_Q14_Text", C.PAGE2_Q14_TEXT, (128, 0, 128)),
    ]

    p3_fields = [
        ("P3_Q1", C.PAGE3_Q1_ROW, (0, 200, 0)),
        ("P3_Q2", C.PAGE3_Q2_ROW, (0, 200, 0)),
        ("P3_Q3", C.PAGE3_Q3_ROW, (0, 200, 0)),
        ("P3_Q4", C.PAGE3_Q4_ROW, (0, 200, 0)),
        ("P3_Q5", C.PAGE3_Q5_ROW, (0, 200, 0)),
        ("P3_Q6", C.PAGE3_Q6_ROW, (0, 200, 0)),
        ("P3_Q7", C.PAGE3_Q7_ROW, (0, 200, 0)),
        ("P3_Q8", C.PAGE3_Q8_ROW, (0, 200, 0)),
        ("P3_Q9", C.PAGE3_Q9_ROW, (0, 200, 0)),
        ("P3_Q10", C.PAGE3_Q10_ROW, (0, 200, 0)),
        ("P3_Q11", C.PAGE3_Q11_ROW, (0, 200, 0)),
        ("P3_Q12", C.PAGE3_Q12_ROW, (0, 200, 0)),
        ("P3_Q13", C.PAGE3_Q13_ROW, (0, 200, 0)),
    ]

    print("[Step 4/6] Annotating full pages with bounding box overlays...")
    p1_annotated = draw_bboxes_on_page(p1, p1_fields)
    p2_annotated = draw_bboxes_on_page(p2, p2_fields)
    cv2.imwrite(os.path.join(out_dir, "Page_1_annotated.png"), p1_annotated)
    cv2.imwrite(os.path.join(out_dir, "Page_2_annotated.png"), p2_annotated)
    if p3 is not None:
        p3_annotated = draw_bboxes_on_page(p3, p3_fields)
        cv2.imwrite(os.path.join(out_dir, "Page_3_annotated.png"), p3_annotated)

    print("[Step 5/6] Extracting individual crops and running field-by-field detections...")
    results_records = []

    def process_field(page_img, tmpl_img, name, bbox, field_type):
        x1, y1, x2, y2 = bbox
        crop = page_img[y1:y2, x1:x2]
        crop_filename = f"{name}.png"
        crop_rel_path = f"crops/{crop_filename}"
        crop_abs_path = os.path.join(crops_dir, crop_filename)
        cv2.imwrite(crop_abs_path, crop)

        val = ""
        mask_rel_path = ""
        flag = ""

        if field_type == "header_date":
            val = read_text_header(page_img, bbox, field_type="date")
        elif field_type == "header_khoji":
            val = read_text_header(page_img, bbox, field_type="khoji")
        elif field_type == "header_city":
            val = read_text_header(page_img, bbox, field_type="city")
        elif field_type == "header_name":
            val = read_text_header(page_img, bbox, field_type="name")
        elif field_type == "checkbox_q1":
            val = read_checkbox(page_img, tmpl_img, C.PAGE1_Q1_HAN_BOX, C.PAGE1_Q1_NA_BOX)
            if not val:
                flag = "Unanswered"
            elif "Yes" in val and "No" in val:
                flag = "Both Marked (Yes, No)"
        elif field_type == "checkbox_q2":
            val = read_checkbox(page_img, tmpl_img, C.PAGE1_Q2_HAN_BOX, C.PAGE1_Q2_NA_BOX)
            if not val:
                flag = "Unanswered"
            elif "Yes" in val and "No" in val:
                flag = "Both Marked (Yes, No)"
        elif field_type == "circled_number":
            val = read_circled_number(page_img, tmpl_img, bbox)
            if not val:
                flag = "Unanswered"
            elif "," in str(val):
                flag = f"Double Mark: {val}"
            mask_img = generate_ink_debug_mask(crop)
            if mask_img is not None:
                mask_filename = f"{name}_mask.png"
                mask_rel_path = f"debug_masks/{mask_filename}"
                cv2.imwrite(os.path.join(masks_dir, mask_filename), mask_img)
        elif field_type == "free_text":
            val = read_text(page_img, bbox)
            if not val:
                flag = "Unanswered"

        results_records.append({
            "name": name,
            "bbox": bbox,
            "val": val,
            "crop_path": crop_rel_path,
            "mask_path": mask_rel_path,
            "flag": flag,
        })
        flag_str = f" [FLAG: {flag}]" if flag else ""
        print(f"  Field [{name:12s}]: Extracted Value = {val!r}{flag_str}")

    # Process Page 1 Headers
    process_field(p1, tmpl_imgs[0], "Date", C.PAGE1_DATE, "header_date")
    process_field(p1, tmpl_imgs[0], "Khoji_No", C.PAGE1_KHOJI, "header_khoji")
    process_field(p1, tmpl_imgs[0], "City", C.PAGE1_CITY, "header_city")
    process_field(p1, tmpl_imgs[0], "Name", C.PAGE1_NAME, "header_name")

    # Process Page 1 Checkboxes
    process_field(p1, tmpl_imgs[0], "P1_Q1", C.PAGE1_Q1_HAN_BOX, "checkbox_q1")
    process_field(p1, tmpl_imgs[0], "P1_Q2", C.PAGE1_Q2_HAN_BOX, "checkbox_q2")

    # Process Page 1 Scale Questions
    for q, bbox in [
        ("P1_Q3", C.PAGE1_Q3_ROW), ("P1_Q4", C.PAGE1_Q4_ROW),
        ("P1_Q5", C.PAGE1_Q5_ROW), ("P1_Q6", C.PAGE1_Q6_ROW),
        ("P1_Q7", C.PAGE1_Q7_ROW), ("P1_Q8", C.PAGE1_Q8_ROW),
        ("P1_Q9", C.PAGE1_Q9_ROW), ("P1_Q10", C.PAGE1_Q10_ROW),
        ("P1_Q11", C.PAGE1_Q11_ROW), ("P1_Q12", C.PAGE1_Q12_ROW)
    ]:
        process_field(p1, tmpl_imgs[0], q, bbox, "circled_number")

    # Process Page 2 Scale Questions & Q14
    for q, bbox in [
        ("P2_Q1", C.PAGE2_Q1_ROW), ("P2_Q2", C.PAGE2_Q2_ROW),
        ("P2_Q3", C.PAGE2_Q3_ROW), ("P2_Q4", C.PAGE2_Q4_ROW),
        ("P2_Q5", C.PAGE2_Q5_ROW), ("P2_Q6", C.PAGE2_Q6_ROW),
        ("P2_Q7", C.PAGE2_Q7_ROW), ("P2_Q8", C.PAGE2_Q8_ROW),
        ("P2_Q9", C.PAGE2_Q9_ROW), ("P2_Q10", C.PAGE2_Q10_ROW),
        ("P2_Q11", C.PAGE2_Q11_ROW), ("P2_Q12", C.PAGE2_Q12_ROW),
        ("P2_Q13", C.PAGE2_Q13_ROW)
    ]:
        process_field(p2, tmpl_imgs[1], q, bbox, "circled_number")

    # Process Page 2 Q14 (Free Text Feedback with updated bounds)
    process_field(p2, tmpl_imgs[1], "P2_Q14", C.PAGE2_Q14_TEXT, "free_text")

    # Process Page 3 if present
    if p3 is not None and len(tmpl_imgs) >= 3:
        for q, bbox in [
            ("P3_Q1", C.PAGE3_Q1_ROW), ("P3_Q2", C.PAGE3_Q2_ROW),
            ("P3_Q3", C.PAGE3_Q3_ROW), ("P3_Q4", C.PAGE3_Q4_ROW),
            ("P3_Q5", C.PAGE3_Q5_ROW), ("P3_Q6", C.PAGE3_Q6_ROW),
            ("P3_Q7", C.PAGE3_Q7_ROW), ("P3_Q8", C.PAGE3_Q8_ROW),
            ("P3_Q9", C.PAGE3_Q9_ROW), ("P3_Q10", C.PAGE3_Q10_ROW),
            ("P3_Q11", C.PAGE3_Q11_ROW), ("P3_Q12", C.PAGE3_Q12_ROW),
            ("P3_Q13", C.PAGE3_Q13_ROW)
        ]:
            process_field(p3, tmpl_imgs[2], q, bbox, "circled_number")

    # Overall Flags list
    all_flags = [f"{r['name']} ({r['flag']})" for r in results_records if r.get('flag')]
    if user_id_str.startswith("QR_DAMAGED"):
        all_flags.insert(0, f"QR Code ({user_id_str})")
    elif user_id_str == "Not Found":
        all_flags.insert(0, "QR Code (Not Found)")

    print("\n[Step 6/6] Generating HTML Inspection Report...")
    html_path = os.path.join(out_dir, "verification_report.html")

    flags_badges = "".join([f"<span class='badge badge-warn' style='margin-right:6px;'>{fl}</span>" for fl in all_flags]) if all_flags else "<span class='badge badge-value'>None</span>"

    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>Pipeline Verification Report - {pdf_name}</title>",
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f4f6f9; color: #333; margin: 0; padding: 20px; }",
        "  h1, h2, h3 { color: #1e293b; }",
        "  .header-card { background: white; padding: 22px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); margin-bottom: 25px; }",
        "  .qr-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-top: 15px; display: flex; gap: 20px; align-items: center; }",
        "  .qr-img { max-height: 120px; max-width: 200px; border: 1px solid #cbd5e1; border-radius: 6px; }",
        "  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin-bottom: 30px; }",
        "  .card { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.06); }",
        "  .card img { max-width: 100%; border: 1px solid #e2e8f0; border-radius: 4px; }",
        "  table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.06); margin-bottom: 30px; }",
        "  th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid #e2e8f0; vertical-align: middle; }",
        "  th { background: #0f172a; color: white; font-weight: 600; }",
        "  tr:hover { background: #f8fafc; }",
        "  tr.flagged { background: #fffbeb; }",
        "  .crop-img { max-height: 55px; max-width: 280px; border: 1px solid #cbd5e1; border-radius: 4px; }",
        "  .mask-img { max-height: 55px; max-width: 500px; border: 1px solid #cbd5e1; border-radius: 4px; }",
        "  .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.9em; background: #e0f2fe; color: #0369a1; }",
        "  .badge-value { background: #dcfce7; color: #15803d; font-size: 1.05em; padding: 4px 12px; }",
        "  .badge-warn { background: #fee2e2; color: #b91c1c; font-size: 0.95em; padding: 4px 10px; }",
        "</style>",
        "</head>",
        "<body>",
        "  <div class='header-card'>",
        f"    <h1>Pipeline Verification Report</h1>",
        f"    <p><strong>Form PDF:</strong> <code>{os.path.basename(pdf_path)}</code> | <strong>Status:</strong> Verification Completed</p>",
        f"    <div class='qr-card'>",
        f"      <img class='qr-img' src='{qr_crop_rel_path}' alt='QR Sticker ROI'>",
        f"      <div>",
        f"        <p style='margin: 4px 0;'><strong>User ID (QR):</strong> <code>{user_id_str}</code></p>",
        f"        <p style='margin: 4px 0;'><strong>API Name:</strong> <span class='badge badge-value'>{api_name if api_name else '[Not Found]'}</span></p>",
        f"        <p style='margin: 4px 0;'><strong>API City:</strong> <span class='badge badge-value'>{api_city if api_city else '[Not Found]'}</span></p>",
        f"        <p style='margin: 6px 0 0 0;'><strong>Flagged Issues:</strong> {flags_badges}</p>",
        f"      </div>",
        f"    </div>",
        "  </div>",
        "",
        "  <h2>1. Full Page Bounding Box Overlays</h2>",
        "  <div class='grid'>",
        f"    <div class='card'><h3>Page 1 Overlays</h3><img src='Page_1_annotated.png' alt='Page 1 Annotations'></div>",
        f"    <div class='card'><h3>Page 2 Overlays</h3><img src='Page_2_annotated.png' alt='Page 2 Annotations'></div>",
        f"    " + (f"<div class='card'><h3>Page 3 Overlays</h3><img src='Page_3_annotated.png' alt='Page 3 Annotations'></div>" if p3 is not None else ""),
        "  </div>",
        "",
        "  <h2>2. Extracted Crops & Field Detections</h2>",
        "  <table>",
        "    <thead>",
        "      <tr>",
        "        <th>Field Name</th>",
        "        <th>Bounding Box (x1,y1,x2,y2)</th>",
        "        <th>Raw Crop Image</th>",
        "        <th>Detected Ink Mask (Side-by-Side)</th>",
        "        <th>Extracted Value</th>",
        "        <th>Status / Flag</th>",
        "      </tr>",
        "    </thead>",
        "    <tbody>"
    ]

    for rec in results_records:
        bbox_str = f"({rec['bbox'][0]}, {rec['bbox'][1]}, {rec['bbox'][2]}, {rec['bbox'][3]})"
        val_display = f"<span class='badge badge-value'>{rec['val']}</span>" if rec['val'] else "<span style='color:#94a3b8;'>[Blank]</span>"
        flag_display = f"<span class='badge badge-warn'>{rec['flag']}</span>" if rec['flag'] else "<span style='color:#15803d;'>OK</span>"
        row_cls = "class='flagged'" if rec['flag'] else ""

        crop_html = f"<img class='crop-img' src='{rec['crop_path']}' alt='{rec['name']}'>" if rec['crop_path'] else "-"
        mask_html = f"<img class='mask-img' src='{rec['mask_path']}' alt='{rec['name']} Mask'>" if rec['mask_path'] else "<span style='color:#94a3b8;'>N/A</span>"

        html_lines.append(f"      <tr {row_cls}>")
        html_lines.append(f"        <td><strong>{rec['name']}</strong></td>")
        html_lines.append(f"        <td><code>{bbox_str}</code></td>")
        html_lines.append(f"        <td>{crop_html}</td>")
        html_lines.append(f"        <td>{mask_html}</td>")
        html_lines.append(f"        <td>{val_display}</td>")
        html_lines.append(f"        <td>{flag_display}</td>")
        html_lines.append("      </tr>")

    html_lines.extend([
        "    </tbody>",
        "  </table>",
        "</body>",
        "</html>"
    ])

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))

    print(f"\n{'='*70}")
    print(f" VERIFICATION COMPLETE!")
    print(f" Report Directory:     {out_dir}")
    print(f" Full Page Overlays:   Page_1_annotated.png, Page_2_annotated.png")
    print(f" Individual Field Crops:{crops_dir}")
    print(f" Circle Ink Masks:     {masks_dir}")
    print(f" HTML Visual Dashboard: {html_path}")
    print(f"{'='*70}\n")


def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Search candidate test PDFs in project
        candidates = (
            glob.glob(os.path.join(PROJECT_ROOT, "input", "*.pdf")) +
            glob.glob(os.path.join(PROJECT_ROOT, "unused_archive", "test_pdfs", "*.pdf")) +
            glob.glob(os.path.join(PROJECT_ROOT, "unused_archive", "test_pdfs", "*", "*.pdf"))
        )
        if candidates:
            pdf_path = candidates[0]
            print(f"No PDF path specified. Using candidate: {pdf_path}")
        else:
            print("Usage: python verify_pipeline_crops.py <path_to_pdf>")
            return

    verify_pdf(pdf_path)


if __name__ == "__main__":
    main()
