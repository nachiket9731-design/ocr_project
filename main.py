"""
main.py - Entry point for the PDF Form Data Extraction tool.

Usage:
    python main.py                  # processes all PDFs in ./input/ folder
    python main.py path/to/file.pdf # processes a single PDF file

Output:
    output.csv  (appended row per form, created if not exists)

CSV columns (43 total):
    Date, Khoji_No, City, Name,
    P1_Q1, P1_Q2,
    P1_Q3..P1_Q12,
    P2_Q1..P2_Q13, P2_Q14,
    P3_Q1..P3_Q13
"""

import sys
import os

# Force UTF-8 output so Hindi text prints correctly on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import glob
import pandas as pd
import numpy as np
import cv2
import base64

from align import pdf_to_images, align_page
import config as C
from detect_qr import extract_qr_user_id
from detect import (
    extract_all_text_fields_single_call,
    read_text,
    read_text_header,
    read_checkbox,
    read_circled_number,
    read_q14_google_vision
)

# ---------------------------------------------------------------------------
# API Integration
# ---------------------------------------------------------------------------
import json
import urllib.request
import urllib.error
import ssl

def fetch_user_profile(user_id: str) -> dict:
    """Fetch user name and city from the backend API using the UUID."""
    if not user_id or user_id.startswith("QR_DAMAGED") or user_id == "Not Found":
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLANK_PDF    = os.path.join(os.path.dirname(__file__), "Nisarg form_Nov 2025.pdf")
INPUT_FOLDER = os.path.join(os.path.dirname(__file__), "input")
OUTPUT_CSV   = os.path.join(os.path.dirname(__file__), "output.csv")

CSV_COLUMNS = [
    "Source_File", "User_ID",
    "Date", "Khoji_No", "City", "Name",
    "P1_Q1", "P1_Q2",
    "P1_Q3",  "P1_Q4",  "P1_Q5",  "P1_Q6",
    "P1_Q7",  "P1_Q8",  "P1_Q9",  "P1_Q10",
    "P1_Q11", "P1_Q12",
    "P2_Q1",  "P2_Q2",  "P2_Q3",  "P2_Q4",
    "P2_Q5",  "P2_Q6",  "P2_Q7",  "P2_Q8",
    "P2_Q9",  "P2_Q10", "P2_Q11", "P2_Q12",
    "P2_Q13", "P2_Q14",
    "Flagged_Fields"
]


# ---------------------------------------------------------------------------
# Per-form extraction
# ---------------------------------------------------------------------------
def extract_form(filled_pdf_path: str, template_images: list) -> dict:
    """Extract all 43 fields from a single 3-page filled PDF form."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(filled_pdf_path)}")

    row = {col: "" for col in CSV_COLUMNS}
    row["Source_File"] = os.path.basename(filled_pdf_path)

    # Convert filled PDF to images
    filled_images = pdf_to_images(filled_pdf_path)

    if len(filled_images) < 2:
        print("  [ERROR] PDF has fewer than 2 pages. Skipping.")
        return row

    # Render & align all 3 pages
    print("  Aligning pages...")
    p1 = align_page(filled_images[0], template_images[0])
    p2 = align_page(filled_images[1], template_images[1])
   

    p1_raw  = filled_images[0]
    p2_raw  = filled_images[1]
    p1_tmpl = template_images[0]
    p2_tmpl = template_images[1]
    p3_tmpl = template_images[2]

    # -----------------------------------------------------------------------
    # QR CODE EXTRACTION & API LOOKUP
    # -----------------------------------------------------------------------
    print("\n  --- Extracting QR Code ---")
    user_id = extract_qr_user_id(p1_raw, debug=False)
    row["User_ID"] = user_id if user_id else "Not Found"
    print(f"  User_ID: {row['User_ID']}")

    row["Name"] = ""
    row["City"] = ""
    row["Date"] = ""
    row["Khoji_No"] = ""

    if user_id and not user_id.startswith("QR_DAMAGED"):
        print("  --- Fetching User Profile from API ---")
        profile = fetch_user_profile(user_id)
        if profile:
            first_name = profile.get("first_name", "")
            last_name = profile.get("last_name", "")
            row["Name"] = f"{first_name} {last_name}".strip()
            row["City"] = profile.get("city", "")
            print(f"  API Name: {row['Name']}")
            print(f"  API City: {row['City']}")
        else:
            print("  [WARN] Could not fetch profile data or profile is empty.")

    # -----------------------------------------------------------------------
    # TEXT EXTRACTION (Page 2 - Q14 Free Text via Google Cloud Vision API)
    # Extracts P2_Q14. Date, Khoji_No, City, Name are handled by QR integration.
    # -----------------------------------------------------------------------
    print("\n  --- Extracting P2_Q14 (Google Cloud Vision API) ---")
    row["P2_Q14"] = read_q14_google_vision(p2, C.PAGE2_Q14_TEXT)
    print(f"  P2_Q14: {row['P2_Q14']!r}")

    # Attach Q14 crop for UI preview
    qx1, qy1, qx2, qy2 = C.PAGE2_Q14_TEXT
    q14_crop_img = p2[qy1:qy2, qx1:qx2]
    if q14_crop_img.size > 0:
        succ, buf = cv2.imencode('.png', q14_crop_img)
        if succ:
            row["q14_crop"] = base64.b64encode(buf).decode('utf-8')

    # -----------------------------------------------------------------------
    # PAGE 1 (Checkboxes & Circled Numbers - 0 Credits / OpenCV Local)
    # -----------------------------------------------------------------------
    print("\n  --- Page 1 Questions ---")
    row["P1_Q1"] = read_checkbox(p1, p1_tmpl, C.PAGE1_Q1_HAN_BOX, C.PAGE1_Q1_NA_BOX)
    print(f"  P1_Q1 (Yes/No): {row['P1_Q1']!r}")

    row["P1_Q2"] = read_checkbox(p1, p1_tmpl, C.PAGE1_Q2_HAN_BOX, C.PAGE1_Q2_NA_BOX)
    print(f"  P1_Q2 (Yes/No): {row['P1_Q2']!r}")

    p1_range_map = {
        "P1_Q3":  C.PAGE1_Q3_ROW,  "P1_Q4":  C.PAGE1_Q4_ROW,
        "P1_Q5":  C.PAGE1_Q5_ROW,  "P1_Q6":  C.PAGE1_Q6_ROW,
        "P1_Q7":  C.PAGE1_Q7_ROW,  "P1_Q8":  C.PAGE1_Q8_ROW,
        "P1_Q9":  C.PAGE1_Q9_ROW,  "P1_Q10": C.PAGE1_Q10_ROW,
        "P1_Q11": C.PAGE1_Q11_ROW, "P1_Q12": C.PAGE1_Q12_ROW,
    }
    for key, bbox in p1_range_map.items():
        val = read_circled_number(p1, p1_tmpl, bbox)
        row[key] = val
        print(f"  {key}: {val!r}")

    # -----------------------------------------------------------------------
    # PAGE 2 (Circled Numbers - 0 Credits / OpenCV Local)
    # -----------------------------------------------------------------------
    print("\n  --- Page 2 Questions ---")
    p2_range_map = {
        "P2_Q1":  C.PAGE2_Q1_ROW,  "P2_Q2":  C.PAGE2_Q2_ROW,
        "P2_Q3":  C.PAGE2_Q3_ROW,  "P2_Q4":  C.PAGE2_Q4_ROW,
        "P2_Q5":  C.PAGE2_Q5_ROW,  "P2_Q6":  C.PAGE2_Q6_ROW,
        "P2_Q7":  C.PAGE2_Q7_ROW,  "P2_Q8":  C.PAGE2_Q8_ROW,
        "P2_Q9":  C.PAGE2_Q9_ROW,  "P2_Q10": C.PAGE2_Q10_ROW,
        "P2_Q11": C.PAGE2_Q11_ROW, "P2_Q12": C.PAGE2_Q12_ROW,
        "P2_Q13": C.PAGE2_Q13_ROW,
    }
    for key, bbox in p2_range_map.items():
        val = read_circled_number(p2, p2_tmpl, bbox)
        row[key] = val
        print(f"  {key}: {val!r}")

   

    # -----------------------------------------------------------------------
    # FLAGGING (Double Marks & Unanswered Fields)
    # -----------------------------------------------------------------------
    flags = []

    # 1. Header fields
    # for hdr in ["Date", "Khoji_No", "City", "Name"]:
    #     if not str(row.get(hdr, "")).strip():
    #         flags.append(f"{hdr} (Missing)")

    # 2. Checkboxes
    for q_name in ["P1_Q1", "P1_Q2"]:
        val = str(row.get(q_name, "")).strip()
        if not val:
            flags.append(f"{q_name} (Unanswered)")
        elif "Yes, No" in val or ("Yes" in val and "No" in val):
            flags.append(f"{q_name} (Both Marked: Yes, No)")

    # 3. Scale questions (1-10)
    scale_keys = (
        [f"P1_Q{i}" for i in range(3, 13)] +
        [f"P2_Q{i}" for i in range(1, 14)] 
       
        )
    for q_name in scale_keys:
        val = str(row.get(q_name, "")).strip()
        if not val:
            flags.append(f"{q_name} (Unanswered)")
        elif "," in val:
            flags.append(f"{q_name} (Double Mark: {val})")

    # 4. Written answer (P2_Q14)
    if not str(row.get("P2_Q14", "")).strip():
        flags.append("P2_Q14 (Unanswered)")

    row["Flagged_Fields"] = "; ".join(flags) if flags else "None"
    if flags:
        print(f"\n  [FLAGGED] Issues detected in: {row['Flagged_Fields']}")

    return row


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def append_to_csv(row: dict, csv_path: str):
    """Append or update a single row dict in the CSV without creating duplicates."""
    df_new = pd.DataFrame([row], columns=CSV_COLUMNS)
    for target_path in [csv_path, csv_path.replace(".csv", "_backup.csv")]:
        try:
            if not os.path.exists(target_path):
                df_new.to_csv(target_path, index=False, encoding="utf-8-sig")
                print(f"\n  [CSV] Created {target_path}")
            else:
                df_existing = pd.read_csv(target_path, encoding="utf-8-sig")
                if "Source_File" in df_existing.columns and row["Source_File"] in df_existing["Source_File"].values:
                    df_existing = df_existing[df_existing["Source_File"] != row["Source_File"]]
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(target_path, index=False, encoding="utf-8-sig")
                    print(f"\n  [CSV] Updated {row['Source_File']} in {target_path}")
                else:
                    df_new.to_csv(target_path, mode="a", index=False, header=False, encoding="utf-8-sig")
                    print(f"\n  [CSV] Appended {row['Source_File']} to {target_path}")
            return
        except PermissionError:
            print(f"\n  [WARN] '{target_path}' is locked (e.g., open in Excel).")
            if target_path == csv_path:
                print("  [WARN] Saving copy to backup CSV instead...")
        except Exception as e:
            print(f"\n  [WARN] Failed to write to {target_path}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Determine which PDF(s) to process
    if len(sys.argv) > 1:
        pdf_files = sys.argv[1:]
    else:
        os.makedirs(INPUT_FOLDER, exist_ok=True)
        pdf_files = glob.glob(os.path.join(INPUT_FOLDER, "*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in '{INPUT_FOLDER}'.")
            print("Usage: python main.py <path_to_filled.pdf> [more_pdfs...]")
            print(f"   OR: Place PDFs in the '{INPUT_FOLDER}' folder and run without arguments.")
            return

    # Load blank template once
    print(f"Loading blank template: {BLANK_PDF}")
    if not os.path.exists(BLANK_PDF):
        print(f"[ERROR] Blank template not found: {BLANK_PDF}")
        return
    template_images = pdf_to_images(BLANK_PDF)
    print(f"Template loaded ({len(template_images)} pages).")

    # Process each filled PDF
    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"[WARN] File not found: {pdf_path} — skipping.")
            continue
        try:
            row = extract_form(pdf_path, template_images)
            append_to_csv(row, OUTPUT_CSV)
        except Exception as e:
            print(f"[ERROR] Failed to process {pdf_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
