"""
app.py — FormScanner local web server.

Usage:
    pip install flask
    python app.py

Opens at: http://localhost:5000

What this does:
  - Serves the review UI (templates/index.html)
  - Accepts a PDF upload, runs your existing extract_form() from main.py
  - Renders Page 1 & Page 2 of the PDF to base64 PNGs for the browser preview
  - Returns extracted fields as JSON → displayed in the editable right panel
  - Does NOT write to output.csv (CSV is generated client-side on Download)
"""

import os
import sys
import base64
import tempfile
import traceback

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz  # PyMuPDF — already in requirements.txt
from flask import Flask, render_template, request, jsonify

from main import extract_form, CSV_COLUMNS
from align import pdf_to_images

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload size

# ---------------------------------------------------------------------------
# Blank template — loaded once at startup so first request is fast
# ---------------------------------------------------------------------------
BLANK_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Nisarg form_Nov 2025.pdf")
_template_images = None


def get_template_images():
    """Load and cache the blank template PDF pages."""
    global _template_images
    if _template_images is None:
        if not os.path.exists(BLANK_PDF):
            raise FileNotFoundError(
                f"Blank template PDF not found at: {BLANK_PDF}\n"
                "Make sure 'Nisarg form_Nov 2025.pdf' is in the project folder."
            )
        print("  Loading blank template PDF...")
        _template_images = pdf_to_images(BLANK_PDF)
        print(f"  Template loaded ({len(_template_images)} pages).")
    return _template_images


# ---------------------------------------------------------------------------
# Helper: render first N pages of a PDF to base64 PNG strings
# ---------------------------------------------------------------------------
def pdf_pages_to_base64(pdf_path: str, page_count: int = 2) -> list:
    """
    Render the first `page_count` pages of a PDF at 2x zoom.
    Returns a list of base64-encoded PNG strings for browser display.
    """
    doc = fitz.open(pdf_path)
    images = []
    mat = fitz.Matrix(2, 2)  # 2x render → crisp but not huge
    for i in range(min(page_count, len(doc))):
        pix = doc[i].get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        images.append(base64.b64encode(img_bytes).decode("utf-8"))
    doc.close()
    return images


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve the main review UI."""
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """
    Accept a PDF upload, run the OCR pipeline, and return:
      {
        "status": "success",
        "fields": { <CSV_COLUMNS dict> },
        "pages":  [ "<base64_page1>", "<base64_page2>" ],
        "filename": "<original filename>"
      }
    """
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded. Use the key 'pdf'."}), 400

    file = request.files["pdf"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted."}), 400

    # Save to a temp file so the pipeline can read it from disk
    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)
    file.save(tmp_path)

    try:
        tmpl = get_template_images()

        # Run your existing OCR pipeline — zero changes to extract_form()
        row = extract_form(tmp_path, tmpl)

        # Render pages for the browser preview
        page_images = pdf_pages_to_base64(tmp_path, page_count=2)

        return jsonify({
            "status":   "success",
            "fields":   row,
            "pages":    page_images,
            "filename": file.filename,
        })

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n[ERROR] {e}\n{tb}")
        return jsonify({"error": str(e), "traceback": tb}), 500

    finally:
        # Always clean up the temp file
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  FormScanner OCR — Local Review UI")
    print("  Open your browser at: http://localhost:5000")
    print("=" * 55)

    # Pre-load the template at startup so the first upload is fast
    try:
        get_template_images()
    except FileNotFoundError as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)

    print("\nServer starting... Press Ctrl+C to stop.\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
