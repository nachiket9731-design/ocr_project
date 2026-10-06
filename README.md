# Handwritten Form OCR & Survey Extraction Pipeline

An automated, **100% offline and local** computer vision & OCR pipeline designed to extract handwritten fields, checkboxes, and rating-scale marks from multi-page scanned/photographed survey forms into structured CSV records.

---

## 🌟 Key Features

- **Robust Computer Vision Alignment**:
  Uses OpenCV ORB feature detection and Homography warping to mathematically align scanned or photographed forms against standard blank templates, correcting for tilt, rotation, and slight scaling variations.
- **100% Local & Offline**:
  Zero cloud dependencies, external API calls, or recurring subscription costs. Runs entirely on local machine CPU/GPU using EasyOCR and OpenCV.
- **Dual-Model OCR Strategy**:
  - **English Model**: Dedicated reader for header fields (`Date`, `Khoji No`, `City`, `Name`) to prevent Roman handwriting from being misinterpreted as Devanagari characters.
  - **Hindi + English Model**: Dedicated reader for free-form handwritten responses (e.g., `Q14`).
  - **Image Enhancement**: Automated contrast enhancement (CLAHE) and sharpening prior to text recognition.
- **Mark & Rating Detection**:
  - **Binary Yes/No Checkboxes**: Evaluated via thresholded dark pixel density per box.
  - **Circled Numbers (1–10 Scale)**: Template subtraction (Absolute Difference) and contour analysis automatically locate handwritten circles and map them to their corresponding numerical rating.
- **Accuracy Auditing**:
  Includes `check_accuracy.py` to benchmark predictions against ground truth datasets (`ground_truth.csv`), generating structured summary metrics and visual audit reports.
- **Interactive Web App**:
  Includes a local Flask-based web interface (`app.py`) for PDF uploads, visual review of form crops, and side-by-side verification.

---

## 📂 Project Structure

```text
ocr_project/
├── align.py                  # PDF-to-image conversion & ORB homography alignment
├── app.py                    # Local Flask web app for form review & upload
├── check_accuracy.py         # Benchmarking script against ground truth
├── config.py                 # Coordinate bounding boxes & configuration constants
├── detect.py                 # OCR extraction & CV mark detection algorithms
├── detect_qr.py              # QR code detection & helper utilities
├── ground_truth.csv          # Benchmark dataset for accuracy testing
├── main.py                   # Main CLI entry point to process form PDFs
├── project_documentation.md  # Detailed system architecture documentation
├── requirements.txt          # Python library dependencies
├── templates/                # Blank reference form templates & HTML UI assets
└── input/                    # Target PDF forms for processing
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ (64-bit recommended)
- `pip`

### 2. Installation
Clone the repository and install dependencies:

```bash
# Clone the repository (once pushed to GitHub)
git clone https://github.com/<your-username>/ocr_project.git
cd ocr_project

# Install required Python packages
pip install -r requirements.txt
```

---

## 💻 Usage

### Command-Line Processing (Batch / Single File)
To process a filled PDF form and append the extracted record to `output.csv`:

```bash
python main.py "input/<your_form>.pdf"
```

### Running the Web Interface
To review forms with an interactive UI in your browser:

```bash
python app.py
```
Then navigate to `http://localhost:5000` in your browser to upload PDFs and inspect extracted values.

### Evaluating Accuracy
To test the pipeline's extraction performance against labeled ground truth:

```bash
python check_accuracy.py
```
This evaluates field-by-field accuracy and generates `accuracy_summary.json` and `accuracy_report.html`.

---

## 📊 Output Schema

The pipeline exports extracted records into `output.csv` with standard columns:
- **Metadata / Headers**: `Source_File`, `Date`, `Khoji_No`, `City`, `Name`
- **Page 1**: `P1_Q1`, `P1_Q2` (Yes/No), `P1_Q3` through `P1_Q12` (Rating 1–10)
- **Page 2**: `P2_Q1` through `P2_Q13` (Rating 1–10), `P2_Q14` (Handwritten text)
- **Page 3**: `P3_Q1` through `P3_Q13` (Rating 1–10)

---

## 🛠️ Architecture & Documentation
For in-depth details on homography alignment, coordinate calculation, and contour filtering, refer to [project_documentation.md](project_documentation.md).
