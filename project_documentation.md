# OCR Project Documentation

## 1. Project Overview
The goal of this project is to automate the extraction of handwritten data from a scanned/photographed 3-page form into a structured CSV file. The form is primarily in the Hindi language, and respondents fill it out using a pen. 

The tool processes the filled forms, aligns them perfectly against a blank template, extracts specific fields (text, checkboxes, and circled numbers), and appends the extracted data to an output file (`output.csv`).

## 2. Technology Stack
*   **Python 3**: Core language used for the logic.
*   **PyMuPDF (fitz)**: Used for rendering PDF files into high-resolution images for processing.
*   **OpenCV (`cv2`)**: Used for image alignment, cropping, template subtraction, and contour detection (checkboxes and circled numbers).
*   **EasyOCR**: Deep-learning OCR engine running 100% locally and offline:
    *   *English-only reader*: Tailored for header fields (`Date`, `Khoji No`, `City`, `Name`) to prevent misclassifying Latin characters as Devanagari.
    *   *Hindi + English reader*: Tailored for free-text fields (`Q14`).
    *   Includes image enhancement preprocessing (CLAHE contrast normalization and sharpening).
*   **Pandas & NumPy**: Used for matrix operations, structuring extracted fields, and saving to CSV.
*   **Zero External API Dependencies**: Entire pipeline operates completely offline with no cloud API keys, credentials, or billing required.

## 3. Project File Structure
The project consists of the following key Python files:

### `main.py`
This is the entry point of the application. It orchestrates the entire pipeline:
*   Parses command-line arguments (target PDF files).
*   Converts the blank template PDF to images.
*   Loops through each page of the filled PDF, calling `align_page()` to align it.
*   Calls specific reading functions (like `read_text`, `read_checkbox`, `read_circled_number`) from `detect.py` using bounding boxes defined in `config.py`.
*   Assembles the final dictionary of results and appends it to `output.csv`.

### `config.py`
This file acts as a central registry for all coordinates (bounding boxes). Since the PDFs are rendered at a 3x zoom (resulting in 1786 x 2526 pixel images), all coordinates here are absolute pixel coordinates corresponding to that resolution:
*   Defines bounding boxes for header fields (`Date`, `Khoji`, `City`, `Name`) on all 3 pages.
*   Defines bounding boxes for "Yes/No" checkboxes (Page 1, Q1 and Q2).
*   Defines bounding boxes for the rows containing circled numbers 1-10 (Questions 3-13).

### `align.py`
Handles the translation of PDFs to images and corrects scanning imperfections:
*   `pdf_to_images()`: Uses PyMuPDF with a custom scale matrix to render the PDF pages into BGR numpy arrays.
*   `align_page()`: Corrects rotation, skew, and zoom of the filled form relative to the blank template. It does this by extracting ORB features from both images, finding matches, and computing a Homography matrix to warp the filled image perfectly over the template.

### `detect.py`
Contains the computer-vision and local OCR extraction logic for individual crops:
*   `read_text()` & `read_text_header()`: Uses multi-pass local EasyOCR with CLAHE contrast enhancement and sharpening. Employs dedicated readers (English for headers, Hindi+English for general text) to optimize character recognition without external APIs.
*   `read_checkbox()`: Converts the crop to grayscale, applies a binary threshold, and checks the percentage of dark pixels. If the box is filled heavily compared to a blank threshold, it counts as "checked".
*   `read_circled_number()`: Takes a row crop containing numbers 1-10. It compares the filled crop against the blank template crop to highlight the pen marks using Absolute Difference. It then finds contours, filters them by area and aspect ratio (to find the circle), and calculates which number was circled based on the x-coordinate of the circle's center.

## 4. Extraction Process Flow

1.  **Input Registration**: The user runs `python main.py <Filled_Form.pdf>`.
2.  **PDF Rendering**: The blank template (`templates/` or `Nisarg form_Nov 2025.pdf`) and the filled PDF are rendered into high-resolution images.
3.  **Image Alignment**: For each of the 3 pages, OpenCV matches structural features of the filled page to the blank page, mathematically warping the filled page so it exactly overlays the blank one. This ensures that coordinates in `config.py` are perfectly accurate regardless of how the form was scanned.
4.  **Field Extraction (Page 1)**:
    *   The top headers (Date, Khoji No, City, Name) are cropped and processed by EasyOCR (English reader).
    *   Q1 and Q2 are cropped specifically at the "Han" (Yes) and "Na" (No) boxes. Pixel density determines the answer.
    *   Q3-Q12 rows are cropped. Contour detection finds the hand-drawn circle and maps it to a score (1-10).
5.  **Field Extraction (Page 2)**:
    *   Headers are scanned (only updated if Page 1's headers were empty).
    *   Q1-Q13 rows are processed for circled numbers.
    *   Q14's large text box is cropped and processed by EasyOCR (Hindi+English reader).
6.  **Field Extraction (Page 3)**:
    *   Q1-Q13 rows are processed for circled numbers.
7.  **Data Export**: A Pandas DataFrame is constructed with 43 columns (including `Source_File`) and appended to `output.csv`.

## 5. Post-Processing Verification & Accuracy Checker (`check_accuracy.py`)

After running `python main.py` to extract forms into `output.csv`, you can evaluate the extraction accuracy against ground-truth data:
```bash
python check_accuracy.py
```
This tool:
1. **Compares `output.csv` against `ground_truth.csv`**:
   - **Checkboxes**: Evaluates Yes/No decisions.
   - **Scale Questions (1–10)**: Calculates Strict Exact Match, Tolerant (+/- 1) Match, False Double-Marks, and Missed Circles.
   - **Written Text (Q14)**: Computes character-level Levenshtein similarity.
2. **Generates an Interactive HTML Report (`accuracy_report.html`)**:
   - Displays KPI cards, form-by-form progress bars, and an interactive field-by-field audit table with zoomable crop previews.
3. **Exports JSON Summary (`accuracy_summary.json`)**:
   - Machine-readable metrics for automated benchmarks.

## 6. How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute extraction**:
   ```bash
   python main.py "input/<sample_form>.pdf"
   ```
3. **Verify results**:
   - Extracted data is saved in `output.csv`.
   - Run `python check_accuracy.py` to compare against ground truth.

## 7. Known Limitations & Future Improvements
*   **Handwriting Variability in Free Text**: The system relies on local EasyOCR models. While headers and clean handwriting are extracted reliably, cursive Hindi handwriting in Q14 can show variability. Cloud Vision/LLM APIs can be introduced in the future as an optional plug-in if higher transcription accuracy is required.
*   **Double-Mark Contour Sensitivity**: Hand-drawn circles can occasionally produce secondary stray contours if the respondent hesitates or re-traces a stroke. Secondary mark suppression is implemented to select the primary marked slot.

