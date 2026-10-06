"""
align.py - Aligns a filled form page to the blank template page using OpenCV.

This corrects for any scanning skew, rotation, translation or zoom so that
our fixed bounding boxes always land on the correct answer regions.
"""

import cv2
import numpy as np
import fitz  # PyMuPDF
import os
from config import ZOOM


def pdf_to_images(pdf_path: str, zoom: int = ZOOM) -> list[np.ndarray]:
    """
    Convert each page of a PDF to a BGR numpy image.

    Args:
        pdf_path: Absolute path to the PDF file.
        zoom:     Render zoom factor (default = ZOOM from config).

    Returns:
        List of BGR images, one per page.
    """
    doc = fitz.open(pdf_path)
    images = []
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        # pix.samples is RGB bytes → convert to BGR for OpenCV
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    doc.close()
    return images


def align_page(filled_img: np.ndarray, template_img: np.ndarray) -> np.ndarray:
    """
    Align filled_img to template_img using ORB feature matching + homography.

    If alignment fails (not enough matches), returns the original filled_img
    resized to match the template's dimensions.

    Args:
        filled_img:   The scanned/photographed filled form page (BGR).
        template_img: The blank template page rendered from the blank PDF (BGR).

    Returns:
        Aligned version of filled_img, same shape as template_img.
    """
    h_tmpl, w_tmpl = template_img.shape[:2]

    # Convert to grayscale
    gray_filled   = cv2.cvtColor(filled_img,   cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    # Detect ORB features
    orb = cv2.ORB_create(nfeatures=3000)
    kp1, des1 = orb.detectAndCompute(gray_filled,   None)
    kp2, des2 = orb.detectAndCompute(gray_template, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print("  [WARN] Not enough keypoints for alignment — using resize only.")
        return cv2.resize(filled_img, (w_tmpl, h_tmpl))

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    # Keep top matches
    good = matches[:min(200, len(matches))]

    if len(good) < 10:
        print("  [WARN] Too few good matches for alignment — using resize only.")
        return cv2.resize(filled_img, (w_tmpl, h_tmpl))

    # Extract point pairs
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Compute homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        print("  [WARN] Homography failed — using resize only.")
        return cv2.resize(filled_img, (w_tmpl, h_tmpl))

    # Warp the filled image to match the template
    aligned = cv2.warpPerspective(filled_img, M, (w_tmpl, h_tmpl))
    return aligned
