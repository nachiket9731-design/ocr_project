"""
config.py - Bounding box coordinates for all form fields.

All coordinates are for images rendered at 3x zoom from the PDF.
Image size: 1786 x 2526 px
Format for text/checkbox fields: (x1, y1, x2, y2)
Format for number rows: (x1, y1, x2, y2) — the entire row containing all 10 numbers

Coordinates verified against ruler-annotated calibration images in coord_finder/.
"""

import os

# ---------------------------------------------------------------------------
# GOOGLE CLOUD VISION CREDENTIALS (1 Credit per Form Optimization)
# ---------------------------------------------------------------------------
# Option 1 (Recommended): Service Account JSON file path
# Place your downloaded service account JSON file in this directory and name it "service_account.json"
GOOGLE_SERVICE_ACCOUNT_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")

# Option 2: API Key (alternative to Service Account JSON)
GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")

# ---------------------------------------------------------------------------
# RENDER SCALE
# ---------------------------------------------------------------------------
ZOOM = 3  # render multiplier used in pdf_to_images()

# ---------------------------------------------------------------------------
# PAGE 1  (1786 x 2526 px at 3x zoom)
# ---------------------------------------------------------------------------

# --- Header text fields ---
# Date field: to the right of "दिनांक :" label
PAGE1_DATE   = (250, 250, 560, 340)
# Khoji No: to the right of "खोजी नंबर :" label
PAGE1_KHOJI  = (750, 260, 1050, 350)
# City: to the right of "शहर :" label
PAGE1_CITY   = (1240, 260, 1760, 350)
# Name: full line after "नाम :" label
PAGE1_NAME   = (250, 345, 1760, 420)

# --- Q1: Yes/No checkboxes ---
# Q1: "...MSY को आ सकती है? हाँ□ ना□"
PAGE1_Q1_HAN_BOX = (600, 545, 650, 595)   # हाँ □ checkbox
PAGE1_Q1_NA_BOX  = (675, 545, 730, 595)   # ना □ checkbox

# --- Q2: Yes/No checkboxes ---
# Q2: "क्या मौन के पालन में आपको कोई दिक्कत आती है? हाँ□ ना□"
PAGE1_Q2_HAN_BOX = (910, 600, 960, 648)   # हाँ □ checkbox
PAGE1_Q2_NA_BOX  = (980, 600, 1035, 648)  # ना □ checkbox

# --- Number rows: Q3–Q12  ---
# The 10 Devanagari numbers are the row BELOW each question text.
# x: from ~170 to ~1440 (numbers spread across page width)

PAGE1_Q3_ROW  = (170, 708, 1445, 752)   # Q3            (shifted upward)
PAGE1_Q4_ROW  = (170, 808, 1445, 852)   # Q4            (shifted upward)
PAGE1_Q5_ROW  = (170, 916, 1445, 960)   # Q5 number row (shifted upward)
PAGE1_Q6_ROW  = (170, 1020, 1445, 1064) # Q6 number row (shifted upward)
PAGE1_Q7_ROW  = (170, 1126, 1445, 1170) # Q7 number row (shifted upward)
PAGE1_Q8_ROW  = (170, 1226, 1445, 1270) # Q8 number row (shifted upward)
PAGE1_Q9_ROW  = (170, 1326, 1445, 1370) # Q9            (shifted upward)
PAGE1_Q10_ROW = (170, 1436, 1445, 1480) # Q10 number row
PAGE1_Q11_ROW = (170, 1532, 1445, 1576) # Q11 number row
PAGE1_Q12_ROW = (170, 1632, 1445, 1676) # Q12 number row


# ---------------------------------------------------------------------------
# PAGE 2  (1786 x 2526 px at 3x zoom)
# ---------------------------------------------------------------------------

# --- Header text fields ---
PAGE2_DATE   = (250, 260, 560, 350)
PAGE2_KHOJI  = (750, 260, 1050, 350)
PAGE2_CITY   = (1240, 260, 1760, 350)
PAGE2_NAME   = (250, 345, 1760, 420)

# --- Number rows: Q1–Q13 ---
PAGE2_Q1_ROW  = (170, 638, 1445, 682)   # Q1 number row (shifted upward)
PAGE2_Q2_ROW  = (170, 765, 1445, 808)   # Q2 number row (shifted upward)
PAGE2_Q3_ROW  = (170, 930, 1445, 974)
PAGE2_Q4_ROW  = (170, 1055, 1445, 1098)  #4
PAGE2_Q5_ROW  = (170, 1168, 1445, 1212)  # Q5           (shifted downward)
PAGE2_Q6_ROW  = (170, 1280, 1445, 1324) #6
PAGE2_Q7_ROW  = (170, 1396, 1445, 1440)
PAGE2_Q8_ROW  = (170, 1508, 1445, 1552) #7
PAGE2_Q9_ROW  = (170, 1676, 1445, 1720) # Q9             (shifted upward)
PAGE2_Q10_ROW = (170, 1790, 1445, 1834) # Q10 number row (shifted upward)
PAGE2_Q11_ROW = (170, 1900, 1445, 1944)
PAGE2_Q12_ROW = (170, 2020, 1445, 2064) #12
PAGE2_Q13_ROW = (170, 2125, 1445, 2169) # Q13           (shifted downward)

# --- Q14: Free text (written answer) ---
PAGE2_Q14_TEXT = (140, 2225, 1650, 2390)


# ---------------------------------------------------------------------------
# PAGE 3  (1786 x 2526 px at 3x zoom)
# ---------------------------------------------------------------------------

# --- Header text fields ---
PAGE3_DATE   = (330, 310, 540, 360)
PAGE3_KHOJI  = (790, 310, 1010, 360)
PAGE3_CITY   = (1260, 310, 1760, 360)
PAGE3_NAME   = (300, 378, 1760, 428)

# --- Number rows: Q1–Q13 ---
PAGE3_Q1_ROW  = (170, 652, 1445, 696)   # Q1             (shifted upward)
PAGE3_Q2_ROW  = (170, 782, 1445, 826)
PAGE3_Q3_ROW  = (170, 910, 1445, 954)   # Q3 number row (shifted upward)
PAGE3_Q4_ROW  = (170, 1032, 1445, 1076)
PAGE3_Q5_ROW  = (170, 1158, 1445, 1202) # Q5            (shifted downward)
PAGE3_Q6_ROW  = (170, 1288, 1445, 1332)
PAGE3_Q7_ROW  = (170, 1410, 1445, 1454) # Q7 number row (shifted downward)
PAGE3_Q8_ROW  = (170, 1536, 1445, 1590) # Q8            (shifted downward)
PAGE3_Q9_ROW  = (170, 1726, 1445, 1770) # Q9            (shifted downward)
PAGE3_Q10_ROW = (170, 1846, 1445, 1900) # Q10           (shifted downward)
PAGE3_Q11_ROW = (170, 1996, 1445, 2040)
PAGE3_Q12_ROW = (170, 2112, 1445, 2156) # Q12           (shifted downward)
PAGE3_Q13_ROW = (170, 2238, 1445, 2282)

# ---------------------------------------------------------------------------
# NUMBER POSITIONS within a number row
# ---------------------------------------------------------------------------
# The 10 numbers (१ to १०) are spread across the row.
# We split the row width into 10 equal slots and find which slot the circle lands in.
NUM_SLOTS = 10
