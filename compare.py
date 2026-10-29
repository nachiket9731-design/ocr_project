import glob, os, jiwer, easyocr
from paddleocr import PaddleOCR

easy = easyocr.Reader(['hi', 'en'], gpu=False)
paddle = PaddleOCR(
    text_recognition_model_name="devanagari_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)

def read_easy(path):
    return " ".join(easy.readtext(path, detail=0))

def read_paddle(path):
    texts = []
    for r in paddle.predict(path):
        texts += list(r["rec_texts"])
    return " ".join(texts)

scores = {"easyocr": [], "paddle": []}
for img in sorted(glob.glob("test/*.png")):
    truth = open(os.path.splitext(img)[0] + ".txt", encoding="utf-8").read().strip()
    scores["easyocr"].append(jiwer.cer(truth, read_easy(img)))
    scores["paddle"].append(jiwer.cer(truth, read_paddle(img)))

for name, vals in scores.items():
    print(name, "error rate:", round(sum(vals) / len(vals), 3))