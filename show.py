import glob, os, jiwer, easyocr
from paddleocr import PaddleOCR

paddle = PaddleOCR(
    text_recognition_model_name="devanagari_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)

def read_paddle(path):
    texts = []
    for r in paddle.predict(path):
        texts += list(r["rec_texts"])
    return " ".join(texts)

with open("results.txt", "w", encoding="utf-8") as f:
    for img in sorted(glob.glob("test/*.png")):
        truth = open(os.path.splitext(img)[0] + ".txt", encoding="utf-8").read().strip()
        pred = read_paddle(img)
        f.write(f"{img}\n  correct : {truth}\n  paddle  : {pred}\n  error   : {jiwer.cer(truth, pred):.2f}\n\n")
print("Done. Open results.txt")