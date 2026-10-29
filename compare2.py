import glob, os, jiwer
from paddleocr import PaddleOCR

paddle = PaddleOCR(
    text_recognition_model_name="devanagari_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)

def norm(s):
    return " ".join(s.split())

def read_paddle(path):
    items = []
    for r in paddle.predict(path):
        for text, box in zip(r["rec_texts"], r["rec_boxes"]):
            x1, y1, x2, y2 = [int(v) for v in box]
            items.append((text, x1, (y1 + y2) / 2, y2 - y1))
    if not items:
        return "", ""
    original = " ".join(t[0] for t in items)
    # group boxes into lines by vertical position, then read each line left to right
    items.sort(key=lambda t: t[2])
    tol = sorted(t[3] for t in items)[len(items) // 2] * 0.6
    lines, cur = [], [items[0]]
    for it in items[1:]:
        if abs(it[2] - sum(c[2] for c in cur) / len(cur)) <= tol:
            cur.append(it)
        else:
            lines.append(cur)
            cur = [it]
    lines.append(cur)
    fixed = " ".join(t[0] for ln in lines for t in sorted(ln, key=lambda t: t[1]))
    return original, fixed

old_scores, new_scores = [], []
with open("results2.txt", "w", encoding="utf-8") as f:
    for img in sorted(glob.glob("test/*.png")):
        truth = norm(open(os.path.splitext(img)[0] + ".txt", encoding="utf-8").read())
        original, fixed = read_paddle(img)
        e_old, e_new = jiwer.cer(truth, norm(original)), jiwer.cer(truth, norm(fixed))
        old_scores.append(e_old)
        new_scores.append(e_new)
        f.write(f"{img}\n  correct : {truth}\n  fixed   : {fixed}\n  error old/new: {e_old:.2f} / {e_new:.2f}\n\n")

print("paddle (original order):", round(sum(old_scores) / len(old_scores), 3))
print("paddle (fixed order)   :", round(sum(new_scores) / len(new_scores), 3))