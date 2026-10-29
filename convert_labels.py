import os, yaml
from collections import Counter

root = r"C:\ocr_imp\paddle_labels"
model_dir = os.path.expanduser(r"~\.paddlex\official_models\devanagari_PP-OCRv5_mobile_rec")

cfg = yaml.safe_load(open(os.path.join(model_dir, "inference.yml"), encoding="utf-8"))
chars = set(cfg["PostProcess"]["character_dict"])
print("characters the model knows:", len(chars))

missing, total = Counter(), 0
for split in ["train", "val", "test"]:
    for line in open(os.path.join(root, f"rec_gt_{split}.txt"), encoding="utf-8"):
        parts = line.rstrip("\n").split("\t", 1)
        if len(parts) < 2:
            continue
        for ch in parts[1]:
            total += 1
            if ch not in chars:
                missing[ch] += 1

print("characters checked:", total)
print("not in the model's list:", sum(missing.values()))
for ch, n in missing.most_common(20):
    print(repr(ch), hex(ord(ch)), n)