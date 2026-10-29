# Colab guide: fine-tune PaddleOCR's Devanagari recognizer on handwriting

Hand this whole file to the agent (Antigravity). It is written so the agent can follow it cell by cell in Google Colab.

---

## 0. Rules for the agent (read first)

1. Work one step at a time. After every step, run the **checkpoint** under it and show the real output. If a checkpoint fails, **stop and report the exact error**. Do not guess a fix and continue.
2. Do not invent config keys, file names, or download URLs. Read them from the cloned PaddleOCR repo (`/content/PaddleOCR`) or its docs. Where this guide says "verify", it means it.
3. Never train on, or tune hyperparameters against, the real-form test crops (`eval_forms`, see section 9). They are the only honest measure of whether this worked.
4. Save everything that matters (checkpoints, logs, the exported model) to Google Drive, not just `/content`. Colab sessions disconnect.
5. Report results honestly. If the fine-tuned model is not better on the real forms, say so.

---

## 1. Goal and success test

**Goal:** make the Devanagari handwriting recognizer more accurate on handwritten Hindi/Marathi lines from real forms, and run completely locally afterward (no cloud OCR).

**Success test:** character error rate (CER) on the real-form test crops goes **down** compared with the pretrained model, and English/number reading does **not** get noticeably worse.

**Baseline already measured** (9 real-form crops, before any training): EasyOCR CER 0.611, PaddleOCR `devanagari_PP-OCRv5_mobile_rec` CER 0.397. Re-measure the baseline with the exact script in section 9 so the before/after comparison is fair.

---

## 2. What is already done (facts)

- Colab runtime: Tesla T4, 15 GB, NVIDIA driver 580 (CUDA 13.0 reported by the driver).
- PaddleOCR repo cloned to `/content/PaddleOCR`.
- Config file for this model (note the `.yaml` extension):
  `/content/PaddleOCR/configs/rec/PP-OCRv5/multi_language/devanagari_PP-OCRv5_mobile_rec.yaml`
- Also present in the repo: `configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml`, `configs/rec/multi_language/rec_devanagari_lite_train.yml` (older), and a PP-OCRv3 Devanagari config. Use the first one above.
- Training data: **IIIT-HW-Hindi v1** (word images, Hindi handwriting, about 12 writers), zipped as `ocr_imp.zip` at the top level of Google Drive (`My Drive/ocr_imp.zip`). After unzip it contains the `HindiSeg` image folders and a `paddle_labels` folder with:
  - `rec_gt_train.txt` (69,853 lines), `rec_gt_val.txt` (12,708), `rec_gt_test.txt` (12,869)
  - Format per line: `relative/path/to/image.jpg<TAB>label`. Paths are relative to the folder that **contains** `paddle_labels` (the dataset root).
- Character coverage was checked: all 570,625 characters in the labels are inside the pretrained model's dictionary (568 characters). The dictionary does **not** need to change.
- Dataset citation (keep it in any write-up): Dutta, Krishnan, Mathew, Jawahar, DAS 2018, IIIT-HW-Dev / word-level Indic handwriting datasets, CVIT IIIT Hyderabad. Read the dataset Readme for usage terms.

**Known limitation to keep in mind:** this data is clean, single *words* from few writers. The target is handwritten *lines* on forms (ruled paper, different pens, Hindi + English + digits). Gains are not guaranteed. Section 7 adds line-level and English/digit samples to reduce this gap.

---

## 3. Setup

### Cell 1: GPU check
```
!nvidia-smi
```
Checkpoint: shows a Tesla T4.

### Cell 2: mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 3: copy and unzip the data, then verify paths
```
!cp /content/drive/MyDrive/ocr_imp.zip /content/
!unzip -q /content/ocr_imp.zip -d /content/data
```
```python
import os, glob, random
lab = glob.glob("/content/data/**/paddle_labels/rec_gt_train.txt", recursive=True)[0]
DATA_DIR = os.path.dirname(os.path.dirname(lab))   # folder that contains paddle_labels
print("label file:", lab)
print("DATA_DIR  :", DATA_DIR)
lines = open(lab, encoding="utf-8").read().splitlines()
print(len(lines), "train lines (expect 69853)")
bad = [l for l in random.sample(lines, 500)
       if not os.path.exists(os.path.join(DATA_DIR, l.split("\t")[0]))]
print("missing among 500 sampled:", len(bad), "(expect 0)")
```
Checkpoint: 69853 lines, 0 missing. If images are missing, the label paths are relative to a different folder. Fix `DATA_DIR`, do not edit the labels blindly. Remember `DATA_DIR`; it is used below.

### Cell 4: repo (skip if `/content/PaddleOCR` already exists)
```
!git clone https://github.com/PaddlePaddle/PaddleOCR.git
```

### Cell 5: install GPU Paddle and dependencies
```
!python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
!cd /content/PaddleOCR && python -m pip install -r requirements.txt
!python -m pip install jiwer pyyaml paddleocr
```
```python
import paddle
paddle.utils.run_check()
print(paddle.device.get_device())
```
Checkpoint: "PaddlePaddle works well on 1 GPU" and device is `gpu:0`. If Colab asks to restart the session, restart, re-run Cell 2 (Drive mount), and continue. Installed packages and `/content/data` remain.

---

## 4. Read the config before changing anything

```
!cat /content/PaddleOCR/configs/rec/PP-OCRv5/multi_language/devanagari_PP-OCRv5_mobile_rec.yaml
```

Find and write down these values (names may differ slightly, so read the real file):

| What | Typically under | Why it matters |
|---|---|---|
| pretrained weights | `Global.pretrained_model` | starting point for fine-tuning |
| character dictionary | `Global.character_dict_path` | must match the pretrained model |
| use space char | `Global.use_space_char` | needed for line-level training (section 7) |
| max text length | `Global.max_text_length` | must cover your longest label (lines are longer than words) |
| epochs, eval interval, save dir | `Global.epoch_num`, `Global.eval_batch_step`, `Global.save_model_dir` | training length and output |
| train/eval data dir and label files | `Train.dataset.data_dir`, `Train.dataset.label_file_list`, same under `Eval` | point these at the Colab data |
| batch size | `Train.loader.batch_size_per_card` (and the sampler section if the config uses one) | memory on the T4 |
| learning rate | `Optimizer.lr.learning_rate` | lower it for fine-tuning |

**Dictionary check (important):**
```python
import os
cfg_dict = "/content/PaddleOCR/<path from Global.character_dict_path>"
print(os.path.exists(cfg_dict))
print(len(open(cfg_dict, encoding="utf-8").read().splitlines()))
```
Checkpoint: the file exists and has about 568 entries (plus or minus one for a space). If it is very different from 568, **stop and report**. The training dictionary must be the same one the pretrained model was trained with, or the output layer will not match.

Prefer overriding values on the command line with `-o key=value` instead of editing the yaml, and keep the yaml untouched so changes are easy to audit.

---

## 5. Pretrained weights (trainable checkpoint)

Fine-tuning needs a **trainable checkpoint** (`.pdparams`), not only the inference files that the laptop downloaded.

1. Look at `Global.pretrained_model` in the config. If it is a URL, download it and check it opens:
```python
import os, urllib.request
url = "<URL from the config>"
os.makedirs("/content/pretrained", exist_ok=True)
dst = "/content/pretrained/devanagari_pretrained.pdparams"
urllib.request.urlretrieve(url, dst)
print(os.path.getsize(dst))
```
2. If it is empty, search the PaddleOCR docs and model pages for the official pretrained weights of `devanagari_PP-OCRv5_mobile_rec`. Do not guess a URL.
3. If no trainable checkpoint can be found, **stop and report to the user**. Training from scratch with the same architecture is possible but will be clearly worse with this little data, so it needs the user's decision.

Checkpoint: a `.pdparams` file with a plausible size (several MB) exists locally.

---

## 6. Smoke test (always before the long run)

Run a very short training to prove everything works: a few hundred iterations, or 1 epoch on a small subset (make a 2,000-line copy of the train labels and a 500-line copy of the val labels). Use the training command from the repo docs. Read these before running it:

- `/content/PaddleOCR/docs/` (find the text-recognition training document; the version 2.x document `ppocr/model_train/recognition` also exists and the commands carry over)
- `python /content/PaddleOCR/tools/train.py --help`

The general shape of the command is:
```
%cd /content/PaddleOCR
!python tools/train.py -c configs/rec/PP-OCRv5/multi_language/devanagari_PP-OCRv5_mobile_rec.yaml \
  -o Global.pretrained_model=/content/pretrained/devanagari_pretrained \
     Global.save_model_dir=/content/drive/MyDrive/ocr_out/smoke \
     Global.epoch_num=1 \
     Train.dataset.data_dir=<DATA_DIR> \
     Train.dataset.label_file_list=[<small train label file>] \
     Eval.dataset.data_dir=<DATA_DIR> \
     Eval.dataset.label_file_list=[<small val label file>] \
     Train.loader.batch_size_per_card=64
```
Use the exact key names from the real config. The pretrained path normally has no `.pdparams` extension in this command (verify in the docs).

Checkpoint: the loss prints and goes down, eval accuracy prints, no out-of-memory error, and a checkpoint is written to the save dir. Record **seconds per iteration** so total training time can be estimated. If out of memory, lower the batch size.

---

## 7. Fix the data mismatch before the long run (recommended)

Training on isolated Hindi words only risks two problems: the model forgets English/digits, and it never sees line-length text. Add these to the training label list (as extra label files merged into one, or use the config's multi-file `label_file_list` with `ratio_list`):

1. **Synthetic Hindi lines.** Stitch 3 to 6 word images **from the same writer** side by side (resize to the same height, fill gaps with the background colour), join their labels with a single space. Needs `use_space_char: True` and a space handled by the dictionary (check this). Make sure `max_text_length` covers the longest stitched label.
2. **English and number samples.** Render English words, phrases, and digit strings (dates, phone-style numbers, 1 to 10) with handwriting-style fonts, with blur, noise, small rotation, and uneven contrast. Aim for a meaningful share of the mix (for example 15 to 25 percent), not a token amount.
3. **Real handwriting from the user**, if provided later (forms filled by friends or classmates). Keep a separate set for testing.
4. Augmentation that suits handwriting (the config's built-in augmentation is usually fine): slight rotation/shear, blur, brightness/contrast jitter, thin/thick stroke changes.

Never put the real-form test crops (section 9) into any training list.

---

## 8. Full training

Start from the smoke-test command and change:

- full train and val label files (plus the extra data from section 7)
- `Global.epoch_num`: start around 20 and let validation decide
- learning rate: lower than the config's default, roughly one fifth to one tenth, because this is fine-tuning (starting suggestion, not a verified value)
- batch size: the largest that fits on the T4 (try 64 to 128)
- `Global.save_model_dir=/content/drive/MyDrive/ocr_out/devanagari_hw` so progress survives disconnects
- evaluate regularly (for example every few thousand iterations) and keep the best checkpoint (`best_accuracy`)

Resume after a disconnect with `Global.checkpoints=<save_dir>/latest` (verify the name in the docs).

Checkpoint (every eval): print validation accuracy and norm edit distance. Stop early if validation stops improving for several evals. Overfitting to 12 writers is the main risk.

---

## 9. Evaluate on the real forms (the test that counts)

The user must upload a zip of real-form crops to Drive as `eval_forms.zip`: folder `test/` with pairs `a1.png` and `a1.txt` (correct text, UTF-8). Unzip to `/content/eval_forms`.

```python
import glob, os, jiwer
from paddleocr import PaddleOCR

def norm(s): return " ".join(s.split())

def make(model_dir=None):
    kw = dict(text_recognition_model_name="devanagari_PP-OCRv5_mobile_rec",
              use_doc_orientation_classify=False, use_doc_unwarping=False,
              use_textline_orientation=False)
    if model_dir:
        kw["text_recognition_model_dir"] = model_dir
    return PaddleOCR(**kw)

def read(ocr, path):
    items = []
    for r in ocr.predict(path):
        for text, box in zip(r["rec_texts"], r["rec_boxes"]):
            x1, y1, x2, y2 = [int(v) for v in box]
            items.append((text, x1, (y1 + y2) / 2, y2 - y1))
    if not items:
        return ""
    items.sort(key=lambda t: t[2])
    tol = sorted(t[3] for t in items)[len(items) // 2] * 0.6
    lines, cur = [], [items[0]]
    for it in items[1:]:
        if abs(it[2] - sum(c[2] for c in cur) / len(cur)) <= tol:
            cur.append(it)
        else:
            lines.append(cur); cur = [it]
    lines.append(cur)
    return " ".join(t[0] for ln in lines for t in sorted(ln, key=lambda t: t[1]))

def cer_report(ocr, folder, out_file):
    scores = []
    with open(out_file, "w", encoding="utf-8") as f:
        for img in sorted(glob.glob(folder + "/*.png")):
            truth = norm(open(os.path.splitext(img)[0] + ".txt", encoding="utf-8").read())
            pred = norm(read(ocr, img))
            e = jiwer.cer(truth, pred)
            scores.append(e)
            f.write(f"{img}\n  correct: {truth}\n  model  : {pred}\n  CER    : {e:.2f}\n\n")
    return sum(scores) / len(scores)

# baseline (pretrained) and fine-tuned, same script, same data
print("baseline :", cer_report(make(), "/content/eval_forms/test", "/content/baseline.txt"))
print("finetuned:", cer_report(make("<exported inference dir>"), "/content/eval_forms/test", "/content/finetuned.txt"))
```

Notes:
- If `rec_boxes` is missing in the result, print one result object and adapt; report what was found.
- With only 9 crops the number is noisy. Report per-image results too, and say so in the summary.
- Evaluate English-heavy crops separately from Hindi-heavy crops so any English regression is visible.

Also run the repo's own evaluation on the IIIT test split (`tools/eval.py` with the best checkpoint) to confirm the model learned the dataset, but do not treat it as the success measure.

---

## 10. Export the model for local use

Export the best checkpoint to an inference model (verify the command in the repo docs):
```
%cd /content/PaddleOCR
!python tools/export_model.py -c configs/rec/PP-OCRv5/multi_language/devanagari_PP-OCRv5_mobile_rec.yaml \
  -o Global.pretrained_model=/content/drive/MyDrive/ocr_out/devanagari_hw/best_accuracy \
     Global.save_inference_dir=/content/drive/MyDrive/ocr_out/inference
```
Checkpoint: the output folder loads with `PaddleOCR(text_recognition_model_name="devanagari_PP-OCRv5_mobile_rec", text_recognition_model_dir="<folder>")` and returns text on one crop. The folder must include the model config with the character dictionary (compare with the cached official model folder, which holds an `inference.yml`). If the exported format differs from what PaddleOCR 3.x expects, report it.

---

## 11. Deliverables

Put in `My Drive/ocr_out/` and list them at the end:

1. `inference/` exported model folder (for the laptop)
2. `devanagari_hw/best_accuracy*` and the latest checkpoint
3. training log with validation accuracy over time
4. `baseline.txt` and `finetuned.txt` (per-image comparisons)
5. a short summary table:

| Measure | Pretrained | Fine-tuned |
|---|---|---|
| CER, all real-form crops | | |
| CER, Hindi-heavy crops | | |
| CER, English-heavy crops | | |
| IIIT test accuracy | | |

and one honest paragraph: did it help, where it still fails, what data to add next.

## 12. Using the result on the laptop (Windows)

Download the `inference` folder to `C:\ocr_project\models\devanagari_hw\`. Then in the project:
```python
paddle = PaddleOCR(
    text_recognition_model_name="devanagari_PP-OCRv5_mobile_rec",
    text_recognition_model_dir=r"C:\ocr_project\models\devanagari_hw",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,   # avoids the CPU crash seen earlier
)
```
Use this only for the free-text answer (Q14). The circled 1 to 10 scale answers and numeric fields should stay on their current OpenCV or digit-only code, because Devanagari models misread handwritten digits.
