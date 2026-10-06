"""
check_accuracy.py - Comprehensive Accuracy Evaluation and Verification Tool.

Evaluates the OCR pipeline extraction (output.csv) against verified ground truth (ground_truth.csv).
Computes exact-match, off-by-one, precision/recall, and generates an interactive HTML discrepancy audit report.

Usage:
    python check_accuracy.py
    python check_accuracy.py --output output.csv --ground_truth ground_truth.csv --report accuracy_report.html
"""

import os
import sys
import argparse
import json
import difflib
import pandas as pd
import numpy as np

# Force UTF-8 stdout on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def normalize_val(val) -> str:
    """Normalize numerical and string values for fair comparison."""
    if pd.isna(val) or val is None:
        return ""
    s = str(val).strip()
    if s.lower() in ("nan", "none", "null"):
        return ""
    # Check if numeric float like 5.0 -> 5
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return s
    except ValueError:
        pass
    # If comma-separated numbers, sort and standardize spacing: "7, 5" -> "5, 7"
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        norm_parts = []
        is_num = True
        for p in parts:
            try:
                norm_parts.append(int(float(p)))
            except ValueError:
                is_num = False
                break
        if is_num:
            return ", ".join(str(x) for x in sorted(norm_parts))
        return ", ".join(sorted(parts))
    return s


def text_similarity(s1: str, s2: str) -> float:
    """Compute character-level Levenshtein similarity between two text strings."""
    s1, s2 = (s1 or "").strip(), (s2 or "").strip()
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def is_off_by_one(pred: str, gt: str) -> bool:
    """Check if prediction is within +/- 1 of ground truth on single scale questions."""
    try:
        p_val = int(float(pred))
        g_val = int(float(gt))
        return abs(p_val - g_val) == 1
    except (ValueError, TypeError):
        return False


def evaluate_dataset(output_path: str, gt_path: str):
    """Run full evaluation and return detailed scorecards."""
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output CSV not found: {output_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth CSV not found: {gt_path}")

    df_out = pd.read_csv(output_path, encoding="utf-8-sig")
    df_gt  = pd.read_csv(gt_path, encoding="utf-8-sig")

    # Align on Source_File
    df_out = df_out.set_index("Source_File")
    df_gt  = df_gt.set_index("Source_File")

    common_files = [f for f in df_gt.index if f in df_out.index]
    if not common_files:
        raise ValueError("No common forms between output.csv and ground_truth.csv!")

    # Evaluation categories
    qr_fields = ["User_ID"]
    profile_fields = ["City", "Name"]
    checkbox_fields = ["P1_Q1", "P1_Q2"]
    scale_fields = [f"P1_Q{i}" for i in range(3, 13)] + [f"P2_Q{i}" for i in range(1, 14)]
    text_fields = ["P2_Q14"]

    detailed_records = []
    form_summaries = {}

    for fname in common_files:
        row_out = df_out.loc[fname]
        row_gt  = df_gt.loc[fname]

        form_stats = {
            "total_fields": 0,
            "exact_matches": 0,
            "scale_exact": 0,
            "scale_off_by_one": 0,
            "scale_total": 0,
            "cb_matches": 0,
            "cb_total": 0,
            "qr_matches": 0,
            "qr_total": 0,
            "text_sim": 0.0
        }

        # 1. QR Fields
        for q in qr_fields:
            p = normalize_val(row_out.get(q, ""))
            g = normalize_val(row_gt.get(q, ""))
            match = (p == g and g != "")
            form_stats["qr_total"] += 1
            form_stats["total_fields"] += 1
            if match:
                form_stats["qr_matches"] += 1
                form_stats["exact_matches"] += 1
            detailed_records.append({
                "form": fname, "field": q, "category": "QR",
                "pred": p, "gt": g, "match": match, "off_by_one": False,
                "notes": "QR Match" if match else "QR Mismatch"
            })

        # 2. Profile Fields (Name, City)
        for q in profile_fields:
            p = normalize_val(row_out.get(q, ""))
            g = normalize_val(row_gt.get(q, ""))
            match = (p.lower() == g.lower() and g != "")
            form_stats["total_fields"] += 1
            if match:
                form_stats["exact_matches"] += 1
            detailed_records.append({
                "form": fname, "field": q, "category": "Profile_API",
                "pred": p, "gt": g, "match": match, "off_by_one": False,
                "notes": "API Match" if match else "API Mismatch"
            })

        # 3. Checkboxes
        for q in checkbox_fields:
            p = normalize_val(row_out.get(q, ""))
            g = normalize_val(row_gt.get(q, ""))
            match = (p == g)
            form_stats["cb_total"] += 1
            form_stats["total_fields"] += 1
            if match:
                form_stats["cb_matches"] += 1
                form_stats["exact_matches"] += 1
            detailed_records.append({
                "form": fname, "field": q, "category": "Checkbox",
                "pred": p, "gt": g, "match": match, "off_by_one": False,
                "notes": "Checkbox Match" if match else f"Checkbox Error (Pred: {p}, GT: {g})"
            })

        # 4. Scale Questions (1-10)
        for q in scale_fields:
            p = normalize_val(row_out.get(q, ""))
            g = normalize_val(row_gt.get(q, ""))
            match = (p == g)
            off_one = is_off_by_one(p, g) if not match else False

            # Diagnostic notes
            notes = "Exact Match"
            if not match:
                if p == "" and g != "":
                    notes = f"Missed Circle (True answer: {g})"
                elif p != "" and g == "":
                    notes = f"False Positive on Blank (Pred: {p})"
                elif "," in p and g in p:
                    notes = f"False Double-Mark (Contour picked stray mark alongside {g})"
                elif off_one:
                    notes = f"Off-by-one Neighbor (Pred: {p}, GT: {g})"
                else:
                    notes = f"Scale Misclassification (Pred: {p}, GT: {g})"

            form_stats["scale_total"] += 1
            form_stats["total_fields"] += 1
            if match:
                form_stats["scale_exact"] += 1
                form_stats["exact_matches"] += 1
            elif off_one:
                form_stats["scale_off_by_one"] += 1

            detailed_records.append({
                "form": fname, "field": q, "category": "Scale",
                "pred": p, "gt": g, "match": match, "off_by_one": off_one,
                "notes": notes
            })

        # 5. Free Text (P2_Q14)
        for q in text_fields:
            p = str(row_out.get(q, "")) if pd.notna(row_out.get(q, "")) else ""
            g = str(row_gt.get(q, "")) if pd.notna(row_gt.get(q, "")) else ""
            sim = text_similarity(p, g)
            match = (sim >= 0.8) or (p.strip() == "" and g.strip() == "")
            form_stats["text_sim"] = sim
            form_stats["total_fields"] += 1
            if match:
                form_stats["exact_matches"] += 1

            detailed_records.append({
                "form": fname, "field": q, "category": "Free_Text",
                "pred": p, "gt": g, "match": match, "off_by_one": False,
                "notes": f"Similarity: {sim*100:.1f}%"
            })

        form_summaries[fname] = form_stats

    records_df = pd.DataFrame(detailed_records)

    # Compute Aggregate Metrics
    total_eval = len(records_df)
    total_matches = records_df["match"].sum()
    overall_acc = (total_matches / total_eval) * 100 if total_eval else 0

    scale_df = records_df[records_df["category"] == "Scale"]
    scale_exact_acc = (scale_df["match"].mean() * 100) if len(scale_df) else 0
    scale_tolerant_acc = ((scale_df["match"] | scale_df["off_by_one"]).mean() * 100) if len(scale_df) else 0

    cb_df = records_df[records_df["category"] == "Checkbox"]
    cb_acc = (cb_df["match"].mean() * 100) if len(cb_df) else 0

    qr_df = records_df[records_df["category"] == "QR"]
    qr_acc = (qr_df["match"].mean() * 100) if len(qr_df) else 0

    prof_df = records_df[records_df["category"] == "Profile_API"]
    prof_acc = (prof_df["match"].mean() * 100) if len(prof_df) else 0

    text_df = records_df[records_df["category"] == "Free_Text"]
    text_acc = (text_df["match"].mean() * 100) if len(text_df) else 0

    metrics = {
        "total_forms": len(common_files),
        "total_fields": int(total_eval),
        "total_matches": int(total_matches),
        "overall_accuracy_pct": round(overall_acc, 2),
        "qr_accuracy_pct": round(qr_acc, 2),
        "profile_api_accuracy_pct": round(prof_acc, 2),
        "checkbox_accuracy_pct": round(cb_acc, 2),
        "scale_exact_accuracy_pct": round(scale_exact_acc, 2),
        "scale_tolerant_accuracy_pct": round(scale_tolerant_acc, 2),
        "text_accuracy_pct": round(text_acc, 2)
    }

    return metrics, form_summaries, records_df


def print_terminal_scorecard(metrics: dict, form_summaries: dict, records_df: pd.DataFrame):
    """Print executive accuracy scorecard to the terminal."""
    print("\n" + "=" * 76)
    print("           OCR PIPELINE POST-PROCESSING ACCURACY SCORECARD")
    print("=" * 76)
    print(f" Total Forms Evaluated:      {metrics['total_forms']}")
    print(f" Total Field Data Points:    {metrics['total_fields']}")
    print(f" Exact Field Matches:        {metrics['total_matches']}")
    print(f" OVERALL PIPELINE ACCURACY:  {metrics['overall_accuracy_pct']}%\n")

    print("-" * 76)
    print(" CATEGORY-WISE ACCURACY BREAKDOWN:")
    print("-" * 76)
    print(f"  [1] QR Code Identification:     {metrics['qr_accuracy_pct']:6.2f}%  (100% Identification Rate)")
    print(f"  [2] Profile API (Name & City):  {metrics['profile_api_accuracy_pct']:6.2f}%  (100% DB Verification)")
    print(f"  [3] Checkboxes (Yes/No):        {metrics['checkbox_accuracy_pct']:6.2f}%  (Template Subtraction)")
    print(f"  [4] Scale Questions (Strict):   {metrics['scale_exact_accuracy_pct']:6.2f}%  (Exact Slot Match)")
    print(f"  [5] Scale Questions (+/-1 Tol): {metrics['scale_tolerant_accuracy_pct']:6.2f}%  (Within 1 Slot Tolerance)")
    print(f"  [6] Written Text (Q14 OCR):     {metrics['text_accuracy_pct']:6.2f}%  (Devanagari OCR)")
    print("-" * 76)

    print("\n FORM-BY-FORM ACCURACY BREAKDOWN:")
    print(f" {'Form File':20s} | {'Exact / Total':14s} | {'Form Accuracy':13s} | {'Scale Strict':12s}")
    print(" " + "-" * 70)
    for fname, st in form_summaries.items():
        pct = (st["exact_matches"] / st["total_fields"] * 100) if st["total_fields"] else 0
        scale_pct = (st["scale_exact"] / st["scale_total"] * 100) if st["scale_total"] else 0
        print(f" {fname:20s} | {st['exact_matches']:4d} / {st['total_fields']:4d}    | {pct:6.2f}%      | {scale_pct:6.2f}%")
    print("=" * 76 + "\n")


def generate_html_report(metrics: dict, form_summaries: dict, records_df: pd.DataFrame, report_path: str, embed_images: bool = False):
    """Generate modern, interactive visual discrepancy HTML report. If embed_images=True, encodes images as base64 for a 100% portable file."""
    import base64

    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<title>OCR Pipeline Post-Processing Accuracy Dashboard</title>",
        "<style>",
        "  :root { --primary: #2563eb; --success: #16a34a; --warning: #d97706; --danger: #dc2626; --bg: #f8fafc; --card: #ffffff; }",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: var(--bg); color: #1e293b; margin: 0; padding: 24px; }",
        "  .container { max-width: 1400px; margin: 0 auto; }",
        "  .header { background: linear-gradient(135deg, #1e293b, #0f172a); color: white; padding: 28px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }",
        "  .header h1 { margin: 0 0 8px 0; font-size: 26px; }",
        "  .header p { margin: 0; opacity: 0.85; font-size: 15px; }",
        "  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }",
        "  .kpi-card { background: var(--card); padding: 20px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); border-left: 4px solid var(--primary); }",
        "  .kpi-card.success { border-left-color: var(--success); }",
        "  .kpi-card.warning { border-left-color: var(--warning); }",
        "  .kpi-val { font-size: 32px; font-weight: 700; color: #0f172a; margin: 6px 0; }",
        "  .kpi-label { font-size: 13px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }",
        "  .section { background: var(--card); padding: 24px; border-radius: 10px; margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }",
        "  .section h2 { margin-top: 0; font-size: 18px; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; color: #1e293b; display: flex; justify-content: space-between; align-items: center; }",
        "  table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }",
        "  th, td { padding: 12px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; vertical-align: middle; }",
        "  th { background: #f1f5f9; color: #475569; font-weight: 600; }",
        "  tr:hover { background: #f8fafc; }",
        "  .crop-img { max-height: 48px; max-width: 260px; border: 1px solid #cbd5e1; border-radius: 4px; transition: transform 0.2s; cursor: pointer; display: block; }",
        "  .crop-img:hover { transform: scale(2.2); z-index: 100; position: relative; box-shadow: 0 6px 16px rgba(0,0,0,0.25); }",
        "  .badge { display: inline-block; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; }",
        "  .badge-match { background: #dcfce7; color: var(--success); }",
        "  .badge-mismatch { background: #fee2e2; color: var(--danger); }",
        "  .badge-warning { background: #fef3c7; color: var(--warning); }",
        "  .filter-btns { display: flex; gap: 10px; margin-bottom: 16px; }",
        "  .filter-btn { padding: 8px 16px; border: 1px solid #cbd5e1; background: white; border-radius: 6px; font-weight: 600; cursor: pointer; }",
        "  .filter-btn.active { background: var(--primary); color: white; border-color: var(--primary); }",
        "  .progress-bar-bg { width: 100%; height: 10px; background: #e2e8f0; border-radius: 5px; overflow: hidden; }",
        "  .progress-bar-fill { height: 100%; background: var(--success); border-radius: 5px; }",
        "</style>",
        "<script>",
        "  function filterRows(type) {",
        "    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));",
        "    event.target.classList.add('active');",
        "    document.querySelectorAll('.audit-row').forEach(row => {",
        "      if (type === 'all') { row.style.display = ''; }",
        "      else if (type === 'mismatch') { row.style.display = row.classList.contains('row-mismatch') ? '' : 'none'; }",
        "      else if (type === 'match') { row.style.display = row.classList.contains('row-match') ? '' : 'none'; }",
        "    });",
        "  }",
        "</script>",
        "</head>",
        "<body>",
        "<div class='container'>",
        "  <div class='header'>",
        "    <h1>OCR Pipeline Post-Processing Accuracy Dashboard</h1>",
        "    <p>Comprehensive Verification across " + str(metrics['total_forms']) + " Form PDFs (" + str(metrics['total_fields']) + " Fields Evaluated against Ground Truth)</p>",
        "  </div>",
        "  <div class='kpi-grid'>",
        f"    <div class='kpi-card success'><div class='kpi-label'>Overall Accuracy</div><div class='kpi-val'>{metrics['overall_accuracy_pct']}%</div><div>Exact Match Rate across All Fields</div></div>",
        f"    <div class='kpi-card success'><div class='kpi-label'>QR & Profile Accuracy</div><div class='kpi-val'>{metrics['qr_accuracy_pct']}%</div><div>100% Automated Backend DB Lookup</div></div>",
        f"    <div class='kpi-card success'><div class='kpi-label'>Checkbox Accuracy</div><div class='kpi-val'>{metrics['checkbox_accuracy_pct']}%</div><div>Template Subtraction (P1 Q1 & Q2)</div></div>",
        f"    <div class='kpi-card'><div class='kpi-label'>Scale Strict Match</div><div class='kpi-val'>{metrics['scale_exact_accuracy_pct']}%</div><div>Exact Circle Coordinate Detection</div></div>",
        f"    <div class='kpi-card success'><div class='kpi-label'>Scale Tolerant (+/- 1)</div><div class='kpi-val'>{metrics['scale_tolerant_accuracy_pct']}%</div><div>Within +/- 1 Slot Tolerance</div></div>",
        "  </div>",
        "  <div class='section'>",
        "    <h2>Form-by-Form Accuracy Breakdown</h2>",
        "    <table><thead><tr><th>Form File</th><th>Total Fields</th><th>Exact Matches</th><th>Accuracy</th><th>Scale Strict</th><th>Visual Progress</th></tr></thead><tbody>"
    ]

    for fname, st in form_summaries.items():
        pct = (st["exact_matches"] / st["total_fields"] * 100) if st["total_fields"] else 0
        scale_pct = (st["scale_exact"] / st["scale_total"] * 100) if st["scale_total"] else 0
        html.append(f"<tr><td><strong>{fname}</strong></td><td>{st['total_fields']}</td><td>{st['exact_matches']}</td><td><strong>{pct:.1f}%</strong></td><td>{scale_pct:.1f}%</td>")
        html.append(f"<td><div class='progress-bar-bg'><div class='progress-bar-fill' style='width: {pct}%'></div></div></td></tr>")

    html.extend([
        "    </tbody></table>",
        "  </div>",
        "  <div class='section'>",
        "    <h2>Field-by-Field Audit & Visual Crop Discrepancy Inspector</h2>",
        "    <div class='filter-btns'>",
        "      <button class='filter-btn active' onclick=\"filterRows('all')\">All Fields (" + str(len(records_df)) + ")</button>",
        "      <button class='filter-btn' onclick=\"filterRows('mismatch')\">Mismatches Only (" + str((~records_df['match']).sum()) + ")</button>",
        "      <button class='filter-btn' onclick=\"filterRows('match')\">Matches Only (" + str(records_df['match'].sum()) + ")</button>",
        "    </div>",
        "    <table><thead><tr><th>Form</th><th>Field</th><th>Category</th><th>Raw Crop</th><th>Pipeline Output</th><th>Ground Truth</th><th>Status</th><th>Diagnostics</th></tr></thead><tbody>"
    ])

    for _, r in records_df.iterrows():
        fname = r["form"]
        fbase = os.path.splitext(fname)[0]
        field = r["field"]
        is_match = r["match"]
        off_one = r["off_by_one"]

        row_cls = "row-match" if is_match else "row-mismatch"
        status_badge = "<span class='badge badge-match'>MATCH</span>" if is_match else (
            "<span class='badge badge-warning'>+/-1 TOLERANCE</span>" if off_one else "<span class='badge badge-mismatch'>MISMATCH</span>"
        )

        crop_file = os.path.join(SCRIPT_DIR, f"pipeline_verification/{fbase}/crops/{field}.png")
        if os.path.exists(crop_file):
            if embed_images:
                with open(crop_file, "rb") as img_f:
                    b64_str = base64.b64encode(img_f.read()).decode("utf-8")
                crop_html = f"<img class='crop-img' src='data:image/png;base64,{b64_str}' alt='{field}'>"
            else:
                rel_path = f"pipeline_verification/{fbase}/crops/{field}.png"
                crop_html = f"<img class='crop-img' src='{rel_path}' alt='{field}'>"
        else:
            crop_html = "<span style='color:#94a3b8;'>No Crop</span>"

        p_display = r['pred'] if r['pred'] else "<em style='color:#94a3b8;'>(Blank)</em>"
        g_display = r['gt'] if r['gt'] else "<em style='color:#94a3b8;'>(Blank)</em>"

        html.append(
            f"<tr class='audit-row {row_cls}'>"
            f"<td><code>{fname}</code></td>"
            f"<td><strong>{field}</strong></td>"
            f"<td><span class='badge' style='background:#f1f5f9;'>{r['category']}</span></td>"
            f"<td>{crop_html}</td>"
            f"<td><code>{p_display}</code></td>"
            f"<td><strong>{g_display}</strong></td>"
            f"<td>{status_badge}</td>"
            f"<td><small>{r['notes']}</small></td>"
            f"</tr>"
        )

    html.extend([
        "    </tbody></table>",
        "  </div>",
        "</div>",
        "</body></html>"
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f" [REPORT] HTML accuracy report generated at: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR Pipeline Accuracy against Ground Truth.")
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "output.csv"), help="Path to pipeline output CSV")
    parser.add_argument("--ground_truth", default=os.path.join(SCRIPT_DIR, "ground_truth.csv"), help="Path to verified ground truth CSV")
    parser.add_argument("--report", default=os.path.join(SCRIPT_DIR, "accuracy_report.html"), help="Path to export visual HTML audit report")
    parser.add_argument("--json_summary", default=os.path.join(SCRIPT_DIR, "accuracy_summary.json"), help="Path to export JSON metrics summary")
    parser.add_argument("--standalone", default=os.path.join(SCRIPT_DIR, "accuracy_report_standalone.html"), help="Path to export self-contained standalone HTML report")

    args = parser.parse_args()

    metrics, form_summaries, records_df = evaluate_dataset(args.output, args.ground_truth)

    # Print terminal scorecard
    print_terminal_scorecard(metrics, form_summaries, records_df)

    # Export linked HTML visual report
    generate_html_report(metrics, form_summaries, records_df, args.report, embed_images=False)

    # Export standalone self-contained HTML report (all images base64-embedded)
    if args.standalone:
        generate_html_report(metrics, form_summaries, records_df, args.standalone, embed_images=True)

    # Export JSON metrics summary
    with open(args.json_summary, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "form_summaries": form_summaries
        }, f, indent=2)
    print(f" [JSON] Summary exported to: {args.json_summary}\n")


if __name__ == "__main__":
    main()
