"""
Summarize evaluation results into CSV tables.

Reads per-image JSON result files produced by image_evaluation.py and the
dataset JSONL (with embedded easy_qidxs / hard_qidxs). Generates two CSVs:

- One grouped by **domain** (slides, webpage, chart, poster, scientific_figure)
- One grouped by **dimension** (layout, attribute, text, knowledge)

Each CSV contains score rows for "easy", "hard", and "all" subsets.
Easy/hard splits are determined per-question using the easy_qidxs and
hard_qidxs fields in the dataset.

Usage:
    python -m evaluation.summarize \
        --data_path assets/bizgeneval.jsonl \
        --result_dir <evaluation_output_dir> \
        --save_dir <summary_output_dir>
"""

import os
import json
import csv
import argparse
from collections import defaultdict
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize evaluation results into CSV tables.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset JSONL (with easy_qidxs/hard_qidxs).")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory containing per-image evaluation JSONs (from image_evaluation.py).")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save summary CSV files.")
    parser.add_argument("--error_alpha", type=float, default=0.1, help="Penalty coefficient: score = max(0, 1 - alpha * errors).")
    return parser.parse_args()


def _load_jsonl(path):
    rows = []
    if not path or not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _infer_image_name(item):
    for k in ("reference_image", "reference image", "image_path", "image", "path"):
        ref = item.get(k)
        if isinstance(ref, str) and ref.strip():
            return os.path.basename(ref.strip())
    domain = str(item.get("domain", "")).strip()
    dimension = str(item.get("dimension", "")).strip()
    idx = item.get("id")
    return f"{domain}_{dimension}_{idx}.png"


def _result_filename(item):
    fname = _infer_image_name(item)
    return fname.rsplit(".", 1)[0] + ".json"


def _load_result(result_dir, item):
    """Load the evaluation result JSON for a dataset item."""
    result_file = _result_filename(item)
    result_path = os.path.join(result_dir, result_file)
    if not os.path.isfile(result_path):
        return None
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _compute_scores(items, result_dir, error_alpha, qidxs_key=None):
    """
    Compute per-item scores and return list of score records.

    If qidxs_key is given (e.g. "easy_qidxs"), only questions at those
    1-based indices are counted. Items without that key are skipped.
    """
    records = []
    for item in items:
        result = _load_result(result_dir, item)
        if result is None:
            continue

        meta_info = result.get("meta_info") or {}
        questions = item.get("questions", [])
        n_total = len(questions)
        if n_total == 0 or not meta_info:
            continue

        # Determine which question indices to evaluate
        if qidxs_key:
            qidxs = item.get(qidxs_key, [])
            if not qidxs:
                continue
        else:
            qidxs = list(range(1, n_total + 1))

        n_questions = len(qidxs)
        errors = 0
        for j in qidxs:
            rec = meta_info.get(str(j), {})
            if rec.get("result") is not True:
                errors += 1

        accuracy = (n_questions - errors) / n_questions if n_questions > 0 else 0.0
        error_score = max(0.0, 1.0 - error_alpha * errors)

        records.append({
            "domain": item.get("domain", ""),
            "dimension": item.get("dimension", ""),
            "accuracy": accuracy,
            "error_score": error_score,
            "n_questions": n_questions,
            "errors": errors,
        })
    return records


def _aggregate_by_key(records, key):
    """Group records by key and compute mean scores."""
    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    agg = {}
    for k, recs in sorted(groups.items()):
        agg[k] = {
            "count": len(recs),
            "accuracy": mean([r["accuracy"] for r in recs]) if recs else 0.0,
            "error_score": mean([r["error_score"] for r in recs]) if recs else 0.0,
        }
    return agg


def _write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(v):
    if v is None:
        return ""
    return f"{v:.4f}"


def _build_grouped_csv(all_records, easy_records, hard_records, group_key, group_order, save_path):
    """Build a CSV with rows for easy/hard/all, columns for each group value + Average."""
    group_display = {k: k.replace("_", " ").title() for k in group_order}
    header = ["Split"] + [group_display.get(k, k) for k in group_order] + ["Average"]

    def _make_row(label, records):
        if not records:
            return {col: "" for col in header} | {"Split": label}
        agg = _aggregate_by_key(records, group_key)
        row = {"Split": label}
        vals = []
        for k in group_order:
            if k in agg:
                v = agg[k]["error_score"]
                row[group_display.get(k, k)] = _fmt(v)
                vals.append(v)
            else:
                row[group_display.get(k, k)] = ""
        row["Average"] = _fmt(mean(vals)) if vals else ""
        return row

    rows = []
    if easy_records:
        rows.append(_make_row("easy", easy_records))
    if hard_records:
        rows.append(_make_row("hard", hard_records))
    rows.append(_make_row("all", all_records))

    _write_csv(save_path, header, rows)
    return save_path


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    all_items = _load_jsonl(args.data_path)

    print(f"Dataset: {len(all_items)} items")

    all_records = _compute_scores(all_items, args.result_dir, args.error_alpha)
    easy_records = _compute_scores(all_items, args.result_dir, args.error_alpha * 2, qidxs_key="easy_qidxs")
    hard_records = _compute_scores(all_items, args.result_dir, args.error_alpha * 2, qidxs_key="hard_qidxs")

    print(f"Scored: {len(all_records)} total, {len(easy_records)} easy, {len(hard_records)} hard")

    domain_order = ["slides", "webpage", "poster", "chart", "scientific_figure"]
    dimension_order = ["layout", "attribute", "text", "knowledge"]

    csv1 = _build_grouped_csv(
        all_records, easy_records, hard_records,
        group_key="domain",
        group_order=domain_order,
        save_path=os.path.join(args.save_dir, "summary_by_domain.csv"),
    )

    csv2 = _build_grouped_csv(
        all_records, easy_records, hard_records,
        group_key="dimension",
        group_order=dimension_order,
        save_path=os.path.join(args.save_dir, "summary_by_dimension.csv"),
    )

    # Also save a JSON summary
    summary = {
        "total_items": len(all_items),
        "total_scored": len(all_records),
        "easy_scored": len(easy_records),
        "hard_scored": len(hard_records),
        "error_alpha": args.error_alpha,
        "by_domain": _aggregate_by_key(all_records, "domain"),
        "by_dimension": _aggregate_by_key(all_records, "dimension"),
    }
    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:")
    print(f"  - {csv1}")
    print(f"  - {csv2}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
