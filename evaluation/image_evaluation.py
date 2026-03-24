"""
Evaluate generated images using checklist-based prompts via Gemini API.

Outputs one JSON result file per image. Supports resume (skips existing results).

Usage:
    # Set your API key (or put it in config/evaluation_config.yaml)
    export GEMINI_API_KEY="your-api-key"

    python -m evaluation.image_evaluation \
        --data_path <input_data_path> \
        --img_dir <input_image_dir> \
        --save_dir <output_save_dir> \
        --only_domain slides \
        --only_ids 12 18 25 \
        --only_dimensions attribute \
        --debug
"""

import sys
sys.path.append(".")
import os
import re
import json
import argparse
import threading
from utils import (
    load_config,
    create_gemini_client,
    request_gemini_i2t,
    parse_json_safe,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from evaluation.evaluate_prompt import EVAL_GENERATION_PROMPTS


def _to_bool(v):
    """Robust boolean coercion for LLM outputs."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv in {"true", "1", "yes"}:
            return True
        if vv in {"false", "0", "no"}:
            return False
    return None


def _extract_results_only(raw_text: str, n_questions: int):
    """
    Fallback parser: extract per-question boolean results from malformed JSON.
    Used when parse_json_safe fails (e.g. unescaped quotes in reason fields).
    """
    if not isinstance(raw_text, str) or n_questions <= 0:
        return None
    matches = re.findall(
        r'"result"\s*:\s*(true|false|True|False|"true"|"false"|1|0)',
        raw_text,
        flags=re.DOTALL,
    )
    if len(matches) < n_questions:
        return None
    vals = []
    for m in matches[:n_questions]:
        mm = str(m).strip().strip('"')
        b = _to_bool(mm)
        if b is None and mm in {"1", "0"}:
            b = (mm == "1")
        if b is None:
            return None
        vals.append(b)
    return {str(i + 1): {"result": vals[i]} for i in range(n_questions)}


def _infer_image_name(item):
    for k in ("reference_image", "reference image", "image_path", "image", "path"):
        ref = item.get(k)
        if isinstance(ref, str) and ref.strip():
            return os.path.basename(ref.strip())
    domain = str(item.get("domain", "")).strip()
    dimension = str(item.get("dimension", "")).strip()
    idx = item.get("id")
    return f"{domain}_{dimension}_{idx}.png"


def _item_matches_filters(item, args):
    domain = item.get("domain", "")
    dimension = item.get("dimension", "")
    item_id = str(item.get("id"))

    if args.only_domain is not None and domain not in args.only_domain:
        return False
    if args.only_dimensions is not None and dimension not in args.only_dimensions:
        return False
    if args.only_ids is not None and item_id not in args.only_ids:
        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated images using checklist-based prompts.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data file (JSONL).")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing generated images.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save evaluation result JSONs.")
    parser.add_argument("--config_path", type=str, default="config/evaluation_config.yaml", help="Path to API configuration YAML file.")
    parser.add_argument("--only_domain", nargs="+", default=None, help="Only evaluate items from the specified domain (e.g. slides webpage).")
    parser.add_argument("--only_dimensions", nargs="+", default=None, help="Only evaluate items from the specified dimension (e.g. attribute layout).")
    parser.add_argument("--only_ids", nargs="+", default=None, help="Only evaluate items with the specified ids.")
    parser.add_argument("--force_rerun", action="store_true", help="Ignore cached results and rerun all evaluations.")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode.")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config_path)
    client = create_gemini_client(config)

    model_name = config.get("model", "gemini-3-flash-preview")
    max_workers = int(config.get("max_workers", 16))
    max_retries = int(config.get("max_retries", 5))
    sleep_time = int(config.get("sleep_time", 5))

    lock = threading.Lock()
    os.makedirs(args.save_dir, exist_ok=True)
    
    data_lists = []
    with open(args.data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data_lists.append(json.loads(line))

    data_lists = [item for item in data_lists if _item_matches_filters(item, args)]
    print(f"[INFO] queued {len(data_lists)} items for evaluation", flush=True)

    def _format_checklist(questions):
        return "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    def _format_required_keys(n: int) -> str:
        return ", ".join(str(i) for i in range(1, n + 1))

    def _render_user_prompt(template: str, checklist: str, expected_count: int) -> str:
        kwargs = {"checklist": checklist}
        if "{expected_count}" in template:
            kwargs["expected_count"] = int(expected_count)
        if "{required_keys}" in template:
            kwargs["required_keys"] = _format_required_keys(int(expected_count))
        return template.format(**kwargs)

    def _is_result_complete(save_path):
        """Check if a previous result exists, succeeded, and has no missing answers."""
        if not os.path.exists(save_path):
            return False
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if obj.get("accuracy") is None:
                return False
            meta = obj.get("meta_info") or {}
            for v in meta.values():
                if isinstance(v, dict) and v.get("reason") == "missing_from_output":
                    return False
            return True
        except Exception:
            return False

    def evaluate_item(item):
        idx = item.get("id")
        eval_tag = item.get("eval_tag", "")
        fname = _infer_image_name(item)

        assert eval_tag in EVAL_GENERATION_PROMPTS, f"Eval tag '{eval_tag}' not found in prompts config"
        
        if "questions" not in item or not item["questions"]:
            print(f"No questions found for {fname}, skipping.")
            return False
        
        save_path = os.path.join(args.save_dir, f"{fname.split('.')[0]}.json")

        # Resume: skip if result already exists
        if not args.force_rerun and _is_result_complete(save_path):
            return True
        
        all_questions = list(item["questions"])
        n_questions = len(all_questions)
        raw_prompt = item.get("prompt", "")
        generated_img_path = os.path.join(args.img_dir, fname)
        EVAL_SYSTEM, EVAL_USER_TEMPLATE = EVAL_GENERATION_PROMPTS[eval_tag]
        
        if not os.path.exists(generated_img_path):
            print(f"Generated image not found: {generated_img_path}")
            return False

        print(
            f"[INFO] {fname} id={idx} questions={n_questions}",
            flush=True,
        )

        checklist = _format_checklist(all_questions)
        user_prompt = _render_user_prompt(EVAL_USER_TEMPLATE, checklist, expected_count=n_questions)

        out_text, ok = request_gemini_i2t(
            client,
            model=model_name,
            img_path=generated_img_path,
            user_prompt=user_prompt,
            system_prompt=EVAL_SYSTEM,
            max_retries=max_retries,
            sleep_time=sleep_time,
            debug=args.debug,
        )

        parsed = {}
        if ok:
            try:
                parsed = parse_json_safe(out_text)
            except Exception:
                parsed = _extract_results_only(out_text, n_questions)
                if parsed is None:
                    parsed = {}
        else:
            if args.debug:
                print(f"  [DEBUG] {fname} API failed: {out_text[:120]}", flush=True)

        n_valid = sum(1 for j in range(1, n_questions + 1)
                      if isinstance(parsed.get(str(j)), dict))
        n_missing = n_questions - n_valid

        if n_missing > 0:
            print(f"[WARN] {fname} {n_missing}/{n_questions} missing (will retry on next resume)", flush=True)

        final_meta = {}
        for j, q in enumerate(all_questions, start=1):
            rec = parsed.get(str(j)) if isinstance(parsed, dict) else None
            if not isinstance(rec, dict):
                final_meta[str(j)] = {
                    "result": False,
                    "raw_description": q,
                    "reason": "missing_from_output",
                }
                continue
            val = _to_bool(rec.get("result"))
            final_meta[str(j)] = {
                "result": val if isinstance(val, bool) else False,
                "raw_description": q,
                "reason": rec.get("reason", ""),
            }

        true_cnt = sum(1 for v in final_meta.values() if v.get("result") is True)
        accuracy = true_cnt / n_questions if n_questions > 0 else 0.0

        out_data = {
            "meta_info": final_meta,
            "prompt": raw_prompt,
            "accuracy": accuracy,
            "is_correct": (accuracy == 1),
            "n_questions": n_questions,
        }

        with lock:
            with open(save_path, "w") as f:
                json.dump(out_data, f, indent=2)

        print(f"[INFO] {fname} done accuracy={accuracy:.4f}", flush=True)
        
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_item, item) for item in data_lists]

        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 10 == 0 or i == len(data_lists):
                print(f"Processed {i}/{len(data_lists)} items")


if __name__ == "__main__":
    main()
