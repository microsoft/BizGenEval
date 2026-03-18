"""
Evaluate generated images using checklist-based prompts.

python image_evaluation.py \
    --data_path <input_data_path> \
    --img_dir <input_image_dir> \
    --save_dir <output_save_dir> \
    --only_domain slides \
    --only_dimensions attribute \
    --debug
"""

import sys
sys.path.append(".")
import os
import json
import argparse
import threading
from utils import (
    config_apis,
    parse_json_safe,
    request_i2t_until_success
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from evaluate_prompt import EVAL_GENERATION_PROMPTS



def _infer_image_name(item):
    # Support multiple dataset field conventions.
    # This repo historically used both "reference_image" and "reference image".
    for k in ("reference_image", "reference image", "image_path", "image", "path"):
        ref = item.get(k)
        if isinstance(ref, str) and ref.strip():
            return os.path.basename(ref.strip())
    application = str(item.get("domain", "")).strip()
    domain = str(item.get("dimension", "")).strip()
    idx = item.get("id")
    return f"{application}_{domain}_{idx}.png"


def parse_args():
    parser = argparse.ArgumentParser(description="Build attribute prompt data for image generation.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data file (JSON or JSONL).")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--config_path", type=str, default="config/default_config.yaml", help="Path to API configuration YAML file.")
    parser.add_argument("--only_domain", nargs="+", default=None, help="Only evaluate items from the specified application.")
    parser.add_argument("--only_dimensions", nargs="+", default=None, help="Only evaluate items from the specified domain.")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode.")
    return parser.parse_args()

def main():
    args = parse_args()
    api_configs = config_apis(args.config_path)

    lock = threading.Lock()
    os.makedirs(args.save_dir, exist_ok=True)
    
    data_lists = []
    with open(args.data_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data_lists.append(item)

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

    def _safe_bool(x):
        return True if x is True else False

    def _call_i2t_with_retries(
        img_path,
        user_prompt,
        system_prompt,
        url,
        headers,
        debug,
        reasoning_effort=None,
        max_attempts=3,
        max_retries=5,
        sleep_time=5,
    ):
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                out, ok = request_i2t_until_success(
                    img_path,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    url=url,
                    api_key=headers,
                    debug=debug,
                    max_retries=max_retries,
                    sleep_time=sleep_time,
                    reasoning_effort=reasoning_effort,
                )
                if not ok:
                    last_err = out
                    continue
                return out, True, None
            except Exception as e:
                last_err = str(e)
        return None, False, last_err

    def evaluate_item(item):
        application = item["domain"]
        domain = item["dimension"]
        idx = item["id"]
        eval_tag = item["eval_tag"]
        fname = _infer_image_name(item)
        if "questions" not in item or not item["questions"]:
            print(f"No questions found for {fname}, skipping.")
            return False
        if args.only_domain is not None and application not in args.only_domain:
            return False
        if args.only_dimensions is not None and domain not in args.only_dimensions:
            return False

        all_questions = list(item["questions"])
        # Preserve original 1-based indices for stable merging.
        orig_qidxs = list(item.get("orig_qidxs") or [])
        if not orig_qidxs:
            orig_qidxs = list(range(1, len(all_questions) + 1))
        total_q_original = int(item.get("total_q_original") or max(orig_qidxs) or len(all_questions))
        raw_prompt = item["prompt"]
        generated_img_path = os.path.join(args.img_dir, f"{fname}")
        save_path = os.path.join(args.save_dir, f"{fname.split('.')[0]}.json")
        EVAL_SYSTEM, EVAL_USER_TEMPLATE = EVAL_GENERATION_PROMPTS[eval_tag]
        
        if not os.path.exists(generated_img_path):
            print(f"Generated image not found: {generated_img_path}")
            return False

        print(
            f"[INFO] {fname} id={idx} azure_q={len(all_questions)} total_q_original={total_q_original}",
            flush=True,
        )

        final_meta = {}

        # 1) Azure batch (Azure-only; Gemini questions are evaluated in a separate script)
        azure_questions = list(all_questions)
        azure_checklist = _format_checklist(azure_questions)
        azure_user_prompt = _render_user_prompt(EVAL_USER_TEMPLATE, azure_checklist, expected_count=len(azure_questions))

        out_text, ok, err = _call_i2t_with_retries(
            generated_img_path,
            user_prompt=azure_user_prompt,
            system_prompt=EVAL_SYSTEM,
            url=api_configs["plain"]["url"],
            headers=api_configs["plain"]["api_key"],
            debug=args.debug,
            reasoning_effort=api_configs["plain"].get("reasoning_effort"),
            max_attempts=3,
            max_retries=5,
            sleep_time=5,
        )

        output_azure = {}
        if ok:
            try:
                output_azure = parse_json_safe(out_text)
            except Exception as e:
                print(f"[WARN] {fname} azure parse_json_safe failed: {e}", flush=True)
                output_azure = {}
        else:
            print(f"[WARN] {fname} azure call failed: {err}", flush=True)

        for j, (qidx, q) in enumerate(zip(orig_qidxs, azure_questions, strict=False), start=1):
            rec = output_azure.get(str(j))
            if not isinstance(rec, dict):
                final_meta[str(qidx)] = {
                    "result": False,
                    "raw_description": q,
                    "reason": "missing_from_azure_output",
                    "use_gemini": 0,
                }
                continue
            final_meta[str(qidx)] = {
                "result": _safe_bool(rec.get("result")),
                "raw_description": q,
                "reason": rec.get("reason", ""),
                "use_gemini": 0,
            }

        true_cnt = sum(1 for k, v in final_meta.items() if v.get("result") is True)
        accuracy_azure_only = true_cnt / len(orig_qidxs) if orig_qidxs else 0.0
        missing_qidxs = [i for i in range(1, total_q_original + 1) if str(i) not in final_meta]
        out_data = {
            "meta_info": final_meta,
            "prompt": raw_prompt,
            "accuracy": accuracy_azure_only,
            "is_correct": (accuracy_azure_only == 1),
            "accuracy_azure_only": accuracy_azure_only,
            "total_q_original": total_q_original,
            "missing_qidxs": missing_qidxs,
        }

        with lock:
            with open(save_path, "w") as f:
                json.dump(out_data, f, indent=2)

        print(f"[INFO] {fname} done accuracy(azure_only)={accuracy_azure_only:.4f} missing={len(missing_qidxs)}", flush=True)
        
        return True

    max_workers = api_configs["plain"]["max_workers"]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_item, item) for item in data_lists]

        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 10 == 0 or i == len(data_lists):
                print(f"Processed {i}/{len(data_lists)} items")


if __name__ == "__main__":
    main()
