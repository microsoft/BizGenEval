import re
import ast
import json
import yaml


def config_apis(config_path: str):
    """configure api keys for different task types."""

    def get_openai_request_url(model_name: str):
        pass

    config = yaml.safe_load(open(config_path, "r"))

    api_keys = {}
    for task_type, config_info in config["models"].items():
        api_model, max_workers = config_info[:2]
        if len(config_info) > 2:
            reasoning_effort = config_info[2]
        else:
            reasoning_effort = "medium" if api_model in ["5.1", "gpt5"] else None
        
        url, api_key = get_openai_request_url(api_model)
        api_keys[task_type] = {
            "url": url,
            "api_key": api_key,
            "max_workers": max_workers,
            "reasoning_effort": reasoning_effort,
        }
        api_keys[task_type].update(**config["config"])
    return api_keys



def parse_json_safe(text):
    """
    Best-effort JSON parser for LLM outputs.
    Handles common formatting issues:
    - Markdown code fences (``` / ```json)
    - Extra text before/after a JSON object (extract outermost {...})
    - Python-style booleans/None (True/False/None) via fallback
    """
    s = (text or "").strip()

    # Strip a single surrounding fenced block, if present.
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
        s = s.strip()

    # 1) Direct JSON.
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) Extract the largest {...} block and try again.
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    s2 = m.group(0).strip() if m else s
    try:
        return json.loads(s2)
    except Exception:
        pass

    # 3) Try JSON after normalizing Python literals -> JSON literals.
    s3 = re.sub(r"\bTrue\b", "true", s2)
    s3 = re.sub(r"\bFalse\b", "false", s3)
    s3 = re.sub(r"\bNone\b", "null", s3)
    try:
        return json.loads(s3)
    except Exception:
        pass

    # 4) Fallback to Python literal eval (handles single quotes / True/False/None).
    s4 = re.sub(r"\bnull\b", "None", s2)
    s4 = re.sub(r"\btrue\b", "True", s4, flags=re.IGNORECASE)
    s4 = re.sub(r"\bfalse\b", "False", s4, flags=re.IGNORECASE)
    return ast.literal_eval(s4)


def request_i2t_until_success(**kwargs):
    pass