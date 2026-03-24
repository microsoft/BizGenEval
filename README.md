# BizGenEval: A Systematic Benchmark for Commercial Visual Content Generation

BizGenEval is a benchmark for evaluating image generation models on real-world commercial design tasks. It covers **5 document types** (slides, charts, webpages, posters, scientific figures) × **4 capability dimensions** (text rendering, layout control, attribute binding, knowledge reasoning) = **20 evaluation tasks**, with 400 curated prompts and 8,000 checklist questions.


![](assets/three_plots_in_one_row.png)

## Installation

```
conda create -n bizgeneval python==3.12 -y
source activate bizgeneval
pip install -r requirements.txt
```

## Dataset Format

Each entry in `assets/bizgeneval.jsonl` follows this schema:

```json
{
  "id": 0,
  "prompt": "Generate a slide with ...",
  "domain": "slides|webpage|chart|poster|scientific_figure",
  "dimension": "layout|attribute|text|knowledge",
  "aspect_ratio": "16:9",
  "reference_image_wh": "2400x1800",
  "questions": ["question_1", "question_2", "..."],
  "eval_tag": "key_in_EVAL_GENERATION_PROMPTS",
  "easy_qidxs": [1, 2, 3],
  "hard_qidxs": [4, 5, 6]
}
```

- `easy_qidxs` / `hard_qidxs`: 1-based indices into `questions`, indicating difficulty split.

## Image Generation

See `config/generation_config.yaml` for model configuration, then run:

```bash
python -m generation.image_generation \
    --data_path assets/bizgeneval.jsonl \
    --save_dir outputs/generated_images \
    --resolution_mode dynamic_original \
    --skip_existing
```

## Evaluation

Set your Gemini API key, then run:

```bash
export GEMINI_API_KEY="your-api-key"

python -m evaluation.image_evaluation \
    --data_path assets/bizgeneval.jsonl \
    --img_dir outputs/generated_images \
    --save_dir outputs/eval_results
```

**Options:**

| Argument | Description |
|---|---|
| `--only_domain` | Filter by domain (e.g. `slides webpage`) |
| `--only_dimensions` | Filter by dimension (e.g. `attribute layout`) |
| `--force_rerun` | Re-evaluate even if results already exist |
| `--debug` | Enable debug output |

The evaluation supports **resume** — existing result files are skipped automatically
unless `--force_rerun` is specified.

## Summary

After evaluation, generate summary CSV tables:

```bash
python -m evaluation.summarize \
    --data_path assets/bizgeneval.jsonl \
    --result_dir outputs/eval_results \
    --save_dir outputs/summary
```

This produces:
- `summary_by_domain.csv` — scores grouped by domain (slides, webpage, etc.)
- `summary_by_dimension.csv` — scores grouped by dimension (layout, attribute, etc.)
- `summary.json` — full summary with per-group statistics

Each CSV includes rows for `easy`, `hard`, and `all` subsets.