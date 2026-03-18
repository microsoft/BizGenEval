# BizGenEval: A Systematic Benchmark for Commercial Visual Content Generation

BizGenEval is a benchmark for evaluating image generation models on real-world commercial design tasks. It covers **5 document types** (slides, charts, webpages, posters, scientific figures) × **4 capability dimensions** (text rendering, layout control, attribute binding, knowledge reasoning) = **20 evaluation tasks**, with 400 curated prompts and 8,000 checklist questions.



![](assets/three_plots_in_one_row.png)

## Dataset Format

Each entry in the `.jsonl` dataset follows this schema:

```json
{
  "id": 0,
  "prompt": "Generate a slide with ...",
  "domain": "slides|webpage|chart|poster|scientific_figure",
  "dimension": "layout|attribute|text|knowledge",
  "aspect_ratio": "16:9",
  "questions": [
    "question_1",
    "question_2"
  ],
  "eval_tag": "key_in_EVAL_GENERATION_PROMPTS"
}
```

## Evaluation

Configure your API in `config/default_config.yaml`, then run:

```bash
python image_evaluation.py \
    --data_path <data.jsonl> \
    --img_dir <generated_images_dir> \
    --save_dir <output_dir>
```

**Options:**

| Argument | Description |
|---|---|
| `--only_domain` | Filter by application domain (e.g. `slides webpage`) |
| `--only_dimensions` | Filter by dimension (e.g. `attribute layout`) |
| `--debug` | Enable debug output |


