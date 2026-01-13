# Scripts Directory

This directory contains standalone Python scripts for evaluation and analysis of trained machine translation models.

## Available Scripts

### evaluation_code.py

**Purpose:** Comprehensive evaluation script for trained MT models

**Description:**
A standalone script to evaluate machine translation models on test data, computing multiple metrics and generating detailed reports. This script can be used independently of the Jupyter notebooks for batch evaluation and automated testing.

**Features:**
- Load and evaluate MBART-50 and NLLB-200 models
- Compute BLEU, chrF++, and BERTScore metrics
- Generate translations for entire test sets
- Export results in multiple formats (CSV, JSON)
- Batch processing support
- Error analysis and reporting

**Usage:**

#### Basic Evaluation
```bash
python scripts/evaluation_code.py \
    --model_path models/mbart50/bidirectional \
    --test_file data/splits/test.csv \
    --output_dir results/
```

#### Advanced Usage
```bash
python scripts/evaluation_code.py \
    --model_path models/nllb200/ne_en \
    --test_file data/splits/test.csv \
    --output_dir results/nllb_eval \
    --source_lang ne_NP \
    --target_lang en_XX \
    --batch_size 16 \
    --max_length 128 \
    --compute_bertscore
```

#### Command-Line Arguments

```
Required Arguments:
  --model_path PATH          Path to trained model checkpoint
  --test_file PATH          Path to test CSV file
  --output_dir PATH         Directory to save results

Optional Arguments:
  --source_lang LANG        Source language code (default: auto-detect)
  --target_lang LANG        Target language code (default: auto-detect)
  --batch_size N            Batch size for evaluation (default: 8)
  --max_length N            Maximum sequence length (default: 128)
  --num_beams N             Number of beams for beam search (default: 5)
  --compute_bertscore       Compute BERTScore (slower but more accurate)
  --save_translations       Save generated translations to file
  --device DEVICE           Device to use: 'cuda' or 'cpu' (default: auto)
```

**Output Files:**

The script generates the following outputs in `--output_dir`:

1. **metrics.json** - Computed metrics
```json
{
  "bleu": 53.97,
  "chrf": 66.23,
  "bertscore_f1": 0.89,
  "length_ratio": 0.918
}
```

2. **translations.csv** - Generated translations
```csv
source,reference,prediction,bleu,chrf
"Source sentence","Reference","Prediction",52.3,65.1
...
```

3. **evaluation_report.txt** - Detailed evaluation report

**Example:**
```bash
# Evaluate MBART bi-directional model
python scripts/evaluation_code.py \
    --model_path models/mbart50/bidirectional \
    --test_file data/splits/test.csv \
    --output_dir results/mbart_bi_eval \
    --compute_bertscore \
    --save_translations

# Output:
# Loading model from models/mbart50/bidirectional...
# Loading test data from data/splits/test.csv...
# Found 503 test examples
# Generating translations...
# 100%|████████████| 503/503 [02:15<00:00, 3.71it/s]
# Computing metrics...
# BLEU: 53.97
# chrF++: 66.23
# BERTScore F1: 0.89
# Results saved to results/mbart_bi_eval/
```

**Python API:**

You can also import and use the evaluation functions in your own scripts:

```python
from scripts.evaluation_code import evaluate_model, compute_metrics

# Evaluate a model
results = evaluate_model(
    model_path="models/mbart50/bidirectional",
    test_file="data/splits/test.csv",
    output_dir="results/",
    batch_size=8,
    device="cuda"
)

print(f"BLEU: {results['bleu']:.2f}")
print(f"chrF++: {results['chrf']:.2f}")

# Or compute metrics for existing translations
from sacrebleu import corpus_bleu

metrics = compute_metrics(
    predictions=["Translation 1", "Translation 2"],
    references=["Reference 1", "Reference 2"]
)
```

**Performance:**
- CPU: ~5-10 minutes for 503 test examples
- GPU (T4): ~2-3 minutes for 503 test examples
- GPU (V100/A100): ~1-2 minutes for 503 test examples

**Requirements:**
```python
transformers>=4.35.0
datasets>=2.14.0
sacrebleu>=2.3.1
evaluate>=0.4.0
bert-score>=0.3.13  # Optional, for BERTScore
pandas>=2.0.0
torch>=2.0.0
tqdm>=4.65.0
```

---

## Adding New Scripts

To add a new script to this directory:

1. **Create the script file:**
```bash
touch scripts/your_script.py
```

2. **Add a main function:**
```python
#!/usr/bin/env python3
"""
Description of your script.
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    # Your code here
    print(f"Processing {args.input}...")

if __name__ == "__main__":
    main()
```

3. **Make it executable:**
```bash
chmod +x scripts/your_script.py
```

4. **Update this README** with documentation

---

## Common Use Cases

### 1. Batch Model Evaluation

Evaluate all models in a loop:

```bash
#!/bin/bash

MODELS=(
    "models/mbart50/bidirectional"
    "models/mbart50/ne_en"
    "models/mbart50/en_ne"
    "models/nllb200/bidirectional"
    "models/nllb200/ne_en"
    "models/nllb200/en_ne"
)

for model in "${MODELS[@]}"; do
    echo "Evaluating $model..."
    python scripts/evaluation_code.py \
        --model_path "$model" \
        --test_file data/splits/test.csv \
        --output_dir "results/eval_$(basename $model)"
done
```

### 2. Compare Multiple Models

```python
import pandas as pd
from scripts.evaluation_code import evaluate_model

models = {
    "MBART Bi-directional": "models/mbart50/bidirectional",
    "MBART Uni NE→EN": "models/mbart50/ne_en",
    "NLLB Bi-directional": "models/nllb200/bidirectional",
    "NLLB Uni NE→EN": "models/nllb200/ne_en",
}

results = []
for name, path in models.items():
    print(f"Evaluating {name}...")
    metrics = evaluate_model(path, "data/splits/test.csv", f"results/{name}")
    results.append({
        "Model": name,
        "BLEU": metrics['bleu'],
        "chrF++": metrics['chrf']
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv("results/model_comparison.csv", index=False)
```

### 3. Custom Evaluation Pipeline

```python
from scripts.evaluation_code import load_model, generate_translations, compute_metrics
import pandas as pd

# Load model
model, tokenizer = load_model("models/mbart50/bidirectional")

# Load test data
df = pd.read_csv("data/splits/test.csv")
sources = df['nepali'].tolist()
references = df['english'].tolist()

# Generate translations
predictions = generate_translations(
    model=model,
    tokenizer=tokenizer,
    sources=sources,
    src_lang="ne_NP",
    tgt_lang="en_XX",
    batch_size=16
)

# Compute metrics
metrics = compute_metrics(predictions, references)
print(f"BLEU: {metrics['bleu']:.2f}")
print(f"chrF++: {metrics['chrf']:.2f}")

# Error analysis
for i, (src, ref, pred) in enumerate(zip(sources, references, predictions)):
    if metrics['sentence_bleu'][i] < 30:  # Low BLEU
        print(f"\nLow-quality translation #{i}:")
        print(f"Source: {src}")
        print(f"Reference: {ref}")
        print(f"Prediction: {pred}")
        print(f"Sentence BLEU: {metrics['sentence_bleu'][i]:.2f}")
```

---

## Troubleshooting

### Common Issues

**1. Model Loading Errors**
```
Error: Can't load model from path
```
**Solution:** Verify model path and ensure all checkpoint files exist:
```bash
ls -la models/mbart50/bidirectional/
# Should contain: pytorch_model.bin, config.json, tokenizer_config.json
```

**2. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size:
```bash
python scripts/evaluation_code.py --batch_size 4  # or 2
```

**3. Slow Evaluation**
```
Evaluation is very slow on CPU
```
**Solution:** Use GPU if available:
```bash
python scripts/evaluation_code.py --device cuda
```

**4. Import Errors**
```
ModuleNotFoundError: No module named 'evaluation_code'
```
**Solution:** Run from repository root:
```bash
cd /path/to/Legal_NLP
python scripts/evaluation_code.py ...
```

---

## Best Practices

1. **Always use absolute paths** when running scripts
2. **Check GPU availability** before large evaluations
3. **Save results with timestamps** for version control
4. **Use meaningful output directory names**
5. **Document any script modifications**
6. **Test on small batches first** before full evaluation

---

## Development

To extend or modify the evaluation script:

1. **Fork the script:**
```bash
cp scripts/evaluation_code.py scripts/evaluation_code_custom.py
```

2. **Modify as needed** while maintaining the core structure

3. **Test thoroughly:**
```bash
python scripts/evaluation_code_custom.py --model_path models/test --test_file data/splits/val.csv --output_dir results/test
```

4. **Document changes** in this README

---

## Future Scripts

Planned additions to this directory:

- **data_preprocessing.py** - Data cleaning and preprocessing
- **model_comparison.py** - Automated model comparison
- **translation_server.py** - REST API for translation service
- **batch_translate.py** - Batch translation for production
- **error_analysis.py** - Detailed error pattern analysis

---

## Contributing

To contribute new scripts:
1. Follow Python best practices (PEP 8)
2. Add argparse for command-line arguments
3. Include docstrings and comments
4. Add error handling
5. Test on both CPU and GPU
6. Update this README with documentation

---

## Citation

If you use these scripts in your research:

```bibtex
@software{legal_nlp_scripts_2025,
  title={Legal NLP Evaluation Scripts},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/Legal_NLP}
}
```

---

**Last Updated:** January 2025

For questions or issues with scripts, please open a GitHub issue or contact [your.email@example.com].
