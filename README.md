# Legal_NLP: English-Nepali Legal Domain Machine Translation

A comprehensive research project focused on neural machine translation between English and Nepali in the legal domain, leveraging state-of-the-art multilingual transformer models.

## Overview

This repository contains the implementation, datasets, and experimental results for training and evaluating neural machine translation models on English-Nepali legal text pairs. The project investigates both uni-directional and bi-directional fine-tuning strategies using MBART-50 and NLLB-200 models.

### Key Results

Our experiments demonstrate superior performance with bi-directional training for NE→EN translation and competitive results across different configurations:

| Model | Configuration | Direction | BLEU | chrF++ | Length Ratio |
|-------|---------------|-----------|------|--------|--------------|
| **MBART** | Bi-directional | NE→EN | **53.97** | **66.23** | 0.918 |
| MBART | Uni-directional | NE→EN | 49.05 | 64.51 | 0.929 |
| MBART | Uni-directional | EN→NE | 36.60 | 64.33 | 0.872 |
| MBART | Bi-directional | EN→NE | 30.26 | 62.36 | 0.887 |
| **NLLB** | Bi-directional | NE→EN | **51.44** | 60.73 | 0.932 |
| NLLB | Uni-directional | NE→EN | 46.51 | 61.35 | 0.929 |
| NLLB | Bi-directional | EN→NE | 27.19 | 62.20 | 0.889 |
| NLLB | Uni-directional | EN→NE | 25.71 | 60.72 | 0.901 |

**Highlights:**
- Best BLEU score: **53.97** (MBART Bi-directional NE→EN)
- Best chrF++ score: **66.23** (MBART Bi-directional NE→EN)
- Bi-directional training shows +4.92 BLEU improvement for MBART NE→EN
- Bi-directional training shows +4.93 BLEU improvement for NLLB NE→EN

## Repository Structure

```
Legal_NLP/
├── data/                           # Dataset files
│   ├── raw/                        # Original raw data
│   ├── processed/                  # Preprocessed data
│   └── splits/                     # Train/val/test splits
│       ├── train.csv              # Training set (4,019 pairs)
│       ├── val.csv                # Validation set (503 pairs)
│       └── test.csv               # Test set (503 pairs)
├── src/                            # Source code
│   ├── preprocessing/              # Data preprocessing modules
│   ├── training/                   # Model training modules
│   ├── evaluation/                 # Evaluation metrics and analysis
│   ├── visualization/              # Plotting and visualization
│   └── utils/                      # Utility functions
├── notebooks/                      # Jupyter notebooks
│   ├── mbart50_training.ipynb     # MBART-50 training notebook
│   ├── nllb200_training.ipynb     # NLLB-200 training notebook
│   ├── results_analysis.ipynb     # Results analysis and visualization
│   └── training_code.ipynb        # General training code
├── scripts/                        # Standalone scripts
│   └── evaluation_code.py         # Evaluation script
├── models/                         # Saved model checkpoints
│   ├── mbart50/                   # MBART-50 models
│   │   ├── en_ne/                 # English to Nepali
│   │   ├── ne_en/                 # Nepali to English
│   │   └── bidirectional/         # Bi-directional model
│   └── nllb200/                   # NLLB-200 models
│       ├── en_ne/                 # English to Nepali
│       ├── ne_en/                 # Nepali to English
│       └── bidirectional/         # Bi-directional model
├── results/                        # Experimental results
│   ├── tables/                     # Result tables (CSV format)
│   ├── figures/                    # Generated figures
│   ├── visualizations/             # Additional visualizations
│   ├── translations/               # Model translations
│   ├── metrics/                    # Detailed metrics
│   └── error_analysis/             # Error analysis reports
├── reference_papers/               # Related research papers
├── paper/                          # Research paper source files
├── requirements.txt                # Python dependencies
├── run_all_experiments.py         # Master experiment script
└── README.md                       # This file
```

## Dataset Statistics

Our legal domain parallel corpus consists of **5,024 English-Nepali sentence pairs** extracted from legal documents, regulations, and judicial texts.

| Statistic | Value |
|-----------|-------|
| Total Pairs | 5,024 |
| Train / Val / Test | 4,019 / 502 / 503 |
| Avg English Length | 34.03 tokens |
| Avg Nepali Length | 24.86 tokens |
| Max English Length | 721 tokens |
| Max Nepali Length | 537 tokens |
| Min English Length | 1 token |
| Min Nepali Length | 1 token |

The dataset is split into:
- **Training set:** 80% (4,019 pairs)
- **Validation set:** 10% (502 pairs)
- **Test set:** 10% (503 pairs)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Legal_NLP
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies

- **PyTorch** (≥2.0.0): Deep learning framework
- **Transformers** (≥4.35.0): Hugging Face model library
- **Datasets** (≥2.14.0): Dataset processing
- **SacreBLEU** (≥2.3.1): BLEU score evaluation
- **Evaluate** (≥0.4.0): Evaluation metrics
- **BERTScore** (≥0.3.13): Semantic similarity metrics

See `requirements.txt` for the complete list.

## Usage

### Training Models

#### MBART-50 Training

Use the provided Jupyter notebooks for interactive training:

```bash
jupyter notebook notebooks/mbart50_training.ipynb
```

Or run programmatically:
```python
from src.training import train_mbart
train_mbart(
    train_file="data/splits/train.csv",
    val_file="data/splits/val.csv",
    output_dir="models/mbart50/bidirectional",
    num_epochs=10,
    batch_size=8
)
```

#### NLLB-200 Training

```bash
jupyter notebook notebooks/nllb200_training.ipynb
```

#### Run All Experiments

To reproduce all experiments:
```bash
python run_all_experiments.py
```

### Evaluation

Evaluate a trained model:

```bash
python scripts/evaluation_code.py \
    --model_path models/mbart50/bidirectional \
    --test_file data/splits/test.csv \
    --output_dir results/
```

Or use the evaluation module:
```python
from src.evaluation import evaluate_model

results = evaluate_model(
    model_path="models/mbart50/bidirectional",
    test_file="data/splits/test.csv"
)
print(f"BLEU: {results['bleu']:.2f}")
print(f"chrF++: {results['chrf']:.2f}")
```

### Analysis and Visualization

Analyze results and generate visualizations:

```bash
jupyter notebook notebooks/results_analysis.ipynb
```

## Model Architectures

### MBART-50
- **Base Model:** `facebook/mbart-large-50-many-to-many-mmt`
- **Parameters:** 610M
- **Languages:** 50 languages including English and Nepali
- **Architecture:** Encoder-decoder transformer

### NLLB-200
- **Base Model:** `facebook/nllb-200-distilled-600M`
- **Parameters:** 600M (distilled version)
- **Languages:** 200 languages including English and Nepali
- **Architecture:** Encoder-decoder transformer

## Training Configurations

### Uni-directional Training
- Separate models for EN→NE and NE→EN
- Optimized for single translation direction
- Lower training time and resource requirements

### Bi-directional Training
- Single model handles both EN→NE and NE→EN
- Improved cross-lingual understanding
- Better performance on NE→EN translation (+4.92 BLEU for MBART)

### Hyperparameters
- **Learning rate:** 2e-5
- **Batch size:** 8 (effective batch size with gradient accumulation)
- **Epochs:** 10
- **Max sequence length:** 128
- **Optimizer:** AdamW
- **Scheduler:** Linear warmup with decay

## Evaluation Metrics

- **BLEU (BiLingual Evaluation Understudy):** Measures n-gram overlap between generated and reference translations
- **chrF++ (Character n-gram F-score):** Character-level metric more robust to morphological variations
- **Length Ratio:** Ratio of generated to reference translation length

## Results and Analysis

Detailed results are available in:
- `results/tables/table1_main_results.csv` - Main BLEU, chrF++, and length ratio scores
- `results/tables/table2_dataset_statistics.csv` - Dataset statistics
- `results/translations/` - Generated translations for all model configurations
- `results/visualizations/` - Performance comparison charts

Key findings:
1. Bi-directional training significantly improves NE→EN translation (+4.92 BLEU for MBART)
2. MBART outperforms NLLB across most configurations
3. EN→NE translation remains challenging (lower BLEU scores)
4. chrF++ scores indicate reasonable character-level accuracy

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{legal_nlp_2025,
  title={Neural Machine Translation for English-Nepali Legal Domain: 
         A Comparative Study of MBART and NLLB Models},
  author={[Your Name]},
  year={2025},
  publisher={GitHub},
  howpublished={\\url{https://github.com/yourusername/Legal_NLP}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MBART-50 model: Facebook AI Research
- NLLB-200 model: Meta AI (No Language Left Behind)
- Hugging Face Transformers library
- Legal domain data sources and contributors

## Contact

For questions, issues, or collaboration opportunities, please:
- Open an issue on GitHub
- Contact: [your.email@example.com]

## References

Related papers are available in the `reference_papers/` directory. Key references include:
- MBART: Multilingual Denoising Pre-training for Neural Machine Translation
- NLLB: No Language Left Behind - Scaling Human-Centered Machine Translation
- Legal NLP and domain-specific translation research

---

**Note:** This is a research project. Model performance may vary based on specific use cases and domains. Always validate translations for critical legal applications.
