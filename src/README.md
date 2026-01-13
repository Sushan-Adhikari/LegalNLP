# Source Code Documentation

This directory contains the core source code modules for the Legal_NLP project, organized into modular components for preprocessing, training, evaluation, and visualization.

## Directory Structure

```
src/
├── __init__.py                 # Package initialization
├── preprocessing/              # Data preprocessing modules
│   ├── __init__.py
│   ├── data_loader.py         # Data loading utilities
│   └── dataset.py             # Custom dataset classes
├── training/                   # Model training modules
│   ├── __init__.py
│   ├── train_mbart.py         # MBART training functions
│   └── train_nllb.py          # NLLB training functions
├── evaluation/                 # Evaluation and metrics
│   ├── __init__.py
│   ├── metrics.py             # Metric computation
│   └── error_analysis.py      # Error analysis tools
├── visualization/              # Plotting and visualization
│   ├── __init__.py
│   └── plot_results.py        # Result visualization
└── utils/                      # Utility functions
```

---

## Module Overview

### 1. Preprocessing (`src/preprocessing/`)

#### data_loader.py
**Purpose:** Load and preprocess parallel corpus data

**Key Functions:**
```python
from src.preprocessing.data_loader import load_data, prepare_datasets

# Load CSV data
train_df, val_df, test_df = load_data(
    train_path="data/splits/train.csv",
    val_path="data/splits/val.csv",
    test_path="data/splits/test.csv"
)

# Prepare HuggingFace datasets
datasets = prepare_datasets(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    tokenizer=tokenizer
)
```

**Features:**
- CSV file loading with validation
- Data cleaning and normalization
- Unicode handling for Devanagari script
- Length filtering and statistics
- Train/val/test split management

---

#### dataset.py
**Purpose:** Custom PyTorch dataset classes for translation

**Key Classes:**
```python
from src.preprocessing.dataset import TranslationDataset

# Create translation dataset
dataset = TranslationDataset(
    data_file="data/splits/train.csv",
    tokenizer=tokenizer,
    source_lang="ne_NP",
    target_lang="en_XX",
    max_length=128
)

# Use with DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch in dataloader:
    input_ids = batch['input_ids']
    labels = batch['labels']
    # Training code here
```

**Features:**
- Efficient data loading with caching
- Dynamic tokenization
- Bi-directional support (simultaneous EN↔NE)
- Batch collation
- Memory-efficient processing

**Classes:**
- `TranslationDataset`: Standard uni-directional translation
- `BidirectionalDataset`: Bi-directional translation dataset
- `DataCollatorForTranslation`: Custom collator for batch creation

---

### 2. Training (`src/training/`)

#### train_mbart.py
**Purpose:** MBART-50 model training functions

**Key Functions:**
```python
from src.training.train_mbart import train_mbart_model, train_bidirectional_mbart

# Uni-directional training (NE→EN)
model = train_mbart_model(
    train_file="data/splits/train.csv",
    val_file="data/splits/val.csv",
    output_dir="models/mbart50/ne_en",
    source_lang="ne_NP",
    target_lang="en_XX",
    num_epochs=10,
    batch_size=8,
    learning_rate=2e-5
)

# Bi-directional training
model = train_bidirectional_mbart(
    train_file="data/splits/train.csv",
    val_file="data/splits/val.csv",
    output_dir="models/mbart50/bidirectional",
    num_epochs=10,
    batch_size=8
)
```

**Features:**
- MBART-50 model initialization
- Custom training loops
- Gradient accumulation
- Learning rate scheduling
- Checkpoint saving
- TensorBoard logging
- Validation during training
- Early stopping support

**Configuration:**
```python
training_args = {
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 500,
    "max_grad_norm": 1.0,
    "fp16": True,  # Mixed precision
    "save_steps": 500,
    "eval_steps": 500,
}
```

---

#### train_nllb.py
**Purpose:** NLLB-200 model training functions

**Key Functions:**
```python
from src.training.train_nllb import train_nllb_model, train_bidirectional_nllb

# Uni-directional training (EN→NE)
model = train_nllb_model(
    train_file="data/splits/train.csv",
    val_file="data/splits/val.csv",
    output_dir="models/nllb200/en_ne",
    source_lang="eng_Latn",
    target_lang="npi_Deva",
    num_epochs=10,
    batch_size=8
)

# Bi-directional training
model = train_bidirectional_nllb(
    train_file="data/splits/train.csv",
    val_file="data/splits/val.csv",
    output_dir="models/nllb200/bidirectional",
    num_epochs=10
)
```

**Features:**
- NLLB-200 model setup
- Language code handling (FLORES-200 format)
- Distilled model support
- Custom tokenizer configuration
- Training progress monitoring
- Automatic mixed precision (AMP)

**NLLB Language Codes:**
- English: `eng_Latn`
- Nepali: `npi_Deva`

---

### 3. Evaluation (`src/evaluation/`)

#### metrics.py
**Purpose:** Metric computation for translation evaluation

**Key Functions:**
```python
from src.evaluation.metrics import (
    compute_bleu,
    compute_chrf,
    compute_bertscore,
    compute_all_metrics
)

# Compute BLEU score
bleu = compute_bleu(
    predictions=["Translation 1", "Translation 2"],
    references=["Reference 1", "Reference 2"]
)

# Compute chrF++ score
chrf = compute_chrf(
    predictions=predictions,
    references=references
)

# Compute all metrics
metrics = compute_all_metrics(
    predictions=predictions,
    references=references,
    compute_bertscore=True
)
# Returns: {'bleu': 53.97, 'chrf': 66.23, 'bertscore_f1': 0.89}
```

**Supported Metrics:**
- **BLEU:** Standard machine translation metric
- **chrF++:** Character-level F-score
- **BERTScore:** Contextual embedding similarity
- **Length Ratio:** Translation length accuracy
- **Sentence-level metrics:** Individual sentence scores

**Features:**
- SacreBLEU integration (official BLEU implementation)
- Corpus-level and sentence-level metrics
- Statistical significance testing
- Bootstrap resampling for confidence intervals

---

#### error_analysis.py
**Purpose:** Detailed error analysis and pattern detection

**Key Functions:**
```python
from src.evaluation.error_analysis import (
    analyze_errors,
    categorize_errors,
    find_problematic_examples,
    generate_error_report
)

# Analyze translation errors
error_stats = analyze_errors(
    predictions=predictions,
    references=references,
    sources=sources
)

# Find low-quality translations
problematic = find_problematic_examples(
    predictions=predictions,
    references=references,
    threshold=30  # BLEU < 30
)

# Generate comprehensive error report
report = generate_error_report(
    predictions=predictions,
    references=references,
    sources=sources,
    output_file="results/error_analysis/report.txt"
)
```

**Analysis Types:**
- Length mismatch errors
- Terminology errors
- Word order errors
- Missing translations
- Over-translation
- Under-translation

**Features:**
- Automatic error categorization
- Pattern detection
- Example extraction
- Statistical summaries
- HTML report generation

---

### 4. Visualization (`src/visualization/`)

#### plot_results.py
**Purpose:** Generate plots and visualizations for results

**Key Functions:**
```python
from src.visualization.plot_results import (
    plot_bleu_comparison,
    plot_training_curves,
    plot_length_distribution,
    plot_error_distribution,
    create_all_plots
)

# Plot BLEU comparison
plot_bleu_comparison(
    results_csv="results/tables/table1_main_results.csv",
    output_file="results/figures/bleu_comparison.png"
)

# Plot training curves
plot_training_curves(
    train_loss=[2.1, 1.8, 1.5, 1.3],
    val_loss=[2.3, 2.0, 1.7, 1.5],
    output_file="results/figures/training_curves.png"
)

# Create all visualizations
create_all_plots(
    results_dir="results/",
    output_dir="results/figures/"
)
```

**Available Plots:**
- BLEU score comparisons (bar charts)
- chrF++ score comparisons
- Training/validation loss curves
- Length distribution histograms
- Error type distributions
- Translation quality heatmaps
- Model performance comparison matrices

**Features:**
- Matplotlib and Seaborn integration
- Publication-quality figures
- Customizable styling
- Multiple output formats (PNG, PDF, SVG)
- Interactive Plotly visualizations

---

### 5. Utils (`src/utils/`)

**Purpose:** Shared utility functions (currently empty, reserved for future use)

**Planned Utilities:**
- Configuration file handling
- Logging utilities
- File I/O helpers
- String processing functions
- Language detection
- Text normalization

---

## Usage Examples

### Complete Training Pipeline

```python
from src.preprocessing.data_loader import load_data
from src.preprocessing.dataset import TranslationDataset
from src.training.train_mbart import train_mbart_model
from src.evaluation.metrics import compute_all_metrics
from src.visualization.plot_results import plot_bleu_comparison

# 1. Load data
train_df, val_df, test_df = load_data(
    train_path="data/splits/train.csv",
    val_path="data/splits/val.csv",
    test_path="data/splits/test.csv"
)

# 2. Train model
model = train_mbart_model(
    train_file="data/splits/train.csv",
    val_file="data/splits/val.csv",
    output_dir="models/mbart50/ne_en",
    source_lang="ne_NP",
    target_lang="en_XX",
    num_epochs=10,
    batch_size=8
)

# 3. Generate predictions
predictions = model.translate(test_df['nepali'].tolist())

# 4. Evaluate
metrics = compute_all_metrics(
    predictions=predictions,
    references=test_df['english'].tolist()
)
print(f"BLEU: {metrics['bleu']:.2f}")

# 5. Visualize
plot_bleu_comparison(
    results_csv="results/tables/table1_main_results.csv",
    output_file="results/figures/comparison.png"
)
```

### Custom Evaluation Pipeline

```python
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.error_analysis import analyze_errors, generate_error_report

# Compute metrics
metrics = compute_all_metrics(predictions, references)

# Analyze errors
error_stats = analyze_errors(predictions, references, sources)

# Generate report
report = generate_error_report(
    predictions=predictions,
    references=references,
    sources=sources,
    output_file="results/error_analysis/detailed_report.html"
)

print(f"Total errors: {error_stats['total_errors']}")
print(f"Average sentence BLEU: {error_stats['avg_sentence_bleu']:.2f}")
```

---

## Installation and Setup

### Install as Package

To use the source modules in your own scripts:

```bash
cd /path/to/Legal_NLP
pip install -e .
```

Then import modules:
```python
from src.preprocessing import load_data
from src.training import train_mbart_model
from src.evaluation import compute_all_metrics
```

### Direct Import (Without Installation)

```python
import sys
sys.path.append('/path/to/Legal_NLP')

from src.preprocessing.data_loader import load_data
from src.training.train_mbart import train_mbart_model
```

---

## Code Style and Conventions

### Python Style
- Follow PEP 8 guidelines
- Use type hints where applicable
- Docstrings for all public functions
- Maximum line length: 100 characters

### Naming Conventions
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Documentation
```python
def compute_bleu(predictions: list, references: list) -> float:
    """
    Compute BLEU score for translations.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        
    Returns:
        BLEU score (0-100 range)
        
    Example:
        >>> bleu = compute_bleu(["hello"], ["hello world"])
        >>> print(f"BLEU: {bleu:.2f}")
    """
    pass
```

---

## Testing

### Unit Tests (Future)

```python
# tests/test_metrics.py
import unittest
from src.evaluation.metrics import compute_bleu

class TestMetrics(unittest.TestCase):
    def test_perfect_bleu(self):
        predictions = ["hello world"]
        references = ["hello world"]
        bleu = compute_bleu(predictions, references)
        self.assertAlmostEqual(bleu, 100.0, places=2)
        
    def test_zero_bleu(self):
        predictions = ["completely different"]
        references = ["hello world"]
        bleu = compute_bleu(predictions, references)
        self.assertLess(bleu, 10.0)
```

### Running Tests

```bash
python -m pytest tests/
```

---

## Contributing

### Adding New Modules

1. **Create module file:**
```bash
touch src/preprocessing/new_module.py
```

2. **Add docstrings and functions:**
```python
"""
New module for [purpose].
"""

def new_function(arg1, arg2):
    """Function description."""
    pass
```

3. **Update `__init__.py`:**
```python
# src/preprocessing/__init__.py
from .data_loader import load_data
from .dataset import TranslationDataset
from .new_module import new_function  # Add this
```

4. **Add tests:**
```bash
touch tests/test_new_module.py
```

5. **Update this README**

---

## API Reference

### Preprocessing

```python
# Data Loading
load_data(train_path, val_path, test_path) -> tuple
prepare_datasets(train_df, val_df, test_df, tokenizer) -> dict

# Dataset Classes
TranslationDataset(data_file, tokenizer, source_lang, target_lang, max_length)
BidirectionalDataset(data_file, tokenizer, max_length)
```

### Training

```python
# MBART Training
train_mbart_model(train_file, val_file, output_dir, **kwargs) -> model
train_bidirectional_mbart(train_file, val_file, output_dir, **kwargs) -> model

# NLLB Training
train_nllb_model(train_file, val_file, output_dir, **kwargs) -> model
train_bidirectional_nllb(train_file, val_file, output_dir, **kwargs) -> model
```

### Evaluation

```python
# Metrics
compute_bleu(predictions, references) -> float
compute_chrf(predictions, references) -> float
compute_bertscore(predictions, references) -> dict
compute_all_metrics(predictions, references, **kwargs) -> dict

# Error Analysis
analyze_errors(predictions, references, sources) -> dict
find_problematic_examples(predictions, references, threshold) -> list
generate_error_report(predictions, references, sources, output_file) -> str
```

### Visualization

```python
# Plotting
plot_bleu_comparison(results_csv, output_file) -> None
plot_training_curves(train_loss, val_loss, output_file) -> None
plot_length_distribution(lengths, output_file) -> None
create_all_plots(results_dir, output_dir) -> None
```

---

## Dependencies

### Required
- `torch>=2.0.0`
- `transformers>=4.35.0`
- `datasets>=2.14.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`

### Evaluation
- `sacrebleu>=2.3.1`
- `evaluate>=0.4.0`
- `bert-score>=0.3.13`

### Visualization
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `plotly>=5.14.0`

---

## Performance Optimization

### Memory Optimization
```python
# Use gradient accumulation for larger effective batch size
training_args = {
    "batch_size": 4,  # Smaller batch size
    "gradient_accumulation_steps": 4,  # Effective batch size = 16
}

# Use mixed precision training
training_args["fp16"] = True
```

### Speed Optimization
```python
# Use multiple data loader workers
dataloader = DataLoader(dataset, num_workers=4)

# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True
```

---

## License

This source code is released under the MIT License. See LICENSE file for details.

---

## Citation

```bibtex
@software{legal_nlp_source_2025,
  title={Legal NLP Source Code: English-Nepali Machine Translation},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/Legal_NLP}
}
```

---

**Last Updated:** January 2025

For questions or contributions, please open a GitHub issue or contact [your.email@example.com].
