# Training and Analysis Notebooks

This directory contains Jupyter notebooks for training, evaluating, and analyzing neural machine translation models for English-Nepali legal domain translation.

## Available Notebooks

### 1. mbart50_training.ipynb
**Purpose:** Train MBART-50 models for English-Nepali translation

**Contents:**
- Data loading and preprocessing
- MBART-50 model initialization
- Uni-directional training (EN→NE and NE→EN)
- Bi-directional training
- Model evaluation and metrics
- Checkpoint saving and management

**Use Cases:**
- Train new MBART-50 models from scratch
- Fine-tune on custom datasets
- Experiment with hyperparameters
- Compare uni-directional vs bi-directional approaches

**Training Time:** ~2-4 hours on GPU (varies by configuration)

---

### 2. nllb200_training.ipynb
**Purpose:** Train NLLB-200 models for English-Nepali translation

**Contents:**
- NLLB-200 model setup and configuration
- Data preparation for NLLB format
- Uni-directional training strategies
- Bi-directional training implementation
- Evaluation metrics computation
- Translation quality assessment

**Use Cases:**
- Train NLLB-200 models
- Compare NLLB vs MBART performance
- Evaluate distilled model efficiency
- Generate translations for test set

**Training Time:** ~2-4 hours on GPU (varies by configuration)

---

### 3. NLLB-200-Fine-tuning.ipynb
**Purpose:** Detailed NLLB-200 fine-tuning workflow

**Contents:**
- Advanced NLLB-200 fine-tuning techniques
- Custom training loops
- Hyperparameter optimization
- Learning rate scheduling
- Gradient accumulation strategies

**Use Cases:**
- Advanced NLLB-200 customization
- Research on training strategies
- Performance optimization
- Ablation studies

---

### 4. results_analysis.ipynb
**Purpose:** Comprehensive analysis and visualization of experimental results

**Contents:**
- Results loading from CSV files
- BLEU and chrF++ score comparison
- Performance visualization (bar charts, line plots)
- Statistical significance testing
- Error analysis and case studies
- Translation quality examples
- Model comparison tables

**Key Outputs:**
- Performance comparison charts
- Statistical analysis reports
- Translation quality samples
- Error pattern identification

**Use Cases:**
- Analyze experimental results
- Generate figures for papers
- Identify model strengths and weaknesses
- Compare different training strategies

---

### 5. training_code.ipynb
**Purpose:** General-purpose training code and utilities

**Contents:**
- Reusable training functions
- Data loading utilities
- Model evaluation helpers
- Metric computation functions
- Visualization tools
- End-to-end training pipeline

**Use Cases:**
- Quick model training experiments
- Code reference and examples
- Modular training components
- Custom pipeline development

---

## Getting Started

### Prerequisites

1. **Install Dependencies:**
```bash
cd /path/to/Legal_NLP
pip install -r requirements.txt
```

2. **GPU Setup (Recommended):**
- CUDA-capable GPU with 16GB+ VRAM recommended
- For CPU-only training, reduce batch sizes significantly

3. **Launch Jupyter:**
```bash
jupyter notebook
```

### Quick Start Guide

#### Training a Model

1. Open `mbart50_training.ipynb` or `nllb200_training.ipynb`
2. Verify data paths in the first cells
3. Choose configuration (uni-directional or bi-directional)
4. Run cells sequentially
5. Monitor training progress via progress bars
6. Evaluate model on test set
7. Save model checkpoint

#### Analyzing Results

1. Open `results_analysis.ipynb`
2. Load results from `results/tables/`
3. Run visualization cells
4. Generate comparison charts
5. Export figures for publication

### Google Colab Usage

All notebooks are compatible with Google Colab for GPU-accelerated training.

#### Setup for Colab

1. **Upload to Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Legal_NLP
```

2. **Install Dependencies:**
```python
!pip install -q transformers datasets sacrebleu evaluate bert-score
```

3. **Enable GPU:**
- Runtime → Change runtime type → Hardware accelerator → GPU (T4 or better)

4. **Run Training:**
Follow notebook instructions

See `INSTALL_CELL_FOR_COLAB.txt` for a ready-to-use installation cell.

---

## Notebook Details

### Common Structure

Most training notebooks follow this structure:

1. **Setup and Imports**
   - Library imports
   - Device configuration (GPU/CPU)
   - Random seed setting

2. **Data Loading**
   - Load train/val/test CSV files
   - Data validation
   - Dataset statistics

3. **Model Configuration**
   - Model initialization
   - Tokenizer setup
   - Training arguments

4. **Training Loop**
   - Fine-tuning execution
   - Progress monitoring
   - Checkpoint saving

5. **Evaluation**
   - Test set evaluation
   - BLEU and chrF++ computation
   - Translation examples

6. **Results Export**
   - Save metrics to CSV
   - Export translations
   - Generate visualizations

### Key Configuration Parameters

```python
# Training hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
NUM_EPOCHS = 10
MAX_LENGTH = 128
WARMUP_STEPS = 500

# Model selection
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
# or
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Training mode
BIDIRECTIONAL = True  # True for bi-directional, False for uni-directional
```

### Memory Requirements

| Configuration | GPU Memory | Recommended GPU |
|---------------|------------|-----------------|
| MBART Uni-directional | ~12GB | T4, V100 |
| MBART Bi-directional | ~14GB | V100, A100 |
| NLLB Uni-directional | ~11GB | T4, V100 |
| NLLB Bi-directional | ~13GB | V100, A100 |

**Tip:** Reduce `BATCH_SIZE` if encountering OOM (Out of Memory) errors.

---

## Utilities

### fix.py
Helper script for notebook maintenance and fixes.

**Usage:**
```bash
python fix.py
```

### INSTALL_CELL_FOR_COLAB.txt
Ready-to-use installation cell for Google Colab notebooks.

**Contents:**
- Dependency installation commands
- GPU verification
- Drive mounting
- Path setup

**Usage:**
Copy and paste into the first cell of your Colab notebook.

---

## Expected Outputs

### Model Checkpoints
Saved to `../models/[model_name]/[configuration]/`
- `pytorch_model.bin` - Model weights
- `config.json` - Model configuration
- `tokenizer_config.json` - Tokenizer settings
- `training_args.bin` - Training arguments

### Results Files
Saved to `../results/`
- `tables/table1_main_results.csv` - Main evaluation metrics
- `translations/[model_config]/translations.csv` - Generated translations
- `metrics/[model_config]_metrics.json` - Detailed metrics

### Visualizations
Saved to `../results/visualizations/` and `../results/figures/`
- Performance comparison charts
- BLEU score trends
- Length distribution plots
- Error analysis visualizations

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solution: Reduce batch size
BATCH_SIZE = 4  # or 2
# Enable gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 2
```

**2. Model Loading Errors**
```python
# Solution: Clear cache and reload
import torch
torch.cuda.empty_cache()
# Reload model
```

**3. Tokenizer Warnings**
```python
# Solution: Set legacy=False if using newer tokenizers
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
```

**4. Slow Training**
```python
# Solution: Enable mixed precision training
from transformers import TrainingArguments
training_args = TrainingArguments(
    fp16=True,  # Enable mixed precision
    dataloader_num_workers=4  # Parallel data loading
)
```

### Performance Tips

1. **Use GPU:** Essential for reasonable training times
2. **Batch size:** Maximize based on available GPU memory
3. **Gradient accumulation:** Simulate larger batches
4. **Mixed precision (fp16):** 2x speedup on modern GPUs
5. **Data caching:** Cache processed datasets to disk

---

## Best Practices

### Before Training
- Verify data paths and file existence
- Check GPU availability and memory
- Set random seeds for reproducibility
- Review hyperparameters

### During Training
- Monitor loss curves for convergence
- Check validation metrics regularly
- Save checkpoints frequently
- Watch for overfitting

### After Training
- Evaluate on held-out test set
- Generate qualitative examples
- Compare with baseline models
- Document configuration and results

---

## Reproducibility

To ensure reproducible results:

```python
import random
import numpy as np
import torch

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Deterministic behavior (may impact performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Citation

If you use these notebooks in your research, please cite:

```bibtex
@software{legal_nlp_notebooks_2025,
  title={Legal NLP Training Notebooks: English-Nepali Machine Translation},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/Legal_NLP}
}
```

---

## Contributing

To contribute new notebooks or improvements:
1. Follow the existing notebook structure
2. Add clear markdown documentation
3. Test on both CPU and GPU
4. Verify Colab compatibility
5. Update this README

---

## Support

For issues or questions:
- Check troubleshooting section above
- Review notebook comments and documentation
- Open a GitHub issue
- Contact: [your.email@example.com]

---

**Last Updated:** January 2025

**Note:** Training times and memory requirements are approximate and may vary based on hardware, batch size, and sequence length.
