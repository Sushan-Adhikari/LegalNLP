# Legal Domain English-Nepali Parallel Corpus

This directory contains the English-Nepali parallel corpus for legal domain machine translation.

## Dataset Overview

The corpus consists of **5,024 parallel sentence pairs** extracted from legal documents, including:
- Court judgments and verdicts
- Legal regulations and statutes
- Administrative legal documents
- Legal terminology and definitions

## Directory Structure

```
data/
├── raw/                    # Original raw data files
├── processed/              # Preprocessed and cleaned data
└── splits/                 # Train/validation/test splits
    ├── train.csv          # Training set (4,019 pairs + 1 header = 4,020 lines)
    ├── val.csv            # Validation set (503 pairs + 1 header = 504 lines)
    └── test.csv           # Test set (503 pairs + 1 header = 504 lines)
```

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| **Total Pairs** | 5,024 |
| **Training Set** | 4,019 pairs (80%) |
| **Validation Set** | 502 pairs (10%) |
| **Test Set** | 503 pairs (10%) |
| | |
| **English Tokens** | |
| Average Length | 34.03 tokens |
| Maximum Length | 721 tokens |
| Minimum Length | 1 token |
| | |
| **Nepali Tokens** | |
| Average Length | 24.86 tokens |
| Maximum Length | 537 tokens |
| Minimum Length | 1 token |

## Data Format

All CSV files follow this structure:

```csv
english,nepali
"English sentence text","Nepali sentence text"
...
```

### Fields

- **english:** English language text (source or target depending on translation direction)
- **nepali:** Nepali language text (source or target depending on translation direction)

### Example Entries

```csv
english,nepali
"The court shall hear the case.","अदालतले मुद्दा सुन्नेछ।"
"Legal proceedings must follow due process.","कानुनी कारबाही उचित प्रक्रिया पालना गर्नुपर्छ।"
```

## Data Preprocessing

The dataset has undergone the following preprocessing steps:

1. **Text Cleaning:**
   - Removal of extra whitespace
   - Normalization of Unicode characters
   - Handling of special characters and punctuation

2. **Quality Filtering:**
   - Removal of duplicate pairs
   - Filtering of incomplete or misaligned sentences
   - Length ratio validation

3. **Tokenization:**
   - Sentence-level tokenization
   - Language-specific tokenization for Nepali (Devanagari script)

4. **Train/Val/Test Split:**
   - 80% training (4,019 pairs)
   - 10% validation (502 pairs)
   - 10% test (503 pairs)
   - Random split with fixed seed for reproducibility

## Data Usage

### Loading the Data

#### Using Pandas

```python
import pandas as pd

# Load training data
train_df = pd.read_csv('data/splits/train.csv')
print(f"Training samples: {len(train_df)}")

# Load validation data
val_df = pd.read_csv('data/splits/val.csv')
print(f"Validation samples: {len(val_df)}")

# Load test data
test_df = pd.read_csv('data/splits/test.csv')
print(f"Test samples: {len(test_df)}")
```

#### Using Hugging Face Datasets

```python
from datasets import load_dataset

# Load as Hugging Face dataset
dataset = load_dataset('csv', data_files={
    'train': 'data/splits/train.csv',
    'validation': 'data/splits/val.csv',
    'test': 'data/splits/test.csv'
})

print(dataset)
```

### Example: Preparing for Translation

```python
import pandas as pd

# Load training data
df = pd.read_csv('data/splits/train.csv')

# English to Nepali translation
source_texts = df['english'].tolist()
target_texts = df['nepali'].tolist()

# Nepali to English translation
source_texts = df['nepali'].tolist()
target_texts = df['english'].tolist()
```

## Data Characteristics

### Language Properties

**English:**
- Script: Latin alphabet
- Average sentence length: 34.03 tokens
- Formal legal language style
- Complex sentence structures common

**Nepali:**
- Script: Devanagari (Unicode range: U+0900 to U+097F)
- Average sentence length: 24.86 tokens
- Agglutinative morphology
- SOV (Subject-Object-Verb) word order

### Domain Characteristics

The legal domain presents unique challenges:
- **Specialized terminology:** Legal jargon and technical terms
- **Formal register:** High formality level in both languages
- **Long sentences:** Complex grammatical structures
- **Precision requirements:** Exact meaning preservation crucial
- **Cultural adaptation:** Legal concepts may differ between jurisdictions

## Data Quality

### Quality Assurance

- Human-verified translations
- Alignment validation
- Consistency checks
- Domain expert review

### Known Limitations

- Length variation: Some sentences are very short (1 token) or very long (700+ tokens)
- Domain coverage: Limited to specific legal document types
- Translation consistency: Minor variations in terminology translation
- Nepali script: Potential Unicode normalization issues

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{legal_nepali_parallel_corpus_2025,
  title={English-Nepali Legal Domain Parallel Corpus},
  author={[Your Name]},
  year={2025},
  publisher={GitHub},
  howpublished={\\url{https://github.com/yourusername/Legal_NLP}}
}
```

## Ethical Considerations

- **Privacy:** All data has been anonymized and reviewed for sensitive information
- **Bias:** Dataset may reflect biases present in legal systems
- **Usage:** Intended for research and educational purposes
- **Validation:** Machine translations should be reviewed by legal experts for critical applications

## License

This dataset is released under [specify license - CC BY 4.0, MIT, etc.]. Please ensure compliance with the license terms when using or distributing this data.

## Contact

For questions about the dataset, data quality issues, or additional data contributions:
- Open an issue on GitHub
- Email: [your.email@example.com]

## Changelog

### Version 1.0 (2025-01)
- Initial release
- 5,024 parallel sentence pairs
- Train/val/test splits created
- Preprocessing and quality filtering applied

---

**Note:** This dataset is for research purposes. For production legal translation systems, additional validation and quality assurance is strongly recommended.
