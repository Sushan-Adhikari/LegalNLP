# Reference Papers

This directory contains academic papers and research references related to machine translation, multilingual NLP, and legal domain language processing.

## Available Papers

### 1. main.pdf
**Title:** [Main Research Paper]

**Description:** Primary paper for this project covering English-Nepali legal domain machine translation using MBART-50 and NLLB-200 models.

**Topics:**
- Neural machine translation
- Legal domain NLP
- English-Nepali translation
- Bi-directional vs uni-directional training

**Relevance:** Core methodology and results for this project

---

### 2. 2024.sigul-1.7-5.pdf
**Title:** SIGUL 2024 Conference Paper

**Description:** Paper from the SIGUL (Special Interest Group on Under-resourced Languages) 2024 conference.

**Topics:**
- Under-resourced languages
- Low-resource NLP
- Language technology
- Multilingual models

**Relevance:** Background on under-resourced language challenges and solutions, particularly relevant for Nepali as a low-resource language

---

### 3. 2212.09811v3.pdf
**Title:** ArXiv Paper (December 2022)

**Description:** Research paper from ArXiv preprint repository.

**Topics:**
- Machine translation
- Transformer models
- Evaluation metrics
- Model architectures

**Relevance:** Technical methodology and evaluation approaches for MT systems

---

### 4. 3034_NepKanun_A_RAG_Based_Nepa.pdf
**Title:** NepKanun: A RAG-Based Nepali Legal System

**Description:** Research on Retrieval-Augmented Generation (RAG) for Nepali legal domain applications.

**Topics:**
- Nepali legal domain
- Retrieval-augmented generation
- Legal question answering
- Domain-specific NLP

**Relevance:** 
- Legal domain challenges in Nepali
- Domain-specific terminology
- Legal text processing
- Related work in Nepali legal NLP

**Key Insights:**
- Legal terminology complexity in Nepali
- Importance of domain adaptation
- RAG approaches for legal applications

---

### 5. acharya18_sltu.pdf
**Title:** SLTU 2018 - Acharya et al.

**Description:** Paper from the SLTU (Spoken Language Technologies for Under-resourced Languages) workshop.

**Authors:** Acharya et al.

**Topics:**
- Nepali language processing
- Speech and language technologies
- Under-resourced language tools
- Nepali computational linguistics

**Relevance:**
- Foundational work on Nepali NLP
- Dataset creation and preprocessing
- Challenges specific to Nepali language
- Baseline approaches for Nepali MT

**Key Contributions:**
- Nepali language resources
- Preprocessing techniques
- Evaluation methodologies

---

## Research Areas Covered

### 1. Machine Translation
- Neural machine translation architectures
- Transformer models (MBART, NLLB)
- Training strategies (uni-directional vs bi-directional)
- Evaluation metrics (BLEU, chrF++)

### 2. Multilingual NLP
- Massively multilingual models
- Cross-lingual transfer learning
- Low-resource language adaptation
- Multilingual tokenization

### 3. Legal Domain NLP
- Legal text processing
- Domain-specific terminology
- Translation accuracy requirements
- Legal document understanding

### 4. Nepali Language Processing
- Devanagari script handling
- Morphological complexity
- SOV word order
- Resource constraints

### 5. Evaluation and Analysis
- Automatic metrics (BLEU, chrF++, BERTScore)
- Human evaluation
- Error analysis
- Quality assessment

## Key Models and Architectures

### MBART-50
- **Paper:** "Multilingual Denoising Pre-training for Neural Machine Translation" (Liu et al., 2020)
- **Model:** facebook/mbart-large-50-many-to-many-mmt
- **Coverage:** 50 languages
- **Size:** 610M parameters

### NLLB-200
- **Paper:** "No Language Left Behind: Scaling Human-Centered Machine Translation" (NLLB Team, 2022)
- **Model:** facebook/nllb-200-distilled-600M
- **Coverage:** 200 languages
- **Size:** 600M parameters (distilled)

## Important Concepts

### Bi-directional Training
Training a single model to translate in both directions (EN↔NE) simultaneously, which has shown superior performance for certain translation directions.

### Low-Resource Languages
Languages with limited digital resources, parallel corpora, and NLP tools. Nepali falls into this category, requiring special attention to:
- Data scarcity
- Lack of pre-trained models
- Limited evaluation resources
- Script and morphological complexity

### Legal Domain Adaptation
Specialized adaptation techniques for legal texts:
- Terminology preservation
- Formal register handling
- Sentence structure complexity
- Cultural and jurisdictional differences

### Evaluation Metrics

**BLEU (BiLingual Evaluation Understudy)**
- N-gram overlap metric
- Range: 0-100
- Higher is better
- Sensitive to exact word matches

**chrF++ (Character n-gram F-score)**
- Character-level metric
- More robust to morphological variation
- Better for morphologically rich languages like Nepali
- Range: 0-100

## Related Research Areas

### Future Reading
For deeper understanding, consider exploring:

1. **Transformer Architecture**
   - "Attention is All You Need" (Vaswani et al., 2017)

2. **Multilingual Pre-training**
   - "Unsupervised Cross-lingual Representation Learning at Scale" (Conneau et al., 2020)

3. **Low-Resource MT**
   - "Findings of the WMT 2021 Shared Task on Machine Translation Using Terminologies"

4. **Legal NLP**
   - "Legal Information Retrieval and Entailment"
   - "Domain Adaptation for Legal Text Processing"

5. **Nepali NLP**
   - "Building NLP Resources for Nepali Language"
   - "Morphological Analysis of Nepali"

## Citation Guidelines

When citing papers from this collection:

```bibtex
@inproceedings{sigul2024,
  title={[Paper Title]},
  booktitle={Proceedings of SIGUL 2024},
  year={2024}
}

@article{arxiv2212,
  title={[Paper Title]},
  author={[Authors]},
  journal={arXiv preprint arXiv:2212.09811},
  year={2022}
}

@inproceedings{nepkanun,
  title={NepKanun: A RAG-Based Nepali Legal System},
  year={2024}
}

@inproceedings{acharya2018sltu,
  title={[Paper Title]},
  author={Acharya et al.},
  booktitle={Proceedings of SLTU 2018},
  year={2018}
}
```

## Paper Summaries

### Key Findings Across Papers

1. **Multilingual Models Work Well for Low-Resource Languages**
   - MBART and NLLB show strong performance
   - Transfer learning is crucial
   - Cross-lingual knowledge helps

2. **Bi-directional Training Can Improve Performance**
   - Single model for both directions
   - Improved representation learning
   - Better resource utilization

3. **Legal Domain Requires Special Attention**
   - Terminology consistency critical
   - High accuracy requirements
   - Domain-specific evaluation needed

4. **Nepali Presents Unique Challenges**
   - Devanagari script complexities
   - Morphological richness
   - Limited resources
   - Word order differences

## Using These Papers

### For Literature Review
1. Read papers in chronological order
2. Focus on methodology sections
3. Compare evaluation approaches
4. Note dataset characteristics

### For Methodology
1. Review training procedures
2. Study hyperparameter choices
3. Analyze evaluation metrics
4. Understand preprocessing steps

### For Results Comparison
1. Compare BLEU scores
2. Check dataset sizes
3. Review model architectures
4. Analyze error patterns

## Additional Resources

### Online Resources
- **Hugging Face Transformers:** https://huggingface.co/transformers/
- **NLLB Project:** https://ai.facebook.com/research/no-language-left-behind/
- **SacreBLEU:** https://github.com/mjpost/sacrebleu
- **WMT Shared Tasks:** http://www.statmt.org/wmt23/

### Datasets
- **FLORES-200:** Multilingual evaluation benchmark
- **WMT Translation Tasks:** Annual MT competitions
- **Legal Domain Corpora:** JRC-Acquis, MultiLegal, etc.

### Tools and Libraries
- **Transformers:** Model implementation
- **Datasets:** Data processing
- **SacreBLEU:** Evaluation metrics
- **Sentencepiece:** Tokenization

## Contributing

To add new papers to this collection:
1. Add PDF to this directory
2. Update this README with paper details
3. Include title, authors, venue, year
4. Add brief description and relevance
5. Update citation format if needed

## Notes

- All papers are for research and educational purposes
- Respect copyright and licensing terms
- Cite appropriately when using ideas or methods
- Keep local copies for reference

---

**Last Updated:** January 2025

For questions about specific papers or to request additional references, please open a GitHub issue.
