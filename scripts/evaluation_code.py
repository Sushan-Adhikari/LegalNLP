# ============================================================================
# EVALUATION & RESULTS GENERATION FOR ICAIL PAPER
# Legal Nepali-English Machine Translation
#
# This script evaluates all 6 trained models (3 NLLB + 3 mBART) and generates:
# - Table 1: Main results (BLEU, chrF++, Length Ratio)
# - Table 2: Dataset statistics
# - Figure 1: Corpus length distribution
# - Figure 2: BLEU comparison
# - Figure 3: Directional asymmetry (KEY FINDING)
#
# Total runtime: ~30-40 minutes
# ============================================================================


# ============================================================================
# CELL 1: Setup & Install Packages
# ============================================================================

# Install evaluation packages
print("Installing evaluation packages...")
get_ipython().system('pip install -q sacrebleu evaluate matplotlib seaborn plotly')

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBart50TokenizerFast
from sacrebleu.metrics import BLEU, CHRF

# Paths
DRIVE_BASE = '/content/drive/MyDrive/Legal_NLP'

# Create output directories
os.makedirs(f"{DRIVE_BASE}/results/tables", exist_ok=True)
os.makedirs(f"{DRIVE_BASE}/results/figures", exist_ok=True)
os.makedirs(f"{DRIVE_BASE}/results/translations", exist_ok=True)
os.makedirs(f"{DRIVE_BASE}/results/error_analysis", exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300

print(f"\n✅ Setup complete!")
print(f"Results will be saved to: {DRIVE_BASE}/results/")


# ============================================================================
# CELL 2: Table 2 - Dataset Statistics
# ============================================================================

# Load all splits
train_df = pd.read_csv(f'{DRIVE_BASE}/data/splits/train.csv')
val_df = pd.read_csv(f'{DRIVE_BASE}/data/splits/val.csv')
test_df = pd.read_csv(f'{DRIVE_BASE}/data/splits/test.csv')

# Combine for overall statistics
df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Compute statistics
df['en_len'] = df['English'].str.split().str.len()
df['ne_len'] = df['Nepali'].str.split().str.len()

stats = {
    'Statistic': [
        'Total Pairs',
        'Train / Val / Test',
        'Avg English Length',
        'Avg Nepali Length',
        'Max English Length',
        'Max Nepali Length',
        'Min English Length',
        'Min Nepali Length'
    ],
    'Value': [
        f"{len(df)}",
        f"{len(train_df)} / {len(val_df)} / {len(test_df)}",
        f"{df['en_len'].mean():.2f}",
        f"{df['ne_len'].mean():.2f}",
        f"{df['en_len'].max()}",
        f"{df['ne_len'].max()}",
        f"{df['en_len'].min()}",
        f"{df['ne_len'].min()}"
    ]
}

table2 = pd.DataFrame(stats)

print("="*80)
print("TABLE 2: DATASET STATISTICS")
print("="*80)
print(table2.to_string(index=False))
print("="*80)

# Save
table2.to_csv(f"{DRIVE_BASE}/results/tables/table2_dataset_statistics.csv", index=False)
print(f"\n✅ Saved to: {DRIVE_BASE}/results/tables/table2_dataset_statistics.csv")


# ============================================================================
# CELL 3: Figure 1 - Corpus Length Distribution
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# English distribution
axes[0].hist(df['en_len'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
axes[0].axvline(df['en_len'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df["en_len"].mean():.1f}')
axes[0].set_xlabel('Sentence Length (words)', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].set_title('English Sentence Length Distribution', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Nepali distribution
axes[1].hist(df['ne_len'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[1].axvline(df['ne_len'].mean(), color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {df["ne_len"].mean():.1f}')
axes[1].set_xlabel('Sentence Length (words)', fontweight='bold')
axes[1].set_ylabel('Frequency', fontweight='bold')
axes[1].set_title('Nepali Sentence Length Distribution', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/results/figures/figure1_corpus_length_distribution.png',
            dpi=300, bbox_inches='tight')
plt.show()

print("✅ Figure 1 saved: figure1_corpus_length_distribution.png")


# ============================================================================
# CELL 4: Evaluation Functions (BLEU + chrF++)
# ============================================================================

def translate_batch(texts, model, tokenizer, src_lang, tgt_lang, model_type,
                    batch_size=16, max_length=128, num_beams=4):
    """
    Translate texts in batches (sorted by length for efficiency).
    Based on working NLLB example.

    Args:
        texts: List of source texts to translate
        model: Loaded translation model
        tokenizer: Loaded tokenizer
        src_lang: Source language code (e.g., 'npi_Deva', 'ne_NP')
        tgt_lang: Target language code (e.g., 'eng_Latn', 'en_XX')
        model_type: 'nllb' or 'mbart'
        batch_size: Number of sentences per batch
        max_length: Max sequence length
        num_beams: Beam search width

    Returns:
        List of translated texts in original order
    """
    # Sort by length for efficient batching
    idxs, texts_sorted = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []

    for i in tqdm(range(0, len(texts_sorted), batch_size), desc="Translating"):
        batch = texts_sorted[i:i+batch_size]

        # Set source language
        if hasattr(tokenizer, 'src_lang'):
            tokenizer.src_lang = src_lang
        if hasattr(tokenizer, 'tgt_lang') and model_type == 'mbart':
            tokenizer.tgt_lang = tgt_lang

        # Tokenize
        inputs = tokenizer(
            batch, return_tensors='pt', padding=True,
            truncation=True, max_length=max_length
        ).to(model.device)

        # Generate translations
        with torch.no_grad():
            if model_type == 'nllb':
                # NLLB requires forced_bos_token_id
                tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_id,
                    max_new_tokens=int(32 + 3 * inputs.input_ids.shape[1]),  # Dynamic length
                    num_beams=num_beams
                )
            else:
                # mBART
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams
                )

        # Decode
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)

    # Restore original order
    return [result for _, result in sorted(zip(idxs, results))]


def evaluate_model(model_path, model_type, direction, test_data_path, output_dir):
    """
    Evaluate a trained model on test set.

    Computes:
    - BLEU score (primary metric)
    - chrF++ score (character-level metric)
    - Length ratio (hypothesis/reference)

    Args:
        model_path: Path to trained model
        model_type: 'nllb' or 'mbart'
        direction: 'ne_en' or 'en_ne'
        test_data_path: Path to test.csv
        output_dir: Directory to save results

    Returns:
        dict: BLEU, chrF++, length_ratio scores
        list: Translated hypotheses
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_type.upper()} - {direction.upper()}")
    print("="*80)

    # Load tokenizer and model
    if model_type == 'mbart':
        tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model = model.to('cuda')
    model.eval()

    print(f"✅ Loaded model from: {model_path}")

    # Load test data
    test_df = pd.read_csv(test_data_path)

    # Set language codes and columns
    if direction == 'ne_en':
        src_col, tgt_col = 'Nepali', 'English'
        if model_type == 'mbart':
            src_lang, tgt_lang = 'ne_NP', 'en_XX'
        else:
            src_lang, tgt_lang = 'npi_Deva', 'eng_Latn'
    else:  # en_ne
        src_col, tgt_col = 'English', 'Nepali'
        if model_type == 'mbart':
            src_lang, tgt_lang = 'en_XX', 'ne_NP'
        else:
            src_lang, tgt_lang = 'eng_Latn', 'npi_Deva'

    sources = test_df[src_col].tolist()
    references = test_df[tgt_col].tolist()

    # Translate using batched function
    print("Translating test set...")
    hypotheses = translate_batch(
        sources, model, tokenizer, src_lang, tgt_lang, model_type,
        batch_size=16, num_beams=4
    )

    # Compute metrics using sacrebleu
    bleu_calc = BLEU()
    chrf_calc = CHRF(word_order=2)  # chrF++

    # sacrebleu expects references as list of lists
    refs = [[r] for r in references]

    bleu_score = bleu_calc.corpus_score(hypotheses, refs)
    chrf_score = chrf_calc.corpus_score(hypotheses, refs)

    # Length ratio
    hyp_lens = [len(h.split()) for h in hypotheses]
    ref_lens = [len(r.split()) for r in references]
    length_ratio = np.mean(hyp_lens) / np.mean(ref_lens)

    results = {
        'model_type': model_type,
        'direction': direction,
        'bleu': bleu_score.score,
        'chrf++': chrf_score.score,
        'length_ratio': length_ratio
    }

    print(f"\nResults:")
    print(f"  BLEU: {results['bleu']:.2f}")
    print(f"  chrF++: {results['chrf++']:.2f}")
    print(f"  Length Ratio: {results['length_ratio']:.3f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(results, f, indent=2)

    pd.DataFrame({
        'source': sources,
        'reference': references,
        'hypothesis': hypotheses
    }).to_csv(f"{output_dir}/translations.csv", index=False)

    print(f"✅ Results saved to: {output_dir}")
    return results, hypotheses

print("✅ Evaluation functions defined (BLEU + chrF++ ready)")


# ============================================================================
# CELL 5: Evaluate All 8 Models - Generate Table 1 (MAIN RESULTS)
# ============================================================================

test_path = f'{DRIVE_BASE}/data/splits/test.csv'
all_results = {}

# Define all 8 models to evaluate
# Format: (model_type, direction, model_path, config)
models_to_eval = [
    ('mbart', 'ne_en', '/content/models/mbart50_ne_en/final_model', 'uni'),
    ('mbart', 'en_ne', '/content/models/mbart50_en_ne/final_model', 'uni'),
    ('mbart', 'ne_en', '/content/models/mbart50_bidirectional/final_model', 'bi'),
    ('mbart', 'en_ne', '/content/models/mbart50_bidirectional/final_model', 'bi'),
    ('nllb', 'ne_en', '/content/models/nllb200_ne_en/final_model', 'uni'),
    ('nllb', 'en_ne', '/content/models/nllb200_en_ne/final_model', 'uni'),
    ('nllb', 'ne_en', '/content/models/nllb200_bidirectional/final_model', 'bi'),
    ('nllb', 'en_ne', '/content/models/nllb200_bidirectional/final_model', 'bi'),
]

print("\n" + "="*80)
print("EVALUATING ALL 8 MODELS (This will take ~30-40 minutes)")
print("="*80 + "\n")

for model_type, direction, model_path, config in models_to_eval:
    if os.path.exists(model_path):
        output_dir = f"{DRIVE_BASE}/results/translations/{model_type}_{config}_{direction}"
        results, _ = evaluate_model(model_path, model_type, direction, test_path, output_dir)
        all_results[f"{model_type}_{config}_{direction}"] = results
    else:
        print(f"⚠️ Model not found: {model_path}")
        print(f"   Skipping {model_type}_{config}_{direction}")

# Save all results to JSON
with open(f"{DRIVE_BASE}/results/all_results.json", 'w') as f:
    json.dump(all_results, f, indent=2)

# Create Table 1 - MAIN RESULTS FOR PAPER
summary = []
for key, res in all_results.items():
    config_type = 'Bi-directional' if '_bi_' in key else 'Uni-directional'
    summary.append({
        'Model': res['model_type'].upper(),
        'Configuration': config_type,
        'Direction': res['direction'].replace('_', '→').upper(),
        'BLEU': f"{res['bleu']:.2f}",
        'chrF++': f"{res['chrf++']:.2f}",
        'Length Ratio': f"{res['length_ratio']:.3f}"
    })

table1 = pd.DataFrame(summary)

print("\n" + "="*80)
print("TABLE 1: MAIN RESULTS (For ICAIL Paper)")
print("="*80)
print(table1.to_string(index=False))
print("="*80)

table1.to_csv(f"{DRIVE_BASE}/results/tables/table1_main_results.csv", index=False)
print(f"\n✅ Table 1 saved: table1_main_results.csv")
print(f"✅ All translations saved to: {DRIVE_BASE}/results/translations/")


# ============================================================================
# CELL 6: Figure 2 - BLEU Comparison Bar Chart
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 6))

labels = []
bleu_scores = []
colors = []

for key, res in all_results.items():
    config = 'Bi-dir' if '_bi_' in key else 'Uni-dir'
    label = f"{res['model_type'].upper()}\n{res['direction'].replace('_','→').upper()}\n({config})"
    labels.append(label)
    bleu_scores.append(res['bleu'])
    colors.append('#e74c3c' if '_bi_' in key else '#3498db')

x = np.arange(len(labels))
bars = ax.bar(x, bleu_scores, color=colors, alpha=0.8, edgecolor='black')

ax.set_xlabel('Model Configuration', fontweight='bold', fontsize=12)
ax.set_ylabel('BLEU Score', fontweight='bold', fontsize=12)
ax.set_title('BLEU Score Comparison Across All Configurations',
             fontweight='bold', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', label='Uni-directional'),
    Patch(facecolor='#e74c3c', label='Bi-directional')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/results/figures/figure2_bleu_comparison.png',
            dpi=300, bbox_inches='tight')
plt.show()

print("✅ Figure 2 saved: figure2_bleu_comparison.png")


# ============================================================================
# CELL 7: Figure 3 - Directional Asymmetry (KEY FINDING)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, model_type in enumerate(['mbart', 'nllb']):
    # Get results for this model
    ne_en_uni = all_results[f"{model_type}_uni_ne_en"]
    en_ne_uni = all_results[f"{model_type}_uni_en_ne"]
    ne_en_bi = all_results[f"{model_type}_bi_ne_en"]
    en_ne_bi = all_results[f"{model_type}_bi_en_ne"]

    metrics = ['BLEU', 'chrF++']
    ne_en_scores = [ne_en_uni['bleu'], ne_en_uni['chrf++']]
    en_ne_scores_uni = [en_ne_uni['bleu'], en_ne_uni['chrf++']]
    en_ne_scores_bi = [en_ne_bi['bleu'], en_ne_bi['chrf++']]

    x = np.arange(len(metrics))
    width = 0.25

    # Plot bars
    axes[idx].bar(x - width, ne_en_scores, width, label='NE→EN (Uni)',
                 color='#2ecc71', alpha=0.8)
    axes[idx].bar(x, en_ne_scores_uni, width, label='EN→NE (Uni)',
                 color='#3498db', alpha=0.8)
    axes[idx].bar(x + width, en_ne_scores_bi, width, label='EN→NE (Bi)',
                 color='#e74c3c', alpha=0.8)

    axes[idx].set_xlabel('Metric', fontweight='bold')
    axes[idx].set_ylabel('Score', fontweight='bold')
    axes[idx].set_title(f'{model_type.upper()} Directional Asymmetry', fontweight='bold')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(metrics)
    axes[idx].legend()
    axes[idx].grid(axis='y', alpha=0.3, linestyle='--')

    # Highlight degradation (KEY FINDING)
    degradation = en_ne_uni['bleu'] - en_ne_bi['bleu']
    axes[idx].text(0.5, 0.95, f'EN→NE Degradation: {degradation:.2f} BLEU',
                  transform=axes[idx].transAxes, ha='center', va='top',
                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                  fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(f'{DRIVE_BASE}/results/figures/figure3_directional_asymmetry.png',
            dpi=300, bbox_inches='tight')
plt.show()

print("✅ Figure 3 saved: figure3_directional_asymmetry.png")
print("   ⚠️ This is your KEY FINDING for the paper!")


# ============================================================================
# CELL 8: Final Summary
# ============================================================================

print("\n" + "="*80)
print("🎉 RESULTS GENERATION COMPLETE! 🎉")
print("="*80)

print("\n✅ TABLES GENERATED:")
print("  • Table 1: Main results (BLEU, chrF++, Length Ratio)")
print("  • Table 2: Dataset statistics")

print("\n✅ FIGURES GENERATED:")
print("  • Figure 1: Corpus length distribution (300 DPI)")
print("  • Figure 2: BLEU comparison bar chart (300 DPI)")
print("  • Figure 3: Directional asymmetry (300 DPI) ⭐ KEY FINDING")

print("\n✅ ALL RESULTS SAVED TO:")
print(f"  {DRIVE_BASE}/results/")
print("\n  Structure:")
print("  ├── tables/")
print("  │   ├── table1_main_results.csv")
print("  │   └── table2_dataset_statistics.csv")
print("  ├── figures/")
print("  │   ├── figure1_corpus_length_distribution.png")
print("  │   ├── figure2_bleu_comparison.png")
print("  │   └── figure3_directional_asymmetry.png")
print("  ├── translations/")
print("  │   └── [8 directories with model outputs]")
print("  └── all_results.json")

print("\n" + "="*80)
print("🎉 READY FOR ICAIL PAPER SUBMISSION! 🎉")
print("="*80)
print("\nAll tables and figures are publication-ready (300 DPI).")
print("Find them in your Google Drive at:")
print(f"{DRIVE_BASE}/results/")
print("\nYou can now download and insert them directly into your LaTeX paper!")
print("="*80)
