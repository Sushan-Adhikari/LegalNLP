"""
Evaluation Metrics Module
Computes BLEU, chrF++, and other MT metrics for legal translation.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF
import json


class TranslationEvaluator:
    """
    Comprehensive evaluator for machine translation models.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = 'mbart',  # 'mbart' or 'nllb'
        device: str = None
    ):
        """
        Initialize the evaluator.

        Args:
            model_path: Path to the trained model
            model_type: Type of model ('mbart' or 'nllb')
            device: Device to use for inference
        """
        self.model_path = model_path
        self.model_type = model_type

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

        print(f"Model loaded on {self.device}")

    def translate_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        max_length: int = 128,
        num_beams: int = 5,
        batch_size: int = 16
    ) -> List[str]:
        """
        Translate a batch of texts.

        Args:
            texts: List of source texts
            src_lang: Source language code
            tgt_lang: Target language code
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            batch_size: Batch size for inference

        Returns:
            List of translated texts
        """
        translations = []

        # Set source language
        if hasattr(self.tokenizer, 'src_lang'):
            self.tokenizer.src_lang = src_lang

        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            # Generate with target language forcing
            with torch.no_grad():
                if self.model_type == 'nllb':
                    # NLLB uses forced_bos_token_id
                    tgt_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
                    generated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=tgt_token_id,
                        max_length=max_length,
                        num_beams=num_beams,
                        early_stopping=True
                    )
                else:
                    # mBART uses decoder_start_token_id
                    self.tokenizer.tgt_lang = tgt_lang
                    generated_tokens = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=num_beams,
                        early_stopping=True
                    )

            # Decode
            batch_translations = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )

            translations.extend(batch_translations)

        return translations

    def compute_bleu(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute BLEU scores.

        Args:
            hypotheses: List of generated translations
            references: List of reference translations

        Returns:
            Dictionary with BLEU scores
        """
        # sacrebleu expects references as list of lists
        refs = [[ref] for ref in references]

        bleu = BLEU()
        bleu_score = bleu.corpus_score(hypotheses, refs)

        return {
            'bleu': bleu_score.score,
            'bleu_precisions': bleu_score.precisions,
            'bleu_bp': bleu_score.bp,
            'bleu_sys_len': bleu_score.sys_len,
            'bleu_ref_len': bleu_score.ref_len
        }

    def compute_chrf(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute chrF++ scores.

        Args:
            hypotheses: List of generated translations
            references: List of reference translations

        Returns:
            Dictionary with chrF++ scores
        """
        refs = [[ref] for ref in references]

        chrf = CHRF(word_order=2)  # chrF++
        chrf_score = chrf.corpus_score(hypotheses, refs)

        return {
            'chrf++': chrf_score.score,
            'chrf++_precision': chrf_score.precision,
            'chrf++_recall': chrf_score.recall
        }

    def compute_length_metrics(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute length-based metrics.

        Args:
            hypotheses: List of generated translations
            references: List of reference translations

        Returns:
            Dictionary with length metrics
        """
        hyp_lengths = [len(h.split()) for h in hypotheses]
        ref_lengths = [len(r.split()) for r in references]

        return {
            'avg_hyp_length': np.mean(hyp_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'length_ratio': np.mean(hyp_lengths) / np.mean(ref_lengths),
            'length_diff_std': np.std([h - r for h, r in zip(hyp_lengths, ref_lengths)])
        }

    def evaluate_on_test_set(
        self,
        test_data_path: str,
        direction: str,  # 'ne_en' or 'en_ne'
        output_dir: str,
        max_length: int = 128,
        num_beams: int = 5,
        batch_size: int = 16
    ) -> Dict:
        """
        Evaluate model on test set.

        Args:
            test_data_path: Path to test CSV file
            direction: Translation direction
            output_dir: Directory to save results
            max_length: Maximum generation length
            num_beams: Number of beams
            batch_size: Batch size for inference

        Returns:
            Dictionary with all evaluation metrics
        """
        # Load test data
        test_df = pd.read_csv(test_data_path)

        # Determine source and target columns
        if direction == 'ne_en':
            src_col, tgt_col = 'Nepali', 'English'
            if self.model_type == 'mbart':
                src_lang, tgt_lang = 'ne_NP', 'en_XX'
            else:  # nllb
                src_lang, tgt_lang = 'npi_Deva', 'eng_Latn'
        else:  # en_ne
            src_col, tgt_col = 'English', 'Nepali'
            if self.model_type == 'mbart':
                src_lang, tgt_lang = 'en_XX', 'ne_NP'
            else:  # nllb
                src_lang, tgt_lang = 'eng_Latn', 'npi_Deva'

        sources = test_df[src_col].tolist()
        references = test_df[tgt_col].tolist()

        print(f"\n{'='*80}")
        print(f"Evaluating on {len(sources)} test examples")
        print(f"Direction: {direction} ({src_lang} -> {tgt_lang})")
        print(f"{'='*80}\n")

        # Generate translations
        hypotheses = self.translate_batch(
            sources,
            src_lang,
            tgt_lang,
            max_length=max_length,
            num_beams=num_beams,
            batch_size=batch_size
        )

        # Compute metrics
        print("\nComputing metrics...")
        bleu_scores = self.compute_bleu(hypotheses, references)
        chrf_scores = self.compute_chrf(hypotheses, references)
        length_metrics = self.compute_length_metrics(hypotheses, references)

        # Combine all metrics
        results = {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'direction': direction,
            'test_size': len(sources),
            **bleu_scores,
            **chrf_scores,
            **length_metrics
        }

        # Print results
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"BLEU Score: {results['bleu']:.2f}")
        print(f"chrF++ Score: {results['chrf++']:.2f}")
        print(f"Length Ratio: {results['length_ratio']:.3f}")
        print("="*80 + "\n")

        # Save results
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics JSON
        with open(os.path.join(output_dir, f'metrics_{direction}.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Save translations
        translations_df = pd.DataFrame({
            'source': sources,
            'reference': references,
            'hypothesis': hypotheses
        })
        translations_df.to_csv(
            os.path.join(output_dir, f'translations_{direction}.csv'),
            index=False
        )

        print(f"Results saved to {output_dir}/")

        return results


def compare_models(results_paths: List[str], output_path: str):
    """
    Compare multiple model evaluation results.

    Args:
        results_paths: List of paths to metrics JSON files
        output_path: Path to save comparison results
    """
    all_results = []

    for path in results_paths:
        with open(path, 'r') as f:
            results = json.load(f)
            all_results.append(results)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results)

    # Select key metrics
    key_metrics = ['model_type', 'direction', 'bleu', 'chrf++', 'length_ratio']
    comparison_df = comparison_df[key_metrics]

    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80 + "\n")

    # Save comparison
    comparison_df.to_csv(output_path, index=False)
    print(f"Comparison saved to {output_path}")

    return comparison_df


if __name__ == "__main__":
    # Example usage
    evaluator = TranslationEvaluator(
        model_path="../../models/mbart50/ne_en/final_model",
        model_type='mbart'
    )

    results = evaluator.evaluate_on_test_set(
        test_data_path="../../data/splits/test.csv",
        direction='ne_en',
        output_dir="../../results/metrics/"
    )
