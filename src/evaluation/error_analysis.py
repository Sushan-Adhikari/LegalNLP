"""
Error Analysis Module
Performs linguistic error analysis on translation outputs.
Focuses on legal terminology, modality, and clause structure errors.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
from collections import Counter, defaultdict
import json


class LegalErrorAnalyzer:
    """
    Analyzes errors in legal translation with focus on legal-specific phenomena.
    """

    def __init__(self):
        """Initialize error analyzer with legal terminology."""
        # Common legal terms in English (can be expanded)
        self.legal_terms_en = {
            'constitution', 'law', 'statute', 'regulation', 'provision',
            'article', 'section', 'clause', 'amendment', 'act',
            'court', 'justice', 'judge', 'tribunal', 'jurisdiction',
            'plaintiff', 'defendant', 'prosecution', 'defense',
            'shall', 'must', 'may', 'liable', 'entitled', 'provided',
            'hereby', 'thereof', 'wherein', 'whereas', 'notwithstanding',
            'punishment', 'imprisonment', 'fine', 'penalty', 'offense',
            'rights', 'duty', 'obligation', 'authority', 'power'
        }

        # Modal verbs indicating legal modality
        self.modal_verbs = {
            'shall', 'must', 'may', 'should', 'ought', 'can', 'will'
        }

        # Legal conjunctions and connectors
        self.legal_connectors = {
            'provided that', 'subject to', 'notwithstanding',
            'in accordance with', 'pursuant to', 'with respect to',
            'in the event', 'unless otherwise', 'except where'
        }

    def analyze_translations(
        self,
        translations_path: str,
        direction: str
    ) -> Dict:
        """
        Perform comprehensive error analysis.

        Args:
            translations_path: Path to CSV with source, reference, hypothesis
            direction: Translation direction ('ne_en' or 'en_ne')

        Returns:
            Dictionary with error analysis results
        """
        df = pd.read_csv(translations_path)

        results = {
            'direction': direction,
            'total_examples': len(df),
            'terminology_analysis': self.analyze_terminology(df, direction),
            'modality_analysis': self.analyze_modality(df, direction),
            'length_analysis': self.analyze_length_patterns(df),
            'sentence_structure': self.analyze_sentence_structure(df, direction),
            'error_examples': self.extract_error_examples(df, direction)
        }

        return results

    def analyze_terminology(
        self,
        df: pd.DataFrame,
        direction: str
    ) -> Dict:
        """
        Analyze legal terminology preservation.

        Args:
            df: DataFrame with translations
            direction: Translation direction

        Returns:
            Dictionary with terminology analysis
        """
        if direction == 'ne_en':
            # For Nepali to English, check if legal terms appear in hypothesis
            term_preservation = []

            for _, row in df.iterrows():
                ref_terms = set([
                    word.lower() for word in row['reference'].split()
                    if word.lower() in self.legal_terms_en
                ])
                hyp_terms = set([
                    word.lower() for word in row['hypothesis'].split()
                    if word.lower() in self.legal_terms_en
                ])

                if ref_terms:
                    # Calculate what % of reference terms appear in hypothesis
                    preservation_rate = len(ref_terms & hyp_terms) / len(ref_terms)
                    term_preservation.append(preservation_rate)

            return {
                'avg_term_preservation': np.mean(term_preservation) if term_preservation else 0,
                'examples_with_legal_terms': len(term_preservation),
                'perfect_preservation_rate': sum(1 for x in term_preservation if x == 1.0) / len(term_preservation) if term_preservation else 0
            }

        else:  # en_ne
            # For English to Nepali, analyze if source legal terms are handled
            term_handling = []
            term_frequency = Counter()

            for _, row in df.iterrows():
                src_terms = [
                    word.lower() for word in row['source'].split()
                    if word.lower() in self.legal_terms_en
                ]

                for term in src_terms:
                    term_frequency[term] += 1

                if src_terms:
                    term_handling.append(len(src_terms))

            return {
                'avg_terms_per_sentence': np.mean(term_handling) if term_handling else 0,
                'sentences_with_legal_terms': len(term_handling),
                'most_common_terms': term_frequency.most_common(10)
            }

    def analyze_modality(
        self,
        df: pd.DataFrame,
        direction: str
    ) -> Dict:
        """
        Analyze modal verb preservation (legal obligations/permissions).

        Args:
            df: DataFrame with translations
            direction: Translation direction

        Returns:
            Dictionary with modality analysis
        """
        modal_preservation = []

        for _, row in df.iterrows():
            if direction == 'ne_en':
                ref_text = row['reference'].lower()
                hyp_text = row['hypothesis'].lower()
            else:
                ref_text = row['source'].lower()
                hyp_text = row['hypothesis'].lower()

            ref_modals = [modal for modal in self.modal_verbs if modal in ref_text.split()]
            hyp_modals = [modal for modal in self.modal_verbs if modal in hyp_text.split()]

            if ref_modals:
                # Check if the same modals appear
                preservation = len(set(ref_modals) & set(hyp_modals)) / len(set(ref_modals))
                modal_preservation.append(preservation)

        return {
            'avg_modal_preservation': np.mean(modal_preservation) if modal_preservation else 0,
            'sentences_with_modals': len(modal_preservation),
            'perfect_modal_preservation': sum(1 for x in modal_preservation if x == 1.0) / len(modal_preservation) if modal_preservation else 0
        }

    def analyze_length_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze length-related patterns and errors.

        Args:
            df: DataFrame with translations

        Returns:
            Dictionary with length analysis
        """
        df['ref_length'] = df['reference'].str.split().str.len()
        df['hyp_length'] = df['hypothesis'].str.split().str.len()
        df['length_ratio'] = df['hyp_length'] / df['ref_length']

        # Identify problematic translations
        too_short = df[df['length_ratio'] < 0.5]  # Hypothesis is less than half reference length
        too_long = df[df['length_ratio'] > 2.0]   # Hypothesis is more than twice reference length

        return {
            'avg_ref_length': df['ref_length'].mean(),
            'avg_hyp_length': df['hyp_length'].mean(),
            'avg_length_ratio': df['length_ratio'].mean(),
            'length_ratio_std': df['length_ratio'].std(),
            'too_short_count': len(too_short),
            'too_long_count': len(too_long),
            'too_short_percentage': len(too_short) / len(df) * 100,
            'too_long_percentage': len(too_long) / len(df) * 100
        }

    def analyze_sentence_structure(
        self,
        df: pd.DataFrame,
        direction: str
    ) -> Dict:
        """
        Analyze complex sentence structure preservation.

        Args:
            df: DataFrame with translations
            direction: Translation direction

        Returns:
            Dictionary with structure analysis
        """
        # Count commas as proxy for clause complexity
        df['ref_commas'] = df['reference'].str.count(',')
        df['hyp_commas'] = df['hypothesis'].str.count(',')

        # Count legal connectors
        connector_preservation = []

        for _, row in df.iterrows():
            ref_text = row['reference'].lower()
            hyp_text = row['hypothesis'].lower()

            ref_connectors = [conn for conn in self.legal_connectors if conn in ref_text]
            hyp_connectors = [conn for conn in self.legal_connectors if conn in hyp_text]

            if ref_connectors:
                preservation = len(set(ref_connectors) & set(hyp_connectors)) / len(set(ref_connectors))
                connector_preservation.append(preservation)

        return {
            'avg_ref_clause_complexity': df['ref_commas'].mean(),
            'avg_hyp_clause_complexity': df['hyp_commas'].mean(),
            'clause_complexity_ratio': df['hyp_commas'].mean() / df['ref_commas'].mean() if df['ref_commas'].mean() > 0 else 0,
            'connector_preservation': np.mean(connector_preservation) if connector_preservation else 0,
            'sentences_with_connectors': len(connector_preservation)
        }

    def extract_error_examples(
        self,
        df: pd.DataFrame,
        direction: str,
        n_examples: int = 10
    ) -> List[Dict]:
        """
        Extract representative error examples.

        Args:
            df: DataFrame with translations
            direction: Translation direction
            n_examples: Number of examples to extract

        Returns:
            List of error examples
        """
        # Calculate length ratios
        df['length_ratio'] = df['hypothesis'].str.split().str.len() / df['reference'].str.split().str.len()

        # Find examples with extreme length ratios (likely errors)
        length_errors = df.nlargest(n_examples // 2, 'length_ratio').head(n_examples // 2)
        short_errors = df.nsmallest(n_examples // 2, 'length_ratio').head(n_examples // 2)

        examples = []

        for _, row in pd.concat([length_errors, short_errors]).iterrows():
            examples.append({
                'source': row['source'],
                'reference': row['reference'],
                'hypothesis': row['hypothesis'],
                'length_ratio': float(row['length_ratio']),
                'error_type': 'too_long' if row['length_ratio'] > 1.5 else 'too_short'
            })

        return examples

    def save_analysis(self, results: Dict, output_path: str):
        """
        Save error analysis results to JSON.

        Args:
            results: Analysis results dictionary
            output_path: Path to save JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Error analysis saved to {output_path}")

    def print_summary(self, results: Dict):
        """
        Print a summary of error analysis.

        Args:
            results: Analysis results dictionary
        """
        print("\n" + "="*80)
        print("ERROR ANALYSIS SUMMARY")
        print("="*80)
        print(f"Direction: {results['direction']}")
        print(f"Total Examples: {results['total_examples']}\n")

        print("TERMINOLOGY ANALYSIS:")
        for key, value in results['terminology_analysis'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

        print("\nMODALITY ANALYSIS:")
        for key, value in results['modality_analysis'].items():
            print(f"  {key}: {value:.3f}")

        print("\nLENGTH ANALYSIS:")
        for key, value in results['length_analysis'].items():
            print(f"  {key}: {value:.3f}")

        print("\nSENTENCE STRUCTURE:")
        for key, value in results['sentence_structure'].items():
            print(f"  {key}: {value:.3f}")

        print("="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    analyzer = LegalErrorAnalyzer()

    results = analyzer.analyze_translations(
        translations_path="../../results/metrics/translations_ne_en.csv",
        direction='ne_en'
    )

    analyzer.print_summary(results)
    analyzer.save_analysis(results, "../../results/error_analysis/error_analysis_ne_en.json")
