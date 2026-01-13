"""
Visualization Module
Creates publication-quality plots for research paper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud


# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class VisualizationGenerator:
    """
    Generates all visualizations for the research paper.
    """

    def __init__(self, output_dir: str):
        """
        Initialize visualization generator.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_model_comparison(
        self,
        results_paths: List[str],
        metric: str = 'bleu',
        save_name: str = 'model_comparison.png'
    ):
        """
        Create bar plot comparing models across configurations.

        Args:
            results_paths: List of paths to metric JSON files
            metric: Metric to compare ('bleu' or 'chrf++')
            save_name: Output filename
        """
        # Load all results
        data = []
        for path in results_paths:
            with open(path, 'r') as f:
                result = json.load(f)
                data.append({
                    'Model': result['model_type'].upper(),
                    'Direction': result['direction'].replace('_', '→').upper(),
                    'Config': 'Bidirectional' if 'bidirectional' in path else 'Uni-directional',
                    'Score': result[metric]
                })

        df = pd.DataFrame(data)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by model and direction
        x = np.arange(len(df))
        width = 0.35

        bars = ax.bar(x, df['Score'], width, alpha=0.8)

        # Color by configuration
        colors = {'Uni-directional': '#2ecc71', 'Bidirectional': '#e74c3c'}
        for bar, config in zip(bars, df['Config']):
            bar.set_color(colors[config])

        ax.set_xlabel('Model Configuration', fontweight='bold')
        ax.set_ylabel(f'{metric.upper()} Score', fontweight='bold')
        ax.set_title(f'{metric.upper()} Score Comparison Across Models and Directions',
                     fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['Model']}\n{row['Direction']}"
                            for _, row in df.iterrows()], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(handles=[plt.Rectangle((0,0),1,1, fc=colors[c]) for c in colors.keys()],
                  labels=colors.keys(), loc='upper right')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_name}")

    def plot_directional_asymmetry(
        self,
        ne_en_results: str,
        en_ne_results: str,
        model_type: str,
        save_name: str = 'directional_asymmetry.png'
    ):
        """
        Plot directional asymmetry for both BLEU and chrF++.

        Args:
            ne_en_results: Path to Nepali→English results JSON
            en_ne_results: Path to English→Nepali results JSON
            model_type: Model type name for title
            save_name: Output filename
        """
        with open(ne_en_results, 'r') as f:
            ne_en = json.load(f)
        with open(en_ne_results, 'r') as f:
            en_ne = json.load(f)

        metrics = ['bleu', 'chrf++']
        ne_en_scores = [ne_en[m] for m in metrics]
        en_ne_scores = [en_ne[m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width/2, ne_en_scores, width, label='Nepali→English',
                       color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, en_ne_scores, width, label='English→Nepali',
                       color='#e67e22', alpha=0.8)

        ax.set_xlabel('Metric', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(f'Directional Asymmetry in {model_type}',
                     fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_name}")

    def plot_training_curves(
        self,
        tensorboard_log_dir: str,
        save_name: str = 'training_curves.png'
    ):
        """
        Plot training and validation loss curves.

        Args:
            tensorboard_log_dir: Directory with TensorBoard logs
            save_name: Output filename
        """
        try:
            from tensorboard.backend.event_processing import event_accumulator

            # Load TensorBoard data
            ea = event_accumulator.EventAccumulator(tensorboard_log_dir)
            ea.Reload()

            # Get loss data
            train_loss = pd.DataFrame(ea.Scalars('train/loss'))
            eval_loss = pd.DataFrame(ea.Scalars('eval/loss'))

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(train_loss['step'], train_loss['value'],
                   label='Training Loss', linewidth=2, color='#3498db')
            ax.plot(eval_loss['step'], eval_loss['value'],
                   label='Validation Loss', linewidth=2, color='#e74c3c')

            ax.set_xlabel('Training Steps', fontweight='bold')
            ax.set_ylabel('Loss', fontweight='bold')
            ax.set_title('Training and Validation Loss Curves', fontweight='bold', pad=20)
            ax.legend()
            ax.grid(alpha=0.3, linestyle='--')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight')
            plt.close()

            print(f"Saved: {save_name}")

        except Exception as e:
            print(f"Could not plot training curves: {e}")

    def plot_length_distribution(
        self,
        data_path: str,
        save_name: str = 'length_distribution.png'
    ):
        """
        Plot sentence length distribution for both languages.

        Args:
            data_path: Path to CSV with English and Nepali columns
            save_name: Output filename
        """
        df = pd.read_csv(data_path)

        df['english_length'] = df['English'].str.split().str.len()
        df['nepali_length'] = df['Nepali'].str.split().str.len()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # English length distribution
        axes[0].hist(df['english_length'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Sentence Length (words)', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_title('English Sentence Length Distribution', fontweight='bold')
        axes[0].axvline(df['english_length'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["english_length"].mean():.1f}')
        axes[0].legend()
        axes[0].grid(alpha=0.3, linestyle='--')

        # Nepali length distribution
        axes[1].hist(df['nepali_length'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Sentence Length (words)', fontweight='bold')
        axes[1].set_ylabel('Frequency', fontweight='bold')
        axes[1].set_title('Nepali Sentence Length Distribution', fontweight='bold')
        axes[1].axvline(df['nepali_length'].mean(), color='blue', linestyle='--',
                       label=f'Mean: {df["nepali_length"].mean():.1f}')
        axes[1].legend()
        axes[1].grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_name}")

    def plot_error_analysis(
        self,
        error_analysis_path: str,
        save_name: str = 'error_analysis.png'
    ):
        """
        Visualize error analysis results.

        Args:
            error_analysis_path: Path to error analysis JSON
            save_name: Output filename
        """
        with open(error_analysis_path, 'r') as f:
            results = json.load(f)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Terminology preservation
        term_data = results['terminology_analysis']
        if 'avg_term_preservation' in term_data:
            metrics = ['Avg Term\nPreservation', 'Perfect\nPreservation']
            values = [
                term_data.get('avg_term_preservation', 0),
                term_data.get('perfect_preservation_rate', 0)
            ]
            axes[0, 0].bar(metrics, values, color=['#3498db', '#2ecc71'], alpha=0.7)
            axes[0, 0].set_ylabel('Rate', fontweight='bold')
            axes[0, 0].set_title('Legal Terminology Preservation', fontweight='bold')
            axes[0, 0].set_ylim([0, 1])
            axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. Modality preservation
        modal_data = results['modality_analysis']
        metrics = ['Avg Modal\nPreservation', 'Perfect\nPreservation']
        values = [
            modal_data.get('avg_modal_preservation', 0),
            modal_data.get('perfect_modal_preservation', 0)
        ]
        axes[0, 1].bar(metrics, values, color=['#e74c3c', '#f39c12'], alpha=0.7)
        axes[0, 1].set_ylabel('Rate', fontweight='bold')
        axes[0, 1].set_title('Modal Verb Preservation', fontweight='bold')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 3. Length analysis
        length_data = results['length_analysis']
        categories = ['Too Short', 'Appropriate', 'Too Long']
        values = [
            length_data.get('too_short_percentage', 0),
            100 - length_data.get('too_short_percentage', 0) - length_data.get('too_long_percentage', 0),
            length_data.get('too_long_percentage', 0)
        ]
        colors = ['#e74c3c', '#2ecc71', '#f39c12']
        axes[1, 0].pie(values, labels=categories, colors=colors, autopct='%1.1f%%',
                      startangle=90)
        axes[1, 0].set_title('Length Appropriateness Distribution', fontweight='bold')

        # 4. Sentence structure
        struct_data = results['sentence_structure']
        metrics = ['Clause\nComplexity\nRatio', 'Connector\nPreservation']
        values = [
            struct_data.get('clause_complexity_ratio', 0),
            struct_data.get('connector_preservation', 0)
        ]
        axes[1, 1].bar(metrics, values, color=['#9b59b6', '#1abc9c'], alpha=0.7)
        axes[1, 1].set_ylabel('Ratio/Rate', fontweight='bold')
        axes[1, 1].set_title('Sentence Structure Analysis', fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_name}")

    def plot_bidirectional_impact(
        self,
        uni_results: List[str],
        bi_results: List[str],
        save_name: str = 'bidirectional_impact.png'
    ):
        """
        Compare uni-directional vs bi-directional training impact.

        Args:
            uni_results: List of paths to uni-directional results
            bi_results: List of paths to bi-directional results
            save_name: Output filename
        """
        data = []

        for path in uni_results:
            with open(path, 'r') as f:
                result = json.load(f)
                data.append({
                    'Direction': result['direction'].replace('_', '→').upper(),
                    'BLEU': result['bleu'],
                    'chrF++': result['chrf++'],
                    'Training': 'Uni-directional'
                })

        for path in bi_results:
            with open(path, 'r') as f:
                result = json.load(f)
                data.append({
                    'Direction': result['direction'].replace('_', '→').upper(),
                    'BLEU': result['bleu'],
                    'chrF++': result['chrf++'],
                    'Training': 'Bi-directional'
                })

        df = pd.DataFrame(data)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, metric in enumerate(['BLEU', 'chrF++']):
            pivot = df.pivot(index='Direction', columns='Training', values=metric)
            pivot.plot(kind='bar', ax=axes[idx], color=['#3498db', '#e74c3c'], alpha=0.7)

            axes[idx].set_xlabel('Translation Direction', fontweight='bold')
            axes[idx].set_ylabel(f'{metric} Score', fontweight='bold')
            axes[idx].set_title(f'Impact of Bi-directional Training on {metric}',
                              fontweight='bold')
            axes[idx].legend(title='Training Mode')
            axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
            axes[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_name}")

    def create_all_visualizations(
        self,
        results_dir: str,
        data_path: str
    ):
        """
        Generate all visualizations for the paper.

        Args:
            results_dir: Directory containing all results
            data_path: Path to the dataset
        """
        print("Generating all visualizations...")
        print("="*80)

        # 1. Length distribution
        self.plot_length_distribution(data_path)

        # 2. Find and plot model comparisons
        # This assumes a specific directory structure
        # Adjust paths as needed based on actual results

        print("\nAll visualizations generated successfully!")
        print(f"Output directory: {self.output_dir}")


if __name__ == "__main__":
    viz = VisualizationGenerator(output_dir="../../results/visualizations")

    # Example: Length distribution
    viz.plot_length_distribution("../../data/splits/train.csv")
