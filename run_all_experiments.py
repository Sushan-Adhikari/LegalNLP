"""
Complete Experiment Runner
Runs all experiments sequentially for the research paper.

Usage:
    python run_all_experiments.py [--skip-training] [--skip-evaluation]

Options:
    --skip-training: Skip training phase (use existing models)
    --skip-evaluation: Skip evaluation phase
    --config: Path to config file (default: config.yaml)
"""

import os
import sys
import argparse
import yaml
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append('src')

from preprocessing.data_loader import LegalCorpusLoader
from training.train_mbart import train_mbart
from training.train_nllb import train_nllb
from evaluation.metrics import TranslationEvaluator
from evaluation.error_analysis import LegalErrorAnalyzer
from visualization.plot_results import VisualizationGenerator


class ExperimentRunner:
    """
    Orchestrates the complete experimental pipeline.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize experiment runner.

        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.results = {}
        self.start_time = datetime.now()

    def log(self, message: str, level: str = 'INFO'):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] [{level}] {message}")

    def step_1_prepare_data(self):
        """Step 1: Data preprocessing and splitting."""
        self.log("="*80)
        self.log("STEP 1: DATA PREPROCESSING")
        self.log("="*80)

        loader = LegalCorpusLoader(
            data_path=self.config['data']['corpus_path'],
            random_seed=self.config['data']['random_seed']
        )

        stats = loader.prepare_dataset(
            output_dir=self.config['data']['output_dir']
        )

        self.results['data_stats'] = stats
        self.log("✓ Data preprocessing complete")

    def step_2_train_models(self):
        """Step 2: Train all model configurations."""
        self.log("="*80)
        self.log("STEP 2: MODEL TRAINING")
        self.log("="*80)

        train_data = os.path.join(self.config['data']['output_dir'], 'train.csv')
        val_data = os.path.join(self.config['data']['output_dir'], 'val.csv')

        for i, exp in enumerate(self.config['experiments'], 1):
            self.log(f"\n[{i}/{len(self.config['experiments'])}] Training: {exp['description']}")
            self.log("-"*80)

            model_type = exp['model']
            direction = exp['direction']
            output_dir = os.path.join(
                self.config['output']['models_dir'],
                model_type,
                direction
            )

            try:
                if model_type == 'mbart':
                    trainer = train_mbart(
                        train_data_path=train_data,
                        val_data_path=val_data,
                        output_dir=output_dir,
                        direction=direction,
                        **self.config['training']
                    )
                elif model_type == 'nllb':
                    trainer = train_nllb(
                        train_data_path=train_data,
                        val_data_path=val_data,
                        output_dir=output_dir,
                        direction=direction,
                        model_name=self.config['models']['nllb']['name'],
                        **self.config['training']
                    )

                self.log(f"✓ Training complete: {exp['description']}")

            except Exception as e:
                self.log(f"✗ Training failed: {exp['description']}", level='ERROR')
                self.log(f"Error: {str(e)}", level='ERROR')
                raise

        self.log("\n✓ All models trained successfully")

    def step_3_evaluate_models(self):
        """Step 3: Evaluate all trained models."""
        self.log("="*80)
        self.log("STEP 3: MODEL EVALUATION")
        self.log("="*80)

        test_data = os.path.join(self.config['data']['output_dir'], 'test.csv')

        # Evaluation configurations
        eval_configs = []

        for exp in self.config['experiments']:
            model_type = exp['model']
            direction_config = exp['direction']
            model_path = os.path.join(
                self.config['output']['models_dir'],
                model_type,
                direction_config,
                'final_model'
            )

            if direction_config == 'bidirectional':
                # Evaluate bidirectional model in both directions
                eval_configs.append((model_path, model_type, 'ne_en', 'bidirectional'))
                eval_configs.append((model_path, model_type, 'en_ne', 'bidirectional'))
            else:
                eval_configs.append((model_path, model_type, direction_config, direction_config))

        all_results = {}

        for i, (model_path, model_type, eval_direction, config_name) in enumerate(eval_configs, 1):
            self.log(f"\n[{i}/{len(eval_configs)}] Evaluating: {model_type} - {eval_direction} ({config_name})")
            self.log("-"*80)

            try:
                evaluator = TranslationEvaluator(
                    model_path=model_path,
                    model_type=model_type
                )

                results = evaluator.evaluate_on_test_set(
                    test_data_path=test_data,
                    direction=eval_direction,
                    output_dir=os.path.join(
                        self.config['output']['metrics_dir'],
                        model_type,
                        config_name
                    ),
                    **self.config['evaluation']
                )

                key = f"{model_type}_{config_name}_{eval_direction}"
                all_results[key] = results

                self.log(f"✓ Evaluation complete: BLEU={results['bleu']:.2f}, chrF++={results['chrf++']:.2f}")

            except Exception as e:
                self.log(f"✗ Evaluation failed: {model_type} - {eval_direction}", level='ERROR')
                self.log(f"Error: {str(e)}", level='ERROR')

        # Save all results
        results_path = os.path.join(self.config['output']['metrics_dir'], 'all_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        self.results['evaluation'] = all_results
        self.log(f"\n✓ All evaluations complete. Results saved to {results_path}")

    def step_4_error_analysis(self):
        """Step 4: Perform error analysis."""
        self.log("="*80)
        self.log("STEP 4: ERROR ANALYSIS")
        self.log("="*80)

        analyzer = LegalErrorAnalyzer()

        for exp in self.config['experiments']:
            model_type = exp['model']
            direction_config = exp['direction']

            # Determine which directions to analyze
            if direction_config == 'bidirectional':
                directions = ['ne_en', 'en_ne']
            else:
                directions = [direction_config]

            for direction in directions:
                self.log(f"\nAnalyzing: {model_type} - {direction} ({direction_config})")
                self.log("-"*80)

                try:
                    translations_path = os.path.join(
                        self.config['output']['metrics_dir'],
                        model_type,
                        direction_config,
                        f'translations_{direction}.csv'
                    )

                    if not os.path.exists(translations_path):
                        self.log(f"⚠ Skipping: {translations_path} not found", level='WARNING')
                        continue

                    results = analyzer.analyze_translations(
                        translations_path=translations_path,
                        direction=direction
                    )

                    analyzer.print_summary(results)

                    # Save analysis
                    output_path = os.path.join(
                        self.config['output']['error_dir'],
                        f"{model_type}_{direction_config}_{direction}_errors.json"
                    )
                    analyzer.save_analysis(results, output_path)

                    self.log(f"✓ Error analysis saved to {output_path}")

                except Exception as e:
                    self.log(f"✗ Error analysis failed: {model_type} - {direction}", level='ERROR')
                    self.log(f"Error: {str(e)}", level='ERROR')

        self.log("\n✓ Error analysis complete")

    def step_5_generate_visualizations(self):
        """Step 5: Generate all visualizations."""
        self.log("="*80)
        self.log("STEP 5: GENERATING VISUALIZATIONS")
        self.log("="*80)

        viz = VisualizationGenerator(
            output_dir=self.config['output']['viz_dir']
        )

        # 1. Corpus statistics
        self.log("\nGenerating corpus visualizations...")
        train_data = os.path.join(self.config['data']['output_dir'], 'train.csv')
        viz.plot_length_distribution(train_data)

        # 2. Model comparisons
        self.log("\nGenerating model comparison plots...")
        results_files = list(Path(self.config['output']['metrics_dir']).rglob('metrics_*.json'))

        if results_files:
            viz.plot_model_comparison(
                results_paths=[str(f) for f in results_files],
                metric='bleu',
                save_name='bleu_comparison.png'
            )

            viz.plot_model_comparison(
                results_paths=[str(f) for f in results_files],
                metric='chrf++',
                save_name='chrf_comparison.png'
            )

        # 3. Directional asymmetry
        self.log("\nGenerating directional asymmetry plots...")
        for model_type in ['mbart', 'nllb']:
            ne_en_path = os.path.join(
                self.config['output']['metrics_dir'],
                model_type, 'ne_en', 'metrics_ne_en.json'
            )
            en_ne_path = os.path.join(
                self.config['output']['metrics_dir'],
                model_type, 'en_ne', 'metrics_en_ne.json'
            )

            if os.path.exists(ne_en_path) and os.path.exists(en_ne_path):
                viz.plot_directional_asymmetry(
                    ne_en_results=ne_en_path,
                    en_ne_results=en_ne_path,
                    model_type=model_type.upper(),
                    save_name=f'{model_type}_directional_asymmetry.png'
                )

        # 4. Error analysis plots
        self.log("\nGenerating error analysis visualizations...")
        error_files = list(Path(self.config['output']['error_dir']).glob('*_errors.json'))

        for error_file in error_files:
            viz.plot_error_analysis(
                error_analysis_path=str(error_file),
                save_name=f"{error_file.stem}_plot.png"
            )

        self.log(f"\n✓ All visualizations saved to {self.config['output']['viz_dir']}")

    def generate_final_report(self):
        """Generate final summary report."""
        self.log("="*80)
        self.log("GENERATING FINAL REPORT")
        self.log("="*80)

        report = {
            'experiment_date': self.start_time.isoformat(),
            'total_duration': str(datetime.now() - self.start_time),
            'configuration': self.config,
            'results': self.results
        }

        report_path = 'experiment_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.log(f"\n✓ Final report saved to {report_path}")

        # Print summary
        self.log("\n" + "="*80)
        self.log("EXPERIMENT COMPLETE!")
        self.log("="*80)
        self.log(f"Total Duration: {datetime.now() - self.start_time}")
        self.log(f"\nResults saved to:")
        self.log(f"  - Models: {self.config['output']['models_dir']}")
        self.log(f"  - Metrics: {self.config['output']['metrics_dir']}")
        self.log(f"  - Visualizations: {self.config['output']['viz_dir']}")
        self.log(f"  - Error Analysis: {self.config['output']['error_dir']}")
        self.log("="*80)

    def run_all(self, skip_training: bool = False, skip_evaluation: bool = False):
        """
        Run complete experimental pipeline.

        Args:
            skip_training: Skip training phase (use existing models)
            skip_evaluation: Skip evaluation phase
        """
        try:
            # Step 1: Data preprocessing
            self.step_1_prepare_data()

            # Step 2: Training (optional)
            if not skip_training:
                self.step_2_train_models()
            else:
                self.log("⚠ Skipping training phase", level='WARNING')

            # Step 3: Evaluation (optional)
            if not skip_evaluation:
                self.step_3_evaluate_models()
            else:
                self.log("⚠ Skipping evaluation phase", level='WARNING')

            # Step 4: Error analysis
            self.step_4_error_analysis()

            # Step 5: Visualizations
            self.step_5_generate_visualizations()

            # Final report
            self.generate_final_report()

        except Exception as e:
            self.log(f"✗ Experiment failed: {str(e)}", level='ERROR')
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run all experiments for Legal MT research')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training phase (use existing models)')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation phase')

    args = parser.parse_args()

    # Create and run experiment
    runner = ExperimentRunner(config_path=args.config)
    runner.run_all(
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation
    )
