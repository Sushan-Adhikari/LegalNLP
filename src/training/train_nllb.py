"""
Training Script for NLLB-200 Model
Supports both uni-directional and bi-directional training.
"""

import os
import sys
import argparse
import torch
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from torch.utils.data import DataLoader
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.dataset import DirectionalTranslationDataset


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_nllb(
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    direction: str = 'ne_en',  # 'ne_en', 'en_ne', or 'bidirectional'
    model_name: str = "facebook/nllb-200-distilled-600M",  # Can also use 1.3B or 3.3B
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    warmup_steps: int = 500,
    max_length: int = 128,
    gradient_accumulation_steps: int = 4,
    fp16: bool = True,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 100,
    seed: int = 42
):
    """
    Train NLLB-200 model for legal translation.

    Args:
        train_data_path: Path to training CSV
        val_data_path: Path to validation CSV
        output_dir: Directory to save model checkpoints
        direction: Translation direction or 'bidirectional'
        model_name: NLLB model variant to use
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        max_length: Maximum sequence length
        gradient_accumulation_steps: Gradient accumulation steps
        fp16: Use mixed precision training
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        logging_steps: Log every N steps
        seed: Random seed
    """
    # Set seed
    set_seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print(f"Training NLLB-200 Model")
    print(f"Model: {model_name}")
    print(f"Direction: {direction}")
    print(f"Output: {output_dir}")
    print("="*80)

    # Load tokenizer and model
    print("\nLoading NLLB tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # NLLB language codes: eng_Latn (English), npi_Deva (Nepali)
    if direction == 'ne_en':
        tokenizer.src_lang = "npi_Deva"
        tokenizer.tgt_lang = "eng_Latn"
    elif direction == 'en_ne':
        tokenizer.src_lang = "eng_Latn"
        tokenizer.tgt_lang = "npi_Deva"

    print(f"Model parameters: {model.num_parameters():,}")

    # Create datasets
    print("\nLoading datasets...")

    if direction == 'bidirectional':
        from torch.utils.data import ConcatDataset

        train_ne_en = DirectionalTranslationDataset(
            data_path=train_data_path,
            tokenizer=tokenizer,
            direction='ne_en',
            max_length=max_length,
            model_type='nllb'
        )

        train_en_ne = DirectionalTranslationDataset(
            data_path=train_data_path,
            tokenizer=tokenizer,
            direction='en_ne',
            max_length=max_length,
            model_type='nllb'
        )

        train_dataset = ConcatDataset([train_ne_en, train_en_ne])

        val_ne_en = DirectionalTranslationDataset(
            data_path=val_data_path,
            tokenizer=tokenizer,
            direction='ne_en',
            max_length=max_length,
            model_type='nllb'
        )

        val_en_ne = DirectionalTranslationDataset(
            data_path=val_data_path,
            tokenizer=tokenizer,
            direction='en_ne',
            max_length=max_length,
            model_type='nllb'
        )

        val_dataset = ConcatDataset([val_ne_en, val_en_ne])

    else:
        # Uni-directional training
        train_dataset = DirectionalTranslationDataset(
            data_path=train_data_path,
            tokenizer=tokenizer,
            direction=direction,
            max_length=max_length,
            model_type='nllb'
        )

        val_dataset = DirectionalTranslationDataset(
            data_path=val_data_path,
            tokenizer=tokenizer,
            direction=direction,
            max_length=max_length,
            model_type='nllb'
        )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16 and torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=max_length,
        report_to=["tensorboard"],
        seed=seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Save training config
    config = {
        'model': model_name,
        'direction': direction,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_length': max_length,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'warmup_steps': warmup_steps,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(output_dir, 'final_model'))
    tokenizer.save_pretrained(os.path.join(output_dir, 'final_model'))

    print(f"\nTraining complete! Model saved to {output_dir}/final_model")

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NLLB-200 for legal translation")

    parser.add_argument('--train_data', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--direction', type=str, default='ne_en',
                        choices=['ne_en', 'en_ne', 'bidirectional'],
                        help='Translation direction')
    parser.add_argument('--model_name', type=str,
                        default='facebook/nllb-200-distilled-600M',
                        help='NLLB model variant')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    train_nllb(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        direction=args.direction,
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        seed=args.seed
    )
