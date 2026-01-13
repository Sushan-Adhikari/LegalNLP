"""
PyTorch Dataset Class for Legal Translation
Handles data loading and tokenization for training.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for translation tasks.
    Supports both uni-directional and bi-directional translation.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        source_lang: str,
        target_lang: str,
        max_length: int = 128,
        bidirectional: bool = False
    ):
        """
        Initialize the translation dataset.

        Args:
            data_path: Path to CSV file with English and Nepali columns
            tokenizer: Hugging Face tokenizer
            source_lang: Source language code ('en_XX' or 'ne_NP')
            target_lang: Target language code ('en_XX' or 'ne_NP')
            max_length: Maximum sequence length
            bidirectional: If True, alternate between both translation directions
        """
        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length
        self.bidirectional = bidirectional

        # Map language codes to column names
        self.lang_to_col = {
            'en_XX': 'English',
            'ne_NP': 'Nepali',
            'eng_Latn': 'English',  # NLLB format
            'npi_Deva': 'Nepali'    # NLLB format
        }

    def __len__(self) -> int:
        """Return dataset size."""
        if self.bidirectional:
            return len(self.df) * 2  # Each pair used in both directions
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Args:
            idx: Index of the example

        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        if self.bidirectional:
            # Alternate between directions
            actual_idx = idx // 2
            direction = idx % 2

            if direction == 0:
                # Original direction
                src_lang, tgt_lang = self.source_lang, self.target_lang
            else:
                # Reverse direction
                src_lang, tgt_lang = self.target_lang, self.source_lang
        else:
            actual_idx = idx
            src_lang, tgt_lang = self.source_lang, self.target_lang

        # Get source and target column names
        src_col = self.lang_to_col.get(src_lang, 'English')
        tgt_col = self.lang_to_col.get(tgt_lang, 'Nepali')

        # Get text
        source_text = str(self.df.iloc[actual_idx][src_col])
        target_text = str(self.df.iloc[actual_idx][tgt_col])

        # Tokenize source
        model_inputs = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        model_inputs['labels'] = labels['input_ids']

        # Squeeze to remove batch dimension
        return {
            'input_ids': model_inputs['input_ids'].squeeze(),
            'attention_mask': model_inputs['attention_mask'].squeeze(),
            'labels': model_inputs['labels'].squeeze()
        }


class DirectionalTranslationDataset(Dataset):
    """
    Dataset that allows switching between translation directions.
    Useful for separate uni-directional models.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        direction: str,  # 'ne_en' or 'en_ne'
        max_length: int = 128,
        model_type: str = 'mbart'  # 'mbart' or 'nllb'
    ):
        """
        Initialize directional dataset.

        Args:
            data_path: Path to CSV file
            tokenizer: Hugging Face tokenizer
            direction: Translation direction ('ne_en' or 'en_ne')
            max_length: Maximum sequence length
            model_type: Model type for language code mapping
        """
        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.direction = direction
        self.max_length = max_length
        self.model_type = model_type

        # Set language codes based on model type
        if model_type == 'mbart':
            self.lang_codes = {'en': 'en_XX', 'ne': 'ne_NP'}
        else:  # nllb
            self.lang_codes = {'en': 'eng_Latn', 'ne': 'npi_Deva'}

        # Determine source and target
        if direction == 'ne_en':
            self.source_col = 'Nepali'
            self.target_col = 'English'
            self.src_lang = self.lang_codes['ne']
            self.tgt_lang = self.lang_codes['en']
        elif direction == 'en_ne':
            self.source_col = 'English'
            self.target_col = 'Nepali'
            self.src_lang = self.lang_codes['en']
            self.tgt_lang = self.lang_codes['ne']
        else:
            raise ValueError(f"Invalid direction: {direction}. Use 'ne_en' or 'en_ne'")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example in the specified direction."""
        source_text = str(self.df.iloc[idx][self.source_col])
        target_text = str(self.df.iloc[idx][self.target_col])

        # Tokenize
        model_inputs = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        model_inputs['labels'] = labels['input_ids']

        return {
            'input_ids': model_inputs['input_ids'].squeeze(),
            'attention_mask': model_inputs['attention_mask'].squeeze(),
            'labels': model_inputs['labels'].squeeze()
        }


if __name__ == "__main__":
    # Example usage
    from transformers import MBartTokenizer

    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50")

    # Test uni-directional dataset
    dataset = DirectionalTranslationDataset(
        data_path="../../data/splits/train.csv",
        tokenizer=tokenizer,
        direction='ne_en',
        model_type='mbart'
    )

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input shape: {sample['input_ids'].shape}")
