"""
Data Loading and Preprocessing Module
Handles loading, cleaning, and splitting of the Nepali-English legal parallel corpus.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import re
import os


class LegalCorpusLoader:
    """
    Loads and preprocesses the Nepali-English legal parallel corpus.
    """

    def __init__(self, data_path: str, random_seed: int = 42):
        """
        Initialize the corpus loader.

        Args:
            data_path: Path to the Excel file containing parallel corpus
            random_seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_seed = random_seed
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the Excel file containing parallel corpus.

        Returns:
            DataFrame with English and Nepali columns
        """
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_excel(self.data_path)
        print(f"Loaded {len(self.df)} sentence pairs")
        print(f"Columns: {self.df.columns.tolist()}")
        return self.df

    def clean_text(self, text: str) -> str:
        """
        Clean individual text entries.

        Args:
            text: Input text string

        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""

        # Convert to string and strip whitespace
        text = str(text).strip()

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing punctuation spaces
        text = text.strip()

        return text

    def clean_corpus(self) -> pd.DataFrame:
        """
        Clean the entire corpus.

        Returns:
            Cleaned DataFrame
        """
        print("Cleaning corpus...")

        # Apply cleaning to both columns
        self.df['English'] = self.df['English'].apply(self.clean_text)
        self.df['Nepali'] = self.df['Nepali'].apply(self.clean_text)

        # Remove empty entries
        initial_count = len(self.df)
        self.df = self.df[(self.df['English'] != "") & (self.df['Nepali'] != "")]
        removed_count = initial_count - len(self.df)

        if removed_count > 0:
            print(f"Removed {removed_count} empty sentence pairs")

        # Remove duplicates
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['English', 'Nepali'])
        removed_count = initial_count - len(self.df)

        if removed_count > 0:
            print(f"Removed {removed_count} duplicate sentence pairs")

        print(f"Final corpus size: {len(self.df)} sentence pairs")

        return self.df

    def get_statistics(self) -> Dict:
        """
        Compute corpus statistics.

        Returns:
            Dictionary containing various statistics
        """
        stats = {
            'total_pairs': len(self.df),
            'avg_english_length': self.df['English'].str.split().str.len().mean(),
            'avg_nepali_length': self.df['Nepali'].str.split().str.len().mean(),
            'max_english_length': self.df['English'].str.split().str.len().max(),
            'max_nepali_length': self.df['Nepali'].str.split().str.len().max(),
            'min_english_length': self.df['English'].str.split().str.len().min(),
            'min_nepali_length': self.df['Nepali'].str.split().str.len().min(),
        }

        return stats

    def split_data(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        print(f"\nSplitting data: {train_ratio:.1%} train, {val_ratio:.1%} val, {test_ratio:.1%} test")

        # First split: separate test set
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=test_ratio,
            random_state=self.random_seed,
            shuffle=True
        )

        # Second split: separate validation from training
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=self.random_seed,
            shuffle=True
        )

        print(f"Train set: {len(train_df)} pairs")
        print(f"Validation set: {len(val_df)} pairs")
        print(f"Test set: {len(test_df)} pairs")

        return train_df, val_df, test_df

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str
    ) -> None:
        """
        Save train/val/test splits to disk.

        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            output_dir: Directory to save the splits
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save as CSV files
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

        print(f"\nSaved splits to {output_dir}/")
        print(f"  - train.csv: {len(train_df)} pairs")
        print(f"  - val.csv: {len(val_df)} pairs")
        print(f"  - test.csv: {len(test_df)} pairs")

    def prepare_dataset(self, output_dir: str) -> Dict:
        """
        Complete pipeline: load, clean, split, and save data.

        Args:
            output_dir: Directory to save processed data

        Returns:
            Dictionary containing statistics
        """
        # Load and clean
        self.load_data()
        self.clean_corpus()

        # Get statistics
        stats = self.get_statistics()
        print("\n=== Corpus Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

        # Split data
        train_df, val_df, test_df = self.split_data()

        # Save splits
        self.save_splits(train_df, val_df, test_df, output_dir)

        return stats


if __name__ == "__main__":
    # Example usage
    loader = LegalCorpusLoader("../../fileours.xlsx")
    stats = loader.prepare_dataset("../../data/splits")
