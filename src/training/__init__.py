"""Model training modules."""

from .train_mbart import train_mbart
from .train_nllb import train_nllb

__all__ = [
    'train_mbart',
    'train_nllb'
]
