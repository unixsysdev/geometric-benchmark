"""
Geometric Benchmark Library

Shared code for all benchmark tasks.
"""

from .models import (
    UnifiedTransformer,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    create_model,
    tiny_transformer,
    small_transformer,
    medium_transformer,
)

from .datasets import (
    BaseDataset,
    ModularArithmeticDataset,
    DigitSumDataset,
    ParityDataset,
    create_dataloader,
    TASK_REGISTRY,
    get_dataset,
)

from .training import (
    Trainer,
    Evaluator,
    create_trainer,
)

__all__ = [
    # Models
    'UnifiedTransformer',
    'SinusoidalPositionalEncoding',
    'LearnedPositionalEncoding',
    'create_model',
    'tiny_transformer',
    'small_transformer',
    'medium_transformer',
    # Datasets
    'BaseDataset',
    'ModularArithmeticDataset',
    'DigitSumDataset',
    'ParityDataset',
    'create_dataloader',
    'TASK_REGISTRY',
    'get_dataset',
    # Training
    'Trainer',
    'Evaluator',
    'create_trainer',
]
