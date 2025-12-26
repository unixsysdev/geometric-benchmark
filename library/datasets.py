"""
Unified Dataset Infrastructure for Geometric Benchmark

All tasks inherit from BaseDataset to ensure consistent interface.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import numpy as np


class BaseDataset(Dataset, ABC):
    """
    Base class for all benchmark datasets.

    All tasks must inherit and implement:
    - __len__: Number of samples
    - __getitem__: Return (input_tokens, target_tokens, metadata)

    Metadata should include:
    - 'task': Task name
    - 'category': Task category (periodic_1d, topological, etc.)
    - Any task-specific info (modulus, curve parameters, etc.)
    """

    def __init__(self, split: str = 'train'):
        """
        Args:
            split: 'train', 'val', or 'test'
        """
        self.split = split
        self.metadata = {
            'task': self.__class__.__name__,
            'split': split,
        }

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Returns:
            input_tokens: Tensor of input token IDs
            target_tokens: Tensor of target token IDs
            metadata: Dictionary with sample info
        """
        pass

    def get_vocab_size(self) -> int:
        """Return vocabulary size for this task."""
        raise NotImplementedError

    def get_output_size(self) -> int:
        """Return output vocabulary size."""
        return self.get_vocab_size()

    def get_seq_len(self) -> Tuple[int, int]:
        """Return (input_seq_len, output_seq_len)."""
        raise NotImplementedError

    def collate_fn(self, batch: list) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.

        Args:
            batch: List of (input, target, metadata) tuples

        Returns:
            Dictionary with 'input', 'target', 'metadata' keys
        """
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        metadata_list = [item[2] for item in batch]

        # Stack inputs and targets (assuming same length within batch)
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)

        return {
            'input': inputs,
            'target': targets,
            'metadata': metadata_list,
        }


class ModularArithmeticDataset(BaseDataset):
    """
    Modular addition/multiplication dataset: a op b mod p

    Args:
        operation: 'add' or 'mul'
        modulus: Prime number p
        train_frac: Fraction of data to use for training (for grokking experiments)
        split: 'train', 'val', or 'test'
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        operation: str = 'add',
        modulus: int = 97,
        train_frac: float = 0.5,
        split: str = 'train',
        seed: int = 42,
    ):
        super().__init__(split)
        self.operation = operation
        self.modulus = modulus
        self.train_frac = train_frac
        self.seed = seed

        # Generate all possible pairs
        self.pairs = []
        for a in range(modulus):
            for b in range(modulus):
                self.pairs.append((a, b))
        self.pairs = np.array(self.pairs)

        # Split into train/val/test
        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.pairs))
        rng.shuffle(indices)

        n_train = int(len(indices) * train_frac)
        n_val = int(len(indices) * (1 - train_frac) / 2)

        if split == 'train':
            self.indices = indices[:n_train]
        elif split == 'val':
            self.indices = indices[n_train:n_train + n_val]
        else:  # test
            self.indices = indices[n_train + n_val:]

        # Filter pairs to split
        self.pairs = self.pairs[self.indices]

        self.metadata.update({
            'operation': operation,
            'modulus': modulus,
            'train_frac': train_frac,
        })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]

        # Compute result
        if self.operation == 'add':
            result = (a + b) % self.modulus
        elif self.operation == 'mul':
            result = (a * b) % self.modulus
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        # Return as token sequences
        # Format: [a, b, op] -> [result]
        # For simplicity, we'll use single-token input and output
        # Format can be overridden by tasks that need sequence input

        input_tokens = torch.tensor([a, b], dtype=torch.long)
        target_tokens = torch.tensor([result], dtype=torch.long)

        metadata = {
            **self.metadata,
            'a': a,
            'b': b,
            'result': result,
            'idx': idx,
        }

        return input_tokens, target_tokens, metadata

    def get_vocab_size(self):
        # Tokens: 0 to modulus-1, plus operation token, plus EOS token
        return self.modulus + 2

    def get_seq_len(self):
        return (2, 1)  # Input: [a, b], Output: [result]


class DigitSumDataset(BaseDataset):
    """
    Sum of digits dataset: Given number, output sum of its digits.

    Args:
        max_digits: Maximum number of digits
        base: Numerical base (default 10)
        split: 'train', 'val', or 'test'
    """

    def __init__(
        self,
        max_digits: int = 3,
        base: int = 10,
        split: str = 'train',
        seed: int = 42,
    ):
        super().__init__(split)
        self.max_digits = max_digits
        self.base = base
        self.seed = seed

        # Generate all numbers up to max_digits
        max_val = base ** max_digits
        self.numbers = np.arange(max_val)

        # Split by range
        n_train = int(max_val * 0.7)
        n_val = int(max_val * 0.15)

        if split == 'train':
            self.numbers = self.numbers[:n_train]
        elif split == 'val':
            self.numbers = self.numbers[n_train:n_train + n_val]
        else:
            self.numbers = self.numbers[n_train + n_val:]

        self.metadata.update({
            'max_digits': max_digits,
            'base': base,
        })

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, idx):
        num = self.numbers[idx]

        # Convert to base and sum digits
        digits = []
        temp = num
        while temp > 0:
            digits.append(temp % self.base)
            temp //= self.base

        # Pad to max_digits
        while len(digits) < self.max_digits:
            digits.append(0)

        digit_sum = sum(digits)

        # Input: digits as tokens, Output: sum
        input_tokens = torch.tensor(digits, dtype=torch.long)
        target_tokens = torch.tensor([digit_sum], dtype=torch.long)

        metadata = {
            **self.metadata,
            'number': num,
            'digits': digits,
            'sum': digit_sum,
        }

        return input_tokens, target_tokens, metadata

    def get_vocab_size(self):
        return self.base

    def get_seq_len(self):
        return (self.max_digits, 1)


class ParityDataset(BaseDataset):
    """
    Parity classification: Is the number even or odd?

    Args:
        max_val: Maximum value
        split: 'train', 'val', or 'test'
    """

    def __init__(self, max_val: int = 1000, split: str = 'train', seed: int = 42):
        super().__init__(split)
        self.max_val = max_val
        self.numbers = np.arange(max_val)

        n_train = int(max_val * 0.7)
        n_val = int(max_val * 0.15)

        if split == 'train':
            self.numbers = self.numbers[:n_train]
        elif split == 'val':
            self.numbers = self.numbers[n_train:n_train + n_val]
        else:
            self.numbers = self.numbers[n_train + n_val:]

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, idx):
        num = self.numbers[idx]
        parity = num % 2

        # For simplicity, represent number in decimal digits
        digits = [int(d) for d in str(num).zfill(4)]  # Pad to 4 digits

        input_tokens = torch.tensor(digits, dtype=torch.long)
        target_tokens = torch.tensor([parity], dtype=torch.long)

        metadata = {
            **self.metadata,
            'number': num,
            'parity': parity,
        }

        return input_tokens, target_tokens, metadata

    def get_vocab_size(self):
        return 10  # Digits 0-9

    def get_seq_len(self):
        return (4, 1)


def create_dataloader(dataset: BaseDataset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Create DataLoader with proper collate function.

    Args:
        dataset: BaseDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )


# Task registry - maps task names to dataset classes
TASK_REGISTRY = {
    'mod_add': ModularArithmeticDataset,
    'mod_mul': ModularArithmeticDataset,
    'digit_sum': DigitSumDataset,
    'parity': ParityDataset,
}


def get_dataset(task_name: str, split: str = 'train', **kwargs) -> BaseDataset:
    """
    Factory function to create dataset by task name.

    Args:
        task_name: Name of task (must be in TASK_REGISTRY)
        split: 'train', 'val', or 'test'
        **kwargs: Task-specific arguments

    Returns:
        BaseDataset instance
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")

    dataset_class = TASK_REGISTRY[task_name]
    return dataset_class(split=split, **kwargs)
