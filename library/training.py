"""
Unified Training Infrastructure for Geometric Benchmark

Provides standardized training loops, checkpointing, and evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import json
import time
from tqdm import tqdm

from .models import UnifiedTransformer
from .datasets import BaseDataset


class Trainer:
    """
    Unified trainer for all benchmark tasks.

    Features:
    - Automatic checkpointing
    - Metric tracking (train/val loss, accuracy)
    - Early stopping (optional)
    - Learning rate scheduling
    - Evaluation hooks for mechanistic analysis
    """

    def __init__(
        self,
        model: UnifiedTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: optim.Optimizer,
        loss_fn: Callable,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        task_name: str = 'task',
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.task_name = task_name
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'steps': [],
            'epochs': [],
        }

        # Callbacks for mechanistic analysis
        self.callbacks = []

    def add_callback(self, callback: Callable):
        """Add a callback function to call during training."""
        self.callbacks.append(callback)

    def train(
        self,
        n_steps: int,
        lr_scheduler: Optional[object] = None,
        eval_fn: Optional[Callable] = None,
    ) -> Dict[str, list]:
        """
        Main training loop.

        Args:
            n_steps: Number of training steps
            lr_scheduler: Optional learning rate scheduler
            eval_fn: Optional custom evaluation function

        Returns:
            Training history dictionary
        """
        self.model.train()
        global_step = 0
        epoch = 0

        pbar = tqdm(total=n_steps, desc=f"Training {self.task_name}")

        while global_step < n_steps:
            for batch in self.train_loader:
                if global_step >= n_steps:
                    break

                # Move to device
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)

                # Forward pass
                logits = self.model(inputs, return_cache=True)

                # Handle different task types based on target shape
                batch_size, seq_len, vocab_size = logits.shape
                target_seq_len = targets.shape[1] if targets.dim() > 1 else 1

                if target_seq_len == 1:
                    # Classification task: use last position only
                    logits_for_loss = logits[:, -1, :]  # [batch, vocab_size]
                    targets_flat = targets.view(-1)  # [batch]
                else:
                    # Sequence-to-sequence task: flatten all positions
                    logits_for_loss = logits.view(-1, vocab_size)
                    targets_flat = targets.view(-1)

                loss = self.loss_fn(logits_for_loss, targets_flat)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (optional)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                if lr_scheduler:
                    lr_scheduler.step()

                # Compute accuracy
                with torch.no_grad():
                    preds = logits_for_loss.argmax(dim=-1)
                    acc = (preds == targets_flat).float().mean()

                # Log metrics
                self.history['train_loss'].append(loss.item())
                self.history['train_acc'].append(acc.item())
                self.history['steps'].append(global_step)

                # Progress bar update
                if global_step % self.log_interval == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{acc.item():.4f}',
                        'step': global_step,
                    })

                # Evaluation
                if self.val_loader and global_step % self.eval_interval == 0:
                    val_metrics = self.evaluate()
                    self.history['val_loss'].append(val_metrics['loss'])
                    self.history['val_acc'].append(val_metrics['acc'])
                    self.history['epochs'].append(global_step)

                    # Update progress bar
                    pbar.write(f"Step {global_step}: Val Loss {val_metrics['loss']:.4f}, Val Acc {val_metrics['acc']:.4f}")

                    # Callbacks
                    for callback in self.callbacks:
                        callback(self.model, global_step, val_metrics)

                # Checkpointing
                if global_step % self.save_interval == 0 and global_step > 0:
                    self.save_checkpoint(global_step)

                global_step += 1
                pbar.update(1)

            epoch += 1

        pbar.close()

        # Final save
        self.save_checkpoint(global_step)
        self.save_history()

        return self.history

    def evaluate(self, eval_fn: Optional[Callable] = None) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Args:
            eval_fn: Optional custom evaluation function

        Returns:
            Dictionary with 'loss' and 'acc' keys
        """
        if eval_fn:
            return eval_fn(self.model, self.val_loader)

        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)

                logits = self.model(inputs)

                # Handle different task types based on target shape
                batch_size, seq_len, vocab_size = logits.shape
                target_seq_len = targets.shape[1] if targets.dim() > 1 else 1

                if target_seq_len == 1:
                    # Classification task: use last position only
                    logits_for_loss = logits[:, -1, :]  # [batch, vocab_size]
                    targets_flat = targets.view(-1)  # [batch]
                else:
                    # Sequence-to-sequence task: flatten all positions
                    logits_for_loss = logits.view(-1, vocab_size)
                    targets_flat = targets.view(-1)

                loss = self.loss_fn(logits_for_loss, targets_flat)
                preds = logits_for_loss.argmax(dim=-1)
                acc = (preds == targets_flat).float().mean()

                total_loss += loss.item()
                total_acc += acc.item()
                n_batches += 1

        self.model.train()

        return {
            'loss': total_loss / n_batches,
            'acc': total_acc / n_batches,
        }

    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'task_name': self.task_name,
                'model_config': {
                    'd_model': self.model.d_model,
                    'n_layers': self.model.n_layers,
                    'n_heads': self.model.n_heads,
                }
            }
        }

        path = self.checkpoint_dir / f'{self.task_name}_step{step}.pt'
        torch.save(checkpoint, path)

        # Also save as "latest"
        latest_path = self.checkpoint_dir / f'{self.task_name}_latest.pt'
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['step']

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / f'{self.task_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_history(self):
        """Load training history from JSON."""
        history_path = self.checkpoint_dir / f'{self.task_name}_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.history = json.load(f)
        return self.history


def create_trainer(
    model: UnifiedTransformer,
    train_dataset: BaseDataset,
    val_dataset: Optional[BaseDataset],
    task_name: str,
    device: str = 'cuda',
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    checkpoint_dir: str = 'checkpoints',
    **kwargs
) -> Trainer:
    """
    Factory function to create trainer with standard defaults.

    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        task_name: Name of task for logging/checkpointing
        device: Device to train on
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        batch_size: Batch size
        checkpoint_dir: Directory for checkpoints
        **kwargs: Additional trainer arguments

    Returns:
        Trainer instance
    """
    from datasets import create_dataloader

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=checkpoint_dir,
        task_name=task_name,
        **kwargs
    )

    return trainer


class Evaluator:
    """
    Unified evaluator for trained models.

    Provides:
    - Accuracy metrics
    - Loss computation
    - Prediction collection
    - Error analysis
    """

    def __init__(
        self,
        model: UnifiedTransformer,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def compute_metrics(
        self,
        dataset: BaseDataset,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive metrics on dataset.

        Returns:
            Dictionary with metrics and predictions
        """
        from datasets import create_dataloader

        loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        all_preds = []
        all_targets = []
        all_losses = []

        for batch in loader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

            logits = self.model(inputs)

            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)

            loss = nn.CrossEntropyLoss(reduction='none')(logits_flat, targets_flat)
            preds = logits_flat.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_targets.append(targets_flat.cpu())
            all_losses.append(loss.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_losses = torch.cat(all_losses)

        # Compute metrics
        accuracy = (all_preds == all_targets).float().mean().item()
        mean_loss = all_losses.mean().item()

        return {
            'accuracy': accuracy,
            'loss': mean_loss,
            'predictions': all_preds,
            'targets': all_targets,
            'losses': all_losses,
        }

    @torch.no_grad()
    def get_predictions(
        self,
        dataset: BaseDataset,
        batch_size: int = 32,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Get model predictions with optional embeddings.

        Args:
            dataset: Dataset to evaluate
            batch_size: Batch size
            return_embeddings: Whether to return cached embeddings

        Returns:
            Dictionary with predictions, targets, and optionally embeddings
        """
        from datasets import create_dataloader

        loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        all_preds = []
        all_targets = []
        all_embeddings = [] if return_embeddings else None

        for batch in loader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

            logits = self.model(inputs, return_cache=return_embeddings)

            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

            if return_embeddings:
                all_embeddings.append(self.model.get_embeddings().cpu())

        result = {
            'predictions': torch.cat(all_preds),
            'targets': torch.cat(all_targets),
        }

        if return_embeddings:
            result['embeddings'] = torch.cat(all_embeddings)

        return result
