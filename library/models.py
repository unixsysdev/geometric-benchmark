"""
Unified Model Architectures for Geometric Benchmark

All tasks use standardized models to ensure fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (SINPE) for periodic tasks.

    Encodes position x as: [sin(2πx/p), cos(2πx/p), sin(4πx/p), cos(4πx/p), ...]

    Args:
        max_val: Maximum value (e.g., modulus p for mod-p tasks)
        d_model: Embedding dimension
        n_freqs: Number of frequency pairs
    """

    def __init__(self, max_val: int, d_model: int, n_freqs: int = 8):
        super().__init__()
        self.max_val = max_val
        self.d_model = d_model
        self.n_freqs = n_freqs

        # Create frequency matrix: [1, 2, 4, 8, ...]
        freqs = 2.0 ** torch.arange(n_freqs, dtype=torch.float32)
        self.register_buffer('freqs', freqs)

        # Project to d_model dimensions if needed
        self.input_dim = 2 * n_freqs  # sin + cos for each frequency
        if self.input_dim != d_model:
            self.projection = nn.Linear(self.input_dim, d_model)
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., seq_len] token IDs (positions)

        Returns:
            [..., seq_len, d_model] positional encodings
        """
        # Normalize to [0, 1]
        x_norm = x.float() / self.max_val

        # Compute sinusoidal features
        angles = x_norm.unsqueeze(-1) * self.freqs * 2 * torch.pi  # [..., seq_len, n_freqs]
        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)

        # Interleave sin and cos
        features = torch.stack([sin_features, cos_features], dim=-1)  # [..., seq_len, n_freqs, 2]
        features = features.reshape(*features.shape[:-2], -1)  # [..., seq_len, 2*n_freqs]

        return self.projection(features)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings.

    Args:
        max_val: Maximum value (vocabulary size)
        d_model: Embedding dimension
    """

    def __init__(self, max_val: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_val + 1, d_model)  # +1 for padding
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., seq_len] token IDs

        Returns:
            [..., seq_len, d_model] embeddings
        """
        return self.embedding(x.clamp(0, self.max_val)) * (self.d_model ** 0.5)


class AttentionHead(nn.Module):
    """Single attention head with attention pattern caching."""

    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_head = d_head
        self.q_proj = nn.Linear(d_model, d_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_head, bias=False)
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        self.out_proj = nn.Linear(d_head, d_model, bias=False)

        # Cache for attention pattern analysis
        self.last_attention_pattern = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            (output, attention_weights)
        """
        batch, seq_len, _ = x.shape

        Q = self.q_proj(x)  # [batch, seq_len, d_head]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        # scores: [batch, seq_len, seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        self.last_attention_pattern = attn.detach()  # Cache for analysis

        out = torch.matmul(attn, V)  # [batch, seq_len, d_head]
        out = self.out_proj(out)

        return out, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention with head-wise outputs."""

    def __init__(self, d_model: int, n_heads: int, d_head: Optional[int] = None):
        super().__init__()
        d_head = d_head or d_model // n_heads
        assert d_head * n_heads == d_model, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_head
        self.heads = nn.ModuleList([AttentionHead(d_model, d_head) for _ in range(n_heads)])

        # Cache all attention patterns
        self.last_attention_patterns = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            [batch, seq_len, d_model]
        """
        head_outputs = []
        attention_patterns = []

        for head in self.heads:
            out, attn = head(x, mask)
            head_outputs.append(out)
            attention_patterns.append(attn)

        # Stack head outputs
        out = torch.stack(head_outputs, dim=0).sum(dim=0)  # [batch, seq_len, d_model]

        # Cache attention patterns [n_heads, batch, seq_len, seq_len]
        self.last_attention_patterns = torch.stack(attention_patterns, dim=0)

        return out


class TransformerBlock(nn.Module):
    """Transformer block with causal attention support."""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            [batch, seq_len, d_model]
        """
        # Self-attention with residual
        attn_out = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)

        # MLP with residual
        mlp_out = self.mlp(self.ln2(x))
        x = x + self.dropout(mlp_out)

        return x


class UnifiedTransformer(nn.Module):
    """
    Unified transformer architecture for all benchmark tasks.

    Supports:
    - Different positional encodings (SINPE, learned)
    - Variable depth/width
    - Classification or sequence-to-sequence tasks
    - Causal or bidirectional attention

    Args:
        vocab_size: Size of input vocabulary
        output_size: Size of output vocabulary (for classification, often vocab_size)
        d_model: Model dimension
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads
        d_mlp: MLP hidden dimension
        pos_enc_type: 'sinusoidal' or 'learned'
        pos_enc_max_val: Max value for positional encoding
        dropout: Dropout rate
        causal: Use causal masking (for autoregressive tasks)
    """

    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        d_mlp: int = 512,
        pos_enc_type: Literal['sinusoidal', 'learned'] = 'learned',
        pos_enc_max_val: Optional[int] = None,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.causal = causal

        # Input embedding (token IDs)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        pos_enc_max_val = pos_enc_max_val or vocab_size
        if pos_enc_type == 'sinusoidal':
            self.pos_encoding = SinusoidalPositionalEncoding(pos_enc_max_val, d_model)
        else:
            self.pos_encoding = LearnedPositionalEncoding(pos_enc_max_val, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, output_size)

        # Cache for analysis
        self.embeddings = None
        self.hidden_states = None

    def forward(self, x: torch.Tensor, return_cache: bool = False) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len] input token IDs
            return_cache: Whether to cache embeddings and hidden states for analysis

        Returns:
            [batch, seq_len, output_size] logits
        """
        batch, seq_len = x.shape

        # Embeddings
        tok_emb = self.token_embedding(x) * (self.d_model ** 0.5)  # [batch, seq_len, d_model]
        pos_emb = self.pos_encoding(x)  # [batch, seq_len, d_model]

        hidden = tok_emb + pos_emb

        # Cache initial embeddings
        if return_cache:
            self.embeddings = hidden.detach()
            self.hidden_states = []

        # Causal mask if needed
        mask = None
        if self.causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # Transformer blocks
        for block in self.blocks:
            hidden = block(hidden, mask)
            if return_cache:
                self.hidden_states.append(hidden.detach())

        # Output projection
        logits = self.output_proj(hidden)  # [batch, seq_len, output_size]

        return logits

    def get_attention_patterns(self, layer: int, head: int) -> torch.Tensor:
        """Get cached attention pattern for specific layer/head."""
        return self.blocks[layer].attn.heads[head].last_attention_pattern

    def get_all_attention_patterns(self) -> torch.Tensor:
        """Get all cached attention patterns [n_layers, n_heads, batch, seq_len, seq_len]."""
        patterns = []
        for block in self.blocks:
            patterns.append(block.attn.last_attention_patterns)
        return torch.stack(patterns, dim=0)

    def get_embeddings(self) -> torch.Tensor:
        """Get cached input embeddings."""
        return self.embeddings

    def get_hidden_states(self) -> list:
        """Get cached hidden states per layer."""
        return self.hidden_states


def create_model(config: dict) -> UnifiedTransformer:
    """
    Factory function to create model from config.

    Args:
        config: Dictionary with model parameters

    Returns:
        UnifiedTransformer instance
    """
    return UnifiedTransformer(
        vocab_size=config['vocab_size'],
        output_size=config['output_size'],
        d_model=config.get('d_model', 128),
        n_layers=config.get('n_layers', 2),
        n_heads=config.get('n_heads', 4),
        d_mlp=config.get('d_mlp', 512),
        pos_enc_type=config.get('pos_enc_type', 'learned'),
        pos_enc_max_val=config.get('pos_enc_max_val', None),
        dropout=config.get('dropout', 0.0),
        causal=config.get('causal', False),
    )


# Convenience functions for common architectures

def tiny_transformer(vocab_size: int, output_size: int, **kwargs) -> UnifiedTransformer:
    """Tiny model: 2 layers, 128 dim, 4 heads."""
    return create_model({
        'vocab_size': vocab_size,
        'output_size': output_size,
        'd_model': 128,
        'n_layers': 2,
        'n_heads': 4,
        'd_mlp': 512,
        **kwargs
    })


def small_transformer(vocab_size: int, output_size: int, **kwargs) -> UnifiedTransformer:
    """Small model: 4 layers, 256 dim, 8 heads."""
    return create_model({
        'vocab_size': vocab_size,
        'output_size': output_size,
        'd_model': 256,
        'n_layers': 4,
        'n_heads': 8,
        'd_mlp': 1024,
        **kwargs
    })


def medium_transformer(vocab_size: int, output_size: int, **kwargs) -> UnifiedTransformer:
    """Medium model: 6 layers, 512 dim, 8 heads."""
    return create_model({
        'vocab_size': vocab_size,
        'output_size': output_size,
        'd_model': 512,
        'n_layers': 6,
        'n_heads': 8,
        'd_mlp': 2048,
        **kwargs
    })
