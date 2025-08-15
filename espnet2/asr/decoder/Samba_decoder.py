import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
import math


class MambaCrossBlock(nn.Module):
    """
    Mamba block with cross-connection mechanism for decoder
    Enables conditioning on encoder features
    """

    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            cross_attn: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.cross_attn = cross_attn

        # Self Mamba block for text sequence
        self.self_mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # Cross-attention to encoder features
        if cross_attn:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.cross_norm = nn.LayerNorm(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x, encoder_output=None, encoder_mask=None, causal_mask=None):
        """
        Forward pass with cross-attention to encoder

        Args:
            x: decoder input (B, L, D)
            encoder_output: encoder output (B, T, D)
            encoder_mask: encoder attention mask
            causal_mask: causal mask for decoder self-attention
        """
        # Self Mamba processing
        residual = x
        x = self.norm1(x)
        x = self.self_mamba(x)
        x = residual + x

        # Cross-attention to encoder if available
        if self.cross_attn and encoder_output is not None:
            residual = x
            x = self.cross_norm(x)

            # Create proper key padding mask for cross-attention
            key_padding_mask = None
            if encoder_mask is not None:
                key_padding_mask = ~encoder_mask.squeeze(1).bool()

            attn_out, _ = self.cross_attention(
                query=x,
                key=encoder_output,
                value=encoder_output,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
            x = residual + attn_out

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class SambaASRDecoder(AbsDecoder):
    """
    Samba-ASR Decoder with Mamba blocks and cross-attention
    Based on the decoder architecture described in paper 2501.02832
    """

    def __init__(
            self,
            vocab_size: int,
            encoder_output_size: int,
            attention_heads: int = 8,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            self_attention_dropout_rate: float = 0.0,
            src_attention_dropout_rate: float = 0.0,
            input_layer: str = "embed",
            use_output_layer: bool = True,
            pos_enc_layer_type: str = "abs_pos",
            normalize_before: bool = True,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.normalize_before = normalize_before
        self.d_model = encoder_output_size

        # Input embedding layer
        if input_layer == "embed":
            self.embed = nn.Sequential(
                nn.Embedding(vocab_size, self.d_model),
                PositionalEncoding(self.d_model, positional_dropout_rate),
            )
        else:
            self.embed = None

        # Mamba decoder blocks with cross-attention
        self.decoders = nn.ModuleList([
            MambaCrossBlock(
                d_model=self.d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                cross_attn=True
            ) for _ in range(num_blocks)
        ])

        # Output layer
        if use_output_layer:
            self.output_layer = nn.Linear(self.d_model, vocab_size)
        else:
            self.output_layer = None

        # Final layer norm
        if normalize_before:
            self.after_norm = nn.LayerNorm(self.d_model)
        else:
            self.after_norm = None

    def forward(
            self,
            hs_pad: torch.Tensor,
            hlens: torch.Tensor,
            ys_in_pad: torch.Tensor,
            ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward decoder pass

        Args:
            hs_pad: encoder output (B, T, D)
            hlens: encoder output lengths (B,)
            ys_in_pad: decoder input tokens (B, L)
            ys_in_lens: decoder input lengths (B,)

        Returns:
            pred_pad: prediction logits (B, L, vocab_size)
            pred_lens: prediction lengths (B,)
        """
        batch_size, max_len = ys_in_pad.size()


        # CRITICAL FIX: Ensure hlens is integer tensor
        if isinstance(hlens, int):
            # If hlens is int, create tensor with correct batch_size
            hlens = torch.full((batch_size,), hlens,
                               dtype=torch.long, device=hs_pad.device)
        elif torch.is_tensor(hlens):
            # Convert float tensor to long tensor - THIS IS THE KEY FIX
            hlens = hlens.long()

            # Ensure correct shape
            if hlens.dim() == 0:
                hlens = hlens.unsqueeze(0).repeat(batch_size)
            elif hlens.size(0) != batch_size:
                # If size doesn't match, repeat first value
                hlens = hlens[0].unsqueeze(0).repeat(batch_size)
        else:
            # If list or other iterable
            hlens = torch.tensor(hlens, dtype=torch.long, device=hs_pad.device)
            if hlens.size(0) != batch_size:
                hlens = hlens[0].unsqueeze(0).repeat(batch_size)

        # Double check: ensure hlens is long tensor
        if hlens.dtype != torch.long:
            hlens = hlens.long()



        # Create encoder attention mask - MANUAL CREATION to avoid make_pad_mask issues
        device = hlens.device
        max_len = hs_pad.size(1)
        batch_size = hlens.size(0)

        # Create padding mask: True for padded positions, False for valid positions
        encoder_mask = torch.arange(max_len, device=device).expand(
            batch_size, max_len
        ) >= hlens.unsqueeze(1)
        encoder_mask = encoder_mask.unsqueeze(1)  # Add attention head dimension



        # Create causal mask for decoder
        causal_mask = self.create_causal_mask(max_len, ys_in_pad.device)

        # Embed input tokens
        if self.embed is not None:
            x = self.embed(ys_in_pad)  # (B, L, D)
        else:
            x = ys_in_pad

        # Process through Mamba decoder blocks
        for decoder in self.decoders:
            x = decoder(
                x=x,
                encoder_output=hs_pad,
                encoder_mask=encoder_mask,
                causal_mask=causal_mask
            )

        # Final normalization
        if self.after_norm is not None:
            x = self.after_norm(x)

        # Output projection
        if self.output_layer is not None:
            x = self.output_layer(x)  # (B, L, vocab_size)

        return x, ys_in_lens

    def create_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.bool()

    def forward_one_step(
            self,
            hs_pad: torch.Tensor,
            hlens: torch.Tensor,
            ys: torch.Tensor,
            cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward one step for autoregressive decoding

        Args:
            hs_pad: encoder output (B, T, D)
            hlens: encoder lengths (B,)
            ys: current token sequence (B, L)
            cache: previous states cache

        Returns:
            logits: next token logits (B, vocab_size)
            new_cache: updated cache
        """
        batch_size = ys.size(0)

        # CRITICAL FIX: Ensure hlens is properly formatted
        if isinstance(hlens, int):
            hlens = torch.full((batch_size,), hlens,
                               dtype=torch.long, device=ys.device)
        elif hlens is None:
            hlens = torch.full((batch_size,), hs_pad.size(1),
                               dtype=torch.long, device=ys.device)
        elif torch.is_tensor(hlens):
            # Convert to long tensor - KEY FIX
            hlens = hlens.long()
            if hlens.dim() == 0:
                hlens = hlens.unsqueeze(0).repeat(batch_size)
            elif hlens.size(0) != batch_size:
                hlens = hlens[0].unsqueeze(0).repeat(batch_size)

        # Create ys_in_lens with correct format
        ys_in_lens = torch.full((batch_size,), ys.size(1),
                                dtype=torch.long, device=ys.device)

        logits, _ = self.forward(hs_pad, hlens, ys, ys_in_lens)

        # Return logits for last position
        return logits[:, -1, :], cache

    def score(self, ys, state, x):
        """Compatibility method for beam search"""
        logits, _ = self.forward_one_step(x, None, ys, state)
        return logits, state


# Import MambaBlock from encoder file
from ..encoder.Samba_encoder import MambaBlock  # Adjust import path as needed


class PositionalEncoding(nn.Module):
    """Positional encoding for decoder embeddings"""

    def __init__(self, d_model: int, dropout_rate: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to embeddings"""
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)
