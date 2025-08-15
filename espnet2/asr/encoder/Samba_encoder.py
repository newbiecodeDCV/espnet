import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class MambaBlock(nn.Module):
    """
    Mamba Block implementation for Samba-ASR
    Based on selective state-space models with input-dependent parameters
    """

    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            dt_rank: Optional[int] = None,
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init: str = "random",
            dt_scale: float = 1.0,
            bias: bool = False,
            conv_bias: bool = True,
            pscan: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank is None else dt_rank
        self.pscan = pscan

        # Input projection - combines x_proj and z_proj from paper
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Convolution layer for local temporal patterns
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Activation function (SiLU as mentioned in paper)
        self.activation = "silu"
        self.act = nn.SiLU()

        # SSM parameters - A, B, C, D, dt projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt projection
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x):
        """
        Forward pass of Mamba block
        Args:
            x: input tensor of shape (B, L, D)
        Returns:
            output tensor of shape (B, L, D)
        """
        batch, seqlen, dim = x.shape

        # Input projections
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # Split into x and z paths

        # Convolution (local temporal modeling)
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv1d(x)[..., :seqlen]  # Causal conv
        x = x.transpose(1, 2)  # (B, L, D)
        x = self.act(x)

        # SSM computation
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)

        # Split into dt, B, C
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)  # (B, L, d_inner)

        # Get A matrix
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Selective SSM computation
        y = self.selective_scan(x, dt, A, B, C, self.D)

        # Apply gate (z path)
        y = y * self.act(z)

        # Output projection
        out = self.out_proj(y)
        return out

    def selective_scan(self, u, delta, A, B, C, D):
        """
        Selective scan implementation - core of Mamba
        """
        batch, seq_len, d_inner = u.shape
        n = A.shape[-1]

        # Discretize A and B
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (B, L, d_inner, d_state)

        # Parallel scan or sequential computation
        if self.pscan and seq_len > 1:
            x = self.parallel_scan(deltaA, deltaB_u)
        else:
            x = self.sequential_scan(deltaA, deltaB_u)

        # Compute output
        y = torch.einsum('blnd,bld->bln', x, C)
        y = y + u * D

        return y

    def parallel_scan(self, A, Bu):
        """Parallel scan implementation for efficiency"""
        # Simplified parallel scan - in practice would use more optimized version
        return self.sequential_scan(A, Bu)

    def sequential_scan(self, A, Bu):
        """Sequential scan implementation"""
        batch, seq_len, d_inner, d_state = A.shape
        x = torch.zeros(batch, d_inner, d_state, device=A.device, dtype=A.dtype)
        xs = []

        for i in range(seq_len):
            x = A[:, i] * x + Bu[:, i]
            xs.append(x)

        return torch.stack(xs, dim=1)  # (B, L, d_inner, d_state)


class SambaASREncoder(AbsEncoder):
    """
    Samba-ASR Encoder implementing the architecture from paper 2501.02832
    Uses Mamba blocks for efficient speech sequence modeling
    """

    def __init__(
            self,
            input_size: int,
            output_size: int = 768,
            num_layers: int = 12,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            dropout_rate: float = 0.1,
            input_layer: str = "conv2d",
            pos_enc_layer_type: str = "abs_pos",
            normalize_before: bool = True,
            **kwargs
    ):
        super().__init__()

        self._output_size = output_size
        self.num_layers = num_layers
        self.normalize_before = normalize_before

        # Input processing layer (from paper: convolutional layers for local patterns)
        if input_layer == "conv2d":
            self.embed = Conv2dSubsamplingForSamba(
                input_size, output_size, dropout_rate
            )
        elif input_layer == "linear":
            self.embed = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.Dropout(dropout_rate)
            )

        # Positional encoding (optional, not explicitly mentioned in paper)
        if pos_enc_layer_type == "abs_pos":
            self.pos_enc = PositionalEncoding(output_size, dropout_rate)
        else:
            self.pos_enc = None

        # Mamba encoder blocks
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                d_model=output_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(num_layers)
        ])

        # Layer normalization layers
        if normalize_before:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(output_size) for _ in range(num_layers)
            ])
        else:
            self.layer_norms = None

        # Final normalization
        self.final_norm = nn.LayerNorm(output_size) if normalize_before else None

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor,
            prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Samba-ASR encoder

        Args:
            xs_pad: Input tensor (B, T, D) - Mel spectrograms
            ilens: Input lengths (B,)
            prev_states: Previous states (not used in current implementation)

        Returns:
            xs_pad: Output tensor (B, T', D')
            olens: Output lengths (B,)
            next_states: Next states (None for current implementation)
        """
        # Input processing and subsampling
        xs_pad, masks = self.embed(xs_pad, masks=None)

        # Apply positional encoding if available
        if self.pos_enc is not None:
            xs_pad = self.pos_enc(xs_pad)

        # Process through Mamba blocks
        for i, mamba_block in enumerate(self.mamba_blocks):
            # Pre-norm if specified
            if self.layer_norms is not None:
                xs_pad = self.layer_norms[i](xs_pad)

            # Apply Mamba block with residual connection
            residual = xs_pad
            xs_pad = mamba_block(xs_pad)
            xs_pad = residual + xs_pad

        # Final normalization
        if self.final_norm is not None:
            xs_pad = self.final_norm(xs_pad)

        # Calculate output lengths based on subsampling
        if masks is not None:
            olens = masks.squeeze(1).sum(1)
        else:
            # Estimate output lengths based on input lengths and subsampling factor
            subsampling_factor = 4  # Typical for conv2d subsampling
            olens = (ilens - 1) // subsampling_factor + 1

        return xs_pad, olens, None


class Conv2dSubsamplingForSamba(nn.Module):
    """
    Convolutional 2D subsampling for Samba-ASR
    Adapted for Mamba architecture requirements
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float):
        super().__init__()
        self.conv = nn.Sequential(
            # First conv layer
            nn.Conv2d(1, odim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Second conv layer
            nn.Conv2d(odim, odim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate output dimension after convolution
        self.linear_out = nn.Linear(odim * ((idim + 1) // 4), odim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, masks=None):
        """
        Args:
            x: Input tensor (B, T, D)
            masks: Input masks
        Returns:
            x: Output tensor (B, T', odim)
            masks: Output masks
        """
        # Add channel dimension
        x = x.unsqueeze(1)  # (B, 1, T, D)

        # Apply convolutions
        x = self.conv(x)  # (B, odim, T', D')

        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)

        # Linear projection
        x = self.linear_out(x)
        x = self.dropout(x)

        # Update masks
        if masks is not None:
            masks = masks[:, :, :-2:2][:, :, :-2:2]
        else:
            masks = torch.ones(b, 1, t, device=x.device, dtype=x.dtype)

        return x, masks


class PositionalEncoding(nn.Module):
    """Simple positional encoding"""

    def __init__(self, d_model: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # For simplicity, just apply dropout
        # In practice, you might want to add sinusoidal positional encodings
        return self.dropout(x)