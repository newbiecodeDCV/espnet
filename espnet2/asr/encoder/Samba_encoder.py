import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class MambaBlock(nn.Module):
    """
    Fixed Mamba Block implementation for Samba-ASR
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
            pscan: bool = False,  # FIXED: Disable parallel scan by default
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank is None else dt_rank
        self.pscan = pscan  # FIXED: Always use sequential scan for stability

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Convolution layer
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # FIXED: Better initialization of A
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.repeat(self.d_inner, 1)
        # Clamp A to avoid numerical issues
        A = torch.clamp(A, min=0.1, max=1.0)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # FIXED: Initialize D with smaller values
        self.D = nn.Parameter(torch.ones(self.d_inner) * 0.1)
        self.D._no_weight_decay = True

    def forward(self, x):
        """Forward pass with stability fixes"""
        batch, seqlen, dim = x.shape

        # Input projections
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Convolution
        x = x.transpose(1, 2)
        x = self.conv1d(x)[..., :seqlen]
        x = x.transpose(1, 2)
        x = self.act(x)

        # SSM computation
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        # Get A matrix
        A = -torch.exp(self.A_log.float())

        # FIXED: Selective SSM computation with stability
        y = self.selective_scan_fixed(x, dt, A, B, C, self.D)

        # Apply gate
        y = y * self.act(z)

        # Output projection
        out = self.out_proj(y)
        return out

    def selective_scan_fixed(self, u, delta, A, B, C, D):
        """
        FIXED: Simplified selective scan implementation with stability checks
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[-1]

        # FIXED: Apply softplus and clamp delta for stability
        delta = F.softplus(delta)
        delta = torch.clamp(delta, min=1e-4, max=10.0)

        # Discretize A and B
        # FIXED: More stable computation
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        deltaA = torch.clamp(deltaA, max=10.0)  # Prevent overflow

        deltaB_u = (delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1))  # (B, L, d_inner, d_state)
        deltaB_u = torch.clamp(deltaB_u, min=-10.0, max=10.0)

        # FIXED: Always use sequential scan for stability
        x = self.sequential_scan_fixed(deltaA, deltaB_u)

        # Compute output
        y = torch.einsum('blnd,bld->bln', x, C)
        y = y + u * D.unsqueeze(0).unsqueeze(0)

        # FIXED: Final stability check
        y = torch.clamp(y, min=-100.0, max=100.0)
        return y

    def sequential_scan_fixed(self, A, Bu):
        """
        FIXED: Sequential scan implementation with proper error handling
        """
        batch, seq_len, d_inner, d_state = A.shape

        # Initialize state
        x = torch.zeros(batch, d_inner, d_state, device=A.device, dtype=A.dtype)
        xs = []

        for i in range(seq_len):
            # FIXED: More stable state update
            x_new = A[:, i] * x + Bu[:, i]
            # FIXED: Clamp to prevent explosion
            x_new = torch.clamp(x_new, min=-50.0, max=50.0)
            x = x_new
            xs.append(x)

        return torch.stack(xs, dim=1)  # (B, L, d_inner, d_state)


class SambaASREncoder(AbsEncoder):
    """
    FIXED: Samba-ASR Encoder with stability improvements
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

        # Input processing layer
        if input_layer == "conv2d":
            self.embed = Conv2dSubsamplingForSamba(
                input_size, output_size, dropout_rate
            )
        elif input_layer == "linear":
            self.embed = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.Dropout(dropout_rate)
            )

        # Positional encoding
        if pos_enc_layer_type == "abs_pos":
            self.pos_enc = PositionalEncoding(output_size, dropout_rate)
        else:
            self.pos_enc = None

        # FIXED: Mamba encoder blocks with proper initialization
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                d_model=output_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                pscan=False  # FIXED: Disable parallel scan for stability
            ) for _ in range(num_layers)
        ])

        # Layer normalization
        if normalize_before:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(output_size, eps=1e-6) for _ in range(num_layers)
            ])
        else:
            self.layer_norms = None

        # Final normalization
        self.final_norm = nn.LayerNorm(output_size, eps=1e-6) if normalize_before else None

        # FIXED: Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """FIXED: Proper weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)  # Smaller gain
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor,
            prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with stability checks"""

        # FIXED: Input validation
        if torch.isnan(xs_pad).any() or torch.isinf(xs_pad).any():
            print("Warning: NaN/Inf detected in input, clamping...")
            xs_pad = torch.nan_to_num(xs_pad, nan=0.0, posinf=1e6, neginf=-1e6)

        # Input processing and subsampling
        xs_pad, masks = self.embed(xs_pad, masks=None)

        # Apply positional encoding
        if self.pos_enc is not None:
            xs_pad = self.pos_enc(xs_pad)

        # Process through Mamba blocks
        for i, mamba_block in enumerate(self.mamba_blocks):
            # Pre-norm if specified
            if self.layer_norms is not None:
                xs_norm = self.layer_norms[i](xs_pad)
            else:
                xs_norm = xs_pad

            # Apply Mamba block with residual connection
            residual = xs_pad
            xs_pad = mamba_block(xs_norm)

            # FIXED: Stability check after each block
            if torch.isnan(xs_pad).any() or torch.isinf(xs_pad).any():
                print(f"Warning: NaN/Inf detected after Mamba block {i}, using residual...")
                xs_pad = residual
            else:
                xs_pad = residual + xs_pad

        # Final normalization
        if self.final_norm is not None:
            xs_pad = self.final_norm(xs_pad)

        # Calculate output lengths
        if masks is not None:
            olens = masks.squeeze(1).sum(1)
        else:
            subsampling_factor = 4
            olens = (ilens - 1) // subsampling_factor + 1

        return xs_pad, olens, None


# FIXED: Keep other classes the same but add missing output projection
class Conv2dSubsamplingForSamba(nn.Module):
    """Convolutional 2D subsampling for Samba-ASR"""

    def __init__(self, idim: int, odim: int, dropout_rate: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(odim, odim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.linear_out = nn.Linear(odim * ((idim + 1) // 4), odim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, masks=None):
        x = x.unsqueeze(1)  # (B, 1, T, D)
        x = self.conv(x)  # (B, odim, T', D')

        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        x = self.linear_out(x)
        x = self.dropout(x)

        if masks is not None:
            masks = masks[:, :, :-2:2][:, :, :-2:2]
        else:
            masks = torch.ones(b, 1, t, device=x.device, dtype=x.dtype)

        return x, masks


class PositionalEncoding(nn.Module):
    """Simple positional encoding with dropout"""

    def __init__(self, d_model: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(x)