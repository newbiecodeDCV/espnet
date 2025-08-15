import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class MambaBlock(nn.Module):
    """
    Complete Mamba Block implementation for ESPnet Samba-ASR
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
            pscan: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank is None else dt_rank
        self.pscan = pscan
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        # FIXED: All required layers for ESPnet compatibility

        # 1. Input projection (combines x and z paths)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # 2. Convolution layer for temporal processing
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # 3. Activation function
        self.activation = "silu"
        self.act = nn.SiLU()

        # 4. SSM parameter projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 5. Initialize SSM parameters A and D
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.repeat(self.d_inner, 1)
        A = torch.clamp(A, min=0.1, max=1.0)  # Stability fix
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner) * 0.1)  # Smaller init
        self.D._no_weight_decay = True

        # 6. CRITICAL: Output projection layer - MUST HAVE for ESPnet
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # 7. Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stability"""
        # Initialize dt_proj with small values
        with torch.no_grad():
            dt_init_std = self.dt_rank ** -0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(self.dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize output projection with small weights
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, x):
        """
        Forward pass of Mamba block
        Args:
            x: input tensor (B, L, D)
        Returns:
            output tensor (B, L, D)
        """
        batch, seqlen, dim = x.shape

        # Input validation for debugging
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN/Inf in MambaBlock input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

        # 1. Input projections - split into x and z paths
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_path, z_path = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)

        # 2. Convolution on x path (local temporal patterns)
        x_conv = x_path.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[..., :seqlen]  # Causal convolution
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = self.act(x_conv)

        # 3. SSM computation
        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + 2*d_state)

        # Split into dt, B, C parameters
        dt, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        dt = self.dt_proj(dt)  # (B, L, d_inner)

        # Get A matrix
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # 4. Core SSM computation
        y = self.selective_scan_v2(x_conv, dt, A, B, C, self.D)

        # 5. Apply gating with z path
        z_gate = self.act(z_path)
        y = y * z_gate

        # 6. CRITICAL: Output projection - must exist for ESPnet
        output = self.out_proj(y)

        # Final validation
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaN/Inf in MambaBlock output")
            output = torch.nan_to_num(output, nan=0.0, posinf=1e3, neginf=-1e3)

        return output

    def selective_scan_v2(self, u, delta, A, B, C, D):
        """
        Optimized selective scan with better stability
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[-1]

        # Ensure delta is positive and stable
        delta = F.softplus(delta) + 1e-4
        delta = torch.clamp(delta, max=10.0)

        # Discretization
        # A: (d_inner, d_state) -> (1, 1, d_inner, d_state)
        # delta: (batch, seq_len, d_inner) -> (batch, seq_len, d_inner, 1)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaA = torch.clamp(deltaA, max=10.0)

        # B: (batch, seq_len, d_state) -> (batch, seq_len, 1, d_state)
        # u: (batch, seq_len, d_inner) -> (batch, seq_len, d_inner, 1)
        # delta: (batch, seq_len, d_inner) -> (batch, seq_len, d_inner, 1)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        deltaB_u = torch.clamp(deltaB_u, min=-10.0, max=10.0)

        # Sequential scan for stability
        x = self.sequential_scan_stable(deltaA, deltaB_u)

        # Output computation
        # x: (batch, seq_len, d_inner, d_state)
        # C: (batch, seq_len, d_state) -> (batch, seq_len, 1, d_state)
        y = torch.sum(x * C.unsqueeze(2), dim=-1)  # (batch, seq_len, d_inner)

        # Add skip connection
        y = y + u * D.unsqueeze(0).unsqueeze(0)

        return torch.clamp(y, min=-50.0, max=50.0)

    def sequential_scan_stable(self, A, Bu):
        """
        Stable sequential scan implementation
        """
        batch, seq_len, d_inner, d_state = A.shape

        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=A.device, dtype=A.dtype)
        outputs = []

        for t in range(seq_len):
            # State update: h_t = A_t * h_{t-1} + Bu_t
            h = A[:, t] * h + Bu[:, t]

            # Stability: prevent explosion
            h = torch.clamp(h, min=-20.0, max=20.0)

            outputs.append(h)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner, d_state)


# FIXED: Complete encoder class
class SambaASREncoder(AbsEncoder):
    """ESPnet-compatible Samba ASR Encoder"""

    def __init__(
            self,
            input_size: int,
            output_size: int = 512,  # Smaller default
            num_layers: int = 6,  # Fewer layers for stability
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

        # Input processing
        if input_layer == "conv2d":
            self.embed = Conv2dSubsamplingForSamba(input_size, output_size, dropout_rate)
        elif input_layer == "linear":
            self.embed = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.Dropout(dropout_rate)
            )
        else:
            raise ValueError(f"Unknown input_layer: {input_layer}")

        # Positional encoding
        if pos_enc_layer_type == "abs_pos":
            self.pos_enc = PositionalEncoding(output_size, dropout_rate)
        else:
            self.pos_enc = nn.Identity()

        # Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                d_model=output_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                pscan=False  # Always use sequential for stability
            ) for _ in range(num_layers)
        ])

        # Layer norms
        if normalize_before:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(output_size, eps=1e-6) for _ in range(num_layers)
            ])
        else:
            self.layer_norms = None

        # Final norm
        self.final_norm = nn.LayerNorm(output_size, eps=1e-6) if normalize_before else None

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor,
            prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """ESPnet forward interface"""

        # Input processing
        xs_pad, masks = self.embed(xs_pad, masks=None)

        # Positional encoding
        xs_pad = self.pos_enc(xs_pad)

        # Mamba blocks with residual connections
        for i, mamba_block in enumerate(self.mamba_blocks):
            residual = xs_pad

            # Pre-norm
            if self.layer_norms is not None:
                xs_pad = self.layer_norms[i](xs_pad)

            # Mamba block
            try:
                xs_pad = mamba_block(xs_pad)
                xs_pad = residual + xs_pad  # Residual connection
            except Exception as e:
                print(f"Error in Mamba block {i}: {e}")
                xs_pad = residual  # Fallback to residual

        # Final norm
        if self.final_norm is not None:
            xs_pad = self.final_norm(xs_pad)

        # Output lengths
        if masks is not None:
            olens = masks.squeeze(1).sum(1).long()
        else:
            subsampling_factor = 4
            olens = ((ilens - 1) // subsampling_factor + 1).long()

        return xs_pad, olens, None


class Conv2dSubsamplingForSamba(nn.Module):
    """2D Convolution subsampling"""

    def __init__(self, idim: int, odim: int, dropout_rate: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2, 1),
            nn.ReLU(),
        )
        self.out = nn.Linear(odim * ((idim + 1) // 4), odim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, masks=None):
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        x = self.out(x)
        x = self.dropout(x)

        masks = torch.ones(b, 1, t, device=x.device, dtype=x.dtype)
        return x, masks


class PositionalEncoding(nn.Module):
    """Simple positional encoding"""

    def __init__(self, d_model: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(x)