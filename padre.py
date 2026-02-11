from typing import Optional

import torch
import torch.nn as nn


class MultiModalPADReProcessor:
    def __init__(self, degree=3, conv_kernel=3):
        super().__init__()
        self.degree = degree
        self.conv_kernel = conv_kernel

    def forward(
        self,
        attn: "Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        # 1. Standard HF Pre-processing (Now including temb)
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        # 2. Time-Embedding Projection
        # If temb is present, we project it to the channel dimension
        # This allows the polynomial to 'know' which timestep we are at.
        temb_offset = 0
        if temb is not None and hasattr(attn, "padre_temb_proj"):
            # temb_offset: (B, 1, D)
            temb_offset = attn.padre_temb_proj(temb).unsqueeze(1)

        # 3. Mode A (Image) Basis
        x_a_t = hidden_states.transpose(1, 2)
        y_a0 = attn.to_q(attn.padre_A_conv[0](x_a_t).transpose(1, 2))
        y_a1 = attn.to_k(attn.padre_A_conv[1](x_a_t).transpose(1, 2))

        # 4. Mode B (Text) Basis
        x_b = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        x_b_t = x_b.transpose(1, 2)
        y_b2 = attn.to_v(attn.padre_B_conv(x_b_t).transpose(1, 2))
        y_b2_global = y_b2.mean(dim=1, keepdim=True)

        # 5. Multi-Modal Recursion (a, a, b)
        # Apply time-awareness to the initial state
        z1 = y_a0 + temb_offset

        z1_mixed = attn.padre_C_conv[0](z1.transpose(1, 2)).transpose(1, 2)
        z1_mixed = attn.padre_D_linear[0](z1_mixed)
        z2 = z1_mixed * y_a1

        z2_mixed = attn.padre_C_conv[1](z2.transpose(1, 2)).transpose(1, 2)
        z2_mixed = attn.padre_D_linear[1](z2_mixed)
        z3 = z2_mixed * y_b2_global

        # 6. Final Summation
        hidden_states = (
            attn.padre_L
            + (z1 * attn.padre_W[0])
            + (z2 * attn.padre_W[1])
            + (z3 * attn.padre_W[2])
        )

        # 7. Final Projections
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        return hidden_states + residual if attn.residual_connection else hidden_states


def inject_multi_modal_padre(attn: nn.Module, temb_dim=1280, degree=3, conv_kernel=3):
    device, dtype = attn.to_q.weight.device, attn.to_q.weight.dtype
    dim = attn.inner_dim

    # Convolutions for Image Mode (A)
    attn.padre_A_conv = nn.ModuleList(
        [
            nn.Conv1d(dim, dim, conv_kernel, padding=conv_kernel // 2, groups=dim)
            for _ in range(2)
        ]
    ).to(device, dtype)

    # Convolution for Text Mode (B)
    attn.padre_B_conv = nn.Conv1d(
        dim, dim, conv_kernel, padding=conv_kernel // 2, groups=dim
    ).to(device, dtype)

    # Mixing layers
    attn.padre_C_conv = nn.ModuleList(
        [
            nn.Conv1d(dim, dim, conv_kernel, padding=conv_kernel // 2, groups=dim)
            for _ in range(degree - 1)
        ]
    ).to(device, dtype)

    attn.padre_D_linear = nn.ModuleList(
        [nn.Linear(dim, dim, bias=False) for _ in range(degree - 1)]
    ).to(device, dtype)

    attn.padre_W = nn.Parameter(torch.randn(degree, dim, device=device, dtype=dtype))
    attn.padre_L = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))

    # Add time embedding projection
    # This maps the global timestep embedding to the attention dimension
    attn.padre_temb_proj = nn.Sequential(
        nn.SiLU(), nn.Linear(temb_dim, attn.inner_dim)
    ).to(attn.to_q.weight.device, attn.to_q.weight.dtype)

    attn.set_processor(MultiModalPADReProcessor(degree=degree, conv_kernel=conv_kernel))
