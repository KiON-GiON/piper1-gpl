import math
import typing
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from .modules import LayerNorm
from .commons import subsequent_mask, fused_add_tanh_sigmoid_multiply

from .attentions import MultiHeadAttention, FFN


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4,
        gin_channels: int = 0,
        cond_layer_idx: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.gin_channels = gin_channels
        self.cond_layer_idx = n_layers
        if gin_channels and gin_channels > 0:
            self.spk_emb_linear = nn.Linear(gin_channels, hidden_channels)
            self.cond_layer_idx = int(cond_layer_idx)
            assert 0 <= self.cond_layer_idx < n_layers

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, g=None):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask

        g_proj = None
        if (
            (g is not None)
            and (self.gin_channels is not None)
            and (self.gin_channels > 0)
            and (0 <= self.cond_layer_idx < self.n_layers)
        ):
            g_proj = self.spk_emb_linear(g.transpose(1, 2)).transpose(1, 2)

        for i in range(self.n_layers):
            if (i == self.cond_layer_idx) and (g_proj is not None):
                x = (x + g_proj) * x_mask

            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)

        return x * x_mask


class FFT(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int = 1,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        proximal_bias: bool = False,
        proximal_init: bool = True,
        isflow: bool = False,
        gin_channels: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.drop = nn.Dropout(p_dropout)

        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()

        self.gin_channels = gin_channels if (isflow and gin_channels > 0) else 0
        if self.gin_channels > 0:
            self.cond_pre = nn.Conv1d(hidden_channels, 2 * hidden_channels, 1)
            self.cond_layer = weight_norm(
                nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1),
                name="weight",
            )

        self.register_buffer(
            "_fused_n_channels", torch.IntTensor([hidden_channels]), persistent=False
        )

        for _ in range(n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, g=None):
        g_all = None
        if (g is not None) and (self.gin_channels > 0):
            g_all = self.cond_layer(g)

        self_attn_mask = subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
        x = x * x_mask

        n_channels = self._fused_n_channels
        if n_channels.device != x.device:
            n_channels = n_channels.to(device=x.device)

        for i in range(self.n_layers):
            if g_all is not None:
                x_in = self.cond_pre(x)
                off = i * 2 * self.hidden_channels
                g_l = g_all[:, off : off + 2 * self.hidden_channels, :]
                x = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels)

            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

        return x * x_mask
