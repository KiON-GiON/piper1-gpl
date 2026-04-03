import torch
from torch import nn
from . import modules


class DurationDiscriminatorMelo(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)

        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)

        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)

        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)

        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.output_layer = nn.Sequential(
            nn.Linear(filter_channels, 1),
            nn.Sigmoid(),
        )

    def forward_probability(self, x, x_mask, dur, g=None):
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)

        x = self.pre_out_conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_1(x)
        x = self.drop(x)

        x = self.pre_out_conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_2(x)
        x = self.drop(x)

        x = (x * x_mask).transpose(1, 2)  # [b, t, c]
        return self.output_layer(x)       # [b, t, 1]

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)

        if g is not None and self.gin_channels != 0:
            g = torch.detach(g)
            x = x + self.cond(g)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)

        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        out_r = self.forward_probability(x, x_mask, dur_r, g=g)
        out_hat = self.forward_probability(x, x_mask, dur_hat, g=g)

        return [out_r], [out_hat]


def build_duration_discriminator(
    kind: str,
    hidden_channels: int,
    gin_channels: int = 0,
) -> nn.Module:
    kind = str(kind)

    if kind in ("dur_disc_2", "melo", "melo_dur_disc"):
        return DurationDiscriminatorMelo(
            hidden_channels,
            hidden_channels,
            3,
            0.1,
            gin_channels=gin_channels,
        )

    raise ValueError(
        "duration_discriminator_type must be one of: 'dur_disc_2', 'melo', 'melo_dur_disc'"
    )