import torch
from torch import nn
from . import modules

class DurationDiscriminatorV1(nn.Module):
    def __init__(self, in_channels: int, filter_channels: int, kernel_size: int, p_dropout: float, gin_channels: int = 0):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        # self.pre_out_norm_1 = modules.LayerNorm(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        # self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

    def forward_probability(self, x, x_mask, dur):
        # x: [b, c, t], dur: [b, 1, t], x_mask: [b, 1, t]
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)

        x = self.pre_out_conv_1(x * x_mask)
        # x = torch.relu(x)
        # x = self.pre_out_norm_1(x)

        x = self.pre_out_conv_2(x * x_mask)
        # x = torch.relu(x)
        # x = self.pre_out_norm_2(x)

        x = (x * x_mask).transpose(1, 2)  # [b, t, c]
        return self.output_layer(x)        # [b, t, 1]

    def forward(self, x, x_mask, dur_real, dur_fake):
        x = torch.detach(x)
        x = self.conv_1(x * x_mask)
        # x = torch.relu(x)
        x = self.conv_2(x * x_mask)
        # x = torch.relu(x)

        out_real = self.forward_probability(x, x_mask, dur_real)
        out_fake = self.forward_probability(x, x_mask, dur_fake)
        return [out_real], [out_fake]


class DurationDiscriminatorV2(nn.Module):
    def __init__(self, in_channels: int, filter_channels: int, kernel_size: int, p_dropout: float, gin_channels: int = 0):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = modules.LayerNorm(filter_channels)

        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

    def forward_probability(self, x, x_mask, dur):
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)

        x = self.pre_out_conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_1(x)

        x = self.pre_out_conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_2(x)

        x = (x * x_mask).transpose(1, 2)   # [b, t, c]
        return self.output_layer(x)         # [b, t, 1]

    def forward(self, x, x_mask, dur_real, dur_fake, g=None):
        x = torch.detach(x)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)

        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)

        out_real = self.forward_probability(x, x_mask, dur_real)
        out_fake = self.forward_probability(x, x_mask, dur_fake)
        return [out_real], [out_fake]


def build_duration_discriminator(kind: str, hidden_channels: int, gin_channels: int = 0) -> nn.Module:
    kind = str(kind)
    if kind == "dur_disc_1":
        return DurationDiscriminatorV1(hidden_channels, hidden_channels, 3, 0.1, gin_channels=gin_channels)
    if kind == "dur_disc_2":
        return DurationDiscriminatorV2(hidden_channels, hidden_channels, 3, 0.1, gin_channels=gin_channels)
    raise ValueError("duration_discriminator_type must be 'dur_disc_1' or 'dur_disc_2'")