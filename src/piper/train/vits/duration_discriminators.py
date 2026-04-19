import torch
from torch import nn
from . import modules
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)
        else:
            self.cond = None

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1))

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

        if (g is not None) and (self.cond is not None):
            g = torch.detach(g)
            x = x + self.cond(g)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)

        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)

        out_real = self.forward_probability(x, x_mask, dur_real)
        out_fake = self.forward_probability(x, x_mask, dur_fake)
        return [out_real], [out_fake]


class DurationDiscriminatorLSTM(nn.Module):
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

        self.lstm = nn.LSTM(
            input_size=2 * filter_channels,
            hidden_size=filter_channels,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)
        else:
            self.cond = None

        self.output_layer = nn.Linear(2 * filter_channels, 1)

    def _encode_x(self, x, x_mask, g=None):
        x = torch.detach(x)

        if (g is not None) and (self.cond is not None):
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

        return x * x_mask

    def forward_probability(self, x, x_mask, dur):
        # x: [B, C, T]
        # dur: [B, 1, T]
        dur = self.dur_proj(dur)
        h = torch.cat([x, dur], dim=1)  # [B, 2C, T]
        h = h.transpose(1, 2)           # [B, T, 2C]

        lengths = x_mask.squeeze(1).sum(dim=1).long()
        lengths = torch.clamp_min(lengths, 1).cpu()

        packed = pack_padded_sequence(
            h, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=h.size(1),
        )

        out = self.output_layer(h)  # [B, T, 1]
        return out

    def forward(self, x, x_mask, dur_real, dur_fake, g=None):
        x = self._encode_x(x, x_mask, g=g)
        out_real = self.forward_probability(x, x_mask, dur_real)
        out_fake = self.forward_probability(x, x_mask, dur_fake)
        return [out_real], [out_fake]


def build_duration_discriminator(kind: str, hidden_channels: int, gin_channels: int = 0) -> nn.Module:
    kind = str(kind)
    if kind == "dur_disc_2":
        return DurationDiscriminatorV2(hidden_channels, hidden_channels, 3, 0.1, gin_channels=gin_channels)
    if kind == "dur_disc_lstm":
        return DurationDiscriminatorLSTM(
            hidden_channels,
            hidden_channels,
            3,
            0.1,
            gin_channels=gin_channels,
        )
    raise ValueError("duration_discriminator_type must be 'dur_disc_2' or 'dur_disc_lstm'")