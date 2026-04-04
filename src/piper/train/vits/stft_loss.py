import torch
from torch import nn
from torch.nn import functional as F


def stft(x, fft_size, hop_size, win_length, window):
    x = x.float()
    window = window.to(device=x.device, dtype=torch.float32)

    spec = torch.stft(
        x,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    return torch.clamp(torch.abs(spec), min=1e-7).transpose(2, 1)


class SpectralConvergenceLoss(nn.Module):
    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.clamp_min(
            torch.norm(y_mag, p="fro"), 1e-7
        )


class LogSTFTMagnitudeLoss(nn.Module):
    def forward(self, x_mag, y_mag):
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(nn.Module):
    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        super().__init__()
        self.fft_size = int(fft_size)
        self.shift_size = int(shift_size)
        self.win_length = int(win_length)
        self.register_buffer("window", getattr(torch, window)(self.win_length), persistent=False)
        self.sc_loss = SpectralConvergenceLoss()
        self.mag_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        return self.sc_loss(x_mag, y_mag), self.mag_loss(x_mag, y_mag)


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=(1024, 2048, 512),
        hop_sizes=(120, 240, 50),
        win_lengths=(600, 1200, 240),
        window="hann_window",
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList(
            [STFTLoss(fs, hs, wl, window) for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)]
        )

    def forward(self, x, y):
        sc, mag = 0.0, 0.0
        for f in self.stft_losses:
            sc_i, mag_i = f(x, y)
            sc = sc + sc_i
            mag = mag + mag_i
        sc = sc / len(self.stft_losses)
        mag = mag / len(self.stft_losses)
        return sc, mag