import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import get_window
from librosa.util import pad_center


class TorchSTFT(nn.Module):
    def __init__(
        self,
        filter_length: int = 800,
        hop_length: int = 200,
        win_length: int = 800,
        window: str = "hann",
    ):
        super().__init__()
        self.filter_length = int(filter_length)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)

        win = get_window(window, self.win_length, fftbins=True).astype(np.float32)
        self.register_buffer("window", torch.from_numpy(win), persistent=False)

    def transform(self, input_data: torch.Tensor):
        spec = torch.stft(
            input_data,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=input_data.device, dtype=input_data.dtype),
            center=True,
            return_complex=True,
        )
        return torch.abs(spec), torch.angle(spec)

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor):
        spec = torch.polar(magnitude, phase)
        wav = torch.istft(
            spec,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=magnitude.device, dtype=magnitude.dtype),
            center=True,
            return_complex=False,
        )
        return wav.unsqueeze(1)  # [B, 1, T]

    def forward(self, input_data: torch.Tensor):
        mag, phase = self.transform(input_data)
        return self.inverse(mag, phase)


class OnnxSTFT(nn.Module):
    def __init__(
        self,
        filter_length: int = 800,
        hop_length: int = 200,
        win_length: int = 800,
        window: str = "hann",
    ):
        super().__init__()
        self.filter_length = int(filter_length)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)

        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int(self.filter_length // 2 + 1)

        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, size=self.filter_length).astype(np.float32)
        fft_window = torch.from_numpy(fft_window)

        forward_basis *= fft_window
        inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis, persistent=False)
        self.register_buffer("inverse_basis", inverse_basis, persistent=False)

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor):
        recombined = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )
        x = F.conv_transpose1d(
            recombined,
            self.inverse_basis.to(device=magnitude.device, dtype=magnitude.dtype),
            stride=self.hop_length,
            padding=0,
        )
        x = x[:, :, self.filter_length // 2 :]
        x = x[:, :, : -self.filter_length // 2]
        return x  # [B, 1, T]