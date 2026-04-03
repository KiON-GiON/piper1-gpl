import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from . import modules
from .commons import init_weights, get_padding
from .pqmf import PQMF
from .stft import TorchSTFT, OnnxSTFT


def _safe_remove_weight_norm(m):
    try:
        remove_weight_norm(m)
    except Exception:
        pass


class _ISTFTDecoderBase(nn.Module):
    LRELU_SLOPE = 0.1

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel: int,
        upsample_kernel_sizes,
        gin_channels: int = 0,
        gen_istft_n_fft: int = 16,
        gen_istft_hop_size: int = 4,
        is_onnx: bool = False,
    ):
        super().__init__()
        self.gen_istft_n_fft = int(gen_istft_n_fft)
        self.gen_istft_hop_size = int(gen_istft_hop_size)
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        resblock_cls = modules.ResBlock1 if str(resblock) == "1" else modules.ResBlock2

        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock_cls(ch, k, d))

        self.post_channels = ch

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
        else:
            self.cond = None

        self.ups.apply(init_weights)
        self._build_stft(is_onnx)

    def _build_stft(self, is_onnx: bool):
        stft_cls = OnnxSTFT if is_onnx else TorchSTFT
        self.stft = stft_cls(
            filter_length=self.gen_istft_n_fft,
            hop_length=self.gen_istft_hop_size,
            win_length=self.gen_istft_n_fft,
        )

    def switch_to_onnx(self):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self._build_stft(is_onnx=True)
        self.stft = self.stft.to(device=device, dtype=dtype)

    def _forward_backbone(self, x, g=None):
        x = self.conv_pre(x)
        if (g is not None) and (self.cond is not None):
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                r = self.resblocks[i * self.num_kernels + j](x)
                xs = r if xs is None else (xs + r)

            x = xs / self.num_kernels

        x = F.leaky_relu(x, self.LRELU_SLOPE)
        return x

    def remove_weight_norm(self):
        _safe_remove_weight_norm(self.conv_pre)
        for l in self.ups:
            _safe_remove_weight_norm(l)
        for l in self.resblocks:
            if hasattr(l, "remove_weight_norm"):
                l.remove_weight_norm()


class ISTFTGenerator(_ISTFTDecoderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.conv_post = weight_norm(
            Conv1d(self.post_channels, self.gen_istft_n_fft + 2, 7, 1, padding=3)
        )
        self.conv_post.apply(init_weights)

    def forward(self, x, g=None):
        x = self._forward_backbone(x, g=g)
        x = self.reflection_pad(x)
        x = self.conv_post(x)

        n_bins = self.gen_istft_n_fft // 2 + 1
        spec = torch.exp(x[:, :n_bins, :])
        phase = math.pi * torch.sin(x[:, n_bins:, :])

        audio = self.stft.inverse(spec, phase)
        return audio, None

    def remove_weight_norm(self):
        super().remove_weight_norm()
        _safe_remove_weight_norm(self.conv_post)


class MultibandISTFTGenerator(_ISTFTDecoderBase):
    def __init__(self, *args, subbands: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.subbands = int(subbands)
        if self.subbands <= 1:
            raise ValueError("MultibandISTFTGenerator requires subbands > 1")

        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.subband_conv_post = weight_norm(
            Conv1d(
                self.post_channels,
                self.subbands * (self.gen_istft_n_fft + 2),
                7,
                1,
                padding=3,
            )
        )
        self.subband_conv_post.apply(init_weights)
        self.pqmf = PQMF(subbands=self.subbands)

    def forward(self, x, g=None):
        x = self._forward_backbone(x, g=g)
        x = self.reflection_pad(x)
        x = self.subband_conv_post(x)

        b, c, t = x.shape
        x = x.view(b, self.subbands, c // self.subbands, t)

        n_bins = self.gen_istft_n_fft // 2 + 1
        spec = torch.exp(x[:, :, :n_bins, :])
        phase = math.pi * torch.sin(x[:, :, n_bins:, :])

        y_mb = self.stft.inverse(
            spec.reshape(b * self.subbands, n_bins, t),
            phase.reshape(b * self.subbands, n_bins, t),
        ).squeeze(1)

        y_mb = y_mb.view(b, self.subbands, -1)
        audio = self.pqmf.synthesis(y_mb)

        return audio, {"subband_audio": y_mb}

    def remove_weight_norm(self):
        super().remove_weight_norm()
        _safe_remove_weight_norm(self.subband_conv_post)


class MultistreamISTFTGenerator(_ISTFTDecoderBase):
    def __init__(self, *args, subbands: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.subbands = int(subbands)
        if self.subbands <= 1:
            raise ValueError("MultistreamISTFTGenerator requires subbands > 1")

        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.subband_conv_post = weight_norm(
            Conv1d(
                self.post_channels,
                self.subbands * (self.gen_istft_n_fft + 2),
                7,
                1,
                padding=3,
            )
        )
        self.subband_conv_post.apply(init_weights)

        updown_filter = torch.zeros((self.subbands, self.subbands, self.subbands)).float()
        for k in range(self.subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter, persistent=False)

        self.multistream_conv_post = weight_norm(
            Conv1d(self.subbands, 1, kernel_size=63, bias=False, padding=get_padding(63, 1))
        )
        self.multistream_conv_post.apply(init_weights)

    def forward(self, x, g=None):
        x = self._forward_backbone(x, g=g)
        x = self.reflection_pad(x)
        x = self.subband_conv_post(x)

        b, c, t = x.shape
        x = x.view(b, self.subbands, c // self.subbands, t)

        n_bins = self.gen_istft_n_fft // 2 + 1
        spec = torch.exp(x[:, :, :n_bins, :])
        phase = math.pi * torch.sin(x[:, :, n_bins:, :])

        y_mb = self.stft.inverse(
            spec.reshape(b * self.subbands, n_bins, t),
            phase.reshape(b * self.subbands, n_bins, t),
        ).squeeze(1)

        y_mb = y_mb.view(b, self.subbands, -1)
        y_stream = F.conv_transpose1d(
            y_mb,
            self.updown_filter.to(device=y_mb.device, dtype=y_mb.dtype) * self.subbands,
            stride=self.subbands,
        )
        audio = self.multistream_conv_post(y_stream)

        return audio, {"stream_audio": y_stream}

    def remove_weight_norm(self):
        super().remove_weight_norm()
        _safe_remove_weight_norm(self.subband_conv_post)
        _safe_remove_weight_norm(self.multistream_conv_post)