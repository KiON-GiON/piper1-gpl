import math
import operator
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from functools import reduce

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
        sample_rate: int = 22050,
        use_explicit_pitch: bool = False,
        periodicity_use_uv: bool = True,
        periodicity_use_noise: bool = False,
        periodicity_noise_std: float = 0.003,
        periodicity_factor: int = 1,
    ):
        super().__init__()
        self.gen_istft_n_fft = int(gen_istft_n_fft)
        self.gen_istft_hop_size = int(gen_istft_hop_size)
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.sample_rate = int(sample_rate)
        self.upsample_rates = list(upsample_rates)
        self.frame_to_backbone_scale = reduce(operator.mul, self.upsample_rates, 1)
        self.use_explicit_pitch = bool(use_explicit_pitch)

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

        self.stage_channels = [
            upsample_initial_channel // (2 ** (i + 1))
            for i in range(len(self.ups))
        ]

        self.post_channels = ch

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
        else:
            self.cond = None

        self.ups.apply(init_weights)
        self._build_stft(is_onnx)

        self.periodicity = None
        if self.use_explicit_pitch:
            effective_sr = float(self.sample_rate) / float(
                self.gen_istft_hop_size * int(periodicity_factor)
            )
            self.periodicity = PeriodicityBranch(
                stage_channels=self.stage_channels,
                upsample_rates=self.upsample_rates,
                frame_to_backbone_scale=self.frame_to_backbone_scale,
                effective_sample_rate=effective_sr,
                use_uv=periodicity_use_uv,
                use_noise=periodicity_use_noise,
                noise_std=periodicity_noise_std,
            )

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

    def _forward_backbone(self, x, g=None, pitch_cond=None):
        x = self.conv_pre(x)
        if (g is not None) and (self.cond is not None):
            x = x + self.cond(g)

        period_feats = None
        if (self.periodicity is not None) and (pitch_cond is not None):
            logf0 = pitch_cond.get("logf0", None)
            uv = pitch_cond.get("uv", None)
            if logf0 is not None:
                period_feats = self.periodicity(logf0, uv)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            x = self.ups[i](x)

            if period_feats is not None and period_feats[i] is not None:
                p = period_feats[i]
                if p.size(-1) != x.size(-1):
                    p = F.interpolate(p, size=x.size(-1), mode="nearest")
                x = x + p

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
        if self.periodicity is not None:
            self.periodicity.remove_weight_norm()


class ISTFTGenerator(_ISTFTDecoderBase):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("periodicity_factor", 1)
        super().__init__(*args, **kwargs)

        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.conv_post = weight_norm(
            Conv1d(self.post_channels, self.gen_istft_n_fft + 2, 7, 1, padding=3)
        )
        self.conv_post.apply(init_weights)

    def forward(self, x, g=None, pitch_cond=None):
        x = self._forward_backbone(x, g=g, pitch_cond=pitch_cond)
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
        kwargs.setdefault("periodicity_factor", int(subbands))
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

    def forward(self, x, g=None, pitch_cond=None):
        x = self._forward_backbone(x, g=g, pitch_cond=pitch_cond)
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
        kwargs.setdefault("periodicity_factor", int(subbands))
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

    def forward(self, x, g=None, pitch_cond=None):
        x = self._forward_backbone(x, g=g, pitch_cond=pitch_cond)
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


class PeriodicityBranch(nn.Module):
    LRELU_SLOPE = 0.1

    def __init__(
        self,
        stage_channels,
        upsample_rates,
        frame_to_backbone_scale: int,
        effective_sample_rate: float,
        use_uv: bool = True,
        use_noise: bool = False,
        noise_std: float = 0.003,
    ):
        super().__init__()
        self.stage_channels = list(stage_channels)
        self.upsample_rates = list(upsample_rates)
        self.frame_to_backbone_scale = int(frame_to_backbone_scale)
        self.effective_sample_rate = float(effective_sample_rate)
        self.use_uv = bool(use_uv)
        self.use_noise = bool(use_noise)
        self.noise_std = float(noise_std)

        in_channels = 1 + int(self.use_uv) + int(self.use_noise)

        self.pre = weight_norm(
            Conv1d(in_channels, self.stage_channels[-1], 7, 1, padding=3)
        )

        self.downs = nn.ModuleList()
        cur_ch = self.stage_channels[-1]

        for stride, out_ch in zip(
            reversed(self.upsample_rates[1:]),
            reversed(self.stage_channels[:-1]),
        ):
            k = 2 * stride
            self.downs.append(
                weight_norm(
                    Conv1d(
                        cur_ch,
                        out_ch,
                        kernel_size=k,
                        stride=stride,
                        padding=(k - stride) // 2,
                    )
                )
            )
            cur_ch = out_ch

    def _build_source(self, logf0, uv):
        target_len = int(logf0.size(-1) * self.frame_to_backbone_scale)

        logf0_up = F.interpolate(logf0, size=target_len, mode="nearest")

        if uv is None:
            uv_up = torch.ones_like(logf0_up)
        else:
            uv_up = F.interpolate(uv, size=target_len, mode="nearest")
            uv_up = uv_up.clamp(0.0, 1.0)

        voiced = (uv_up > 0.5).to(dtype=logf0_up.dtype)

        max_f0 = max(self.effective_sample_rate * 0.5 - 1.0, 1.0)
        f0_hz = torch.where(
            voiced > 0.0,
            torch.exp(logf0_up).clamp(min=0.0, max=max_f0),
            torch.zeros_like(logf0_up),
        )

        rad = (2.0 * math.pi / self.effective_sample_rate) * f0_hz
        phase = torch.cumsum(rad, dim=-1)
        sine = torch.sin(phase) * uv_up

        feats = [sine]
        if self.use_uv:
            feats.append(uv_up)

        if self.use_noise:
            if self.training and (not torch.onnx.is_in_onnx_export()):
                noise = torch.randn_like(sine) * self.noise_std
            else:
                # ONNX-friendly
                noise = torch.zeros_like(sine)
            feats.append(noise)

        return torch.cat(feats, dim=1)

    @staticmethod
    def _match_time(x, target_t: int):
        if x.size(-1) == target_t:
            return x
        return F.interpolate(x, size=target_t, mode="nearest")

    def forward(self, logf0, uv=None):
        src = self._build_source(logf0, uv)
        cur = self.pre(src)

        feats = [None] * len(self.stage_channels)
        feats[-1] = cur

        for idx, down in enumerate(self.downs):
            cur = F.leaky_relu(cur, self.LRELU_SLOPE)
            cur = down(cur)
            stage_idx = len(self.stage_channels) - 2 - idx
            feats[stage_idx] = cur

        return feats

    def remove_weight_norm(self):
        _safe_remove_weight_norm(self.pre)
        for l in self.downs:
            _safe_remove_weight_norm(l)
