import math
import torch
from torch import nn

from . import commons, modules, attentions
from .models import (
    Generator,
    PosteriorEncoder,
    DurationPredictor,
    StochasticDurationPredictor,
    ResidualCouplingBlock,
    TextEncoder as TextEncoderVits1,
)

from .attentions_vits2 import Encoder as EncoderSpkCond
from .attentions_vits2 import FFT as FFTBlock


class DurationPredictorVits2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
        noise_channels: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.noise_channels = noise_channels

        self.drop = nn.Dropout(p_dropout)

        self.noise_proj = nn.Conv1d(noise_channels, in_channels, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.conv_1 = nn.Conv1d(
            in_channels,
            filter_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.norm_1 = modules.LayerNorm(filter_channels)

        self.conv_2 = nn.Conv1d(
            filter_channels,
            filter_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.norm_2 = modules.LayerNorm(filter_channels)

        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(
        self,
        x,
        x_mask,
        g=None,
        z=None,
        noise_scale: float = 1.0,
    ):
        """
        x: [B, C, T]
        x_mask: [B, 1, T]
        g: [B, gin_channels, 1] or None
        z: optional external noise [B, noise_channels, T]
        returns: log-duration prediction [B, 1, T]
        """
        x = torch.detach(x)

        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)

        if z is None:
            z = torch.randn(
                x.size(0),
                self.noise_channels,
                x.size(2),
                device=x.device,
                dtype=x.dtype,
            )

        z = z * float(noise_scale)
        x = x + self.noise_proj(z) * x_mask

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)

        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoderSpkConditioned(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int,
        cond_layer_idx: int = 2,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = EncoderSpkCond(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=gin_channels,
            cond_layer_idx=cond_layer_idx,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).type_as(x)

        x = self.encoder(x * x_mask, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingTransformersLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0.0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        super().__init__()
        assert channels % 2 == 0
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre_transformer = attentions.Encoder(
            self.half_channels,
            self.half_channels,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
            window_size=None,
        )

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - int(mean_only)), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        if not reverse:
            x0_ = self.pre_transformer(x0 * x_mask, x_mask)
            x0_ = x0_ + x0  # residual VITS2
            h = self.pre(x0_) * x_mask
            h = self.enc(h, x_mask, g=g)

            stats = self.post(h) * x_mask
            if not self.mean_only:
                m, logs = torch.split(stats, [self.half_channels] * 2, 1)
            else:
                m = stats
                logs = torch.zeros_like(m)

            x1 = m + x1 * torch.exp(logs) * x_mask
            x_out = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs * x_mask, [1, 2])
            return x_out, logdet

        # reverse
        x0_ = self.pre_transformer(x0 * x_mask, x_mask)
        x0_ = x0_ + x0
        h = self.pre(x0_) * x_mask
        h = self.enc(h, x_mask, g=g)

        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1], 1)


class ResidualCouplingTransformersLayer2(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0.0,
        gin_channels: int = 0,
        mean_only: bool = True,
    ):
        super().__init__()
        assert channels % 2 == 0
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.pre_transformer = attentions.Encoder(
            hidden_channels,
            hidden_channels,
            n_heads=2,
            n_layers=1,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            # window_size=None,
        )
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - int(mean_only)), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        if not reverse:
            h = self.pre(x0) * x_mask
            h = h + self.pre_transformer(h * x_mask, x_mask)  # residual sobre h
            h = self.enc(h, x_mask, g=g)

            stats = self.post(h) * x_mask
            if not self.mean_only:
                m, logs = torch.split(stats, [self.half_channels] * 2, 1)
            else:
                m = stats
                logs = torch.zeros_like(m)

            x1 = m + x1 * torch.exp(logs) * x_mask
            x_out = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs * x_mask, [1, 2])
            return x_out, logdet

        # reverse
        h = self.pre(x0) * x_mask
        h = h + self.pre_transformer(h * x_mask, x_mask)
        h = self.enc(h, x_mask, g=g)

        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1], 1)


class FFTransformerCouplingLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        n_layers: int,
        n_heads: int,
        p_dropout: float = 0.0,
        filter_channels: int = 768,
        mean_only: bool = False,
        gin_channels: int = 0,
    ):
        super().__init__()
        assert channels % 2 == 0
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = FFTBlock(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            isflow=True,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - int(mean_only)), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        if not reverse:
            h = self.pre(x0) * x_mask
            h_ = self.enc(h, x_mask, g=g)
            h = h + h_

            stats = self.post(h) * x_mask
            if not self.mean_only:
                m, logs = torch.split(stats, [self.half_channels] * 2, 1)
            else:
                m = stats
                logs = torch.zeros_like(m)

            x1 = m + x1 * torch.exp(logs) * x_mask
            x_out = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs * x_mask, [1, 2])
            return x_out, logdet

        # reverse
        h = self.pre(x0) * x_mask
        h_ = self.enc(h, x_mask, g=g)
        h = h + h_

        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1], 1)


class MonoTransformerFlowLayer(nn.Module):
    def __init__(self, channels: int, mean_only: bool = False, residual_connection: bool = False):
        super().__init__()
        assert channels % 2 == 0
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.residual_connection = residual_connection

        self.pre_transformer = attentions.Encoder(
            self.half_channels,
            self.half_channels,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
            window_size=None,
        )
        self.post = nn.Conv1d(self.half_channels, self.half_channels * (2 - int(mean_only)), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        if self.residual_connection:
            if not reverse:
                x0_ = self.pre_transformer(x0, x_mask)
                stats = self.post(x0_) * x_mask
                if not self.mean_only:
                    m, logs = torch.split(stats, [self.half_channels] * 2, 1)
                else:
                    m = stats
                    logs = torch.zeros_like(m)

                x1_ = m + x1 * torch.exp(logs) * x_mask
                x_ = torch.cat([x0, x1_], 1)
                x_out = x + x_
                logdet = torch.sum(torch.log(torch.exp(logs) + 1.0), [1, 2])
                logdet = logdet + math.log(2.0) * (x0.shape[1] * x0.shape[2])
                return x_out, logdet

            # reverse (residual_connection=True)
            x0 = x0 / 2.0
            x0_ = self.pre_transformer(x0, x_mask)
            stats = self.post(x0_) * x_mask
            if not self.mean_only:
                m, logs = torch.split(stats, [self.half_channels] * 2, 1)
            else:
                m = stats
                logs = torch.zeros_like(m)

            x1_ = ((x1 - m) / (1.0 + torch.exp(-logs))) * x_mask
            return torch.cat([x0, x1_], 1)

        # Original (non-residual) mono transformer flow path
        if not reverse:
            x0_ = self.pre_transformer(x0 * x_mask, x_mask)
            h = x0_ + x0
            stats = self.post(h) * x_mask
            if not self.mean_only:
                m, logs = torch.split(stats, [self.half_channels] * 2, 1)
            else:
                m = stats
                logs = torch.zeros_like(m)

            x1_ = m + x1 * torch.exp(logs) * x_mask
            x_ = torch.cat([x0, x1_], 1)
            logdet = torch.sum(logs * x_mask, [1, 2])
            return x_, logdet

        # reverse (non-residual)
        x0_ = self.pre_transformer(x0 * x_mask, x_mask)
        h = x0_ + x0
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        x1_ = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1_], 1)


class ResidualCouplingTransformersBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
        use_transformer_flows: bool = False,
        transformer_flow_type: str = "mono_layer_post_residual",
    ):
        super().__init__()
        self.flows = nn.ModuleList()

        if not use_transformer_flows:
            for _ in range(n_flows):
                self.flows.append(
                    modules.ResidualCouplingLayer(
                        channels,
                        hidden_channels,
                        kernel_size,
                        dilation_rate,
                        n_layers,
                        gin_channels=gin_channels,
                        mean_only=True,
                    )
                )
                self.flows.append(modules.Flip())
            return

        t = str(transformer_flow_type)

        if t == "pre_conv":
            for _ in range(n_flows):
                self.flows.append(
                    ResidualCouplingTransformersLayer(
                        channels,
                        hidden_channels,
                        kernel_size,
                        dilation_rate,
                        n_layers,
                        gin_channels=gin_channels,
                        mean_only=True,
                    )
                )
                self.flows.append(modules.Flip())

        elif t == "pre_conv2":
            for _ in range(n_flows):
                self.flows.append(
                    ResidualCouplingTransformersLayer2(
                        channels,
                        hidden_channels,
                        kernel_size,
                        dilation_rate,
                        n_layers,
                        gin_channels=gin_channels,
                        mean_only=True,
                    )
                )
                self.flows.append(modules.Flip())

        elif t == "fft":
            for _ in range(n_flows):
                self.flows.append(
                    FFTransformerCouplingLayer(
                        channels,
                        hidden_channels,
                        kernel_size,
                        n_layers=2,
                        n_heads=2,
                        p_dropout=0.1,
                        filter_channels=768,
                        mean_only=True,
                        gin_channels=gin_channels,
                    )
                )
                self.flows.append(modules.Flip())

        elif t == "mono_layer_inter_residual":
            for _ in range(n_flows):
                self.flows.append(
                    modules.ResidualCouplingLayer(
                        channels,
                        hidden_channels,
                        kernel_size,
                        dilation_rate,
                        n_layers,
                        gin_channels=gin_channels,
                        mean_only=True,
                    )
                )
                self.flows.append(modules.Flip())
                self.flows.append(MonoTransformerFlowLayer(channels, mean_only=True, residual_connection=False))

        elif t == "mono_layer_post_residual":
            for _ in range(n_flows):
                self.flows.append(
                    modules.ResidualCouplingLayer(
                        channels,
                        hidden_channels,
                        kernel_size,
                        dilation_rate,
                        n_layers,
                        gin_channels=gin_channels,
                        mean_only=True,
                    )
                )
                self.flows.append(modules.Flip())
                self.flows.append(MonoTransformerFlowLayer(channels, mean_only=True, residual_connection=True))

        else:
            raise ValueError(
                "Invalid transformer_flow_type. Use one of: "
                "pre_conv, pre_conv2, fft, mono_layer_inter_residual, mono_layer_post_residual"
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for f in self.flows:
                r = f(x, x_mask, g=g, reverse=False)
                x = r[0] if isinstance(r, tuple) else r
            return x

        for f in reversed(self.flows):
            r = f(x, x_mask, g=g, reverse=True)
            x = r[0] if isinstance(r, tuple) else r
        return x


class SynthesizerTrnVits2(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel: int,
        upsample_kernel_sizes,
        n_speakers: int = 1,
        gin_channels: int = 0,
        use_sdp: bool = True,
        # VITS2 flags
        vits2_use_spk_conditioned_encoder: bool = False,
        vits2_cond_layer_idx: int = 2,
        vits2_use_transformer_flows: bool = False,
        vits2_transformer_flow_type: str = "mono_layer_post_residual",
        vits2_use_noise_scaled_mas: bool = False,
        vits2_mas_noise_scale_initial: float = 0.01,
        vits2_noise_scale_delta: float = 2e-6,
        vits2_use_dp: bool = False,
        vits2_dp_noise_channels: int = 1,
        vits2_dp_train_noise_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp

        # VITS2 features
        self.vits2_use_spk_conditioned_encoder = bool(vits2_use_spk_conditioned_encoder)
        self.vits2_cond_layer_idx = int(vits2_cond_layer_idx)
        self.vits2_use_transformer_flows = bool(vits2_use_transformer_flows)
        self.vits2_transformer_flow_type = str(vits2_transformer_flow_type)

        self.vits2_use_noise_scaled_mas = bool(vits2_use_noise_scaled_mas)
        self.mas_noise_scale_initial = float(vits2_mas_noise_scale_initial)
        self.noise_scale_delta = float(vits2_noise_scale_delta)
        self.current_mas_noise_scale = self.mas_noise_scale_initial
        self.use_noise_scaled_mas = self.vits2_use_noise_scaled_mas

        self.vits2_use_dp = bool(vits2_use_dp)
        self.vits2_dp_train_noise_scale = float(vits2_dp_train_noise_scale)

        if self.n_speakers > 1:
            self.emb_g = nn.Embedding(self.n_speakers, gin_channels)

        if self.vits2_use_spk_conditioned_encoder and (gin_channels > 0) and (self.n_speakers > 1):
            self.enc_p = TextEncoderSpkConditioned(
                n_vocab, inter_channels, hidden_channels, filter_channels,
                n_heads, n_layers, kernel_size, p_dropout,
                gin_channels=gin_channels,
                cond_layer_idx=self.vits2_cond_layer_idx,
            )
        else:
            self.enc_p = TextEncoderVits1(
                n_vocab, inter_channels, hidden_channels, filter_channels,
                n_heads, n_layers, kernel_size, p_dropout
            )

        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels
        )

        if self.vits2_use_transformer_flows:
            self.flow = ResidualCouplingTransformersBlock(
                inter_channels, hidden_channels, 5, 1, 4,
                gin_channels=gin_channels,
                use_transformer_flows=True,
                transformer_flow_type=self.vits2_transformer_flow_type,
            )
        else:
            self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        if use_sdp:
            self.dp = StochasticDurationPredictor(
                hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
            )
        else:
            if self.vits2_use_dp:
                self.dp = DurationPredictorVits2(
                    hidden_channels,
                    256,
                    3,
                    0.5,
                    gin_channels=gin_channels,
                    noise_channels=vits2_dp_noise_channels,
                )
            else:
                self.dp = DurationPredictor(
                    hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
                )

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        from . import monotonic_align

        if self.n_speakers > 1:
            g = self.emb_g(sid).unsqueeze(-1)
        else:
            g = None

        hidden_x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            if self.use_noise_scaled_mas:
                eps = torch.std(neg_cent) * torch.randn_like(neg_cent) * float(self.current_mas_noise_scale)
                neg_cent = neg_cent + eps

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        w = attn.sum(2)  # [b, 1, t_x]

        logw_ = torch.log(w + 1e-6) * x_mask

        if self.use_sdp:
            l_length = self.dp(hidden_x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
            logw = self.dp(hidden_x, x_mask, g=g, reverse=True, noise_scale=1.0)
        else:
            if self.vits2_use_dp:
                logw = self.dp(
                    hidden_x,
                    x_mask,
                    g=g,
                    noise_scale=self.vits2_dp_train_noise_scale,
                )
            else:
                logw = self.dp(hidden_x, x_mask, g=g)

            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (hidden_x, logw, logw_),
        )

    def infer(self, x, x_lengths, sid=None, noise_scale=0.667, length_scale=1, noise_scale_w=0.8, max_len=None):
        if self.n_speakers > 1:
            assert sid is not None, "Missing speaker id"
            g = self.emb_g(sid).unsqueeze(-1)
        else:
            g = None

        x_enc, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)

        if self.use_sdp:
            logw = self.dp(x_enc, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            if self.vits2_use_dp:
                logw = self.dp(
                    x_enc,
                    x_mask,
                    g=g,
                    noise_scale=noise_scale_w,
                )
            else:
                logw = self.dp(x_enc, x_mask, g=g)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_lengths.max()), 1).type_as(x_mask)

        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)
