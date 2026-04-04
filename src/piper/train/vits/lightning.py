"""PyTorch Lightning module."""
import os
import numpy as np
from scipy.io.wavfile import write as write_wav

import ast
import logging
import operator
from functools import reduce
from dataclasses import dataclass
from typing import Optional, Any, Tuple

import lightning as L
import torch
from torch import autocast
from torch.nn import functional as F

from .commons import slice_segments
from .dataset import Batch
from .losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
    masked_discriminator_loss,
    masked_generator_loss,
)
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .models import MultiPeriodDiscriminator, SynthesizerTrn
from .models_vits2 import SynthesizerTrnVits2
from .duration_discriminators import build_duration_discriminator

from .decoder_factory import resolve_decoder_type, decoder_output_hop_length
from .pqmf import PQMF
from .stft_loss import MultiResolutionSTFTLoss

_LOGGER = logging.getLogger(__name__)


@dataclass
class _ForwardPack:
    y: torch.Tensor
    y_hat: torch.Tensor
    y_mel: torch.Tensor
    y_hat_mel: torch.Tensor
    x_mask: torch.Tensor
    z_mask: torch.Tensor
    l_length: torch.Tensor
    z_p: torch.Tensor
    m_p: torch.Tensor
    logs_p: torch.Tensor
    logs_q: torch.Tensor
    extra: Optional[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
        ]
    ] = None
    decoder_aux: Optional[Any] = None


class VitsModel(L.LightningModule):
    def __init__(
        self,
        batch_size: int = 32,
        sample_rate: int = 22050,
        num_symbols: int = 256,
        num_speakers: int = 1,
        # audio
        resblock="2",
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=(
            (1, 2),
            (2, 6),
            (3, 12),
        ),
        upsample_rates=(8, 8, 4),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16, 8),
        # mel
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_channels: int = 80,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
        # model
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        n_layers_q: int = 3,
        use_spectral_norm: bool = False,
        gin_channels: int = 0,
        use_sdp: bool = True,
        segment_size: int = 8192,
        # VITS2
        use_vits2: bool = False,
        vits2_use_spk_conditioned_encoder: bool = False,
        vits2_cond_layer_idx: int = 2,
        vits2_use_transformer_flows: bool = False,
        vits2_transformer_flow_type: str = "mono_layer_post_residual",
        vits2_use_noise_scaled_mas: bool = False,
        vits2_mas_noise_scale_initial: float = 0.01,
        vits2_noise_scale_delta: float = 2e-6,
        vits2_infer_sdp_ratio: float = 0.2,
        use_mel_posterior_encoder: bool = False,
        use_duration_discriminator: bool = False,
        duration_discriminator_type: str = "dur_disc_2",
        log_vits2_features: bool = True,
        # Decoder + subbands
        decoder_type: Optional[str] = None,
        istft_vits: bool = False,
        mb_istft_vits: bool = False,
        ms_istft_vits: bool = False,
        gen_istft_n_fft: int = 16,
        gen_istft_hop_size: int = 4,
        subbands: int = 4,
        use_subband_stft_loss: bool = False,
        subband_stft_loss_weight: float = 1.0,
        subband_stft_fft_sizes=(384, 683, 171),
        subband_stft_hop_sizes=(30, 60, 10),
        subband_stft_win_lengths=(150, 300, 60),
        # training
        learning_rate: float = 2e-4,
        learning_rate_d: float = 1e-4,
        betas: tuple[float, float] = (0.8, 0.99),
        betas_d: tuple[float, float] = (0.5, 0.9),
        eps: float = 1e-9,
        lr_decay: float = 0.999875,
        lr_decay_d: float = 0.9999,
        init_lr_ratio: float = 1.0,
        warmup_epochs: int = 0,
        c_mel: int = 45,
        c_kl: float = 1.0,
        grad_clip: Optional[float] = None,
        init_from_checkpoint: Optional[str] = None,
        # unused
        dataset: object = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if isinstance(self.hparams.resblock_kernel_sizes, str):
            self.hparams.resblock_kernel_sizes = ast.literal_eval(
                self.hparams.resblock_kernel_sizes
            )

        if isinstance(self.hparams.resblock_dilation_sizes, str):
            self.hparams.resblock_dilation_sizes = ast.literal_eval(
                self.hparams.resblock_dilation_sizes
            )

        if isinstance(self.hparams.upsample_rates, str):
            self.hparams.upsample_rates = ast.literal_eval(self.hparams.upsample_rates)

        if isinstance(self.hparams.upsample_kernel_sizes, str):
            self.hparams.upsample_kernel_sizes = ast.literal_eval(
                self.hparams.upsample_kernel_sizes
            )

        if isinstance(self.hparams.betas, str):
            self.hparams.betas = ast.literal_eval(self.hparams.betas)

        if isinstance(self.hparams.betas_d, str):
            self.hparams.betas_d = ast.literal_eval(self.hparams.betas_d)

        if isinstance(self.hparams.subband_stft_fft_sizes, str):
            self.hparams.subband_stft_fft_sizes = ast.literal_eval(self.hparams.subband_stft_fft_sizes)
        if isinstance(self.hparams.subband_stft_hop_sizes, str):
            self.hparams.subband_stft_hop_sizes = ast.literal_eval(self.hparams.subband_stft_hop_sizes)
        if isinstance(self.hparams.subband_stft_win_lengths, str):
            self.hparams.subband_stft_win_lengths = ast.literal_eval(self.hparams.subband_stft_win_lengths)

        if isinstance(self.hparams.subbands, bool):
            self.hparams.subbands = 1 if not self.hparams.subbands else 4

        effective_decoder_type = resolve_decoder_type(
            decoder_type=self.hparams.decoder_type,
            istft_vits=self.hparams.istft_vits,
            mb_istft_vits=self.hparams.mb_istft_vits,
            ms_istft_vits=self.hparams.ms_istft_vits,
        )

        expected_hop_length = decoder_output_hop_length(
            decoder_type=effective_decoder_type,
            upsample_rates=self.hparams.upsample_rates,
            gen_istft_hop_size=self.hparams.gen_istft_hop_size,
            subbands=self.hparams.subbands,
        )

        self._effective_decoder_type = effective_decoder_type

        self._uses_istft_decoder = self._effective_decoder_type in {
            "istft",
            "mb_istft",
            "ms_istft",
        }
        self._uses_multiband_decoder = self._effective_decoder_type == "mb_istft"
        self._uses_multistream_decoder = self._effective_decoder_type == "ms_istft"

        self._use_subband_loss = (
            self.hparams.use_vits2
            and self._uses_multiband_decoder
            and bool(self.hparams.use_subband_stft_loss)
        )

        self.hparams.resblock = str(self.hparams.resblock)

        # Need to use manual optimization because we have multiple optimizers
        self.automatic_optimization = False

        self.batch_size = batch_size
        self._mas_batch_step = 0  # batch-based counter, not optimizer-step-based

        if (self.hparams.num_speakers > 1) and (self.hparams.gin_channels <= 0):
            self.hparams.gin_channels = 512

        if self.hparams.use_duration_discriminator and not self.hparams.use_vits2:
            _LOGGER.warning(
                "use_duration_discriminator=True ignored because use_vits2=False"
            )
            self.hparams.use_duration_discriminator = False

        SynthClass = SynthesizerTrnVits2 if self.hparams.use_vits2 else SynthesizerTrn

        spec_channels = (
            self.hparams.mel_channels
            if self.hparams.use_mel_posterior_encoder
            else (self.hparams.filter_length // 2 + 1)
        )

        if self.hparams.log_vits2_features:
            _LOGGER.info("Generator: %s", SynthClass.__name__)
            _LOGGER.info(
                "use_mel_posterior_encoder=%s (spec_channels=%s)",
                self.hparams.use_mel_posterior_encoder,
                spec_channels,
            )
            _LOGGER.info(
                "decoder_type=%s, gen_istft_n_fft=%s, gen_istft_hop_size=%s, subbands=%s",
                effective_decoder_type,
                self.hparams.gen_istft_n_fft,
                self.hparams.gen_istft_hop_size,
                self.hparams.subbands,
            )
            if self.hparams.use_vits2:
                _LOGGER.info(
                    "VITS2 flags: spk_cond_enc=%s (idx=%s), transformer_flows=%s (type=%s), "
                    "noise_scaled_mas=%s (init=%s, delta=%s), dur_disc=%s (type=%s)",
                    self.hparams.vits2_use_spk_conditioned_encoder,
                    self.hparams.vits2_cond_layer_idx,
                    self.hparams.vits2_use_transformer_flows,
                    self.hparams.vits2_transformer_flow_type,
                    self.hparams.vits2_use_noise_scaled_mas,
                    self.hparams.vits2_mas_noise_scale_initial,
                    self.hparams.vits2_noise_scale_delta,
                    self.hparams.use_duration_discriminator,
                    self.hparams.duration_discriminator_type,
                )

        model_g_kwargs = dict(
            n_vocab=num_symbols,
            spec_channels=spec_channels,
            segment_size=self.hparams.segment_size // self.hparams.hop_length,
            inter_channels=self.hparams.inter_channels,
            hidden_channels=self.hparams.hidden_channels,
            filter_channels=self.hparams.filter_channels,
            n_heads=self.hparams.n_heads,
            n_layers=self.hparams.n_layers,
            kernel_size=self.hparams.kernel_size,
            p_dropout=self.hparams.p_dropout,
            resblock=self.hparams.resblock,
            resblock_kernel_sizes=self.hparams.resblock_kernel_sizes,
            resblock_dilation_sizes=self.hparams.resblock_dilation_sizes,
            upsample_rates=self.hparams.upsample_rates,
            upsample_initial_channel=self.hparams.upsample_initial_channel,
            upsample_kernel_sizes=self.hparams.upsample_kernel_sizes,
            n_speakers=self.hparams.num_speakers,
            gin_channels=self.hparams.gin_channels,
            use_sdp=self.hparams.use_sdp,
        )

        if self.hparams.use_vits2:
            model_g_kwargs.update(
                dict(
                    vits2_use_spk_conditioned_encoder=self.hparams.vits2_use_spk_conditioned_encoder,
                    vits2_cond_layer_idx=self.hparams.vits2_cond_layer_idx,
                    vits2_use_transformer_flows=self.hparams.vits2_use_transformer_flows,
                    vits2_transformer_flow_type=self.hparams.vits2_transformer_flow_type,
                    vits2_use_noise_scaled_mas=self.hparams.vits2_use_noise_scaled_mas,
                    vits2_mas_noise_scale_initial=self.hparams.vits2_mas_noise_scale_initial,
                    vits2_noise_scale_delta=self.hparams.vits2_noise_scale_delta,
                    vits2_infer_sdp_ratio=self.hparams.vits2_infer_sdp_ratio,
                    decoder_type=self.hparams.decoder_type,
                    istft_vits=self.hparams.istft_vits,
                    mb_istft_vits=self.hparams.mb_istft_vits,
                    ms_istft_vits=self.hparams.ms_istft_vits,
                    gen_istft_n_fft=self.hparams.gen_istft_n_fft,
                    gen_istft_hop_size=self.hparams.gen_istft_hop_size,
                    subbands=self.hparams.subbands,
                )
            )

        self.model_g = SynthClass(**model_g_kwargs)

        self.mb_pqmf = None
        self.subband_stft_loss = None

        if self._use_subband_loss:
            self.mb_pqmf = PQMF(subbands=int(self.hparams.subbands))
            self.subband_stft_loss = MultiResolutionSTFTLoss(
                fft_sizes=tuple(self.hparams.subband_stft_fft_sizes),
                hop_sizes=tuple(self.hparams.subband_stft_hop_sizes),
                win_lengths=tuple(self.hparams.subband_stft_win_lengths),
            )

        self.model_d = MultiPeriodDiscriminator(
            use_spectral_norm=self.hparams.use_spectral_norm
        )

        self.model_dur = None
        if self.hparams.use_duration_discriminator:
            dur_gin_channels = (
                self.hparams.gin_channels if self.hparams.num_speakers > 1 else 0
            )
            self.model_dur = build_duration_discriminator(
                self.hparams.duration_discriminator_type,
                hidden_channels=self.hparams.hidden_channels,
                gin_channels=dur_gin_channels,
            )

        if init_from_checkpoint:
            self._load_generator_weights(init_from_checkpoint)
            self.hparams.init_from_checkpoint = None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["mas_batch_step"] = int(self._mas_batch_step)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self._mas_batch_step = int(checkpoint.get("mas_batch_step", 0))

    def _load_generator_weights(self, ckpt_path: str):

        if not os.path.isfile(ckpt_path):
            _LOGGER.warning(
                "init_from_checkpoint: file not found (%s), weight loading is omitted",
                ckpt_path,
            )
            return

        _LOGGER.info("init_from_checkpoint: loading weights from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt

        model_state = self.model_g.state_dict()

        def _strip_known_prefixes(name: str) -> str:
            prefixes = ("model_g.", "module.", "net_g.")
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                        changed = True
            return name

        normalized_state = {}
        for full_name, param in state.items():
            name = _strip_known_prefixes(full_name)
            normalized_state[name] = param

        model_has_sdp = any(k.startswith("sdp.") for k in model_state.keys())
        ckpt_has_sdp = any(k.startswith("sdp.") for k in normalized_state.keys())

        ckpt_old_sdp_under_dp = (
            model_has_sdp
            and (not ckpt_has_sdp)
            and any(
                k.startswith("dp.log_flow")
                or k.startswith("dp.flows.")
                or k.startswith("dp.post_pre.")
                or k.startswith("dp.post_proj.")
                or k.startswith("dp.post_convs.")
                or k.startswith("dp.post_flows.")
                or k.startswith("dp.pre.")
                or k.startswith("dp.proj.")
                or k.startswith("dp.convs.")
                for k in normalized_state.keys()
            )
        )

        if ckpt_old_sdp_under_dp:
            _LOGGER.info(
                "init_from_checkpoint: detected legacy checkpoint with SDP stored under dp.*"
            )

        ckpt_has_speaker_emb = any("emb_g" in k for k in normalized_state.keys())
        model_is_multispeaker = self.hparams.num_speakers > 1

        transferred = []
        skipped_missing = []
        skipped_speaker_cond = []
        skipped_shape = []
        remapped_dp_to_sdp = 0

        speaker_cond_patterns = (
            "emb_g",
            "dec.cond",
            "dp.cond",
            "sdp.cond",
            "cond_layer",
        )

        for name, param in normalized_state.items():
            original_name = name

            if ckpt_old_sdp_under_dp and name.startswith("dp."):
                mapped_name = "sdp." + name[3:]
                if (mapped_name in model_state) and (model_state[mapped_name].shape == param.shape):
                    name = mapped_name
                    remapped_dp_to_sdp += 1

            if any(pattern in name for pattern in speaker_cond_patterns):
                if not ckpt_has_speaker_emb and model_is_multispeaker:
                    skipped_speaker_cond.append(name)
                    continue

                if name in model_state and model_state[name].shape != param.shape:
                    skipped_speaker_cond.append(name)
                    continue

            if name not in model_state:
                skipped_missing.append(original_name)
                continue

            if model_state[name].shape != param.shape:
                skipped_shape.append((name, param.shape, model_state[name].shape))
                continue

            transferred.append(name)
            model_state[name] = param

        self.model_g.load_state_dict(model_state, strict=False)

        _LOGGER.info("init_from_checkpoint: %d transferred parameters", len(transferred))

        if remapped_dp_to_sdp > 0:
            _LOGGER.info(
                "init_from_checkpoint: remapped %d legacy SDP parameters from dp.* to sdp.*",
                remapped_dp_to_sdp,
            )

        if skipped_speaker_cond:
            _LOGGER.info(
                "init_from_checkpoint: %d speaker conditioning parameters omitted "
                "(single→multi speaker or different number of speakers): %s",
                len(skipped_speaker_cond),
                skipped_speaker_cond[:20],
            )

        if skipped_missing:
            _LOGGER.debug(
                "init_from_checkpoint: %d parameters not found in destination model: %s",
                len(skipped_missing),
                skipped_missing[:20],
            )

        if skipped_shape:
            _LOGGER.info("init_from_checkpoint: parameters omitted due to incompatible shape:")
            for name, src_shape, dst_shape in skipped_shape[:20]:
                _LOGGER.info("  %s: origin=%s, destination=%s", name, src_shape, dst_shape)
            if len(skipped_shape) > 20:
                _LOGGER.info("  ... and %d more", len(skipped_shape) - 20)

    def forward(self, text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio, *_ = self.model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )
        return audio

    def _set_requires_grad(self, module: Optional[torch.nn.Module], flag: bool) -> None:
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = flag

    def _forward_g_and_prepare(self, batch: Batch) -> _ForwardPack:
        x, x_lengths, y, _, spec, spec_lengths, speaker_ids = (
            batch.phoneme_ids,
            batch.phoneme_lengths,
            batch.audios,
            batch.audio_lengths,
            batch.spectrograms,
            batch.spectrogram_lengths,
            batch.speaker_ids if batch.speaker_ids is not None else None,
        )

        spec_f = spec.float()

        if getattr(self.hparams, "use_mel_posterior_encoder", False):
            y_in = spec_to_mel_torch(
                spec_f,
                self.hparams.filter_length,
                self.hparams.mel_channels,
                self.hparams.sample_rate,
                self.hparams.mel_fmin,
                self.hparams.mel_fmax,
            )
            mel_gt = y_in
        else:
            y_in = spec
            mel_gt = spec_to_mel_torch(
                spec_f,
                self.hparams.filter_length,
                self.hparams.mel_channels,
                self.hparams.sample_rate,
                self.hparams.mel_fmin,
                self.hparams.mel_fmax,
            )

        out = self.model_g(x, x_lengths, y_in, spec_lengths, speaker_ids)

        extra = None
        decoder_aux = None
        if isinstance(out, (tuple, list)) and (len(out) == 9):
            (
                y_hat,
                l_length,
                _attn,
                ids_slice,
                x_mask,
                z_mask,
                (_z, z_p, m_p, logs_p, _m_q, logs_q),
                extra,
                decoder_aux,
            ) = out
        else:
            (
                y_hat,
                l_length,
                _attn,
                ids_slice,
                x_mask,
                z_mask,
                (_z, z_p, m_p, logs_p, _m_q, logs_q),
            ) = out

        seg_frames = self.hparams.segment_size // self.hparams.hop_length
        y_mel = slice_segments(mel_gt, ids_slice, seg_frames)
        y_slice = slice_segments(
            y, ids_slice * self.hparams.hop_length, self.hparams.segment_size
        )

        y_hat = y_hat[..., : y_slice.shape[-1]]

        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            self.hparams.filter_length,
            self.hparams.mel_channels,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.mel_fmin,
            self.hparams.mel_fmax,
        )

        if extra is not None:
            try:
                if len(extra) == 4:
                    hidden_x, logw, logw_, g = extra
                else:
                    hidden_x, logw, logw_ = extra
                    g = None
                extra = (hidden_x, logw, logw_, g)
            except Exception:
                extra = None

        return _ForwardPack(
            y=y_slice,
            y_hat=y_hat,
            y_mel=y_mel,
            y_hat_mel=y_hat_mel,
            x_mask=x_mask,
            z_mask=z_mask,
            l_length=l_length,
            z_p=z_p,
            m_p=m_p,
            logs_p=logs_p,
            logs_q=logs_q,
            extra=extra,
            decoder_aux=decoder_aux,
        )

    def _loss_d_audio(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        y_d_hat_r, y_d_hat_g, _, _ = self.model_d(y, y_hat.detach())
        with autocast(self.device.type, enabled=False):
            loss_d, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
        return loss_d

    def _loss_g_audio_adv_fm(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.model_d(y, y_hat)
        with autocast(self.device.type, enabled=False):
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_adv, _ = generator_loss(y_d_hat_g)
        return loss_adv + loss_fm

    def _loss_g_fixed_terms(
        self, pack: _ForwardPack
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with autocast(self.device.type, enabled=False):
            loss_dur = torch.sum(pack.l_length.float())
            loss_mel = F.l1_loss(pack.y_mel, pack.y_hat_mel) * self.hparams.c_mel
            loss_kl = (
                kl_loss(pack.z_p, pack.logs_q, pack.m_p, pack.logs_p, pack.z_mask)
                * self.hparams.c_kl
            )
            loss_fixed = loss_mel + loss_dur + loss_kl
        return loss_fixed, loss_mel, loss_dur, loss_kl

    def _loss_d_dur(
        self,
        extra: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_x, logw, logw_, g = extra
        y_dur_r, y_dur_g = self.model_dur(
            hidden_x.detach(),
            x_mask.detach(),
            logw_.detach(),
            logw.detach(),
            g.detach() if g is not None else None,
        )
        with autocast(self.device.type, enabled=False):
            loss_dur_disc, _, _ = masked_discriminator_loss(y_dur_r, y_dur_g, x_mask)
        return loss_dur_disc

    def _loss_g_dur(
        self,
        extra: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_x, logw, logw_, g = extra
        _y_dur_r2, y_dur_g2 = self.model_dur(hidden_x, x_mask, logw_, logw, g)
        with autocast(self.device.type, enabled=False):
            loss_dur_gen, _ = masked_generator_loss(y_dur_g2, x_mask)
        return loss_dur_gen

    def _loss_g_subband(self, y: torch.Tensor, decoder_aux: Optional[Any]):
        if not self._use_subband_loss:
            return None

        if (self.mb_pqmf is None) or (self.subband_stft_loss is None) or (decoder_aux is None):
            return None

        if isinstance(decoder_aux, dict):
            y_hat_mb = decoder_aux.get("subband_audio", None)
        else:
            y_hat_mb = None

        if y_hat_mb is None:
            return None

        y_mb = self.mb_pqmf.analysis(y)

        y_mb = y_mb.contiguous().view(-1, y_mb.size(-1))
        y_hat_mb = y_hat_mb.contiguous().view(-1, y_hat_mb.size(-1))

        min_len = min(y_mb.size(-1), y_hat_mb.size(-1))
        y_mb = y_mb[:, :min_len]
        y_hat_mb = y_hat_mb[:, :min_len]

        with autocast(self.device.type, enabled=False):
            sc_loss, mag_loss = self.subband_stft_loss(y_hat_mb.float(), y_mb.float())
            return (sc_loss + mag_loss) * float(self.hparams.subband_stft_loss_weight)

    def training_step(self, batch: Batch, batch_idx: int):
        opts = self.optimizers()
        if not isinstance(opts, (list, tuple)):
            opts = [opts]

        opt_g = opts[0]
        opt_d = opts[1]
        opt_dur = opts[2] if (len(opts) > 2) else None

        batch_size = int(batch.phoneme_ids.size(0))

        if (
            getattr(self.hparams, "use_vits2", False)
            and getattr(self.hparams, "vits2_use_noise_scaled_mas", False)
        ):
            if (
                hasattr(self.model_g, "current_mas_noise_scale")
                and hasattr(self.model_g, "mas_noise_scale_initial")
                and hasattr(self.model_g, "noise_scale_delta")
            ):
                cur = float(self.model_g.mas_noise_scale_initial) - (
                    float(self.model_g.noise_scale_delta) * float(self._mas_batch_step)
                )
                self.model_g.current_mas_noise_scale = max(cur, 0.0)

        pack = self._forward_g_and_prepare(batch)

        self._set_requires_grad(self.model_d, True)
        opt_d.zero_grad(set_to_none=True)
        loss_d = self._loss_d_audio(pack.y, pack.y_hat)
        self.manual_backward(loss_d)
        opt_d.step()
        self.log("loss_d", loss_d, batch_size=batch_size)

        loss_dur_disc = None
        if (
            (opt_dur is not None)
            and (getattr(self, "model_dur", None) is not None)
            and (pack.extra is not None)
        ):
            self._set_requires_grad(self.model_dur, True)
            opt_dur.zero_grad(set_to_none=True)
            loss_dur_disc = self._loss_d_dur(pack.extra, pack.x_mask)
            self.manual_backward(loss_dur_disc)
            opt_dur.step()
            self.log("loss_dur_disc", loss_dur_disc, batch_size=batch_size)

        self._set_requires_grad(self.model_d, False)
        if getattr(self, "model_dur", None) is not None:
            self._set_requires_grad(self.model_dur, False)

        opt_g.zero_grad(set_to_none=True)

        loss_adv_fm = self._loss_g_audio_adv_fm(pack.y, pack.y_hat)
        loss_fixed, loss_mel, loss_dur, loss_kl = self._loss_g_fixed_terms(pack)
        loss_g = loss_adv_fm + loss_fixed
        loss_subband = self._loss_g_subband(pack.y, pack.decoder_aux)
        if loss_subband is not None:
            loss_g = loss_g + loss_subband
            self.log("loss_subband", loss_subband, batch_size=batch_size)

        if (getattr(self, "model_dur", None) is not None) and (pack.extra is not None):
            loss_g = loss_g + self._loss_g_dur(pack.extra, pack.x_mask)

        self.manual_backward(loss_g)
        opt_g.step()

        self.log("loss_g", loss_g, batch_size=batch_size)
        self.log("loss_mel", loss_mel, batch_size=batch_size)
        self.log("loss_dur", loss_dur, batch_size=batch_size)
        self.log("loss_kl", loss_kl, batch_size=batch_size)

        self._set_requires_grad(self.model_d, True)
        if getattr(self, "model_dur", None) is not None:
            self._set_requires_grad(self.model_dur, True)

        self._mas_batch_step += 1

    def validation_step(self, batch: Batch, batch_idx: int):
        pack = self._forward_g_and_prepare(batch)
        batch_size = int(batch.phoneme_ids.size(0))

        with torch.no_grad():
            loss_adv_fm = self._loss_g_audio_adv_fm(pack.y, pack.y_hat)
            loss_fixed, _loss_mel, _loss_dur, _loss_kl = self._loss_g_fixed_terms(pack)
            val_loss = loss_adv_fm + loss_fixed

            loss_subband = self._loss_g_subband(pack.y, pack.decoder_aux)
            if loss_subband is not None:
                val_loss = val_loss + loss_subband
                self.log("val_loss_subband", loss_subband, batch_size=batch_size)

            if (getattr(self, "model_dur", None) is not None) and (pack.extra is not None):
                val_loss = val_loss + self._loss_g_dur(pack.extra, pack.x_mask)

        self.log("val_loss", val_loss, batch_size=batch_size)
        return val_loss

    def on_train_epoch_end(self) -> None:
        scheds = self.lr_schedulers()
        if scheds is None:
            return

        if not isinstance(scheds, (list, tuple)):
            scheds = [scheds]

        for sch in scheds:
            if sch is not None:
                sch.step()

    def on_validation_end(self) -> None:
        if self.trainer.sanity_checking:
            return super().on_validation_end()

        output_dir = os.path.join(self.trainer.default_root_dir, "val_samples")
        os.makedirs(output_dir, exist_ok=True)

        dataset = getattr(self.trainer.datamodule, "test_dataset", None)
        if dataset is None:
            return super().on_validation_end()

        logger = getattr(self, "logger", None)
        has_tb_audio = (
            logger is not None
            and hasattr(logger, "experiment")
            and hasattr(logger.experiment, "add_audio")
        )

        for utt_idx, test_utt in enumerate(dataset):
            text = test_utt.phoneme_ids.unsqueeze(0).to(self.device)
            text_lengths = torch.LongTensor([len(test_utt.phoneme_ids)]).to(self.device)
            scales = [0.667, 1.0, 0.8]
            sid = (
                test_utt.speaker_id.to(self.device)
                if test_utt.speaker_id is not None
                else None
            )
            if sid is not None and sid.dim() == 0:
                sid = sid.unsqueeze(0)

            with torch.no_grad():
                audio, *_ = self.model_g.infer(
                    text,
                    text_lengths,
                    noise_scale=scales[0],
                    length_scale=scales[1],
                    noise_scale_w=scales[2],
                    sid=sid,
                    max_len=2000,
                )

            audio = audio.squeeze()
            audio_np = audio.squeeze().cpu().numpy()

            max_val = np.abs(audio_np).max()
            if max_val > 0.9:
                audio_np = audio_np / max_val * 0.9

            audio_int16 = (audio_np * 32767).astype(np.int16)

            filename = f"sample_{utt_idx}.wav"
            filepath = os.path.join(output_dir, filename)

            write_wav(filepath, self.hparams.sample_rate, audio_int16)

            if has_tb_audio:
                audio_tb = torch.from_numpy(audio_np).unsqueeze(0)
                tag = test_utt.text or str(utt_idx)

                logger.experiment.add_audio(
                    tag=tag,
                    snd_tensor=audio_tb,
                    sample_rate=self.hparams.sample_rate,
                    global_step=self.global_step,
                )

        return super().on_validation_end()

    def configure_optimizers(self):
        optim_g = torch.optim.AdamW(
            self.model_g.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
        )
        optim_d = torch.optim.AdamW(
            self.model_d.parameters(),
            lr=self.hparams.learning_rate_d,
            betas=self.hparams.betas_d,
            eps=self.hparams.eps,
        )

        optimizers = [optim_g, optim_d]
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=self.hparams.lr_decay),
            torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=self.hparams.lr_decay_d),
        ]

        if self.model_dur is not None:
            optim_dur = torch.optim.AdamW(
                self.model_dur.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                eps=self.hparams.eps,
            )
            optimizers.append(optim_dur)
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    optim_dur, gamma=self.hparams.lr_decay
                )
            )

        return optimizers, schedulers
