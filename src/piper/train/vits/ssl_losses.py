import torch
import torchaudio
from torch import nn
from transformers import AutoModel


class WavLMFeatureLoss(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        source_sr: int = 22050,
        target_sr: int = 16000,
        layers: tuple[int, ...] | None = (6, 9, 12),
    ):
        super().__init__()
        self.model_name = model_name
        self.source_sr = int(source_sr)
        self.target_sr = int(target_sr)
        self.layers = tuple(layers) if layers is not None else None

        self.ssl = AutoModel.from_pretrained(model_name)
        self.ssl.eval()
        for p in self.ssl.parameters():
            p.requires_grad = False

        if self.source_sr != self.target_sr:
            self.resample = torchaudio.transforms.Resample(
                self.source_sr, self.target_sr
            )
        else:
            self.resample = nn.Identity()

    def train(self, mode: bool = True):
        super().train(mode)
        self.ssl.eval()
        return self

    def _prepare_audio(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)

        wav = wav.float().clamp(-1.0, 1.0)
        wav = self.resample(wav)
        return wav

    def _extract(self, wav: torch.Tensor, requires_grad: bool):
        wav = self._prepare_audio(wav)

        if requires_grad:
            out = self.ssl(input_values=wav, output_hidden_states=True)
        else:
            with torch.no_grad():
                out = self.ssl(input_values=wav, output_hidden_states=True)

        hs = out.hidden_states
        if self.layers is not None:
            hs = [hs[i] for i in self.layers]

        return hs

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        self.ssl.eval()

        ref_hs = self._extract(y, requires_grad=False)
        hat_hs = self._extract(y_hat, requires_grad=True)

        loss = y_hat.new_zeros([])
        n = 0
        for ref, hat in zip(ref_hs, hat_hs):
            t = min(ref.size(1), hat.size(1))
            loss = loss + torch.mean(torch.abs(ref[:, :t].detach() - hat[:, :t]))
            n += 1

        return loss / max(n, 1)