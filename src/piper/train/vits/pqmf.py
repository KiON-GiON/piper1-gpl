import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal.windows import kaiser


def design_prototype_filter(taps=62, cutoff_ratio=0.15, beta=9.0):
    assert taps % 2 == 0
    assert 0.0 < cutoff_ratio < 1.0

    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid="ignore"):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (
            np.pi * (np.arange(taps + 1) - 0.5 * taps)
        )
    h_i[taps // 2] = cutoff_ratio

    w = kaiser(taps + 1, beta)
    return h_i * w


class PQMF(nn.Module):
    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.15, beta=9.0):
        super().__init__()
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)

        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))

        for k in range(subbands):
            h_analysis[k] = 2 * h_proto * np.cos(
                (2 * k + 1)
                * (np.pi / (2 * subbands))
                * (np.arange(taps + 1) - ((taps - 1) / 2))
                + ((-1) ** k) * np.pi / 4
            )
            h_synthesis[k] = 2 * h_proto * np.cos(
                (2 * k + 1)
                * (np.pi / (2 * subbands))
                * (np.arange(taps + 1) - ((taps - 1) / 2))
                - ((-1) ** k) * np.pi / 4
            )

        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)

        updown_filter = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0

        self.register_buffer("analysis_filter", analysis_filter, persistent=False)
        self.register_buffer("synthesis_filter", synthesis_filter, persistent=False)
        self.register_buffer("updown_filter", updown_filter, persistent=False)

        self.subbands = int(subbands)
        self.pad_fn = nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x: torch.Tensor):
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x: torch.Tensor):
        x = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)