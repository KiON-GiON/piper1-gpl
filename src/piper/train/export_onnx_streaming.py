import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.utils import remove_weight_norm

from .vits import commons
from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__name__)
OPSET_VERSION = 15


def recursive_remove_weight_norm(module: nn.Module) -> None:
    try:
        remove_weight_norm(module)
    except ValueError:
        pass

    for child in module.children():
        recursive_remove_weight_norm(child)


class VitsEncoder(nn.Module):

    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def forward(
        self,
        x: torch.LongTensor,
        x_lengths: torch.LongTensor,
        scales: torch.FloatTensor,
        sid: Optional[torch.LongTensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]

        gen = self.gen

        if gen.n_speakers > 1:
            assert sid is not None, "Missing speaker id"
            g = gen.emb_g(sid).unsqueeze(-1)  # [B, gin, 1]
        else:
            g = None

        x_enc, m_p, logs_p, x_mask = gen.enc_p(x, x_lengths, g=g)

        if hasattr(gen, "sdp") and (gen.sdp is not None):
            logw_sdp = gen.sdp(
                x_enc,
                x_mask,
                g=g,
                reverse=True,
                noise_scale=noise_scale_w,
            )
            logw_dp = gen.dp(x_enc, x_mask, g=g)

            sdp_ratio = float(getattr(gen, "vits2_infer_sdp_ratio", 0.2))
            sdp_ratio = max(0.0, min(1.0, sdp_ratio))

            logw = logw_sdp * sdp_ratio + logw_dp * (1.0 - sdp_ratio)

        elif getattr(gen, "use_sdp", False):
            logw = gen.dp(
                x_enc,
                x_mask,
                g=g,
                reverse=True,
                noise_scale=noise_scale_w,
            )
        else:
            logw = gen.dp(x_enc, x_mask, g=g)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(
            commons.sequence_mask(y_lengths, y_lengths.max()), 1
        ).type_as(x_mask)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p_exp = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_exp = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )

        z_p = m_p_exp + torch.randn_like(m_p_exp) * torch.exp(logs_p_exp) * noise_scale

        if gen.n_speakers > 1:
            return z_p, y_mask, g
        else:
            return z_p, y_mask


class VitsDecoder(nn.Module):

    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def forward(
        self,
        z: torch.Tensor,
        y_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.gen.flow(z, y_mask, g=g, reverse=True)
        output = self.gen.dec(z * y_mask, g=g)
        return output


def main() -> None:
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = VitsModel.load_from_checkpoint(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model_g = model.model_g
    model_g.eval()

    with torch.no_grad():
        if hasattr(model_g.dec, "remove_weight_norm"):
            model_g.dec.remove_weight_norm()
        recursive_remove_weight_norm(model_g)

    _LOGGER.info("Exporting encoder...")
    decoder_input = export_encoder(output_dir, model_g)
    _LOGGER.info("Exporting decoder...")
    export_decoder(output_dir, model_g, decoder_input)
    _LOGGER.info("Exported streaming model to %s", str(output_dir))


def export_encoder(output_dir: Path, model_g) -> tuple:
    model = VitsEncoder(model_g)
    model.eval()

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers

    dummy_input_length = 50
    sequences = torch.randint(
        low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long
    )
    sequence_lengths = torch.LongTensor([sequences.size(1)])

    scales = torch.FloatTensor([0.667, 1.0, 0.8])

    if num_speakers > 1:
        sid = torch.LongTensor([0])
        dummy_input = (sequences, sequence_lengths, scales, sid)
        input_names = ["input", "input_lengths", "scales", "sid"]
        dynamic_axes = {
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "scales": {0: "scale_dim"},
            "sid": {0: "batch_size"},
        }
        output_names = ["z", "y_mask", "g"]
        dynamic_axes.update(
            {
                "z": {0: "batch_size", 2: "time"},
                "y_mask": {0: "batch_size", 2: "time"},
                "g": {0: "batch_size"},
            }
        )
    else:
        dummy_input = (sequences, sequence_lengths, scales)
        input_names = ["input", "input_lengths", "scales"]
        dynamic_axes = {
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "scales": {0: "scale_dim"},
        }
        output_names = ["z", "y_mask"]
        dynamic_axes.update(
            {
                "z": {0: "batch_size", 2: "time"},
                "y_mask": {0: "batch_size", 2: "time"},
            }
        )

    onnx_path = str(output_dir.joinpath("encoder.onnx"))

    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=onnx_path,
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    _LOGGER.info("Exported encoder to %s", onnx_path)

    with torch.no_grad():
        encoder_outputs = model(*dummy_input)

    if isinstance(encoder_outputs, tuple):
        return encoder_outputs
    else:
        return (encoder_outputs,)


def export_decoder(output_dir: Path, model_g, decoder_input: tuple) -> None:
    model = VitsDecoder(model_g)
    model.eval()

    num_speakers = model_g.n_speakers

    if num_speakers > 1:
        input_names = ["z", "y_mask", "g"]
        dynamic_axes = {
            "z": {0: "batch_size", 2: "time"},
            "y_mask": {0: "batch_size", 2: "time"},
            "g": {0: "batch_size"},
            "output": {0: "batch_size", 2: "time"},
        }
    else:
        input_names = ["z", "y_mask"]
        dynamic_axes = {
            "z": {0: "batch_size", 2: "time"},
            "y_mask": {0: "batch_size", 2: "time"},
            "output": {0: "batch_size", 2: "time"},
        }

    onnx_path = str(output_dir.joinpath("decoder.onnx"))

    torch.onnx.export(
        model=model,
        args=decoder_input,
        f=onnx_path,
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    _LOGGER.info("Exported decoder to %s", onnx_path)


if __name__ == "__main__":
    main()