import operator
from functools import reduce

from .istft_decoders import (
    ISTFTGenerator,
    MultibandISTFTGenerator,
    MultistreamISTFTGenerator,
)

VALID_DECODER_TYPES = {"hifigan", "istft", "mb_istft", "ms_istft"}


def resolve_decoder_type(
    decoder_type=None,
    istft_vits: bool = False,
    mb_istft_vits: bool = False,
    ms_istft_vits: bool = False,
) -> str:
    if decoder_type is not None:
        t = str(decoder_type).lower()
    elif mb_istft_vits:
        t = "mb_istft"
    elif ms_istft_vits:
        t = "ms_istft"
    elif istft_vits:
        t = "istft"
    else:
        t = "hifigan"

    if t not in VALID_DECODER_TYPES:
        raise ValueError(f"decoder_type must be one of {sorted(VALID_DECODER_TYPES)}")

    return t


def decoder_output_hop_length(
    decoder_type: str,
    upsample_rates,
    gen_istft_hop_size: int = 1,
    subbands: int = 1,
) -> int:
    base = reduce(operator.mul, upsample_rates, 1)
    decoder_type = resolve_decoder_type(decoder_type=decoder_type)

    if decoder_type == "hifigan":
        return base
    if decoder_type == "istft":
        return base * int(gen_istft_hop_size)
    if decoder_type in {"mb_istft", "ms_istft"}:
        return base * int(gen_istft_hop_size) * int(subbands)

    raise ValueError(f"Unsupported decoder_type={decoder_type}")


def build_decoder(
    decoder_type: str,
    initial_channel: int,
    resblock,
    resblock_kernel_sizes,
    resblock_dilation_sizes,
    upsample_rates,
    upsample_initial_channel: int,
    upsample_kernel_sizes,
    gin_channels: int = 0,
    gen_istft_n_fft: int = 16,
    gen_istft_hop_size: int = 4,
    subbands: int = 4,
    is_onnx: bool = False,
):
    decoder_type = resolve_decoder_type(decoder_type=decoder_type)

    if decoder_type == "hifigan":
        from .models import Generator
        return Generator(
            initial_channel,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

    common = dict(
        initial_channel=initial_channel,
        resblock=resblock,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        upsample_rates=upsample_rates,
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=upsample_kernel_sizes,
        gin_channels=gin_channels,
        gen_istft_n_fft=gen_istft_n_fft,
        gen_istft_hop_size=gen_istft_hop_size,
        is_onnx=is_onnx,
    )

    if decoder_type == "istft":
        return ISTFTGenerator(**common)
    if decoder_type == "mb_istft":
        return MultibandISTFTGenerator(**common, subbands=subbands)
    if decoder_type == "ms_istft":
        return MultistreamISTFTGenerator(**common, subbands=subbands)

    raise ValueError(f"Unsupported decoder_type={decoder_type}")