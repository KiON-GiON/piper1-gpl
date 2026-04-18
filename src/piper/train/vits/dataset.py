"""PyTorch Lightning dataset."""

import os
import csv
import itertools
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import lightning as L
import numpy as np
import torch
from pysilero_vad import SileroVoiceActivityDetector
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import pyworld as pw
except Exception:
    pw = None

from piper.config import PhonemeType, PiperConfig
from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP, phonemes_to_ids
from piper.phonemize_espeak import EspeakPhonemizer

from .mel_processing import spectrogram_torch
from .utils import get_cache_id

_LOGGER = logging.getLogger(__name__)
VAD_SAMPLE_RATE = 16000

_WORKER_VAD = None

def _init_prepare_worker():
    """Initializer for each ProcessPool worker."""
    global _WORKER_VAD
    from pysilero_vad import SileroVoiceActivityDetector
    _WORKER_VAD = SileroVoiceActivityDetector()

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _trim_silence_worker(
    audio_original_array: np.ndarray,
    audio_16khz_array: np.ndarray,
    threshold: float,
    keep_seconds_before_silence: float,
    keep_seconds_after_silence: float,
) -> np.ndarray:
    """Equivalent trimming logic, but for worker processes."""
    global _WORKER_VAD
    vad = _WORKER_VAD
    if vad is None:
        return audio_original_array

    vad.reset()

    first_chunk = None
    last_chunk = None

    samples_per_chunk = vad.chunk_samples()
    seconds_per_chunk = samples_per_chunk / VAD_SAMPLE_RATE
    num_chunks = len(audio_16khz_array) // samples_per_chunk

    for chunk_idx in range(num_chunks):
        off = chunk_idx * samples_per_chunk
        chunk = audio_16khz_array[off : off + samples_per_chunk]
        if len(chunk) < samples_per_chunk:
            continue

        prob = vad.process_array(chunk)
        is_speech = prob >= threshold

        if is_speech:
            if first_chunk is None:
                first_chunk = chunk_idx
            last_chunk = chunk_idx

    if (first_chunk is None) or (last_chunk is None):
        return audio_original_array

    num_original_samples = len(audio_original_array)
    audio_seconds = len(audio_16khz_array) / VAD_SAMPLE_RATE

    first_sec = first_chunk * seconds_per_chunk
    first_sec = max(0.0, first_sec - keep_seconds_before_silence)
    first_sample = int(math.floor(num_original_samples * (first_sec / audio_seconds)))

    last_sec = (last_chunk + 1) * seconds_per_chunk
    last_sec = min(audio_seconds, last_sec + keep_seconds_after_silence)
    last_sample = int(math.ceil(num_original_samples * (last_sec / audio_seconds)))

    return audio_original_array[first_sample:last_sample]


def _fit_feature_length_np(values: np.ndarray, target_length: int, pad_value: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] == target_length:
        return values
    if values.shape[0] > target_length:
        return values[:target_length]
    if values.shape[0] == 0:
        return np.full((target_length,), pad_value, dtype=np.float32)
    pad = np.full((target_length - values.shape[0],), pad_value, dtype=np.float32)
    return np.concatenate([values, pad], axis=0)


def _extract_log_f0_and_voiced_pyworld(
    audio_array: np.ndarray,
    sample_rate: int,
    hop_length: int,
    pitch_fmin: float,
    pitch_fmax: float,
    target_frames: int,
):
    import pyworld as pw

    x = np.asarray(audio_array, dtype=np.float64)
    frame_period_ms = (hop_length / sample_rate) * 1000.0

    f0, t = pw.dio(
        x,
        fs=int(sample_rate),
        f0_floor=float(pitch_fmin),
        f0_ceil=float(pitch_fmax),
        frame_period=frame_period_ms,
    )
    f0 = pw.stonemask(x, f0, t, fs=int(sample_rate))
    f0 = np.asarray(f0, dtype=np.float32)

    voiced = (f0 > 0.0).astype(np.float32)

    if np.any(voiced > 0.5):
        idx = np.where(voiced > 0.5)[0]
        cf0 = np.interp(np.arange(len(f0)), idx, f0[idx]).astype(np.float32)
    else:
        cf0 = np.zeros_like(f0, dtype=np.float32)

    log_f0 = np.log(np.clip(cf0, a_min=1.0, a_max=None)).astype(np.float32)

    log_pad_value = float(log_f0[-1]) if log_f0.size > 0 else 0.0
    log_f0 = _fit_feature_length_np(log_f0, target_frames, pad_value=log_pad_value)
    voiced = _fit_feature_length_np(voiced, target_frames, pad_value=0.0)

    return torch.from_numpy(log_f0), torch.from_numpy(voiced)


def _process_audio_spec_pitch_task(task: dict) -> tuple[bool, str]:
    """
    Worker task:
      - load + (optional) trim + save normalized audio
      - compute + save spectrogram
      - compute + save log_f0/voiced (optional)
    """
    try:
        import librosa
        import torch
        from .mel_processing import spectrogram_torch  # local import (worker safe)

        audio_path = task["audio_path"]
        norm_audio_path = task["norm_audio_path"]
        spec_path = task["spec_path"]
        do_pitch = task["do_pitch"]
        logf0_path = task.get("logf0_path")
        voiced_path = task.get("voiced_path")

        sample_rate = int(task["sample_rate"])
        filter_length = int(task["filter_length"])
        hop_length = int(task["hop_length"])
        win_length = int(task["win_length"])

        trim_silence = bool(task["trim_silence"])
        keep_before = float(task["keep_before"])
        keep_after = float(task["keep_after"])
        vad_threshold = float(task.get("vad_threshold", 0.2))

        pitch_fmin = float(task.get("pitch_fmin", 40.0))
        pitch_fmax = float(task.get("pitch_fmax", 1100.0))

        if os.path.exists(norm_audio_path) and os.path.exists(spec_path):
            if not do_pitch:
                return True, audio_path
            if os.path.exists(logf0_path) and os.path.exists(voiced_path):
                return True, audio_path

        audio, _sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        audio = np.asarray(audio, dtype=np.float32)

        if trim_silence:
            if sample_rate != VAD_SAMPLE_RATE:
                audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=VAD_SAMPLE_RATE)
            else:
                audio_16k = audio

            audio = _trim_silence_worker(
                audio_original_array=audio,
                audio_16khz_array=np.asarray(audio_16k, dtype=np.float32),
                threshold=vad_threshold,
                keep_seconds_before_silence=keep_before,
                keep_seconds_after_silence=keep_after,
            )
            audio = np.asarray(audio, dtype=np.float32)

        if not os.path.exists(norm_audio_path):
            torch.save(torch.FloatTensor(audio), norm_audio_path)

        if not os.path.exists(spec_path):
            with torch.no_grad():
                audio_t = torch.FloatTensor(audio).unsqueeze(0)
                spec = spectrogram_torch(
                    y=audio_t,
                    n_fft=filter_length,
                    sampling_rate=sample_rate,
                    hop_size=hop_length,
                    win_size=win_length,
                    center=False,
                ).squeeze(0)
            torch.save(spec, spec_path)
            target_frames = int(spec.size(1))
        else:
            target_frames = None

        if do_pitch:
            assert logf0_path is not None and voiced_path is not None

            need_logf0 = not os.path.exists(logf0_path)
            need_voiced = not os.path.exists(voiced_path)
            if need_logf0 or need_voiced:
                if target_frames is None:
                    spec = torch.load(spec_path, map_location="cpu")
                    target_frames = int(spec.size(1))

                log_f0, voiced = _extract_log_f0_and_voiced_pyworld(
                    audio_array=audio,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                    pitch_fmin=pitch_fmin,
                    pitch_fmax=pitch_fmax,
                    target_frames=target_frames,
                )
                if need_logf0:
                    torch.save(log_f0.contiguous(), logf0_path)
                if need_voiced:
                    torch.save(voiced.contiguous(), voiced_path)

        return True, audio_path

    except Exception as e:
        return False, f"{task.get('audio_path','?')}: {repr(e)}"


@dataclass
class CachedUtterance:
    phoneme_ids_path: Path
    audio_norm_path: Path
    audio_spec_path: Path
    pitch_log_f0_path: Optional[Path] = None
    pitch_voiced_path: Optional[Path] = None
    text: Optional[str] = None
    speaker_id: Optional[int] = None


class VitsDataModule(L.LightningDataModule):
    def __init__(
        self,
        csv_path: Union[str, Path],
        cache_dir: Union[str, Path],
        espeak_voice: str,
        config_path: Union[str, Path],
        voice_name: str,
        sample_rate: int = 22050,
        audio_dir: Optional[Union[str, Path]] = None,
        alignments_dir: Optional[Union[str, Path]] = None,
        num_symbols: int = 256,
        num_speakers: int = 1,
        batch_size: int = 32,
        validation_split: float = 0.1,
        num_test_examples: int = 5,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        segment_size: int = 8192,
        num_workers: int = 1,
        trim_silence: bool = True,
        keep_seconds_before_silence: float = 0.25,
        keep_seconds_after_silence: float = 0.25,
        prepare_num_workers: int = 0,
        compute_pitch_features: bool = False,
        pitch_fmin: float = 40.0,
        pitch_fmax: float = 1100.0,
    ) -> None:
        super().__init__()

        self.csv_path = Path(csv_path)
        self.cache_dir = Path(cache_dir)
        self.espeak_voice = espeak_voice
        self.config_path = Path(config_path)
        self.voice_name = voice_name

        self.sample_rate = sample_rate
        self.num_symbols = num_symbols
        self.num_speakers = num_speakers

        if audio_dir is not None:
            self.audio_dir = Path(audio_dir)
        else:
            self.audio_dir = self.csv_path.parent

        if alignments_dir is not None:
            self.alignments_dir = Path(alignments_dir)
        else:
            self.alignments_dir = self.csv_path.parent / "alignments"

        self.batch_size = batch_size
        self.validation_split = validation_split
        self.num_test_examples = num_test_examples

        # Mel
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

        self.segment_size = segment_size
        self.num_workers = num_workers

        # Silence trimming
        self.trim_silence = trim_silence
        self.keep_seconds_before_silence = keep_seconds_before_silence
        self.keep_seconds_after_silence = keep_seconds_after_silence

        self.piper_config: Optional[PiperConfig] = None
        self.is_multispeaker = self.num_speakers > 1
        self.prepare_num_workers = int(prepare_num_workers)

        self.compute_pitch_features = compute_pitch_features
        self.pitch_fmin = float(pitch_fmin)
        self.pitch_fmax = float(pitch_fmax)

    @staticmethod
    def _fit_feature_length(
        values: np.ndarray,
        target_length: int,
        pad_value: float = 0.0,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)

        if values.shape[0] == target_length:
            return values

        if values.shape[0] > target_length:
            return values[:target_length]

        if values.shape[0] == 0:
            return np.full((target_length,), pad_value, dtype=np.float32)

        pad = np.full((target_length - values.shape[0],), pad_value, dtype=np.float32)
        return np.concatenate([values, pad], axis=0)

    def _extract_log_f0_and_voiced(
        self,
        audio_array: np.ndarray,
        target_frames: int,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        if pw is None:
            raise RuntimeError(
                "pyworld is not installed. Run: pip install pyworld"
            )

        x = np.asarray(audio_array, dtype=np.float64)

        frame_period_ms = (self.hop_length / self.sample_rate) * 1000.0

        f0, t = pw.dio(
            x,
            fs=int(self.sample_rate),
            f0_floor=float(self.pitch_fmin),
            f0_ceil=float(self.pitch_fmax),
            frame_period=frame_period_ms,
        )
        f0 = pw.stonemask(x, f0, t, fs=int(self.sample_rate))

        f0 = np.asarray(f0, dtype=np.float32)

        # voiced/unvoiced
        voiced = (f0 > 0.0).astype(np.float32)

        # continuous f0
        if np.any(voiced > 0.5):
            idx = np.where(voiced > 0.5)[0]
            cf0 = np.interp(np.arange(len(f0)), idx, f0[idx]).astype(np.float32)
        else:
            cf0 = np.zeros_like(f0, dtype=np.float32)

        log_f0 = np.log(np.clip(cf0, a_min=1.0, a_max=None)).astype(np.float32)

        log_pad_value = float(log_f0[-1]) if log_f0.size > 0 else 0.0
        log_f0 = self._fit_feature_length(log_f0, target_frames, pad_value=log_pad_value)
        voiced = self._fit_feature_length(voiced, target_frames, pad_value=0.0)

        return torch.FloatTensor(log_f0), torch.FloatTensor(voiced)

    def prepare_data(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.piper_config = PiperConfig(
            num_symbols=self.num_symbols,
            num_speakers=self.num_speakers,
            sample_rate=self.sample_rate,
            espeak_voice=self.espeak_voice,
            phoneme_id_map=DEFAULT_PHONEME_ID_MAP,
            phoneme_type=PhonemeType.ESPEAK,
            piper_version="1.3.0",
        )

        speaker_id_map: Dict[str, int] = {}
        if self.is_multispeaker:
            with open(self.csv_path, "r", encoding="utf-8") as csv_file:
                reader = csv.reader(csv_file, delimiter="|")
                for row in reader:
                    assert len(row) >= 3, "Expected CSV columns for multi-speaker metadata: wav|speaker|text"
                    speaker_name = row[1]
                    if speaker_name not in speaker_id_map:
                        speaker_id_map[speaker_name] = len(speaker_id_map)

            assert len(speaker_id_map) <= self.num_speakers, "More speakers in metadata than num_speakers"
            if len(speaker_id_map) != self.num_speakers:
                _LOGGER.warning("Expected %s speakers in the dataset, got %s", self.num_speakers, len(speaker_id_map))

            self.piper_config.speaker_id_map = speaker_id_map

        # Write config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as config_file:
            json.dump(self.piper_config.to_dict(), config_file, ensure_ascii=False, indent=2)

        phonemizer = EspeakPhonemizer()

        tasks: list[dict] = []
        num_utterances = 0
        report_prepare: Optional[bool] = None

        with open(self.csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for row_number, row in enumerate(reader, start=1):
                utt_id, text = row[0], row[-1]

                speaker_id: Optional[int] = None
                if self.is_multispeaker:
                    assert len(row) >= 3, "Expected CSV columns for multi-speaker metadata: wav|speaker|text"
                    speaker_name = row[1]
                    speaker_id = speaker_id_map[speaker_name]

                audio_path = self.audio_dir / utt_id
                if not audio_path.exists():
                    audio_path = self.audio_dir / f"{utt_id}.wav"

                if not audio_path.exists():
                    _LOGGER.warning("Missing audio file: %s", audio_path)
                    continue

                cache_id = get_cache_id(row_number, text, speaker_id=speaker_id)

                # text
                text_path = self.cache_dir / f"{cache_id}.txt"
                if not text_path.exists():
                    text_path.write_text(text, encoding="utf-8")

                # phonemes
                phonemes: Optional[List[List[str]]] = None
                phonemes_path = self.cache_dir / f"{cache_id}.phonemes.txt"
                if not phonemes_path.exists():
                    phonemes = phonemizer.phonemize(self.espeak_voice, text)
                    with open(phonemes_path, "w", encoding="utf-8") as f:
                        for sentence_phonemes in phonemes:
                            print("".join(sentence_phonemes), file=f)
                    if report_prepare is None:
                        report_prepare = True

                # phoneme ids
                phoneme_ids_path = self.cache_dir / f"{cache_id}.phonemes.pt"
                if not phoneme_ids_path.exists():
                    if phonemes is None:
                        phonemes = phonemizer.phonemize(self.espeak_voice, text)

                    phoneme_ids = list(
                        itertools.chain(
                            *(
                                phonemes_to_ids(sentence_phonemes, id_map=self.piper_config.phoneme_id_map)
                                for sentence_phonemes in phonemes
                            )
                        )
                    )
                    torch.save(torch.LongTensor(phoneme_ids), phoneme_ids_path)
                    if report_prepare is None:
                        report_prepare = True

                norm_audio_path = self.cache_dir / f"{cache_id}.audio.pt"
                audio_spec_path = self.cache_dir / f"{cache_id}.spec.pt"

                pitch_log_f0_path = self.cache_dir / f"{cache_id}.log_f0.pt"
                pitch_voiced_path = self.cache_dir / f"{cache_id}.voiced.pt"

                do_pitch = bool(self.compute_pitch_features)
                # Skip task if everything already exists
                if norm_audio_path.exists() and audio_spec_path.exists():
                    if (not do_pitch) or (pitch_log_f0_path.exists() and pitch_voiced_path.exists()):
                        num_utterances += 1
                        continue

                tasks.append(
                    dict(
                        audio_path=str(audio_path),
                        norm_audio_path=str(norm_audio_path),
                        spec_path=str(audio_spec_path),
                        do_pitch=do_pitch,
                        logf0_path=str(pitch_log_f0_path) if do_pitch else None,
                        voiced_path=str(pitch_voiced_path) if do_pitch else None,
                        sample_rate=int(self.sample_rate),
                        filter_length=int(self.filter_length),
                        hop_length=int(self.hop_length),
                        win_length=int(self.win_length),
                        trim_silence=bool(self.trim_silence),
                        keep_before=float(self.keep_seconds_before_silence),
                        keep_after=float(self.keep_seconds_after_silence),
                        vad_threshold=0.2,
                        pitch_fmin=float(self.pitch_fmin),
                        pitch_fmax=float(self.pitch_fmax),
                    )
                )

                num_utterances += 1
                if report_prepare:
                    _LOGGER.info("Preparing utterances (serial text/phonemes, parallel audio/spec/pitch)...")
                    report_prepare = False

        if tasks:
            max_workers = self.prepare_num_workers if self.prepare_num_workers > 0 else (os.cpu_count() or 1)
            max_workers = max(1, int(max_workers))
            _LOGGER.info("prepare_data: processing %s utterance(s) with %s worker(s)", len(tasks), max_workers)

            ok_count = 0
            fail_count = 0

            with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_prepare_worker) as ex:
                futures = [ex.submit(_process_audio_spec_pitch_task, t) for t in tasks]
                for fut in as_completed(futures):
                    ok, msg = fut.result()
                    if ok:
                        ok_count += 1
                    else:
                        fail_count += 1
                        _LOGGER.warning("prepare_data worker failed: %s", msg)

            _LOGGER.info("prepare_data: parallel stage done (ok=%s, failed=%s)", ok_count, fail_count)

        _LOGGER.info("Processed %s utterance(s)", num_utterances)

    def setup(self, stage: str) -> None:
        if self.piper_config is None:
            if not self.config_path.exists():
                raise RuntimeError(
                    f"piper_config is None and config_path doesn't exist: {self.config_path}"
                )

            with open(self.config_path, "r", encoding="utf-8") as config_file:
                cfg_dict = json.load(config_file)

            if hasattr(PiperConfig, "from_dict"):
                self.piper_config = PiperConfig.from_dict(cfg_dict)
            else:
                self.piper_config = PiperConfig(**cfg_dict)

        assert self.piper_config is not None

        all_utts: list[CachedUtterance] = []
        speaker_id_map = self.piper_config.speaker_id_map

        with open(self.csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for row_number, row in enumerate(reader, start=1):
                utt_id, text = row[0], row[-1]
                speaker_id: Optional[int] = None
                if self.is_multispeaker:
                    assert (
                        len(row) >= 3
                    ), "Expected CSV columns for multi-speaker metadata: wav|speaker|text"
                    speaker_name = row[1]
                    speaker_id = speaker_id_map[speaker_name]

                audio_path = self.audio_dir / utt_id
                if not audio_path.exists():
                    audio_path = self.audio_dir / f"{utt_id}.wav"

                if not audio_path.exists():
                    _LOGGER.warning("Missing audio file: %s", audio_path)
                    continue

                cache_id = get_cache_id(row_number, text, speaker_id=speaker_id)

                phoneme_ids_path = self.cache_dir / f"{cache_id}.phonemes.pt"
                if not phoneme_ids_path.exists():
                    _LOGGER.warning(
                        "Missing phoneme ids for %s: %s",
                        audio_path,
                        phoneme_ids_path,
                    )
                    continue

                audio_norm_path = self.cache_dir / f"{cache_id}.audio.pt"
                if not audio_norm_path.exists():
                    _LOGGER.warning(
                        "Missing normalized audio for %s: %s",
                        audio_path,
                        audio_norm_path,
                    )
                    continue

                audio_spec_path = self.cache_dir / f"{cache_id}.spec.pt"
                if not audio_spec_path.exists():
                    _LOGGER.warning(
                        "Missing mel spec for %s: %s",
                        audio_path,
                        audio_spec_path,
                    )
                    continue

                pitch_log_f0_path: Optional[Path] = None
                pitch_voiced_path: Optional[Path] = None

                if self.compute_pitch_features:
                    pitch_log_f0_path = self.cache_dir / f"{cache_id}.log_f0.pt"
                    pitch_voiced_path = self.cache_dir / f"{cache_id}.voiced.pt"

                    if not pitch_log_f0_path.exists():
                        _LOGGER.warning(
                            "Missing log_f0 for %s: %s",
                            audio_path,
                            pitch_log_f0_path,
                        )
                        continue

                    if not pitch_voiced_path.exists():
                        _LOGGER.warning(
                            "Missing voiced flags for %s: %s",
                            audio_path,
                            pitch_voiced_path,
                        )
                        continue

                text: Optional[str] = None
                text_path = self.cache_dir / f"{cache_id}.txt"
                if text_path.exists():
                    text = text_path.read_text(encoding="utf-8")

                all_utts.append(
                    CachedUtterance(
                        phoneme_ids_path=phoneme_ids_path,
                        audio_norm_path=audio_norm_path,
                        audio_spec_path=audio_spec_path,
                        pitch_log_f0_path=pitch_log_f0_path,
                        pitch_voiced_path=pitch_voiced_path,
                        text=text,
                        speaker_id=speaker_id,
                    )
                )

        full_dataset = VitsDataset(all_utts)

        valid_set_size = int(len(full_dataset) * self.validation_split)
        train_set_size = len(full_dataset) - valid_set_size - self.num_test_examples
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(
            full_dataset, [train_set_size, self.num_test_examples, valid_set_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=(self.num_speakers > 1), segment_size=self.segment_size
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=(self.num_speakers > 1), segment_size=self.segment_size
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=(self.num_speakers > 1), segment_size=self.segment_size
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def _trim_silence(
        self,
        audio_original_array: np.ndarray,
        audio_16khz_array: np.ndarray,
        vad: SileroVoiceActivityDetector,
        threshold: float = 0.2,
    ) -> np.ndarray:
        """Trims silence from original array."""
        vad.reset()

        first_chunk: Optional[int] = None
        last_chunk: Optional[int] = None
        first_sample: Optional[int] = None
        last_sample: Optional[int] = None

        samples_per_chunk = vad.chunk_samples()
        seconds_per_chunk: float = samples_per_chunk / VAD_SAMPLE_RATE
        num_chunks = len(audio_16khz_array) // samples_per_chunk

        # Determine main block of speech
        for chunk_idx in range(num_chunks):
            chunk_offset = chunk_idx * samples_per_chunk
            chunk = audio_16khz_array[chunk_offset : chunk_offset + samples_per_chunk]
            if len(chunk) < samples_per_chunk:
                continue

            prob = vad.process_array(chunk)
            is_speech = prob >= threshold

            if is_speech:
                if first_chunk is None:
                    first_chunk = chunk_idx
                last_chunk = chunk_idx

        if (first_chunk is not None) and (last_chunk is not None):
            num_original_samples = len(audio_original_array)
            audio_seconds = len(audio_16khz_array) / VAD_SAMPLE_RATE

            first_sec = first_chunk * seconds_per_chunk
            first_sec = max(0.0, first_sec - self.keep_seconds_before_silence)
            first_sample = int(
                math.floor(num_original_samples * (first_sec / audio_seconds))
            )

            last_sec = (last_chunk + 1) * seconds_per_chunk
            last_sec = min(audio_seconds, last_sec + self.keep_seconds_after_silence)
            last_sample = int(
                math.ceil(num_original_samples * (last_sec / audio_seconds))
            )

        return audio_original_array[first_sample:last_sample]


@dataclass
class UtteranceTensors:
    phoneme_ids: LongTensor
    spectrogram: FloatTensor
    audio_norm: FloatTensor
    log_f0: Optional[FloatTensor] = None
    voiced: Optional[FloatTensor] = None
    speaker_id: Optional[LongTensor] = None
    text: Optional[str] = None

    @property
    def spec_length(self) -> int:
        return self.spectrogram.size(1)


@dataclass
class Batch:
    phoneme_ids: LongTensor
    phoneme_lengths: LongTensor
    spectrograms: FloatTensor
    spectrogram_lengths: LongTensor
    audios: FloatTensor
    audio_lengths: LongTensor
    speaker_ids: Optional[LongTensor] = None
    log_f0: Optional[FloatTensor] = None
    voiced: Optional[FloatTensor] = None


class VitsDataset(Dataset):
    def __init__(self, utts: list[CachedUtterance]):
        self.utts = utts

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, idx) -> UtteranceTensors:
        utt = self.utts[idx]
        return UtteranceTensors(
            phoneme_ids=torch.load(utt.phoneme_ids_path),
            audio_norm=torch.load(utt.audio_norm_path),
            spectrogram=torch.load(utt.audio_spec_path),
            log_f0=(
                torch.load(utt.pitch_log_f0_path)
                if utt.pitch_log_f0_path is not None
                else None
            ),
            voiced=(
                torch.load(utt.pitch_voiced_path)
                if utt.pitch_voiced_path is not None
                else None
            ),
            speaker_id=(
                LongTensor([utt.speaker_id]) if utt.speaker_id is not None else None
            ),
            text=utt.text,
        )


class UtteranceCollate:
    def __init__(self, is_multispeaker: bool, segment_size: int):
        self.is_multispeaker = is_multispeaker
        self.segment_size = segment_size

    def __call__(self, utterances: Sequence[UtteranceTensors]) -> Batch:
        num_utterances = len(utterances)
        assert num_utterances > 0, "No utterances"

        max_phonemes_length = 0
        max_spec_length = 0
        max_audio_length = 0

        num_mels = 0

        has_pitch = any(utt.log_f0 is not None for utt in utterances)
        if has_pitch:
            assert all(
                (utt.log_f0 is not None) and (utt.voiced is not None)
                for utt in utterances
            ), "Mixed pitch availability in batch"

        # Determine lengths
        for utt_idx, utt in enumerate(utterances):
            assert utt.spectrogram is not None
            assert utt.audio_norm is not None

            phoneme_length = utt.phoneme_ids.size(0)
            spec_length = utt.spectrogram.size(1)
            audio_length = utt.audio_norm.size(0)

            max_phonemes_length = max(max_phonemes_length, phoneme_length)
            max_spec_length = max(max_spec_length, spec_length)
            max_audio_length = max(max_audio_length, audio_length)

            num_mels = utt.spectrogram.size(0)
            if self.is_multispeaker:
                assert utt.speaker_id is not None, "Missing speaker id"

        # Audio cannot be smaller than segment size (8192)
        max_audio_length = max(max_audio_length, self.segment_size)

        # Create padded tensors
        phonemes_padded = LongTensor(num_utterances, max_phonemes_length)
        spec_padded = FloatTensor(num_utterances, num_mels, max_spec_length)
        audio_padded = FloatTensor(num_utterances, 1, max_audio_length)

        phonemes_padded.zero_()
        spec_padded.zero_()
        audio_padded.zero_()

        log_f0_padded: Optional[FloatTensor] = None
        voiced_padded: Optional[FloatTensor] = None
        if has_pitch:
            log_f0_padded = FloatTensor(num_utterances, max_spec_length)
            voiced_padded = FloatTensor(num_utterances, max_spec_length)
            log_f0_padded.zero_()
            voiced_padded.zero_()

        phoneme_lengths = LongTensor(num_utterances)
        spec_lengths = LongTensor(num_utterances)
        audio_lengths = LongTensor(num_utterances)

        speaker_ids: Optional[LongTensor] = None
        if self.is_multispeaker:
            speaker_ids = LongTensor(num_utterances)

        # Sort by decreasing spectrogram length
        sorted_utterances = sorted(
            utterances, key=lambda u: u.spectrogram.size(1), reverse=True
        )
        for utt_idx, utt in enumerate(sorted_utterances):
            phoneme_length = utt.phoneme_ids.size(0)

            spec_length = utt.spectrogram.size(1)
            audio_length = utt.audio_norm.size(0)

            phonemes_padded[utt_idx, :phoneme_length] = utt.phoneme_ids
            phoneme_lengths[utt_idx] = phoneme_length

            spec_padded[utt_idx, :, :spec_length] = utt.spectrogram
            spec_lengths[utt_idx] = spec_length

            audio_padded[utt_idx, :, :audio_length] = utt.audio_norm
            audio_lengths[utt_idx] = audio_length

            if utt.speaker_id is not None:
                assert speaker_ids is not None
                speaker_ids[utt_idx] = int(utt.speaker_id.item())

            if has_pitch:
                assert log_f0_padded is not None
                assert voiced_padded is not None
                assert utt.log_f0 is not None
                assert utt.voiced is not None

                log_f0_padded[utt_idx, :spec_length] = utt.log_f0[:spec_length]
                voiced_padded[utt_idx, :spec_length] = utt.voiced[:spec_length]

        return Batch(
            phoneme_ids=phonemes_padded,
            phoneme_lengths=phoneme_lengths,
            spectrograms=spec_padded,
            spectrogram_lengths=spec_lengths,
            audios=audio_padded,
            audio_lengths=audio_lengths,
            speaker_ids=speaker_ids,
            log_f0=log_f0_padded,
            voiced=voiced_padded,
        )
