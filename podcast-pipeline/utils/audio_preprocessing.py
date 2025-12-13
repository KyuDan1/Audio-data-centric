"""
Audio preprocessing utilities for podcast pipeline.
Includes audio standardization and chunking functionality.
"""

import os
import numpy as np
import librosa
import tempfile
from pydub import AudioSegment
from utils.logger import time_logger, Logger

# Global audio counter
audio_count = 0

# Constants
MIN_SPLIT_SILENCE = 1.0  # seconds of silence required for splitting
MAX_DIA_CHUNK_DURATION = 5 * 60  # 5 minutes

# Logger will be initialized from main module
logger = None

def set_logger(log_instance):
    """Set logger instance from main module."""
    global logger
    logger = log_instance


@time_logger
def standardization(audio, cfg):
    """
    Preprocess the audio file, including setting sample rate, bit depth, channels, and volume normalization.

    Args:
        audio (str or AudioSegment): Audio file path or AudioSegment object, the audio to be preprocessed.
        cfg (dict): Configuration dictionary containing sample rate settings.

    Returns:
        dict: A dictionary containing the preprocessed audio waveform, audio file name, and sample rate, formatted as:
              {
                  "waveform": np.ndarray, the preprocessed audio waveform, dtype is np.float32, shape is (num_samples,)
                  "name": str, the audio file name
                  "sample_rate": int, the audio sample rate
              }

    Raises:
        ValueError: If the audio parameter is neither a str nor an AudioSegment.
    """
    global audio_count
    name = "audio"

    if isinstance(audio, str):
        name = os.path.basename(audio)
        audio = AudioSegment.from_file(audio)
    elif isinstance(audio, AudioSegment):
        name = f"audio_{audio_count}"
        audio_count += 1
    else:
        raise ValueError("Invalid audio type")

    logger.debug("Entering the preprocessing of audio")

    # Convert the audio file to WAV format
    audio = audio.set_frame_rate(cfg["entrypoint"]["SAMPLE_RATE"])
    audio = audio.set_sample_width(2)  # Set bit depth to 16bit
    audio = audio.set_channels(1)  # Set to mono

    logger.debug("Audio file converted to WAV format")

    # Calculate the gain to be applied
    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    logger.info(f"Calculating the gain needed for the audio: {gain} dB")

    # Normalize volume and limit gain range to between -3 and 3
    normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)

    # Ensure waveform is 1D (mono)
    if waveform.ndim > 1:
        logger.warning(f"Waveform has {waveform.ndim} dimensions with shape {waveform.shape}, converting to mono")
        waveform = waveform.flatten()

    max_amplitude = np.max(np.abs(waveform))
    if max_amplitude > 0:
        waveform /= max_amplitude  # Normalize
    else:
        logger.warning("Audio has zero amplitude, skipping normalization")

    logger.debug(f"waveform shape: {waveform.shape}")
    logger.debug("waveform in np ndarray, dtype=" + str(waveform.dtype))

    return {
        "waveform": waveform,
        "name": name,
        "sample_rate": cfg["entrypoint"]["SAMPLE_RATE"],
        "audio_segment": normalized_audio,
    }


def _build_silence_intervals(waveform, sample_rate, min_silence, vad_model, silero_vad):
    """
    Use VAD to find silence intervals that can be used as cut points.
    """
    if vad_model is None:
        return len(waveform) / sample_rate, []

    if len(waveform) == 0:
        return 0.0, []

    resampled = librosa.resample(
        waveform, orig_sr=sample_rate, target_sr=silero_vad.SAMPLING_RATE
    )
    if resampled.size == 0:
        return len(waveform) / sample_rate, []

    speech_ts = vad_model.get_speech_timestamps(
        resampled,
        vad_model.vad_model,
        sampling_rate=silero_vad.SAMPLING_RATE,
    )
    total_duration = len(waveform) / sample_rate
    if not speech_ts:
        return total_duration, [(0.0, total_duration)]

    silence = []
    first_start = speech_ts[0]["start"] / silero_vad.SAMPLING_RATE
    if first_start >= min_silence:
        silence.append((0.0, first_start))

    for prev_seg, next_seg in zip(speech_ts[:-1], speech_ts[1:]):
        sil_start = prev_seg["end"] / silero_vad.SAMPLING_RATE
        sil_end = next_seg["start"] / silero_vad.SAMPLING_RATE
        if sil_end - sil_start >= min_silence:
            silence.append((sil_start, sil_end))

    last_end = speech_ts[-1]["end"] / silero_vad.SAMPLING_RATE
    trailing = total_duration - last_end
    if trailing >= min_silence:
        silence.append((last_end, last_end + trailing))
    return total_duration, silence


def _build_chunk_ranges(total_duration, silence_intervals, max_duration):
    """
    Determine chunk ranges capped at max_duration, preferring silence points.
    """
    epsilon = 1e-3
    if total_duration <= max_duration + epsilon:
        return [(0.0, total_duration)]

    silence_points = sorted([(start + end) / 2.0 for start, end in silence_intervals])
    chunk_ranges = []
    chunk_start = 0.0

    while chunk_start < total_duration - epsilon:
        limit = min(chunk_start + max_duration, total_duration)
        candidates = [p for p in silence_points if chunk_start + epsilon < p <= limit]
        chunk_end = candidates[-1] if candidates else limit
        if chunk_end - chunk_start < epsilon:
            chunk_end = limit
            if chunk_end - chunk_start < epsilon:
                break
        chunk_ranges.append((chunk_start, chunk_end))
        chunk_start = chunk_end

    if not chunk_ranges:
        chunk_ranges.append((0.0, total_duration))
    return chunk_ranges


def prepare_diarization_chunks(
    audio_path,
    audio_info,
    vad_model,
    silero_vad,
    max_duration=MAX_DIA_CHUNK_DURATION,
    min_silence=MIN_SPLIT_SILENCE,
):
    """
    Split long audio files prior to diarization using silence from VAD.
    Returns chunk metadata and optional temp directory for cleanup.
    """
    waveform = audio_info["waveform"]
    sample_rate = audio_info["sample_rate"]
    total_duration, silence_intervals = _build_silence_intervals(
        waveform, sample_rate, min_silence, vad_model, silero_vad
    )
    chunk_ranges = _build_chunk_ranges(total_duration, silence_intervals, max_duration)

    epsilon = 1e-3
    normalized_audio = audio_info.get("audio_segment")

    if (
        len(chunk_ranges) == 1
        and chunk_ranges[0][0] <= epsilon
        and abs(chunk_ranges[0][1] - total_duration) <= epsilon
    ):
        # Single chunk case: use normalized_audio if available, otherwise create temp mono file
        if normalized_audio is not None:
            # Create a temporary file with the normalized mono audio
            temp_dir = tempfile.mkdtemp(prefix="pre_diar_")
            temp_path = os.path.join(temp_dir, "full_audio.wav")
            normalized_audio.export(temp_path, format="wav", parameters=["-ac", "1"])
            return [{"path": temp_path, "offset": 0.0, "duration": total_duration}], temp_dir
        else:
            # Fallback: load and ensure mono
            temp_audio = AudioSegment.from_file(audio_path).set_channels(1)
            temp_dir = tempfile.mkdtemp(prefix="pre_diar_")
            temp_path = os.path.join(temp_dir, "full_audio.wav")
            temp_audio.export(temp_path, format="wav", parameters=["-ac", "1"])
            return [{"path": temp_path, "offset": 0.0, "duration": total_duration}], temp_dir


    if normalized_audio is None:
        normalized_audio = AudioSegment.from_file(audio_path)
        # Ensure mono audio for diarization
        normalized_audio = normalized_audio.set_channels(1)
    temp_dir = tempfile.mkdtemp(prefix="pre_diar_")
    chunk_entries = []

    for idx, (start_sec, end_sec) in enumerate(chunk_ranges):
        start_ms = max(0, int(round(start_sec * 1000)))
        end_ms = max(start_ms, int(round(end_sec * 1000)))
        chunk_audio = normalized_audio[start_ms:end_ms]
        chunk_path = os.path.join(temp_dir, f"chunk_{idx:03d}.wav")
        # Explicitly export as mono WAV with 24kHz sample rate
        chunk_audio.export(chunk_path, format="wav", parameters=["-ac", "1"])
        chunk_entries.append(
            {
                "path": chunk_path,
                "offset": start_sec,
                "duration": end_sec - start_sec,
            }
        )

    logger.info(
        f"Pre-diarization chunking created {len(chunk_entries)} chunks "
        f"(max {max_duration}s) from {os.path.basename(audio_path)}"
    )
    return chunk_entries, temp_dir
