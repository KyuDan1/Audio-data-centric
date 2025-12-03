# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
##################################
# CREDENTIAL 
##################################
import argparse
import json
import librosa
import numpy as np
import ast
import sys
import os
import shutil
import tqdm
import re
import warnings
import tempfile
from openai import OpenAI
import torch
import requests
from pyannote.audio.core.task import Specifications, Problem, Resolution

# 기존에 작성하신 TorchVersion 외에 pyannote 관련 클래스들도 추가합니다.
torch.serialization.add_safe_globals([
    torch.torch_version.TorchVersion,
    Specifications, 
    Problem, 
    Resolution
])
from pydub import AudioSegment
import os
from tritony import InferenceClient
import numpy as np
import librosa
from pyannote.audio import Pipeline
import pandas as pd
#from prompt import DIAR_PROMPT, WEAK_DIAR_PROMPT, NEW_DIAR_PROMPT, SPK_SUMMERIZE_PROMPT, NEW_DIAR_PROMPT_with_spk_inform, DIAR_PROMPT_KO
from utils.tool import (
    export_to_mp3,
    load_cfg,
    get_audio_files,
    detect_gpu,
    check_env,
    calculate_audio_stats,
)
from utils.logger import Logger, time_logger
from models import separate_fast, dnsmos, whisper_asr, silero_vad
import time
import datetime
from panns_inference import AudioTagging
import soundfile as sf

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.speechlm2.models import SALM

import json
import re
import argparse
from g2pk import G2p
import difflib
import collections
from typing import List, Tuple
from itertools import zip_longest


warnings.filterwarnings("ignore")
audio_count = 0
MAX_DIA_CHUNK_DURATION = 5 * 60  # 5 minutes
MIN_SPLIT_SILENCE = 1.0  # seconds of silence required for splitting
QWEN_3_OMNI_PORT = "11500"
class RoverEnsembler:
    """
    ROVER(Recognizer Output Voting Error Reduction) 앙상블 구현.
    여러 ASR 모델의 출력을 결합하여 더 정확한 전사를 생성합니다.
    """

    @staticmethod
    def align_and_vote(transcripts: List[str]) -> str:
        """
        여러 전사 결과를 정렬하고 다수결 투표를 수행합니다.

        Args:
            transcripts: ASR 모델들의 전사 결과 리스트 (예: [whisper, canary, parakeet])

        Returns:
            앙상블된 최종 전사 결과
        """
        if not transcripts:
            return ""

        # 빈 문자열 제거
        transcripts = [t.strip() for t in transcripts if t and t.strip()]
        if not transcripts:
            return ""

        if len(transcripts) == 1:
            return transcripts[0]

        # 단어 단위로 토큰화
        base_tokens = transcripts[0].split()  # Whisper
        candidate_1 = transcripts[1].split() if len(transcripts) > 1 else []  # Canary
        candidate_2 = transcripts[2].split() if len(transcripts) > 2 else []  # Parakeet

        final_output = []

        # 3개의 리스트를 동시에 순회하며 다수결 투표
        for w1, w2, w3 in zip_longest(base_tokens, candidate_1, candidate_2, fillvalue=""):
            votes = collections.Counter([w for w in [w1, w2, w3] if w])  # 빈 문자열 제외

            if not votes:
                continue

            # 최다 득표 단어 선정
            best_word, count = votes.most_common(1)[0]

            # 2개 이상이 동의하면 채택, 그렇지 않으면 첫 번째(Whisper) 채택
            if count >= 2:
                final_output.append(best_word)
            else:
                final_output.append(w1 if w1 else best_word)

        return " ".join(final_output)


class RepetitionFilter:
    """
    반복되는 n-gram을 감지하여 저품질 전사를 필터링합니다.
    논문: 15-gram이 5회 초과 등장하면 샘플을 제거합니다.
    """

    def __init__(self, use_mock_tokenizer=True):
        self.use_mock_tokenizer = use_mock_tokenizer

    def tokenize(self, text: str) -> List[str]:
        """단순 공백 기반 토큰화 (실제로는 SentencePiece 사용)"""
        if self.use_mock_tokenizer:
            return text.split()
        else:
            # 실제 구현 시 SentencePiece 사용
            pass

    def filter(self, text: str) -> bool:
        """
        필터링 조건:
        1. 빈 텍스트 제거
        2. 15-gram이 5회 초과 등장 시 제거

        Returns:
            bool: True이면 유지, False이면 제거
        """
        # 빈 텍스트 체크
        if not text or not text.strip():
            logger.debug(f"[RepetitionFilter] Empty text detected.")
            return False

        tokens = self.tokenize(text)

        # 15-gram 반복 체크
        N = 15
        THRESHOLD = 5

        if len(tokens) < N:
            return True  # 짧은 텍스트는 통과

        # n-gram 생성
        ngrams = [tuple(tokens[i:i+N]) for i in range(len(tokens) - N + 1)]

        # 빈도수 계산
        counts = collections.Counter(ngrams)

        # 5회 초과 체크
        for ngram, count in counts.items():
            if count > THRESHOLD:
                logger.debug(f"[RepetitionFilter] Repetition detected! Span '{' '.join(ngram[:3])}...' occurs {count} times.")
                return False

        return True


@time_logger
def standardization(audio):
    """
    Preprocess the audio file, including setting sample rate, bit depth, channels, and volume normalization.

    Args:
        audio (str or AudioSegment): Audio file path or AudioSegment object, the audio to be preprocessed.

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


# Step 2: Speaker Diarization
@time_logger
def detect_background_music(audio, threshold=0.3):
    """
    PANNs를 사용하여 배경음악이 있는지 검출합니다.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.
        threshold (float): Music 확률 임계값. 이 값 이상이면 배경음악이 있다고 판단.

    Returns:
        tuple: (has_music: bool, music_prob: float)
    """
    logger.debug("Detecting background music using PANNs")

    # PANNs는 32kHz 오디오를 기대하므로 리샘플링
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    if sample_rate != 32000:
        waveform_32k = librosa.resample(waveform, orig_sr=sample_rate, target_sr=32000)
    else:
        waveform_32k = waveform

    # Set custom PANNs data directory
    panns_data_dir = '/mnt/ddn/kyudan/panns_data'
    os.makedirs(panns_data_dir, exist_ok=True)

    # Set environment variable for PANNs to use custom directory
    os.environ['PANNS_DATA'] = panns_data_dir

    # PANNs inference with custom checkpoint path
    checkpoint_path = os.path.join(panns_data_dir, 'Cnn14_mAP=0.431.pth')
    at = AudioTagging(checkpoint_path=checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    (clipwise_output, embedding) = at.inference(waveform_32k[None, :])

    # Get labels
    labels = at.labels

    # Find Music probability
    music_idx = labels.index('Music') if 'Music' in labels else None
    if music_idx is not None:
        music_prob = float(clipwise_output[0, music_idx])
        logger.info(f"Music probability: {music_prob:.3f}")
        has_music = music_prob > threshold
        return has_music, music_prob
    else:
        logger.warning("Music label not found in PANNs output")
        return False, 0.0


def detect_segment_background_music(segment_audio, sample_rate, threshold=0.3):
    """
    세그먼트 오디오에 대해 배경음악이 있는지 검출합니다.

    Args:
        segment_audio (np.ndarray): 세그먼트 오디오 waveform.
        sample_rate (int): 샘플 레이트.
        threshold (float): Music 확률 임계값.

    Returns:
        tuple: (has_music: bool, music_prob: float)
    """
    # PANNs는 32kHz 오디오를 기대하므로 리샘플링
    if sample_rate != 32000:
        waveform_32k = librosa.resample(segment_audio, orig_sr=sample_rate, target_sr=32000)
    else:
        waveform_32k = segment_audio

    # PANNs 모델이 요구하는 최소 길이 체크 (약 1초 = 32000 samples)
    # Cnn14 모델의 pooling 계층을 통과하려면 최소한의 길이가 필요
    min_length = 32000  # 1 second at 32kHz
    if len(waveform_32k) < min_length:
        logger.warning(f"Segment too short for music detection ({len(waveform_32k)/32000:.2f}s < 1.0s), skipping music detection")
        return False, 0.0

    # Set custom PANNs data directory
    panns_data_dir = '/mnt/ddn/kyudan/panns_data'
    os.makedirs(panns_data_dir, exist_ok=True)
    os.environ['PANNS_DATA'] = panns_data_dir

    # PANNs inference with custom checkpoint path
    checkpoint_path = os.path.join(panns_data_dir, 'Cnn14_mAP=0.431.pth')
    at = AudioTagging(checkpoint_path=checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    (clipwise_output, embedding) = at.inference(waveform_32k[None, :])

    # Get labels
    labels = at.labels

    # Find Music probability
    music_idx = labels.index('Music') if 'Music' in labels else None
    if music_idx is not None:
        music_prob = float(clipwise_output[0, music_idx])
        has_music = music_prob > threshold
        return has_music, music_prob
    else:
        return False, 0.0


def remove_segment_background_music_demucs(segment_audio, sample_rate):
    """
    세그먼트 오디오에서 Demucs를 사용하여 배경음악을 제거하고 vocal만 추출합니다.

    Args:
        segment_audio (np.ndarray): 세그먼트 오디오 waveform.
        sample_rate (int): 샘플 레이트.

    Returns:
        np.ndarray: Vocal-only waveform 또는 실패 시 원본.
    """
    # Create temporary directory for demucs output
    temp_dir = tempfile.mkdtemp(prefix="demucs_seg_")

    try:
        # Save segment audio to temporary file
        temp_input = os.path.join(temp_dir, "segment.wav")
        sf.write(temp_input, segment_audio, sample_rate)

        # Run demucs to separate vocals
        import subprocess
        demucs_output_dir = os.path.join(temp_dir, "separated")

        cmd = [
            "python", "-m", "demucs.separate",
            "-n", "htdemucs",
            "--two-stems", "vocals",
            "-o", demucs_output_dir,
            temp_input
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Demucs failed for segment: {result.stderr}")
            return segment_audio

        # Load separated vocal track
        vocal_path = os.path.join(demucs_output_dir, "htdemucs", "segment", "vocals.wav")

        if not os.path.exists(vocal_path):
            logger.error(f"Vocal track not found at {vocal_path}")
            return segment_audio

        # Load vocal-only audio
        vocal_waveform, _ = librosa.load(vocal_path, sr=sample_rate, mono=True)
        return vocal_waveform.astype(np.float32)

    except Exception as e:
        logger.error(f"Error during segment Demucs processing: {e}")
        return segment_audio
    finally:
        # Cleanup temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@time_logger
def preprocess_segments_with_demucs(segment_list, audio, use_demucs=False):
    """
    ASR 전에 세그먼트별로 배경음악을 검출하고 demucs를 적용합니다.

    Args:
        segment_list (list): 세그먼트 리스트 (start, end, speaker 포함)
        audio (dict): 오디오 딕셔너리 (waveform, sample_rate 포함)
        use_demucs (bool): demucs 적용 여부

    Returns:
        tuple: (updated_audio, segment_demucs_flags)
            - updated_audio: demucs가 적용된 오디오 딕셔너리
            - segment_demucs_flags: 각 세그먼트의 demucs 적용 여부 리스트
    """
    if not use_demucs:
        logger.info("Demucs preprocessing skipped (flag disabled)")
        return audio, [False] * len(segment_list)

    logger.info(f"Preprocessing {len(segment_list)} segments with background music detection and removal")

    waveform = audio["waveform"].copy()
    sample_rate = audio["sample_rate"]
    segment_demucs_flags = []

    for idx, segment in enumerate(segment_list):
        start_time = segment["start"]
        end_time = segment["end"]
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)

        segment_audio = waveform[start_frame:end_frame]

        # Detect background music
        has_music, music_prob = detect_segment_background_music(segment_audio, sample_rate, threshold=0.3)
        logger.debug(f"Segment {idx} ({start_time:.2f}s - {end_time:.2f}s): Music probability = {music_prob:.3f}")

        if has_music:
            logger.info(f"Segment {idx}: Background music detected (prob={music_prob:.3f}), applying Demucs...")
            # Apply demucs to remove background music
            vocal_audio = remove_segment_background_music_demucs(segment_audio, sample_rate)
            # Replace the segment in the waveform
            waveform[start_frame:end_frame] = vocal_audio[:len(segment_audio)]  # Ensure same length
            segment_demucs_flags.append(True)
            logger.info(f"Segment {idx}: Demucs applied successfully")
        else:
            logger.debug(f"Segment {idx}: No significant background music (prob={music_prob:.3f})")
            segment_demucs_flags.append(False)

    # Update audio dictionary with processed waveform
    updated_audio = audio.copy()
    updated_audio["waveform"] = waveform

    # Also update audio_segment for export
    from pydub import AudioSegment as PydubAudioSegment
    # Convert numpy array to AudioSegment
    # Normalize to int16 range
    waveform_int16 = (waveform * 32767).astype(np.int16)
    updated_audio_segment = PydubAudioSegment(
        waveform_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    updated_audio["audio_segment"] = updated_audio_segment

    logger.info(f"Demucs preprocessing completed: {sum(segment_demucs_flags)}/{len(segment_list)} segments processed")

    return updated_audio, segment_demucs_flags


@time_logger
def speaker_diarization(audio):
    """
    Perform speaker diarization on the given audio.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.

    Returns:
        pd.DataFrame: A dataframe containing segments with speaker labels.
    """
    logger.debug(f"Start speaker diarization")
    logger.debug(f"audio waveform shape: {audio['waveform'].shape}")

    waveform = torch.tensor(audio["waveform"]).to(device)
    waveform = torch.unsqueeze(waveform, 0)

    segments = dia_pipeline(
        {
            "waveform": waveform,
            "sample_rate": audio["sample_rate"],
            "channel": 0,
        },
        max_speakers=4
    )

    diarize_df = pd.DataFrame(
        segments.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

    logger.debug(f"diarize_df: {diarize_df}")

    return diarize_df


@time_logger
def cut_by_speaker_label(vad_list):
    """
    Merge and trim VAD segments by speaker labels, enforcing constraints on segment length and merge gaps.

    Args:
        vad_list (list): List of VAD segments with start, end, and speaker labels.

    Returns:
        list: A list of updated VAD segments after merging and trimming.
    """
    MERGE_GAP = args.merge_gap  # merge gap in seconds, if smaller than this, merge
    MIN_SEGMENT_LENGTH = 3  # min segment length in seconds
    MAX_SEGMENT_LENGTH = 30  # max segment length in seconds

    updated_list = []

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None
        last_speaker = updated_list[-1]["speaker"] if updated_list else None

        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            current_start = vad["start"]
            segment_end = vad["end"]
            logger.warning(
                f"cut_by_speaker_label > segment longer than 30s, force trimming to 30s smaller segments"
            )
            while segment_end - current_start >= MAX_SEGMENT_LENGTH:
                vad["end"] = current_start + MAX_SEGMENT_LENGTH  # update end time
                updated_list.append(vad)
                vad = vad.copy()
                current_start += MAX_SEGMENT_LENGTH
                vad["start"] = current_start  # update start time
                vad["end"] = segment_end
            updated_list.append(vad)
            continue

        if (
            last_speaker is None
            or last_speaker != vad["speaker"]
            or vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
            continue

        if (
            vad["start"] - last_end_time >= MERGE_GAP
            or vad["end"] - last_start_time >= MAX_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
        else:
            updated_list[-1]["end"] = vad["end"]  # merge the time

    logger.debug(
        f"cut_by_speaker_label > merged {len(vad_list) - len(updated_list)} segments"
    )

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]

    logger.debug(
        f"cut_by_speaker_label > removed: {len(updated_list) - len(filter_list)} segments by length"
    )

    return filter_list


@time_logger
def asr(vad_segments, audio):
    """
    Perform Automatic Speech Recognition (ASR) on the VAD segments of the given audio.

    Args:
        vad_segments (list): List of VAD segments with start and end times.
        audio (dict): A dictionary containing the audio waveform and sample rate.

    Returns:
        list: A list of ASR results with transcriptions and language details.
    """
    if len(vad_segments) == 0:
        return []

    temp_audio = audio["waveform"]
    start_time = vad_segments[0]["start"]
    end_time = vad_segments[-1]["end"]
    start_frame = int(start_time * audio["sample_rate"])
    end_frame = int(end_time * audio["sample_rate"])
    temp_audio = temp_audio[start_frame:end_frame]  # remove silent start and end

    # update vad_segments start and end time (this is a little trick for batched asr:)
    for idx, segment in enumerate(vad_segments):
        vad_segments[idx]["start"] -= start_time
        vad_segments[idx]["end"] -= start_time

    # resample to 16k
    temp_audio = librosa.resample(
        temp_audio, orig_sr=audio["sample_rate"], target_sr=16000
    )

    if multilingual_flag:
        logger.debug("Multilingual flag is on")
        valid_vad_segments, valid_vad_segments_language = [], []
        # get valid segments to be transcripted
        for idx, segment in enumerate(vad_segments):
            start_frame = int(segment["start"] * 16000)
            end_frame = int(segment["end"] * 16000)
            segment_audio = temp_audio[start_frame:end_frame]
            language, prob = asr_model.detect_language(segment_audio)
            # 1. if language is in supported list, 2. if prob > 0.8
            if language in supported_languages and prob > 0.8:
                valid_vad_segments.append(vad_segments[idx])
                valid_vad_segments_language.append(language)

        # if no valid segment, return empty
        if len(valid_vad_segments) == 0:
            return []
        all_transcribe_result = []
        logger.debug(f"valid_vad_segments_language: {valid_vad_segments_language}")
        unique_languages = list(set(valid_vad_segments_language))
        logger.debug(f"unique_languages: {unique_languages}")
        # process each language one by one
        for language_token in unique_languages:
            language = language_token
            # filter out segments with different language
            vad_segments = [
                valid_vad_segments[i]
                for i, x in enumerate(valid_vad_segments_language)
                if x == language
            ]
            # bacthed trascription

            transcribe_result_temp = asr_model.transcribe(
                temp_audio,
                vad_segments,
                batch_size=batch_size,
                language=language,
                print_progress=True,
            )
            result = transcribe_result_temp["segments"]
            # restore the segment annotation
            for idx, segment in enumerate(result):
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result_temp["language"]
            all_transcribe_result.extend(result)
        # sort by start time
        all_transcribe_result = sorted(all_transcribe_result, key=lambda x: x["start"])
        return all_transcribe_result
    else:
        logger.debug("Multilingual flag is off")
        language, prob = asr_model.detect_language(temp_audio)
        if language in supported_languages and prob > 0.8:
            transcribe_result = asr_model.transcribe(
                temp_audio,
                vad_segments,
                batch_size=batch_size,
                language=language,
                print_progress=True,
            )
            result = transcribe_result["segments"]
            for idx, segment in enumerate(result):
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result["language"]
            return result
        else:
            return []

@time_logger
def asr_MoE(vad_segments, audio, segment_demucs_flags=None, enable_word_timestamps=False, device="cuda"):
    """
    Perform Automatic Speech Recognition (ASR) on the VAD segments of the given audio.

    Args:
        vad_segments (list): List of VAD segments with start and end times.
        audio (dict): A dictionary containing the audio waveform and sample rate.
        segment_demucs_flags (list): List of booleans indicating which segments had demucs applied.
        enable_word_timestamps (bool): Whether to enable WhisperX word-level timestamp alignment.
        device (str): Device to use for alignment ("cuda" or "cpu").

    Returns:
        tuple: (list of ASR results, whisper processing time in seconds, alignment time in seconds)
    """
    if len(vad_segments) == 0:
        return [], 0.0, 0.0

    if segment_demucs_flags is None:
        segment_demucs_flags = [False] * len(vad_segments)

    temp_audio = audio["waveform"]
    start_time = vad_segments[0]["start"]
    end_time = vad_segments[-1]["end"]
    start_frame = int(start_time * audio["sample_rate"])
    end_frame = int(end_time * audio["sample_rate"])
    temp_audio = temp_audio[start_frame:end_frame]  # remove silent start and end

    # update vad_segments start and end time (this is a little trick for batched asr:)
    for idx, segment in enumerate(vad_segments):
        vad_segments[idx]["start"] -= start_time
        vad_segments[idx]["end"] -= start_time

    # resample to 16k
    temp_audio = librosa.resample(
        temp_audio, orig_sr=audio["sample_rate"], target_sr=16000
    )

    if multilingual_flag:
        logger.debug("Multilingual flag is on")
        valid_vad_segments, valid_vad_segments_language = [], []
        # get valid segments to be transcripted
        for idx, segment in enumerate(vad_segments):
            start_frame = int(segment["start"] * 16000)
            end_frame = int(segment["end"] * 16000)
            segment_audio = temp_audio[start_frame:end_frame]
            language, prob = asr_model.detect_language(segment_audio)
            # 1. if language is in supported list, 2. if prob > 0.8
            if language in supported_languages and prob > 0.8:
                valid_vad_segments.append(vad_segments[idx])
                valid_vad_segments_language.append(language)

        # if no valid segment, return empty
        if len(valid_vad_segments) == 0:
            return [], 0.0, 0.0
        all_transcribe_result = []
        total_whisper_time = 0.0
        total_alignment_time = 0.0
        logger.debug(f"valid_vad_segments_language: {valid_vad_segments_language}")
        unique_languages = list(set(valid_vad_segments_language))
        logger.debug(f"unique_languages: {unique_languages}")
        # process each language one by one
        for language_token in unique_languages:
            language = language_token
            # filter out segments with different language
            vad_segments = [
                valid_vad_segments[i]
                for i, x in enumerate(valid_vad_segments_language)
                if x == language
            ]
            # bacthed trascription
            whisper_start_time = time.time()
            transcribe_result_temp = asr_model.transcribe(
                temp_audio,
                vad_segments,
                batch_size=batch_size,
                language=language,
                print_progress=True,
            )
            whisper_end_time = time.time()
            whisper_time = whisper_end_time - whisper_start_time
            total_whisper_time += whisper_time
            result = transcribe_result_temp["segments"]

            # WhisperX alignment for word-level timestamps
            if enable_word_timestamps:
                logger.info(f"Performing WhisperX alignment for language: {language}")
                alignment_start = time.time()
                aligned_result = asr_model.align_segments(
                    result,
                    temp_audio,
                    language,
                    device=device
                )
                alignment_end = time.time()
                alignment_time = alignment_end - alignment_start
                total_alignment_time += alignment_time
                result = aligned_result.get("segments", result)
                logger.info(f"Alignment completed in {alignment_time:.2f}s")

            # restore the segment annotation
            for idx, segment in enumerate(result):
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result_temp["language"]
                # Add speaker information from vad_segments
                if idx < len(vad_segments) and "speaker" in vad_segments[idx]:
                    result[idx]["speaker"] = vad_segments[idx]["speaker"]
                # Adjust word timestamps if present
                if enable_word_timestamps and "words" in result[idx]:
                    for word in result[idx]["words"]:
                        word["start"] += start_time
                        word["end"] += start_time
            all_transcribe_result.extend(result)
        # sort by start time
        all_transcribe_result = sorted(all_transcribe_result, key=lambda x: x["start"])
        return all_transcribe_result, total_whisper_time, total_alignment_time
    else:
        logger.debug("Multilingual flag is off")
        language, prob = asr_model.detect_language(temp_audio)
        logger.info(f"Detected language: {language} with probability: {prob}")
        logger.info(f"Supported languages: {supported_languages}")
        if language in supported_languages and prob > 0.8:
            # Whisper transcription
            whisper_start_time = time.time()
            transcribe_result = asr_model.transcribe(
                temp_audio,
                vad_segments,
                batch_size=batch_size,
                language=language,
                task="transcribe",
                print_progress=True,
            )
            whisper_end_time = time.time()
            whisper_time = whisper_end_time - whisper_start_time

            # Build result with both Whisper, Parakeet, and Canary outputs
            result = transcribe_result["segments"]

            # WhisperX alignment for word-level timestamps
            alignment_time = 0.0
            if enable_word_timestamps:
                logger.info(f"Performing WhisperX alignment for language: {language}")
                alignment_start = time.time()
                aligned_result = asr_model.align_segments(
                    result,
                    temp_audio,
                    language,
                    device=device
                )
                alignment_end = time.time()
                alignment_time = alignment_end - alignment_start
                result = aligned_result.get("segments", result)
                logger.info(f"Alignment completed in {alignment_time:.2f}s")

            rover = RoverEnsembler()

            for idx, segment in enumerate(result):
                # Extract segment audio for Parakeet and Canary
                seg_start_frame = int(segment["start"] * 16000)
                seg_end_frame = int(segment["end"] * 16000)
                segment_audio = temp_audio[seg_start_frame:seg_end_frame]

                # Parakeet transcription for this segment
                transcribe_result_parakeet = asr_model_2.transcribe(segment_audio)
                result_parakeet = transcribe_result_parakeet[0].text if transcribe_result_parakeet else ""

                # Canary transcription for this segment
                # Save segment audio to temporary file for Canary
                temp_wav_path = f"/tmp/temp_segment_{idx}.wav"
                import soundfile as sf
                sf.write(temp_wav_path, segment_audio, 16000)

                answer_ids = canary_model.generate(
                    prompts=[
                        [{"role": "user", "content": f"Transcribe the following: {canary_model.audio_locator_tag}", "audio": [temp_wav_path]}]
                    ],
                    max_new_tokens=128,
                )
                result_canary = canary_model.tokenizer.ids_to_text(answer_ids[0].cpu())

                # Clean up temporary file
                import os
                os.remove(temp_wav_path)

                # ROVER ensemble: combine whisper, canary, parakeet
                text_whisper = segment["text"]
                ensembled_text = rover.align_and_vote([text_whisper, result_canary, result_parakeet])

                # Update result with all model outputs
                result[idx]["text_whisper"] = text_whisper
                result[idx]["text_canary"] = result_canary
                result[idx]["text_parakeet"] = result_parakeet
                result[idx]["text_ensemble"] = ensembled_text  # ROVER ensembled result
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result["language"]
                # Add speaker information from vad_segments
                if idx < len(vad_segments) and "speaker" in vad_segments[idx]:
                    result[idx]["speaker"] = vad_segments[idx]["speaker"]
                # Add demucs flag from preprocessing
                result[idx]["demucs"] = segment_demucs_flags[idx] if idx < len(segment_demucs_flags) else False
                # Adjust word timestamps if present
                if enable_word_timestamps and "words" in result[idx]:
                    for word in result[idx]["words"]:
                        word["start"] += start_time
                        word["end"] += start_time
            return result, whisper_time, alignment_time
        else:
            return [], 0.0, 0.0

def add_qwen3omni_caption(filtered_list, audio, save_path):
    """
    ASR 결과의 각 세그먼트에 대해 Qwen3-Omni API를 호출하여 audio caption을 추가합니다.

    Args:
        filtered_list (list): ASR 결과 세그먼트 리스트
        audio (dict): 오디오 딕셔너리 (waveform, sample_rate 포함)
        save_path (str): 임시 오디오 파일을 저장할 경로

    Returns:
        tuple: (qwen3omni_caption이 추가된 세그먼트 리스트, 처리 시간(초))
    """
    import soundfile as sf

    logger.info(f"Adding Qwen3-Omni captions to {len(filtered_list)} segments...")
    caption_start_time = time.time()

    for idx, segment in enumerate(filtered_list):
        try:
            # 세그먼트 오디오 추출
            start_time = segment["start"]
            end_time = segment["end"]
            sample_rate = audio["sample_rate"]
            start_frame = int(start_time * sample_rate)
            end_frame = int(end_time * sample_rate)
            segment_audio = audio["waveform"][start_frame:end_frame]

            # 임시 오디오 파일로 저장
            temp_audio_path = os.path.join(save_path, f"temp_segment_{idx:05d}.wav")
            sf.write(temp_audio_path, segment_audio, sample_rate)

            # Qwen3-Omni API 호출
            url = f"http://localhost:{QWEN_3_OMNI_PORT}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}

            # 로컬 파일을 base64로 인코딩하거나 URL로 제공해야 합니다
            # 여기서는 임시 파일 경로를 사용 (실제로는 서버가 접근 가능한 URL이 필요할 수 있음)
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio_url",
                                "audio_url": {"url": f"file://{temp_audio_path}"}
                            }
                        ]
                    }
                ]
            }

            response = requests.post(url, headers=headers, json=data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                # Extract content from response
                caption = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                segment["qwen3omni_caption"] = caption
                logger.debug(f"Segment {idx}: Successfully added Qwen3-Omni caption")
            else:
                logger.warning(f"Segment {idx}: API call failed with status {response.status_code}")
                segment["qwen3omni_caption"] = ""

            # 임시 파일 삭제
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        except Exception as e:
            logger.error(f"Segment {idx}: Error calling Qwen3-Omni API: {e}")
            segment["qwen3omni_caption"] = ""

    caption_end_time = time.time()
    caption_processing_time = caption_end_time - caption_start_time

    logger.info("Qwen3-Omni caption addition completed")
    return filtered_list, caption_processing_time

# 비용 계산 함수
def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing = {
        "gpt-4.1": {
            "input": 2.00 / 1_000_000,
            "cached_input": 0.50 / 1_000_000,
            "output": 8.00 / 1_000_000,
        },
        "gpt-4.1-mini": {
            "input": 0.40 / 1_000_000,
            "cached_input": 0.10 / 1_000_000,
            "output": 1.60 / 1_000_000,
        },
        "gpt-4.1-nano": {
            "input": 0.10 / 1_000_000,
            "cached_input": 0.025 / 1_000_000,
            "output": 0.40 / 1_000_000,
        },
        "openai-o3": {
            "input": 2.00 / 1_000_000,
            "cached_input": 0.50 / 1_000_000,
            "output": 8.00 / 1_000_000,
        },
        "openai-o4-mini": {
            "input": 1.10 / 1_000_000,
            "cached_input": 0.275 / 1_000_000,
            "output": 4.40 / 1_000_000,
        },
    }

    if model_name not in pricing:
        raise ValueError(f"Model '{model_name}' not found in pricing table.")

    rates = pricing[model_name]
    input_cost = input_tokens * rates["input"]
    output_cost = output_tokens * rates["output"]
    total_cost = input_cost + output_cost

    return total_cost
import json
from collections import defaultdict



def speaker_tagged_text(data):
    """
    주어진 데이터에 화자 태그를 추가하고, 텍스트에 나타나는 순서대로
    화자 번호를 s0, s1, s2... 순으로 다시 매깁니다.
    """
    # 1. 초기 태그를 생성하고, 고유 화자가 나타나는 순서를 기록합니다.
    initially_tagged_data = []
    unique_speakers_in_order = []
    seen_speakers = set()

    for item in data:
        # 'SPEAKER_01' -> '[s1]' 형식으로 원래 태그 생성
        speaker_num = item['speaker'].replace('SPEAKER_', '')
        original_tag = f"[s{int(speaker_num)}]"

        # 새로운 고유 화자 태그가 나타나면 순서대로 리스트에 추가
        if original_tag not in seen_speakers:
            unique_speakers_in_order.append(original_tag)
            seen_speakers.add(original_tag)
        
        # 나중에 매핑을 적용하기 위해 원본 텍스트와 태그를 임시 저장
        initially_tagged_data.append({
            'text': item['text'],
            'start': item['start'],
            'end': item['end'],
            'original_tag': original_tag
        })

    # 2. 원래 태그를 새 순차 태그로 변환하는 매핑(규칙)을 생성합니다.
    # 예: {'[s2]': '[s0]', '[s0]': '[s1]', '[s1]': '[s2]'}
    speaker_map = {
        original_tag: f"[s{i}]" 
        for i, original_tag in enumerate(unique_speakers_in_order)
    }

    # 3. 생성된 매핑을 적용하여 최종 결과를 만듭니다.
    result = []
    for item in initially_tagged_data:
        original_tag = item['original_tag']
        new_tag = speaker_map[original_tag]  # 매핑에서 새 태그 가져오기
        
        final_item = {
            'text': f"{new_tag}{item['text']}",  # 새 태그를 텍스트 앞에 추가
            'start': item['start'],
            'end': item['end']
        }
        result.append(final_item)
        
    return result

import re
import json
import ast

def parse_speaker_summary(llm_output: str) -> list | None:
    """
    LLM이 출력한 문자열에서 JSON 배열을 추출하고 파싱합니다.
    'json' 접두사, 코드 블록(```), 앞뒤 공백 등을 처리합니다.
    """
    if not llm_output:
        return None

    try:
        # ```json ... ``` 또는 ``` ... ``` 같은 코드 블록 제거
        # 정규 표현식을 사용하여 대괄호 '[' 와 ']' 사이의 내용을 찾음
        match = re.search(r'\[.*\]', llm_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            # JSON 문자열을 파이썬 객체로 변환 (list of dicts)
            return json.loads(json_str)
        else:
            print("Parsing Error: 유효한 JSON 배열 형식([])을 찾을 수 없습니다.")
            return None
            
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 에러: {e}")
        return None
    except Exception as e:
        print(f"알 수 없는 파싱 에러: {e}")
        return None

def process_llm_diarization_output(llm_output: str) -> list[dict]:

    # 1. LLM 출력에서 ```json ... ``` 코드 블록 찾기
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_output)
    if not json_match:
        # 만약 ```json 블록이 없다면, 문자열 전체를 파싱 시도
        json_string = llm_output
    else:
        json_string = json_match.group(1)

    # 2. JSON 문자열을 파이썬 객체로 파싱
    try:
        llm_data = json.loads(json_string)
    except json.JSONDecodeError:
        # LLM이 Python 리스트 형식('[{"text":...}]')으로 출력했을 경우를 대비
        try:
            # ast.literal_eval은 보안에 더 안전한 eval 버전입니다.
            import ast
            llm_data = ast.literal_eval(json_string)
        except (ValueError, SyntaxError) as e:
            print(f"오류: JSON 및 Python 리터럴 파싱에 모두 실패했습니다. {e!r}")
            return []


    return llm_data


# def llm_speaker_summerize(chunk):
#     max_retries = 3
#     spk_inform_list = None  # 최종 결과를 저장할 변수

#     for attempt in range(1, max_retries + 1):
#         try:
#             input_text = SPK_SUMMERIZE_PROMPT + str(chunk)
#             #print("="*30)
#             #print(input_text)
#             # 1. API 요청
#             response = client.responses.create(
#                 model=model_name,
#                 temperature=0.2,
#                 input=input_text
#             )
#             #print("*"*30)
#             #print("RESPONSE")
#             #print(response.output_text)

#             print(f"--- [Attempt {attempt}/{max_retries}] LLM Response---")
#             spk_inform_list = parse_speaker_summary(response.output_text)
#             if not spk_inform_list:
#                 print(f"speaker summerize Parsing failed: The function returned an empty list. Retrying...")
#                 continue
            

#             print(f"Successfully parsed and generated segments on attempt {attempt}.")
#             break 

#         except Exception as e:
#             print(f"[speaker summerize  Attempt {attempt}/{max_retries}] An error occurred: {e!r}. Retrying...")

#     if spk_inform_list is None:
#         raise RuntimeError(f"speaker summerize Failed to parse and build segments after {max_retries} attempts")
#     return spk_inform_list




# def llm_inference(asr_result, spk_inform = None, spk_inf_turn = False):

#     max_retries = 1
#     filtered_list = None  # 최종 결과를 저장할 변수

#     if asr_result[0]['language'] == "ko":
#         print("한국어.")
#         kor = True 

#     for attempt in range(1, max_retries + 1):
#         try:
#             # 1. API 요청
#             if spk_inf_turn:
#                 response = client.responses.create(
#                 model=model_name,
#                 temperature=0.2,
#                 input=NEW_DIAR_PROMPT_with_spk_inform.format(spk_inform = str(spk_inform)) + str(speaker_tagged_text(asr_result))
#                 )


#             else:
#                 print("api 요청 중")
#                 response = client.responses.create(
#                 model=model_name,
#                 temperature=0.2,
#                 input=(DIAR_PROMPT_KO+ str(speaker_tagged_text(asr_result))) if kor else (NEW_DIAR_PROMPT + str(speaker_tagged_text(asr_result)))
#                 )
#                 print("받음")

#             print(f"--- [Attempt {attempt}/{max_retries}] LLM Response Success---")
#             output_list = process_llm_diarization_output(response.output_text)
#             if not output_list:
#                 print(f"Parsing failed: The function returned an empty list. Retrying...")
#                 continue
#             filtered_list = output_list

#             print(f"Successfully parsed and generated segments on attempt {attempt}.")
#             break 

#         except Exception as e:
#             print(f"[Attempt {attempt}/{max_retries}] An error occurred: {e!r}. Retrying...")

#     if filtered_list is None:
#         raise RuntimeError(f"Failed to parse and build segments after {max_retries} attempts")
#     return filtered_list

def sortformer_dia(predicted_segments):
    lists = [x for x in predicted_segments if isinstance(x, (list, tuple))]
    if not lists:
        lists = predicted_segments
    segs = [s for sub in lists for s in sub]

    rows = []
    for idx, seg in enumerate(segs):
        start_s, end_s, sp = seg.split()
        start, end = float(start_s), float(end_s)
        # SPEAKER 형식 변환
        num = int(sp.split('_')[1])
        speaker = f"SPEAKER_{num:02d}"
        # 레이블(A, B, C, ...)
        label = chr(ord('A') + idx)
        # 시간 포맷 함수
        def fmt(sec):
            td = datetime.timedelta(seconds=sec)
            hrs = td.seconds // 3600 + td.days * 24
            mins = (td.seconds // 60) % 60
            secs = td.seconds % 60
            ms = int(td.microseconds / 1000)
            return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"
        segment_str = f"[ {fmt(start)} --> {fmt(end)}]"
        rows.append({
            'segment': segment_str,
            'label': label,
            'speaker': speaker,
            'start': start,
            'end': end
        })

    df = pd.DataFrame(rows, columns=['segment','label','speaker','start','end'])
    df = df.sort_values(by='start').reset_index(drop=True)
    return df

def df_to_list(df: pd.DataFrame) -> list[dict]:
    """
    DataFrame의 각 행을 아래 형식의 dict로 변환한 리스트를 반환합니다.
      - index: 5자리 0패딩 문자열
      - start: float
      - end: float
      - speaker: str
    """
    records = []
    for i, row in df.iterrows():
        records.append({
            'index': f"{i:05d}",
            'start': float(row['start']),
            'end': float(row['end']),
            'speaker': row['speaker']
        })
    return records

def split_long_segments(segment_list, max_duration=30.0):
    """
    세그먼트 리스트에서 max_duration보다 긴 세그먼트를
    시간 기준으로 분할합니다. (VAD 사용 안 함)

    Args:
        segment_list (list): 분할할 세그먼트 딕셔너리의 리스트.
        max_duration (float): 세그먼트의 최대 허용 길이 (초).

    Returns:
        list: 분할이 완료된 새로운 세그먼트 리스트.
    """
    new_segments = []
    new_index = 0

    for segment in segment_list:
        start_time = segment['start']
        end_time = segment['end']
        speaker = segment['speaker']
        duration = end_time - start_time

        # 세그먼트 길이가 최대 길이보다 짧거나 같으면 그대로 추가
        if duration <= max_duration:
            segment['index'] = str(new_index).zfill(5)
            new_segments.append(segment)
            new_index += 1
        # 세그먼트 길이가 최대 길이보다 길면 분할
        else:
            current_start = start_time
            # 현재 시작 시간이 원본 세그먼트의 종료 시간보다 작을 때까지 반복
            while current_start < end_time:
                # 다음 분할 지점 계산 (최대 길이를 더하거나, 원본 종료 시간을 넘지 않도록)
                chunk_end = min(current_start + max_duration, end_time)
                
                new_segments.append({
                    'index': str(new_index).zfill(5),
                    'start': round(current_start, 3),
                    'end': round(chunk_end, 3),
                    'speaker': speaker
                })
                new_index += 1
                # 다음 청크의 시작 시간을 현재 청크의 종료 시간으로 업데이트
                current_start = chunk_end
                
    return new_segments


def _build_silence_intervals(waveform, sample_rate, min_silence):
    """
    Use VAD to find silence intervals that can be used as cut points.
    """
    vad_model = globals().get("vad")
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
        waveform, sample_rate, min_silence
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

def ko_transliterate_english(text: str) -> str:
    """
    입력 문자열에서 영어 구간만 찾아 한글 발음으로 변환합니다.
    """
    def _repl(m: re.Match) -> str:
        segment = m.group(0)
        return G2P(segment)
    return ENG_PATTERN.sub(_repl, text)


def ko_process_json(input_list: str) -> None:
    for entry in input_list:
        text = entry.get("text", "")
        # 영어 포함 시 변환
        if re.search(r"[A-Za-z]", text):
            entry["text"] = ko_transliterate_english(text)


        
def main_process(audio_path, save_path=None, audio_name=None,
                 do_vad = False,
                 LLM = "",
                 use_demucs = False):

    if not audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac")):
        logger.warning(f"Unsupported file type: {audio_path}")
    # 오디오 convert_mono and split_wav_files_2min temp file 만들기.

    # for a single audio from path Ïaaa/bbb/ccc.wav ---> save to aaa/bbb_processed/ccc/ccc_0.wav
    audio_name = audio_name or os.path.splitext(os.path.basename(audio_path))[0]
    suffix = "dia3" if args.dia3 else "ori"
    save_path = save_path or os.path.join(
        os.path.dirname(audio_path), "_final", "_processed_llm-twelve-cases" + f"-vad-{do_vad}"+ f"-diaModel-{suffix}"
        # initial prompt off or on
        + f"-initPrompt-{args.initprompt}"
        + f"-merge_gap-{args.merge_gap}" +f"-seg_th-{args.seg_th}"+ f"-cl_min-{args.min_cluster_size}" +f"-cl-th-{args.clust_th}"+ f"-LLM-{LLM}", audio_name
    )
    os.makedirs(save_path, exist_ok=True)
    logger.debug(
        f"Processing audio: {audio_name}, from {audio_path}, save to: {save_path}"
    )

    logger.info(
        "Step 0: Preprocess all audio files --> 24k sample rate + wave format + loudnorm + bit depth 16"
    )
    audio = standardization(audio_path)
    diar_chunks, temp_chunk_dir = prepare_diarization_chunks(audio_path, audio)

    # Calculate total audio duration
    audio_duration = len(audio["waveform"]) / audio["sample_rate"]
    logger.info(f"Total audio duration: {audio_duration:.2f} seconds")

    logger.info("Step 2: Speaker Diarization")
    dia_start = time.time()


    diarization_frames = []
    try:
        for chunk in diar_chunks:
            predicted_segments, _ = diar_model.diarize(
                audio=chunk["path"], batch_size=1, include_tensor_outputs=True
            )
            chunk_df = sortformer_dia(predicted_segments)
            if not chunk_df.empty:
                chunk_df["start"] += chunk["offset"]
                chunk_df["end"] += chunk["offset"]
            diarization_frames.append(chunk_df)
    finally:
        if temp_chunk_dir:
            shutil.rmtree(temp_chunk_dir, ignore_errors=True)

    if diarization_frames:
        speakerdia = pd.concat(diarization_frames, ignore_index=True)
    else:
        speakerdia = pd.DataFrame(columns=["segment", "label", "speaker", "start", "end"])
    ori_list = df_to_list(speakerdia)
    dia_end = time.time()

    # Calculate VAD + Sortformer RT factor
    vad_sortformer_processing_time = dia_end - dia_start
    vad_sortformer_rt = vad_sortformer_processing_time / audio_duration if audio_duration > 0 else 0
    logger.info(f"VAD + Sortformer - Processing time: {vad_sortformer_processing_time:.2f}s, RT factor: {vad_sortformer_rt:.4f}")

    # TEST
    ######################
    segment_list = ori_list
    segment_list = split_long_segments(segment_list)
    ######################

    # Step 3: Background Music Detection and Removal (preprocess segments before ASR)
    logger.info("Step 3: Background Music Detection and Removal")
    audio, segment_demucs_flags = preprocess_segments_with_demucs(segment_list, audio, use_demucs=use_demucs)

    logger.info("Step 4: ASR (Automatic Speech Recognition)")
    if args.ASRMoE:
        asr_start = time.time()

        asr_result, whisper_time, alignment_time = asr_MoE(
            segment_list,
            audio,
            segment_demucs_flags=segment_demucs_flags,
            enable_word_timestamps=args.whisperx_word_timestamps,
            device=device_name
        )

        asr_end = time.time()

        dia_time = dia_end-dia_start
        asr_time = whisper_time
    else:
        asr_start = time.time()

        asr_result = asr(segment_list, audio)

        asr_end = time.time()

        dia_time = dia_end-dia_start
        asr_time = asr_end-asr_start
        alignment_time = 0.0

    # Calculate Whisper large v3 RT factor
    whisper_processing_time = asr_time
    whisper_rt = whisper_processing_time / audio_duration if audio_duration > 0 else 0

    # Calculate WhisperX alignment RT factor
    alignment_rt = alignment_time / audio_duration if audio_duration > 0 else 0

    if LLM == "case_0":
        print("LLM case_0")
        filtered_list = asr_result
        print(f"ASR result contains {len(filtered_list)} segments")




    # LLM post diarization start
    ####################################################################################################
    # "LLM 불러서 post-processing 하는 것"
    elif LLM == "case_2":
        print(f"asr_result len: {len(asr_result)}")
        print("Warning: llm_inference functions are commented out. Using ASR results directly.")
        filtered_list = asr_result
        # if len(asr_result) <200:
        #     print("asr_result is less than 200. llm inference start.")
        #     filtered_list = llm_inference(asr_result)
        # else:
        #     print("asr_result is larger than 200. chunking in 200.")
        #     chunk_asr_result = []
        #     chunk_filtered_list = []
        #     for i in range(0, len(asr_result), 200):
        #         chunk = asr_result[i:i+200]
        #         chunk_asr_result.append(chunk)
        #     print(f"number of chunk_list: {len(chunk_asr_result)}")
        #     for i, chunk_200 in enumerate(chunk_asr_result):
        #
        #         # 첫번째 청크에 대해서는 바로 llm에 입력.
        #         if i==0:
        #             print("first chunk llm_inference")
        #             chunk_filtered_list.append(llm_inference(chunk_200))
        #         else:
        #             # 직전까지의 모든 chunk로 speaker 정보 뽑고 (llm input length 제한 (context length)는 1M 토큰. 24시간 이상 오디오 분량임.)
        #             print("summerizing speaker identity.")
        #             #print(chunk_filtered_list[:i])
        #             spk_inform = llm_speaker_summerize(chunk_filtered_list[:i])
        #             # 직전 chunk의 speaker 정보와 현재 chunk 같이 넣어서 inference.
        #             chunk_filtered_list.append(llm_inference(chunk_200, spk_inform, spk_inf_turn=True))

        #     filtered_list=[]
        #     for sublist in chunk_filtered_list:
        #         filtered_list.extend(sublist)

    else:
        raise ValueError("LLM 변수는 case_0, case_1, case_2 중 하나여야 한다.")

    ############################################################################################################
        # LLM post diarization end

    # Step 4.5: Add Qwen3-Omni captions (if enabled)
    caption_time = 0.0
    if args.qwen3omni:
        logger.info("Step 4.5: Adding Qwen3-Omni captions")
        filtered_list, caption_time = add_qwen3omni_caption(filtered_list, audio, save_path)
    else:
        logger.info("Step 4.5: Qwen3-Omni caption generation skipped (flag disabled)")

    # Calculate Qwen3-Omni RT factor
    caption_rt = caption_time / audio_duration if audio_duration > 0 else 0

    # Print all timing information
    print(f"\n{'='*60}")
    print(f"Audio duration: {audio_duration:.2f} seconds ({audio_duration/60:.2f} minutes)")
    print(f"{'='*60}")
    print(f"VAD + Sortformer:")
    print(f"  - Processing time: {dia_time:.2f} seconds")
    print(f"  - RT factor: {vad_sortformer_rt:.4f}")
    print(f"{'='*60}")
    print(f"Whisper large v3:")
    print(f"  - Processing time: {asr_time:.2f} seconds")
    print(f"  - RT factor: {whisper_rt:.4f}")
    print(f"{'='*60}")
    if args.whisperx_word_timestamps:
        print(f"WhisperX Alignment:")
        print(f"  - Processing time: {alignment_time:.2f} seconds")
        print(f"  - RT factor: {alignment_rt:.4f}")
        print(f"{'='*60}")
    if args.qwen3omni:
        print(f"Qwen3-Omni Caption:")
        print(f"  - Processing time: {caption_time:.2f} seconds")
        print(f"  - RT factor: {caption_rt:.4f}")
        print(f"{'='*60}")
    print()

    logger.info("Step 5: Write result into MP3 and JSON file")
    print(f"Exporting {len(filtered_list)} segments to MP3 and JSON...")
    export_to_mp3(audio, filtered_list, save_path, audio_name)


    # 한국어 g2p 후처리
    if args.korean:
        ko_process_json(filtered_list)

    # Prepare output with RT factor metrics
    output_data = {
        "metadata": {
            "audio_duration_seconds": audio_duration,
            "audio_duration_minutes": audio_duration / 60,
            "vad_sortformer": {
                "processing_time_seconds": vad_sortformer_processing_time,
                "rt_factor": vad_sortformer_rt
            },
            "whisper_large_v3": {
                "processing_time_seconds": whisper_processing_time,
                "rt_factor": whisper_rt
            },
            "total_segments": len(filtered_list)
        },
        "segments": filtered_list
    }

    # Add WhisperX alignment metadata if enabled
    if args.whisperx_word_timestamps:
        output_data["metadata"]["whisperx_alignment"] = {
            "processing_time_seconds": alignment_time,
            "rt_factor": alignment_rt,
            "enabled": True
        }

    # Add Qwen3-Omni caption metadata if enabled
    if args.qwen3omni:
        output_data["metadata"]["qwen3omni_caption"] = {
            "processing_time_seconds": caption_time,
            "rt_factor": caption_rt,
            "enabled": True
        }

    final_path = os.path.join(save_path, audio_name + ".json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"All done, Saved to: {final_path}")
    print(f"Processing complete! Results saved to: {final_path}")
    print(f"Total segments processed: {len(filtered_list)}")
    return final_path, filtered_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default="",
        help="input folder path, this will override config if set",
    )
    parser.add_argument(
        "--config_path", type=str, default="config.json", help="config path"
    )
    
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    # 32 정도 괜찮을듯?

    parser.add_argument(
        "--compute_type",
        type=str,
        default="float16",
        help="The compute type to use for the model",
    )
    parser.add_argument(
        "--whisper_arch",
        type=str,
        default="large-v3",
        help="The name of the Whisper model to load.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="The number of CPU threads to use per worker, e.g. will be multiplied by num workers.",
    )
    parser.add_argument(
        "--exit_pipeline",
        type=bool,
        default=False,
        help="Exit pipeline when task done.",
    )
    parser.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Turning on vad.",
    )
    parser.add_argument(
        "--dia3",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Turning on diaization 3.0 model",
    )
    parser.add_argument(
        "--LLM",
        type=str,
        default="case_2",
        help="LLM diarization cases",
    )

    # hyperparameter
    parser.add_argument(
        "--seg_th",
        type=float,
        default=0.15,
        help="diarization model segmentation threshold",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=10,
        help="diarization model clustering min_cluster_size",
    )
    parser.add_argument(
        "--clust_th",
        type=float,
        default=0.5,
        help="diarization model clustering threshold",
    )

    parser.add_argument(
        "--merge_gap",
        type=float,
        default=2,
        help="merge gap in seconds, if smaller than this, merge",
    )

    parser.add_argument(
        "--initprompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Turning on initial prompt on whisper model",
    )
    parser.add_argument(
        "--korean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="korean_g2p",
    )

    parser.add_argument(
        "--ASRMoE",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="parakeet",
    )

    parser.add_argument(
        "--demucs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable background music detection and removal using PANNs and Demucs",
    )

    parser.add_argument(
        "--whisperx_word_timestamps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable WhisperX word-level timestamps with alignment",
    )

    parser.add_argument(
        "--qwen3omni",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen3-Omni audio captioning for each segment",
    )


    args = parser.parse_args()

    batch_size = args.batch_size
    cfg = load_cfg(args.config_path)

    logger = Logger.get_logger()

    if args.input_folder_path:
        logger.info(f"Using input folder path: {args.input_folder_path}")
        cfg["entrypoint"]["input_folder_path"] = args.input_folder_path

    logger.debug("Loading models...")

    # Load models
    if detect_gpu():
        logger.info("Using GPU")
        device_name = "cuda"
        device = torch.device(device_name)
    else:
        logger.info("Using CPU")
        device_name = "cpu"
        device = torch.device(device_name)
        # whisperX expects compute type: int8
        logger.info("Overriding the compute type to int8")
        args.compute_type = "int8"

    check_env(logger)

    # Speaker Diarization
    logger.debug(" * Loading Speaker Diarization Model")
    if not cfg["huggingface_token"].startswith("hf"):
        raise ValueError(
            "huggingface_token must start with 'hf', check the config file. "
            "You can get the token at https://huggingface.co/settings/tokens. "
            "Remeber grant access following https://github.com/pyannote/pyannote-audio?tab=readme-ov-file#tldr"
        )
    if args.dia3 == True:
        print("Using diarization-3.1 model")
        dia_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        #"pyannote/speaker-diarization",
        use_auth_token=cfg["huggingface_token"],

    )
        dia_pipeline.to(device)
        
    else:
        dia_pipeline = Pipeline.from_pretrained(
            #"pyannote/speaker-diarization-3.1",
            "pyannote/speaker-diarization",
            use_auth_token=cfg["huggingface_token"]
        )
        dia_pipeline.to(device)

        # hyperparameters
        dia_pipeline.instantiate({
        "segmentation": {
            "min_duration_off": 0.0, 
            "threshold": args.seg_th
        },
        "clustering": {
            "method": "centroid",
            "min_cluster_size": args.min_cluster_size,
            "threshold": args.clust_th   
        }
    })
    # ASR
    logger.debug(" * Loading ASR Model")

    if args.initprompt == True:
        asr_options_dict = {
            #"log_prob_threshold": -1.0,
            #"no_speech_threshold": 0.6,
            # 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음

            # 원래 코드의 initial prompt
            # "initial_prompt": "ha. heh. Mm, hmm. Mm hm. uh. Uh huh. Mm huh. Uh. hum Uh. Ah. Uh hu. Like. you know. Yeah. I mean. right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. Hoo. Hu. Hoo, hoo. Heah. Ha. Yu. Nah. Uh-huh. No way. Uh-oh. Jeez. Whoa. Dang. Gosh. Duh. Whoops. Phew. Woo. Ugh. Er. Geez. Oh wow. Oh man. Uh yeah. Uh huh. For real?",

            #"initial_prompt": "ha. heh. Mm, hmm. uh. "
            #"initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. Hoo. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음.",

            # 아예 initial_prompt 안 넣어 줄 때: ASR이 안 되는 구간이 생기진 않지만 추임새 인식이 잘 안 됨.
            # 원래 코드의 initial prompt를 넣어 줄 때:  ASR이 안 되는 구간이 생기진 않지만 '원하는' 추임새 인식이 잘 안 됨.
            # 완전히 다른 initial prompt를 넣어 줄 때 (뒤에 일본어 한자 다 지울 때): ASR이 안 되는 구간이 드문 드문 생김.
            # 원래 코드의 initial prompt에서 원하는 추임새만 3단어 이하로 수정할 때: 적당히 추가하면 ASR이 안 되는 구간이 생기지 않으면서 원하는 추임새 인식이 잘 됨.


            "initial_prompt": "Um. Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. Mm. So. Oh. Hoo hoo.生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음.",

        }
        # Add word_timestamps if flag is enabled
        if args.whisperx_word_timestamps:
            asr_options_dict["word_timestamps"] = True

        asr_model = whisper_asr.load_asr_model(
            "large-v3",
            device_name,
            compute_type=args.compute_type,
            threads=args.threads,
            language="en", # 언어 지정 한국어.

        # whisper_asr.py 의 default_asr_options 수정으로 asr 모델 수정 가능.

            asr_options=asr_options_dict,
        )
    else:
        asr_options_dict = {}
        # Add word_timestamps if flag is enabled
        if args.whisperx_word_timestamps:
            asr_options_dict["word_timestamps"] = True

        asr_model = whisper_asr.load_asr_model(
            "large-v3",
            device_name,
            compute_type=args.compute_type,
            threads=args.threads,

            language="en",
            asr_options=asr_options_dict if asr_options_dict else None,

            )
    if args.ASRMoE:
        import nemo.collections.asr as nemo_asr
        asr_model_2 = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

        # Load Canary model
        logger.debug(" * Loading Canary Model")
        canary_model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
        canary_model = canary_model.to(device)
        canary_model.eval()
        logger.debug(f" * Canary model loaded on {device}")
        # 클라이언트 초기화
    #client = OpenAI(api_key=OPENAI_API_KEY)
    model_name = "gpt-4.1"
    # VAD
    logger.debug(" * Loading VAD Model")
    vad = silero_vad.SileroVAD(device=device)
    
    # g2p 인스턴스 초기화
    G2P = G2p()

    # 영어 구간 탐지 패턴: 연속된 알파벳과 apostrophe, 공백으로 연결된 단어 그룹
    ENG_PATTERN = re.compile(r"[A-Za-z][A-Za-z']*(?: [A-Za-z][A-Za-z']*)*")

    # load model from Hugging Face model card directly (You need a Hugging Face token)
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
    diar_model.eval()

    logger.debug("All models loaded")

    supported_languages = cfg["language"]["supported"]
    multilingual_flag = cfg["language"]["multilingual"]
    logger.debug(f"supported languages multilingual {supported_languages}")
    logger.debug(f"using multilingual asr {multilingual_flag}")

    input_folder_path = cfg["entrypoint"]["input_folder_path"]

    if not os.path.exists(input_folder_path):
        raise FileNotFoundError(f"input_folder_path: {input_folder_path} not found")

    audio_paths = get_audio_files(input_folder_path)  # Get all audio files
    logger.debug(f"Scanning {len(audio_paths)} audio files in {input_folder_path}")

    start_time = time.time()
    for path in audio_paths:


        # 폴더가 이미 있으면 넘어가는 로직
        
    #    # 1. main_process와 동일한 로직으로 예상 출력 폴더 경로를 생성합니다.
    #     audio_name = os.path.splitext(os.path.basename(path))[0]
    #     suffix = "dia3" if args.dia3 else "ori"
    #     save_path_dir = os.path.join(
    #         os.path.dirname(path) + "_processed_llm-twelve-cases" + f"-vad-{args.vad}"+ f"-diaModel-{suffix}" 
    #         + f"-merge_gap-{args.merge_gap}" + f"-seg_th-{args.seg_th}"+ f"-cl_min-{args.min_cluster_size}" +f"-cl-th-{args.clust_th}"+ f"-LLM-{args.LLM}", audio_name
    #     )

    #     # 2. 해당 폴더가 이미 존재하는지 확인합니다.
    #     if os.path.exists(save_path_dir):
    #         logger.info(f"Output directory already exists, skipping: {save_path_dir}")
    #         continue  # 폴더가 존재하면 다음 오디오 파일로 넘어갑니다.


        main_process(path, do_vad=args.vad, LLM=args.LLM, use_demucs=args.demucs)
end_time = time.time()
print("Total time:", end_time - start_time)