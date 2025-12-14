# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

# Fix for PyTorch 2.6+ weights_only=True default breaking pyannote model loading
# Patch lightning_fabric's _load function to use weights_only=False
import lightning_fabric.utilities.cloud_io as cloud_io
from pathlib import Path
from typing import Union, IO, Any

_original_load = cloud_io._load

def _patched_load(path_or_url: Union[IO, str, Path], map_location=None) -> Any:
    """Patched version of lightning_fabric's _load that uses weights_only=False for pyannote compatibility"""
    if not isinstance(path_or_url, (str, Path)):
        return torch.load(path_or_url, map_location=map_location, weights_only=False)

    if str(path_or_url).startswith("http"):
        return torch.hub.load_state_dict_from_url(str(path_or_url), map_location=map_location)

    from lightning_fabric.utilities.cloud_io import get_filesystem
    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(f, map_location=map_location, weights_only=False)

cloud_io._load = _patched_load

# Continue with other imports
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

import requests
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
    export_to_mp3_new,
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
import collections
import difflib
from typing import List, Tuple, Dict
from itertools import zip_longest

# Import FlowSE denoising class
# Use relative path for better portability
flowse_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FlowSE")
sys.path.insert(0, flowse_path)
from simple_denoise import FlowSEDenoiser

warnings.filterwarnings("ignore")


def _apply_sortformer_segment_padding_from_args(
    df: pd.DataFrame, args, logger, audio_duration: float | None = None
) -> pd.DataFrame:
    """
    Shift diarization segment boundaries (frame-level tweak) outside NeMo's internal post-processing.
    When --sortformer-param is set, this ensures observable timing changes even if model cfg overrides are ignored.
    """
    if df is None or df.empty:
        return df
    if not getattr(args, "sortformer_param", False):
        return df

    pad_onset = float(getattr(args, "sortformer_pad_onset", 0.0))
    pad_offset = float(getattr(args, "sortformer_pad_offset", 0.0))

    if pad_onset == 0.0 and pad_offset == 0.0:
        return df

    df = df.copy()
    df["start"] = (df["start"].astype(float) + pad_onset).clip(lower=0.0)
    df["end"] = df["end"].astype(float) + pad_offset
    if audio_duration is not None and audio_duration > 0:
        df["end"] = df["end"].clip(lower=0.0, upper=float(audio_duration))
    else:
        df["end"] = df["end"].clip(lower=0.0)
    df["end"] = df[["start", "end"]].max(axis=1)

    return df
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
    def build_confusion_network(all_tokens: List[List[str]]) -> List[List[str]]:
        """
        여러 토큰 시퀀스를 Confusion Network로 구성합니다.
        모든 시퀀스를 동시에 고려하여 통합된 정렬을 생성합니다.

        Args:
            all_tokens: 모든 transcript의 토큰 리스트들 [[tok1, tok2, ...], ...]

        Returns:
            각 위치별 토큰 후보들의 리스트 [[cand1, cand2, ...], [cand1, cand2, ...], ...]
        """
        if not all_tokens:
            return []

        if len(all_tokens) == 1:
            return [[tok] for tok in all_tokens[0]]

        # 가장 긴 시퀀스를 pivot으로 선택 (보통 가장 정확함)
        pivot_idx = max(range(len(all_tokens)), key=lambda i: len(all_tokens[i]))
        pivot = all_tokens[pivot_idx]

        # Confusion network 초기화: pivot의 각 위치에 해당 토큰으로 시작
        confusion_net = [[pivot[i]] for i in range(len(pivot))]

        # 다른 모든 시퀀스를 pivot에 정렬하여 confusion network에 추가
        for idx, tokens in enumerate(all_tokens):
            if idx == pivot_idx:
                continue

            matcher = difflib.SequenceMatcher(None, pivot, tokens)

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    # 일치: 해당 위치에 토큰 추가
                    for i, j in zip(range(i1, i2), range(j1, j2)):
                        if i < len(confusion_net):
                            confusion_net[i].append(tokens[j])

                elif tag == 'replace':
                    # 치환: 각 pivot 위치에 대응하는 candidate 토큰 추가
                    pivot_len = i2 - i1
                    cand_len = j2 - j1

                    if pivot_len == cand_len:
                        # 1:1 대응
                        for i, j in zip(range(i1, i2), range(j1, j2)):
                            if i < len(confusion_net):
                                confusion_net[i].append(tokens[j])
                    elif pivot_len > cand_len:
                        # pivot이 더 긺: candidate를 분산 배치
                        for i in range(i1, i2):
                            offset = (i - i1) * cand_len // pivot_len
                            if j1 + offset < j2 and i < len(confusion_net):
                                confusion_net[i].append(tokens[j1 + offset])
                    else:
                        # candidate가 더 긺: pivot 위치에 여러 candidate 병합
                        # 첫 번째 pivot 위치에 모든 candidate 추가
                        if i1 < len(confusion_net):
                            for j in range(j1, j2):
                                confusion_net[i1].append(tokens[j])

                elif tag == 'delete':
                    # pivot에만 존재: 이미 confusion_net에 있음 (다른 시퀀스는 빈 값)
                    pass

                elif tag == 'insert':
                    # candidate에만 존재: 가장 가까운 pivot 위치에 삽입
                    # 이전 매칭 위치 다음에 추가
                    insert_pos = min(i1, len(confusion_net) - 1) if confusion_net else 0
                    if insert_pos >= 0 and insert_pos < len(confusion_net):
                        for j in range(j1, j2):
                            confusion_net[insert_pos].append(tokens[j])

        return confusion_net

    @staticmethod
    def has_local_repetition(output: List[str], word: str, window: int = 3) -> bool:
        """
        최근 window 내에 같은 단어가 반복되는지 확인합니다.

        Args:
            output: 현재까지 출력된 단어 리스트
            word: 확인할 단어
            window: 확인할 윈도우 크기

        Returns:
            반복이면 True, 아니면 False
        """
        if len(output) < 1:
            return False

        # 최근 window개 단어 확인
        recent = output[-window:] if len(output) >= window else output

        # 같은 단어가 이미 2번 이상 나왔으면 반복으로 판단
        return recent.count(word) >= 2

    @staticmethod
    def calculate_transcript_similarity(t1_tokens: List[str], t2_tokens: List[str]) -> float:
        """
        두 transcript의 유사도를 계산합니다 (Jaccard similarity 기반).

        Args:
            t1_tokens: 첫 번째 transcript 토큰
            t2_tokens: 두 번째 transcript 토큰

        Returns:
            0~1 사이의 유사도 점수
        """
        if not t1_tokens or not t2_tokens:
            return 0.0

        set1 = set(t1_tokens)
        set2 = set(t2_tokens)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def align_and_vote(transcripts: List[str]) -> str:
        """
        여러 전사 결과를 Confusion Network로 정렬 후 개선된 투표를 수행합니다.
        - Confusion Network로 통합 정렬
        - 반복 패턴 감지 및 필터링
        - 유사도 기반 아웃라이어 다운웨이팅

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
        all_tokens = [t.split() for t in transcripts]

        # Transcript 간 유사도 계산하여 아웃라이어 감지
        similarities = []
        for i in range(len(all_tokens)):
            sim_scores = []
            for j in range(len(all_tokens)):
                if i != j:
                    sim = RoverEnsembler.calculate_transcript_similarity(all_tokens[i], all_tokens[j])
                    sim_scores.append(sim)
            avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0
            similarities.append(avg_sim)

        # 평균 유사도가 너무 낮은 transcript는 신뢰도 감소
        # 임계값: 평균 유사도가 0.3 미만이면 아웃라이어로 간주
        outlier_threshold = 0.3
        trusted_indices = [i for i, sim in enumerate(similarities) if sim >= outlier_threshold]

        # 모든 transcript가 아웃라이어면 가장 유사도 높은 것들만 사용
        if len(trusted_indices) == 0:
            trusted_indices = [i for i, _ in sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:2]]

        # Confusion Network 구성
        confusion_net = RoverEnsembler.build_confusion_network(all_tokens)

        # 각 위치별 개선된 투표
        final_output = []
        for pos_idx, candidates in enumerate(confusion_net):
            if not candidates:
                continue

            # 빈 문자열 제거 후 투표
            valid_candidates = [c for c in candidates if c]
            if not valid_candidates:
                continue

            # 신뢰할 수 있는 transcript의 후보들만 필터링
            # (confusion_net의 첫 번째는 pivot, 나머지는 순서대로)
            trusted_candidates = []
            for i, cand in enumerate(valid_candidates):
                # i번째 candidate가 어느 transcript에서 왔는지 추정
                # (간단히 pivot은 항상 신뢰, 나머지는 순서대로)
                if i == 0 or (i - 1) in trusted_indices:
                    trusted_candidates.append(cand)

            # 신뢰할 수 있는 후보가 없으면 원래 후보 사용
            if not trusted_candidates:
                trusted_candidates = valid_candidates

            # 최다 득표 단어 선정
            votes = collections.Counter(trusted_candidates)
            best_word, count = votes.most_common(1)[0]

            # 반복 패턴 체크: 최근에 같은 단어가 반복되면 스킵
            if RoverEnsembler.has_local_repetition(final_output, best_word):
                # 반복이면 다음으로 많은 후보 선택
                if len(votes) > 1:
                    best_word = votes.most_common(2)[1][0]
                else:
                    # 다른 후보가 없으면 스킵
                    continue

            # 과반수 이상이면 채택, 아니면 pivot 우선
            if count >= len(trusted_candidates) / 2:
                final_output.append(best_word)
            else:
                # pivot의 토큰 우선
                pivot_word = candidates[0] if candidates[0] else best_word
                # pivot도 반복 체크
                if RoverEnsembler.has_local_repetition(final_output, pivot_word):
                    if pivot_word != best_word:
                        final_output.append(best_word)
                else:
                    final_output.append(pivot_word)

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
def detect_background_music(audio, panns_model, threshold=0.3):
    """
    PANNs를 사용하여 배경음악이 있는지 검출합니다.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.
        panns_model (AudioTagging): 로드된 PANNs 모델 인스턴스.
        threshold (float): Music 확률 임계값. 이 값 이상이면 배경음악이 있다고 판단.

    Returns:
        tuple: (has_music: bool, music_prob: float)
    """
    if panns_model is None:
        logger.warning("PANNs model is not loaded, skipping music detection")
        return False, 0.0

    logger.debug("Detecting background music using PANNs")

    # PANNs는 32kHz 오디오를 기대하므로 리샘플링
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    if sample_rate != 32000:
        waveform_32k = librosa.resample(waveform, orig_sr=sample_rate, target_sr=32000)
    else:
        waveform_32k = waveform

    # PANNs inference (모델은 이미 로드됨)
    (clipwise_output, embedding) = panns_model.inference(waveform_32k[None, :])

    # Get labels
    labels = panns_model.labels

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


def detect_segment_background_music(segment_audio, sample_rate, panns_model, threshold=0.3):
    """
    세그먼트 오디오에 대해 배경음악이 있는지 검출합니다.

    Args:
        segment_audio (np.ndarray): 세그먼트 오디오 waveform.
        sample_rate (int): 샘플 레이트.
        panns_model (AudioTagging): 로드된 PANNs 모델 인스턴스.
        threshold (float): Music 확률 임계값.

    Returns:
        tuple: (has_music: bool, music_prob: float)
    """
    if panns_model is None:
        logger.warning("PANNs model is not loaded, skipping music detection")
        return False, 0.0

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

    # PANNs inference (모델은 이미 로드됨)
    (clipwise_output, embedding) = panns_model.inference(waveform_32k[None, :])

    # Get labels
    labels = panns_model.labels

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

        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Running Demucs on device: {device}")

        cmd = [
            "python", "-m", "demucs.separate",
            "-n", "htdemucs",
            "--two-stems", "vocals",
            "-d", device,  # Explicitly specify device (cuda or cpu)
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
def preprocess_segments_with_demucs(segment_list, audio, panns_model=None, use_demucs=False, padding=0.5):
    """
    ASR 전에 세그먼트별로 배경음악을 검출하고 demucs를 적용합니다.
    (Padding을 추가하여 ASR 타임스탬프가 늘어날 경우를 대비합니다)
    """
    if not use_demucs:
        logger.info("Demucs preprocessing skipped (flag disabled)")
        return audio, [False] * len(segment_list)

    logger.info(f"Preprocessing {len(segment_list)} segments with background music detection and removal (padding={padding}s)")

    waveform = audio["waveform"].copy()
    sample_rate = audio["sample_rate"]
    total_samples = len(waveform)
    segment_demucs_flags = []

    for idx, segment in enumerate(segment_list):
        # Apply padding to cover ASR boundary shifts
        start_time = max(0, segment["start"] - padding)
        end_time = segment["end"] + padding # 경계를 넘는 것은 슬라이싱에서 자동 처리됨
        
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        
        # 전체 길이 넘어가는 것 방지
        end_frame = min(end_frame, total_samples)

        segment_audio = waveform[start_frame:end_frame]
        
        # 세그먼트가 너무 짧으면 패스
        if len(segment_audio) < 16000: # 0.5초 미만
            segment_demucs_flags.append(False)
            continue

        # Detect background music (padding이 포함된 구간으로 검사)
        has_music, music_prob = detect_segment_background_music(segment_audio, sample_rate, panns_model, threshold=0.3)
        
        if has_music:
            logger.info(f"Segment {idx} (with padding): Background music detected (prob={music_prob:.3f}), applying Demucs...")
            # Apply demucs
            vocal_audio = remove_segment_background_music_demucs(segment_audio, sample_rate)
            
            # Replace the segment in the waveform
            # vocal_audio 길이가 segment_audio와 다를 수 있으므로 길이를 맞춤
            target_length = len(segment_audio)
            if len(vocal_audio) >= target_length:
                waveform[start_frame:end_frame] = vocal_audio[:target_length]
            else:
                # 드문 경우지만 vocal_audio가 짧을 경우 패딩 처리
                waveform[start_frame : start_frame + len(vocal_audio)] = vocal_audio

            segment_demucs_flags.append(True)
            logger.info(f"Segment {idx}: Demucs applied successfully")
        else:
            segment_demucs_flags.append(False)

    # Update audio dictionary with processed waveform
    updated_audio = audio.copy()
    updated_audio["waveform"] = waveform

    # Also update audio_segment for export
    from pydub import AudioSegment as PydubAudioSegment
    waveform_clipped = np.clip(waveform, -1.0, 1.0)
    waveform_int16 = (waveform_clipped * 32767).astype(np.int16)
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
def detect_overlapping_segments(segment_list, overlap_threshold=0.2):
    """
    Detect segments that overlap for more than overlap_threshold seconds.

    Args:
        segment_list (list): List of segments with 'start', 'end', and 'speaker' keys
        overlap_threshold (float): Minimum overlap duration in seconds to be considered

    Returns:
        list: List of overlapping segment pairs with overlap info
            [{'seg1': segment1, 'seg2': segment2, 'overlap_start': float, 'overlap_end': float, 'overlap_duration': float}]
    """
    overlapping_pairs = []

    # Sort segments by start time
    sorted_segments = sorted(segment_list, key=lambda x: x['start'])

    for i in range(len(sorted_segments)):
        for j in range(i + 1, len(sorted_segments)):
            seg1 = sorted_segments[i]
            seg2 = sorted_segments[j]

            # If seg2 starts after seg1 ends, no more overlaps possible for seg1
            if seg2['start'] >= seg1['end']:
                break

            # Calculate overlap
            overlap_start = max(seg1['start'], seg2['start'])
            overlap_end = min(seg1['end'], seg2['end'])
            overlap_duration = overlap_end - overlap_start

            # Check if overlap exceeds threshold
            if overlap_duration >= overlap_threshold:
                overlapping_pairs.append({
                    'seg1': seg1,
                    'seg2': seg2,
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end,
                    'overlap_duration': overlap_duration
                })
                logger.info(f"Overlap detected: {overlap_duration:.2f}s between "
                           f"[{seg1['start']:.2f}-{seg1['end']:.2f}] and "
                           f"[{seg2['start']:.2f}-{seg2['end']:.2f}]")

    return overlapping_pairs


class SepReformerSeparator:
    """
    SepReformer 모델을 한 번만 로드하고 여러 번 inference할 수 있는 클래스
    """
    def __init__(self, sepreformer_path, device):
        """
        SepReformer 모델 초기화 및 로드

        Args:
            sepreformer_path: SepReformer 모델 디렉토리 경로
            device: torch device (cuda/cpu)
        """
        import sys
        import yaml

        self.sepreformer_path = sepreformer_path
        self.device = device

        print(f"[SepReformer] Initializing on device: {self.device}")

        # Store original sys.path to restore later
        original_sys_path = sys.path.copy()

        try:
            # Save the current 'models' and 'utils' modules if they exist
            original_models = sys.modules.get('models', None)
            original_utils = sys.modules.get('utils', None)

            # Remove podcast-pipeline from sys.path temporarily
            podcast_pipeline_path = os.path.dirname(os.path.abspath(__file__))
            paths_to_remove = [p for p in sys.path if podcast_pipeline_path in p]
            for path in paths_to_remove:
                sys.path.remove(path)

            # Add SepReformer to path
            if sepreformer_path not in sys.path:
                sys.path.insert(0, sepreformer_path)

            # Clear conflicting modules
            modules_to_clear = [key for key in sys.modules.keys()
                              if key.startswith('models.') or key.startswith('utils.') or key in ['models', 'utils']]
            cleared_modules = {}
            for module_name in modules_to_clear:
                cleared_modules[module_name] = sys.modules[module_name]
                del sys.modules[module_name]

            # Import SepReformer's model
            from models.SepReformer_Base_WSJ0.model import Model

            # Restore the original modules
            for module_name, module_obj in cleared_modules.items():
                sys.modules[module_name] = module_obj

            # Load SepReformer config
            config_path = os.path.join(sepreformer_path, "models/SepReformer_Base_WSJ0/configs.yaml")
            with open(config_path, 'r') as f:
                yaml_dict = yaml.safe_load(f)
            self.config = yaml_dict["config"]

            # Load model
            print("[SepReformer] Loading model...")
            self.model = Model(**self.config["model"])

            # Load checkpoint
            checkpoint_dir = os.path.join(sepreformer_path, "models/SepReformer_Base_WSJ0/log/pretrain_weights")
            if not os.path.exists(checkpoint_dir) or not os.listdir(checkpoint_dir):
                checkpoint_dir = os.path.join(sepreformer_path, "models/SepReformer_Base_WSJ0/log/scratch_weights")

            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pt', '.pth'))]
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(device)
            self.model.eval()

            print("[SepReformer] Model initialization complete!")

        finally:
            # Restore original sys.path
            sys.path = original_sys_path

    def separate(self, audio_segment, sample_rate):
        """
        오디오 분리 수행

        Args:
            audio_segment (np.ndarray): 분리할 오디오 세그먼트
            sample_rate (int): 오디오 샘플레이트

        Returns:
            tuple: (separated_audio_1, separated_audio_2) as numpy arrays
        """
        try:
            # Resample to 8kHz if needed
            if sample_rate != 8000:
                audio_8k = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=8000)
            else:
                audio_8k = audio_segment

            # Prepare tensor
            mixture_tensor = torch.tensor(audio_8k, dtype=torch.float32).unsqueeze(0)

            # Padding
            stride = self.config["model"]["module_audio_enc"]["stride"]
            remains = mixture_tensor.shape[-1] % stride
            if remains != 0:
                padding = stride - remains
                mixture_padded = torch.nn.functional.pad(mixture_tensor, (0, padding), "constant", 0)
            else:
                mixture_padded = mixture_tensor

            # Inference
            with torch.inference_mode():
                nnet_input = mixture_padded.to(self.device)
                estim_src, _ = self.model(nnet_input)

                # Extract separated sources
                src1 = estim_src[0][..., :mixture_tensor.shape[-1]].squeeze().cpu().numpy()
                src2 = estim_src[1][..., :mixture_tensor.shape[-1]].squeeze().cpu().numpy()

            # Resample back to original sample rate if needed
            if sample_rate != 8000:
                src1 = librosa.resample(src1, orig_sr=8000, target_sr=sample_rate)
                src2 = librosa.resample(src2, orig_sr=8000, target_sr=sample_rate)

            return src1, src2

        except Exception as e:
            logger.error(f"SepReformer separation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return audio_segment, audio_segment


@time_logger
def identify_speaker_with_embedding(audio_segment, sample_rate, reference_embeddings, speaker_labels, embedding_model):
    """
    Identify which speaker an audio segment belongs to using speaker embeddings.

    Args:
        audio_segment (np.ndarray): Audio segment to identify
        sample_rate (int): Sample rate of the audio
        reference_embeddings (dict): Dictionary of {speaker_label: embedding_tensor}
        speaker_labels (list): List of possible speaker labels
        embedding_model: 미리 로드된 pyannote embedding 모델

    Returns:
        str: Identified speaker label
    """

    # Extract embedding from audio segment
    # Resample to 16kHz if needed (pyannote expects 16kHz)
    if sample_rate != 16000:
        audio_16k = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=16000)
    else:
        audio_16k = audio_segment

    # Convert to tensor
    audio_tensor = torch.tensor(audio_16k, dtype=torch.float32).unsqueeze(0).to(device)

    # Extract embedding
    with torch.inference_mode():
        embedding = embedding_model(audio_tensor)

    # Compare with reference embeddings using cosine similarity
    best_speaker = None
    best_similarity = -1.0

    for speaker_label in speaker_labels:
        if speaker_label in reference_embeddings:
            ref_embedding = reference_embeddings[speaker_label]
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                embedding.mean(dim=1),
                ref_embedding.mean(dim=1),
                dim=0
            ).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_label

    logger.debug(f"Speaker identification: {best_speaker} (similarity: {best_similarity:.3f})")
    return best_speaker


@time_logger
def process_overlapping_segments_with_separation(segment_list, audio, overlap_threshold=1.0,
                                                 separator=None, embedding_model=None):
    """
    Process overlapping segments by separating them with SepReformer.
    [Updated] Matches the volume of separated audio to the original overlap audio to prevent volume jumps.

    Args:
        segment_list: 세그먼트 리스트
        audio: 오디오 딕셔너리
        overlap_threshold: 오버랩 임계값
        separator: 미리 로드된 SepReformerSeparator 객체
        embedding_model: 미리 로드된 pyannote embedding 모델
    """
    if separator is None:
        logger.warning("SepReformer separator not provided, skipping separation")
        return audio, segment_list

    if embedding_model is None:
        logger.warning("Embedding model not provided, skipping separation")
        return audio, segment_list

    logger.info(f"Processing overlapping segments with SepReformer (threshold: {overlap_threshold}s)")

    # -------------------------------------------------------------------------
    # [추가] 볼륨 매칭 헬퍼 함수
    # -------------------------------------------------------------------------
    def match_target_amplitude(source_wav, target_wav):
        """
        source_wav의 볼륨(RMS)을 target_wav의 볼륨에 맞춥니다.
        """
        # 0으로 나누기 방지용 엡실론
        epsilon = 1e-10
        
        # RMS(Root Mean Square) 에너지 계산
        src_rms = np.sqrt(np.mean(source_wav**2))
        tgt_rms = np.sqrt(np.mean(target_wav**2))
        
        if src_rms < epsilon:
            return source_wav
        
        # 비율 계산 (Target이 Source보다 얼마나 큰지/작은지)
        gain = tgt_rms / (src_rms + epsilon)
        
        # Gain 적용
        adjusted_wav = source_wav * gain
        
        # 클리핑 방지 (-1.0 ~ 1.0)
        return np.clip(adjusted_wav, -1.0, 1.0)
    # -------------------------------------------------------------------------

    # 1. 모든 세그먼트에 대해 초기 'enhanced_audio'를 원본 오디오로 초기화
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    
    for seg in segment_list:
        if 'enhanced_audio' not in seg:
            start_frame = int(seg['start'] * sample_rate)
            end_frame = int(seg['end'] * sample_rate)
            seg['enhanced_audio'] = waveform[start_frame:end_frame].copy()
        
        if 'sepreformer' not in seg:
            seg['sepreformer'] = False

    # Detect overlapping segments
    overlapping_pairs = detect_overlapping_segments(segment_list, overlap_threshold)

    if not overlapping_pairs:
        logger.info("No overlapping segments found")
        return audio, segment_list

    logger.info(f"Found {len(overlapping_pairs)} overlapping segment pairs")

    # (Reference Embeddings 추출 로직 - 기존 유지)
    reference_embeddings = {}
    all_speakers = list(set([seg['speaker'] for seg in segment_list]))

    # ... (Reference Embedding 추출 - 기존 유지) ...
    for speaker in all_speakers:
        speaker_segments = [seg for seg in segment_list if seg['speaker'] == speaker]
        for seg in speaker_segments:
            is_overlapping = any(pair['seg1'] == seg or pair['seg2'] == seg for pair in overlapping_pairs)
            if not is_overlapping and (seg['end'] - seg['start']) >= 2.0:
                start_frame = int(seg['start'] * sample_rate)
                end_frame = int(seg['end'] * sample_rate)
                seg_audio = waveform[start_frame:end_frame]
                if sample_rate != 16000:
                    seg_audio_16k = librosa.resample(seg_audio, orig_sr=sample_rate, target_sr=16000)
                else:
                    seg_audio_16k = seg_audio
                seg_tensor = torch.tensor(seg_audio_16k, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.inference_mode():
                    embedding = embedding_model(seg_tensor)
                reference_embeddings[speaker] = embedding
                break

    # 2. Process overlap pairs
    for pair_idx, pair in enumerate(overlapping_pairs):
        overlap_start = pair['overlap_start']
        overlap_end = pair['overlap_end']
        seg1 = pair['seg1']
        seg2 = pair['seg2']
        
        seg1_speaker = seg1['speaker']
        seg2_speaker = seg2['speaker']

        # Extract overlapping audio (Original Mixture)
        start_frame = int(overlap_start * sample_rate)
        end_frame = int(overlap_end * sample_rate)
        overlap_audio = waveform[start_frame:end_frame]

        # Separate with SepReformer
        separated_src1, separated_src2 = separator.separate(
            overlap_audio, sample_rate
        )

        # Identify speakers
        speaker1_identity = identify_speaker_with_embedding(
            separated_src1, sample_rate, reference_embeddings, [seg1_speaker, seg2_speaker], embedding_model
        )
        
        if speaker1_identity == seg1_speaker:
            seg1_part = separated_src1
            seg2_part = separated_src2
        else:
            seg1_part = separated_src2
            seg2_part = separated_src1

        # ---------------------------------------------------------------------
        # [수정] 볼륨 보정 적용 (Overlap 구간의 원본 볼륨에 맞춤)
        # ---------------------------------------------------------------------
        # 분리된 오디오가 원본 Overlap 구간(두 사람이 섞인 소리)의 RMS 에너지와 비슷해지도록 조정
        # (주의: 원본은 2명분이 섞여 있어서 1명분으로 분리된 것보다 에너지가 클 수밖에 없지만,
        #  SepReformer 출력이 0dB로 튀는 것보다는 이 기준이 훨씬 자연스럽습니다.)
        
        logger.debug(f"   Adjusting volume for overlap {pair_idx+1}...")
        seg1_part = match_target_amplitude(seg1_part, overlap_audio)
        seg2_part = match_target_amplitude(seg2_part, overlap_audio)
        # ---------------------------------------------------------------------

        # 1) Seg1 업데이트
        seg1_start_global = int(seg1['start'] * sample_rate)
        rel_start_1 = start_frame - seg1_start_global
        
        limit_len_1 = min(len(seg1_part), len(seg1['enhanced_audio'][rel_start_1:]))
        if limit_len_1 > 0:
            seg1['enhanced_audio'][rel_start_1 : rel_start_1 + limit_len_1] = seg1_part[:limit_len_1]
            seg1['sepreformer'] = True
            logger.info(f"  ✓ Updated Seg1 enhanced_audio with volume-adjusted separated audio") 

        # 2) Seg2 업데이트
        seg2_start_global = int(seg2['start'] * sample_rate)
        rel_start_2 = start_frame - seg2_start_global

        limit_len_2 = min(len(seg2_part), len(seg2['enhanced_audio'][rel_start_2:]))
        if limit_len_2 > 0:
            seg2['enhanced_audio'][rel_start_2 : rel_start_2 + limit_len_2] = seg2_part[:limit_len_2]
            seg2['sepreformer'] = True
            logger.info(f"  ✓ Updated Seg2 enhanced_audio with volume-adjusted separated audio")

    return audio, segment_list


@time_logger
def asr(vad_segments, audio):
    """
    Perform Automatic Speech Recognition (ASR) on the VAD segments of the given audio.
    [Updated] Now processes segments iteratively exactly like asr_MoE to ensure 'enhanced_audio' 
    is correctly utilized without relying on global buffer sandwiching.
    """
    if len(vad_segments) == 0:
        return []

    # 전체 오디오 (Fallback 용)
    full_waveform = audio["waveform"]
    global_sample_rate = audio["sample_rate"]
    
    final_results = []
    
    supported_languages = cfg["language"]["supported"]
    multilingual_flag = cfg["language"]["multilingual"]
    # asr_MoE 방식(개별 처리)을 따르므로 배치 사이즈는 1로 처리하거나, 
    # 라이브러리 지원 여부에 따라 조정 가능하지만 여기서는 정확성을 위해 개별 처리를 우선합니다.
    batch_size = 1 

    if multilingual_flag:
        # ... (기존 Multilingual 로직 유지 또는 필요시 동일한 루프 구조로 변경 필요) ...
        # 현재 코드 맥락상 Multilingual은 구현이 방대하여 Placeholder로 남김
        pass 
        return []

    logger.info(f"ASR Processing: {len(vad_segments)} segments (Iterative Mode)")

    for idx, segment in enumerate(vad_segments):
        start_time = segment["start"]
        end_time = segment["end"]
        speaker = segment.get("speaker", "Unknown")

        # ---------------------------------------------------------------------
        # 1. Audio Selection Logic (Identical to asr_MoE)
        # ---------------------------------------------------------------------
        segment_audio = None
        is_enhanced = False

        if "enhanced_audio" in segment:
            # SepReformer로 분리된 오디오가 있으면 우선 사용
            raw_audio = segment["enhanced_audio"]
            is_enhanced = True
        else:
            # 없으면 전체 오디오에서 해당 구간만 잘라냄
            start_frame = int(start_time * global_sample_rate)
            end_frame = int(end_time * global_sample_rate)
            raw_audio = full_waveform[start_frame:end_frame]
            is_enhanced = False

        # 16kHz 리샘플링 (Whisper 입력용)
        if global_sample_rate != 16000:
            segment_audio_16k = librosa.resample(raw_audio, orig_sr=global_sample_rate, target_sr=16000)
        else:
            segment_audio_16k = raw_audio

        # 너무 짧은 오디오 건너뛰기
        if len(segment_audio_16k) < 160: 
            continue

        # ---------------------------------------------------------------------
        # 2. Prepare Dummy VAD & Transcribe
        # ---------------------------------------------------------------------
        # 이미 잘라낸 오디오 조각을 입력하므로 상대 시간은 0 ~ duration 입니다.
        duration_sec = len(segment_audio_16k) / 16000
        dummy_vad = [{"start": 0.0, "end": duration_sec}]

        try:
            # 언어 감지 (필요 시 세그먼트마다 수행하거나, 'en'으로 고정)
            # 여기서는 기존 흐름에 따라 'en'을 기본으로 하되, 감지가 필요하면 detect_language 사용 가능
            # language, prob = asr_model.detect_language(segment_audio_16k)
            language = "en" 

            transcribe_result = asr_model.transcribe(
                segment_audio_16k,
                dummy_vad,
                batch_size=batch_size,
                language=language,
                print_progress=False,
            )
            
            # 결과 처리
            if transcribe_result and "segments" in transcribe_result:
                for res_seg in transcribe_result["segments"]:
                    # 1. 텍스트가 비어있지 않은 경우만 처리
                    if res_seg["text"].strip():
                        # 2. 상대 시간(0~duration)을 절대 시간(start_time~)으로 변환
                        res_seg["start"] += start_time
                        res_seg["end"] += start_time
                        
                        # 3. 메타데이터 복원
                        res_seg["speaker"] = speaker
                        res_seg["language"] = transcribe_result.get("language", language)
                        res_seg["sepreformer"] = segment.get("sepreformer", False)
                        res_seg["is_separated"] = is_enhanced
                        
                        if is_enhanced:
                            res_seg["enhanced_audio"] = raw_audio

                        # 4. 워드 타임스탬프가 있는 경우 시간 보정
                        if "words" in res_seg:
                            for w in res_seg["words"]:
                                w["start"] += start_time
                                w["end"] += start_time

                        final_results.append(res_seg)

        except Exception as e:
            logger.error(f"ASR failed for segment {idx} ({start_time:.2f}-{end_time:.2f}): {e}")
            continue

    return final_results

import concurrent.futures

@time_logger
def asr_MoE(vad_segments, audio, segment_demucs_flags=None, enable_word_timestamps=False, device="cuda"):
    """
    Perform Automatic Speech Recognition (ASR) on the VAD segments using MoE with Parallel Execution.
    [Updated] Runs Whisper, Parakeet, and Canary in parallel using ThreadPoolExecutor.
    """
    if len(vad_segments) == 0:
        return [], 0.0, 0.0

    if segment_demucs_flags is None:
        segment_demucs_flags = [False] * len(vad_segments)

    # 전체 오디오 (Fallback 용)
    full_waveform = audio["waveform"]
    global_sample_rate = audio["sample_rate"]
    
    final_results = []
    total_whisper_time = 0.0
    total_alignment_time = 0.0
    
    rover = RoverEnsembler()

    # --- Helper Functions for Parallel Execution ---
    def run_whisper_task(segment_audio_16k, dummy_vad):
        w_start = time.time()
        try:
            transcribe_result = asr_model.transcribe(
                segment_audio_16k, 
                dummy_vad, 
                batch_size=1, 
                print_progress=False
            )
            
            text_whisper = ""
            detected_language = "en"
            words = []

            if transcribe_result and "segments" in transcribe_result and len(transcribe_result["segments"]) > 0:
                text_whisper = " ".join([s["text"] for s in transcribe_result["segments"]]).strip()
                detected_language = transcribe_result.get("language", "en")
                if enable_word_timestamps:
                    for s in transcribe_result["segments"]:
                        if "words" in s: words.extend(s["words"])
            
            w_end = time.time()
            return {
                "text": text_whisper,
                "language": detected_language,
                "words": words,
                "time": w_end - w_start
            }
        except Exception as e:
            logger.error(f"Whisper failed: {e}")
            return {"text": "", "language": "en", "words": [], "time": 0.0}

    def run_parakeet_task(segment_audio_16k):
        try:
            # Parakeet input requires list
            p_res = asr_model_2.transcribe([segment_audio_16k])
            
            text_parakeet = ""
            if p_res:
                first_result = p_res[0]
                if isinstance(first_result, str):
                    text_parakeet = first_result
                elif hasattr(first_result, 'text'):
                    text_parakeet = first_result.text
                else:
                    text_parakeet = str(first_result)
            return text_parakeet
        except Exception as e:
            logger.error(f"Parakeet failed: {e}")
            return ""

    def run_canary_task(segment_audio_16k):
        try:
            # Canary requires a file path usually, creating temp file safely inside thread
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                sf.write(temp_wav.name, segment_audio_16k, 16000)
                # Ensure write is flushed
                temp_wav.flush()
                
                answer_ids = canary_model.generate(
                    prompts=[[{"role": "user", "content": f"Transcribe the following: {canary_model.audio_locator_tag}", "audio": [temp_wav.name]}]],
                    max_new_tokens=128,
                )
                text_canary = canary_model.tokenizer.ids_to_text(answer_ids[0].cpu())
                return text_canary
        except Exception as e:
            logger.error(f"Canary failed: {e}")
            return ""
    # ---------------------------------------------

    # Create a ThreadPoolExecutor
    # max_workers=3 allows all three models to be attempted roughly at the same time.
    # Note: Python GIL exists, but since these calls release GIL for C++/CUDA ops, it works for parallelization.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:

        for idx, segment in enumerate(vad_segments):
            start_time = segment["start"]
            end_time = segment["end"]
            speaker = segment.get("speaker", "Unknown")
            
            # 1. Audio Selection Logic
            segment_audio = None
            is_enhanced = False

            if "enhanced_audio" in segment:
                raw_audio = segment["enhanced_audio"]
                is_enhanced = True
            else:
                start_frame = int(start_time * global_sample_rate)
                end_frame = int(end_time * global_sample_rate)
                raw_audio = full_waveform[start_frame:end_frame]
            
            # 16kHz 리샘플링
            if global_sample_rate != 16000:
                segment_audio_16k = librosa.resample(raw_audio, orig_sr=global_sample_rate, target_sr=16000)
            else:
                segment_audio_16k = raw_audio

            if len(segment_audio_16k) < 160: 
                continue

            # Dummy VAD for Whisper
            duration_sec = len(segment_audio_16k) / 16000
            dummy_vad = [{"start": 0.0, "end": duration_sec}]

            # ---------------------------------------------------------------------
            # Submit Tasks in Parallel
            # ---------------------------------------------------------------------
            future_whisper = executor.submit(run_whisper_task, segment_audio_16k, dummy_vad)
            future_parakeet = executor.submit(run_parakeet_task, segment_audio_16k)
            future_canary = executor.submit(run_canary_task, segment_audio_16k)

            # ---------------------------------------------------------------------
            # Wait for results (Barrier)
            # ---------------------------------------------------------------------
            # .result() blocks until the future is done
            whisper_res = future_whisper.result()
            text_parakeet = future_parakeet.result()
            text_canary = future_canary.result()

            # Unpack Whisper results
            text_whisper = whisper_res["text"]
            detected_language = whisper_res["language"]
            words = whisper_res["words"]
            total_whisper_time += whisper_res["time"]

            # ---------------------------------------------------------------------
            # 5. Ensemble & Result Construction
            # ---------------------------------------------------------------------
            text_ensemble = rover.align_and_vote([text_whisper, text_canary, text_parakeet])

            seg_result = {
                "start": start_time,
                "end": end_time,
                "text": text_ensemble,
                "text_whisper": text_whisper,
                "text_parakeet": text_parakeet,
                "text_canary": text_canary,
                "speaker": speaker,
                "language": detected_language,
                "demucs": segment_demucs_flags[idx] if idx < len(segment_demucs_flags) else False,
                "is_separated": is_enhanced, 
                "sepreformer": segment.get("sepreformer", False)
            }
            
            if is_enhanced:
                seg_result["enhanced_audio"] = raw_audio

            if enable_word_timestamps and words:
                for w in words:
                    w["start"] += start_time
                    w["end"] += start_time
                seg_result["words"] = words

            final_results.append(seg_result)

    return final_results, total_whisper_time, total_alignment_time

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
            # [CRITICAL] SepReformer로 처리된 오디오가 있으면 우선 사용 (저장될 파일과 일치시키기 위함)
            if "enhanced_audio" in segment:
                segment_audio = segment["enhanced_audio"]
                sample_rate = audio["sample_rate"]
            else:
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


def apply_flowse_denoising(filtered_list, audio, save_path, denoiser=None, use_asr_moe=False):
    """
    sepreformer==True인 세그먼트에 대해 FlowSE 디노이징을 적용합니다.

    Args:
        filtered_list (list): ASR 결과 세그먼트 리스트
        audio (dict): 오디오 딕셔너리 (waveform, sample_rate 포함)
        save_path (str): 디노이즈된 오디오 파일을 저장할 경로
        denoiser (FlowSEDenoiser): 미리 로드된 FlowSE denoiser 객체
        use_asr_moe (bool): ASRMoE 모드 여부 (True면 ensemble_text, False면 whisper_text 사용)

    Returns:
        tuple: (디노이징이 적용된 세그먼트 리스트, 처리 시간(초))
    """
    if denoiser is None:
        logger.warning("FlowSE denoiser not provided, skipping denoising")
        return filtered_list, 0.0
    logger.info(f"Applying FlowSE denoising to sepreformer segments...")
    denoise_start_time = time.time()

    # 디노이즈 디렉토리 생성
    denoise_dir = os.path.join(save_path, "denoised_audio")
    os.makedirs(denoise_dir, exist_ok=True)

    denoised_count = 0

    for idx, segment in enumerate(filtered_list):
        # sepreformer==True인 세그먼트만 처리
        if not segment.get("sepreformer", False):
            continue

        try:
            # 텍스트 선택: ASRMoE이면 ensemble_text (또는 text), 아니면 whisper_text
            if use_asr_moe:
                # asr_MoE에서는 "text"에 ensemble 결과가 저장됨
                text = segment.get("text", "")
            else:
                # 일반 asr에서는 "text"에 whisper 결과가 저장됨
                text = segment.get("text", "")

            if not text or not text.strip():
                logger.warning(f"Segment {idx}: No text available for denoising, skipping")
                continue

            # 세그먼트 오디오 추출
            # enhanced_audio가 있으면 우선 사용 (SepReformer로 분리된 오디오)
            if "enhanced_audio" in segment:
                segment_audio = segment["enhanced_audio"]
                sample_rate = audio["sample_rate"]
            else:
                start_time = segment["start"]
                end_time = segment["end"]
                sample_rate = audio["sample_rate"]
                start_frame = int(start_time * sample_rate)
                end_frame = int(end_time * sample_rate)
                segment_audio = audio["waveform"][start_frame:end_frame]

            # 원본 오디오의 RMS 레벨 측정 (볼륨 유지를 위해)
            original_rms = np.sqrt(np.mean(segment_audio ** 2))

            # 임시 입력 오디오 파일로 저장
            temp_input_path = os.path.join(denoise_dir, f"temp_input_{idx:05d}.wav")
            sf.write(temp_input_path, segment_audio, sample_rate)

            # 출력 파일 경로 설정
            output_path = os.path.join(denoise_dir, f"denoised_{idx:05d}.wav")

            # FlowSE 디노이징 수행
            logger.debug(f"Segment {idx}: Denoising with text: '{text[:50]}...'")
            denoised_path = denoiser.denoise(
                audio_path=temp_input_path,
                text=text,
                output_path=output_path
            )

            # 디노이즈된 오디오를 세그먼트에 추가
            denoised_audio, denoised_sr = sf.read(denoised_path)

            # 볼륨 매칭: 디노이징된 오디오를 원본과 같은 RMS 레벨로 조정
            if original_rms > 1e-8:  # 무음이 아닌 경우에만
                denoised_rms = np.sqrt(np.mean(denoised_audio ** 2))
                if denoised_rms > 1e-8:  # 디노이징된 오디오도 무음이 아닌 경우
                    volume_scale = original_rms / denoised_rms
                    denoised_audio = denoised_audio * volume_scale
                    # 클리핑 방지
                    denoised_audio = np.clip(denoised_audio, -1.0, 1.0)
                    # 볼륨 조정된 오디오를 다시 저장
                    sf.write(denoised_path, denoised_audio, denoised_sr)
                    logger.debug(f"Segment {idx}: Volume matched (scale: {volume_scale:.3f})")

            segment["denoised_audio_path"] = denoised_path
            segment["flowse_denoised"] = True

            # 임시 입력 파일 삭제
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)

            denoised_count += 1
            logger.debug(f"Segment {idx}: Successfully denoised and saved to {denoised_path}")

        except Exception as e:
            logger.error(f"Segment {idx}: Error during FlowSE denoising: {e}")
            segment["flowse_denoised"] = False

    denoise_end_time = time.time()
    denoise_processing_time = denoise_end_time - denoise_start_time

    logger.info(f"FlowSE denoising completed: {denoised_count} segments processed")
    return filtered_list, denoise_processing_time


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

def export_segments_with_enhanced_audio(audio_info, segment_list, save_dir, audio_name):
    """
    Export segments to MP3 files.
    If 'enhanced_audio' exists in the segment (processed by SepReformer), use it.
    Otherwise, slice from the original audio.
    """
    import os
    from pydub import AudioSegment as PydubAudioSegment
    
    # 세그먼트 저장용 폴더 생성
    segments_dir = os.path.join(save_dir, audio_name)
    os.makedirs(segments_dir, exist_ok=True)
    
    # 원본 전체 오디오 (Pydub 객체)
    full_audio_segment = audio_info.get("audio_segment")
    sample_rate = audio_info["sample_rate"]
    
    if full_audio_segment is None:
        # 만약 audio_segment가 없으면 waveform에서 생성 (fallback)
        waveform_int16 = (audio_info["waveform"] * 32767).astype(np.int16)
        full_audio_segment = PydubAudioSegment(
            waveform_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )

    logger.info(f"Exporting {len(segment_list)} segments with enhanced audio check...")

    for i, seg in enumerate(segment_list):
        # 파일명 생성 (예: 00001_SPEAKER_01.mp3)
        idx_str = seg.get("index", f"{i:05d}")
        spk = seg.get("speaker", "Unknown")
        filename = f"{idx_str}_{spk}.mp3"
        file_path = os.path.join(segments_dir, filename)

        # 1. FlowSE 디노이징된 오디오가 있는지 확인 (최우선)
        if seg.get("flowse_denoised", False) and "denoised_audio_path" in seg:
            denoised_path = seg["denoised_audio_path"]
            if os.path.exists(denoised_path):
                # 디노이징된 오디오 파일 로드
                denoised_waveform, denoised_sr = sf.read(denoised_path)

                # float32 (-1.0 ~ 1.0) -> int16 범위로 변환
                denoised_waveform = np.clip(denoised_waveform, -1.0, 1.0)
                wav_int16 = (denoised_waveform * 32767).astype(np.int16)

                target_segment = PydubAudioSegment(
                    wav_int16.tobytes(),
                    frame_rate=denoised_sr,
                    sample_width=2,
                    channels=1
                )
                logger.debug(f"Segment {idx_str}: Saved using FlowSE denoised output.")

        # 2. SepReformer가 적용된 'enhanced_audio'가 있는지 확인
        elif seg.get("is_separated", False) and "enhanced_audio" in seg:
            # Numpy array -> Pydub AudioSegment 변환
            enhanced_waveform = seg["enhanced_audio"]

            # float32 (-1.0 ~ 1.0) -> int16 범위로 변환
            # 클리핑 방지를 위해 clip 적용
            enhanced_waveform = np.clip(enhanced_waveform, -1.0, 1.0)
            wav_int16 = (enhanced_waveform * 32767).astype(np.int16)

            target_segment = PydubAudioSegment(
                wav_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )
            # logger.debug(f"Segment {idx_str}: Saved using SepReformer output.")

        else:
            # 3. 적용되지 않은 경우 원본에서 추출
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            target_segment = full_audio_segment[start_ms:end_ms]

        # MP3 저장
        target_segment.export(file_path, format="mp3")
        
def main_process(audio_path, save_path=None, audio_name=None,
                 do_vad = False,
                 LLM = "",
                 use_demucs = False,
                 use_sepreformer = False,
                 overlap_threshold = 1.0,
                 flowse_denoiser = None,
                 sepreformer_separator = None,
                 embedding_model = None,
                 panns_model = None):

    if not audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac")):
        logger.warning(f"Unsupported file type: {audio_path}")
    # 오디오 convert_mono and split_wav_files_2min temp file 만들기.

    # for a single audio from path Ïaaa/bbb/ccc.wav ---> save to aaa/bbb_processed/ccc/ccc_0.wav
    audio_name = audio_name or os.path.splitext(os.path.basename(audio_path))[0]
    suffix = "dia3" if args.dia3 else "ori"
    save_path = save_path or os.path.join(
        os.path.dirname(audio_path), "_final", f"-sepreformer-{args.sepreformer}" +f"-demucs-{args.demucs}"  + f"-vad-{do_vad}"+ f"-diaModel-{suffix}"
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
                chunk_df = _apply_sortformer_segment_padding_from_args(
                    chunk_df, args=args, logger=logger, audio_duration=audio_duration
                )
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

    # [수정됨] Step 3를 Step 2.5보다 먼저 실행!
    # Step 3: Background Music Detection and Removal
    # SepReformer가 실행되기 전에 전체 오디오를 먼저 깨끗하게 만듭니다.
    logger.info("Step 3: Background Music Detection and Removal")
    # padding을 주어 ASR 타임스탬프 오차 범위를 커버
    audio, segment_demucs_flags = preprocess_segments_with_demucs(segment_list, audio, panns_model=panns_model, use_demucs=use_demucs, padding=0.5)

    # [수정됨] 이제 깨끗해진 audio를 가지고 SepReformer 실행
    # Step 2.5: Overlap control using SepReformer
    logger.info("Step 2.5: Overlap Control with SepReformer")
    separation_time = 0.0
    if use_sepreformer and sepreformer_separator is not None and embedding_model is not None:
        separation_start = time.time()
        # 여기서 audio는 이미 Demucs 처리가 된 상태입니다.
        audio, segment_list = process_overlapping_segments_with_separation(
            segment_list,
            audio,
            overlap_threshold=overlap_threshold,
            separator=sepreformer_separator,
            embedding_model=embedding_model
        )
        separation_end = time.time()
        separation_time = separation_end - separation_start

        # Calculate SepReformer RT factor
        separation_rt = separation_time / audio_duration if audio_duration > 0 else 0
        logger.info(f"SepReformer separation - Processing time: {separation_time:.2f}s, RT factor: {separation_rt:.4f}")
    else:
        logger.info("SepReformer overlap separation skipped (flag disabled)")
        
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

    # Step 4.6: Apply FlowSE denoising to sepreformer segments (if enabled)
    denoise_time = 0.0
    if args.sepreformer and flowse_denoiser is not None:
        logger.info("Step 4.6: Applying FlowSE denoising to sepreformer segments")
        filtered_list, denoise_time = apply_flowse_denoising(
            filtered_list,
            audio,
            save_path,
            denoiser=flowse_denoiser,
            use_asr_moe=args.ASRMoE
        )
    else:
        logger.info("Step 4.6: FlowSE denoising skipped (sepreformer flag disabled or denoiser not loaded)")

    # Calculate FlowSE denoising RT factor
    denoise_rt = denoise_time / audio_duration if audio_duration > 0 else 0

    # Print all timing information
    print(f"\n{'='*60}")
    print(f"Audio duration: {audio_duration:.2f} seconds ({audio_duration/60:.2f} minutes)")
    print(f"{'='*60}")
    print(f"VAD + Sortformer:")
    print(f"  - Processing time: {dia_time:.2f} seconds")
    print(f"  - RT factor: {vad_sortformer_rt:.4f}")
    print(f"{'='*60}")
    if use_sepreformer:
        print(f"SepReformer Overlap Separation:")
        print(f"  - Processing time: {separation_time:.2f} seconds")
        print(f"  - RT factor: {separation_rt:.4f}")
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
    if args.sepreformer and denoise_time > 0:
        print(f"FlowSE Denoising:")
        print(f"  - Processing time: {denoise_time:.2f} seconds")
        print(f"  - RT factor: {denoise_rt:.4f}")
        print(f"{'='*60}")
    print()

    logger.info("Step 5: Write result into MP3 and JSON file")
    print(f"Exporting {len(filtered_list)} segments to MP3 and JSON...")
    export_segments_with_enhanced_audio(audio, filtered_list, save_path, audio_name)

    # 한국어 g2p 후처리
    if args.korean:
        ko_process_json(filtered_list)

    cleaned_list = []
    for item in filtered_list:
        # 얕은 복사(copy)를 통해 원본 filtered_list에는 영향 주지 않도록 함
        clean_item = item.copy()
        
        # 1. 'enhanced_audio' (오디오 행렬) 키가 있으면 삭제
        if "enhanced_audio" in clean_item:
            del clean_item["enhanced_audio"]
        # 1-2. denoised_audio_path는 최종 JSON에 포함하지 않음
        if "denoised_audio_path" in clean_item:
            del clean_item["denoised_audio_path"]
            
        # 2. (혹시 모를 에러 방지) Numpy float/int 타입을 Python native 타입으로 변환
        for k, v in clean_item.items():
            if hasattr(v, 'item'):  # numpy 타입인 경우
                clean_item[k] = v.item()
                
        cleaned_list.append(clean_item)

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
            # [수정] filtered_list 대신 cleaned_list 길이 사용
            "total_segments": len(cleaned_list)
        },
        # [수정] 여기서 filtered_list가 아닌 cleaned_list를 넣어야 합니다.
        "segments": cleaned_list  
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

    # Add SepReformer separation metadata if enabled
    if use_sepreformer:
        output_data["metadata"]["sepreformer_separation"] = {
            "processing_time_seconds": separation_time,
            "rt_factor": separation_rt,
            "overlap_threshold_seconds": overlap_threshold,
            "enabled": True
        }

    # Add FlowSE denoising metadata if enabled
    if args.sepreformer and denoise_time > 0:
        output_data["metadata"]["flowse_denoising"] = {
            "processing_time_seconds": denoise_time,
            "rt_factor": denoise_rt,
            "enabled": True
        }

    final_path = os.path.join(save_path, audio_name + ".json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Cleanup: remove intermediate FlowSE denoised_audio directory
    denoise_dir = os.path.join(save_path, "denoised_audio")
    if os.path.isdir(denoise_dir):
        shutil.rmtree(denoise_dir, ignore_errors=True)
        logger.info(f"Removed temporary denoised_audio directory: {denoise_dir}")

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

    parser.add_argument(
        "--sepreformer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable SepReformer for overlapping speech separation",
    )

    parser.add_argument(
        "--overlap_threshold",
        type=float,
        default=1.0,
        help="Minimum overlap duration in seconds to trigger SepReformer separation",
    )

    # Sortformer diarization segment boundary adjustment (optional)
    parser.add_argument(
        "--sortformer-param",
        "--sortformerParam",
        "--sortformerParma",
        dest="sortformer_param",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Sortformer segment boundary adjustment (pad_offset/pad_onset applied after model output).",
    )
    parser.add_argument(
        "--sortformer-pad-offset",
        type=float,
        default=-0.24,
        help="Seconds to add to segment end time (negative pulls ends earlier). Used with --sortformer-param.",
    )
    parser.add_argument(
        "--sortformer-pad-onset",
        type=float,
        default=0.0,
        help="Seconds to add to segment start time (negative pulls starts earlier). Used with --sortformer-param.",
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

    # Pyannote embedding model 초기화 (sepreformer가 활성화된 경우에만)
    embedding_model = None
    if args.sepreformer:
        logger.debug(" * Loading Pyannote Embedding Model")
        try:
            from pyannote.audio import Model as PyannoteModel
            embedding_model = PyannoteModel.from_pretrained("pyannote/embedding", use_auth_token=cfg["huggingface_token"])
            embedding_model = embedding_model.to(device)
            logger.debug(" * Pyannote Embedding Model loaded successfully")
        except Exception as e:
            logger.error(f" * Failed to load Pyannote Embedding Model: {e}")
            embedding_model = None

    # SepReformer separator 초기화 (sepreformer가 활성화된 경우에만)
    sepreformer_separator = None
    if args.sepreformer:
        logger.debug(" * Loading SepReformer Separator Model")
        try:
            sepreformer_separator = SepReformerSeparator(
                sepreformer_path="/mnt/ddn/kyudan/Audio-data-centric/SepReformer",
                device=device
            )
            logger.debug(" * SepReformer Separator loaded successfully")
        except Exception as e:
            logger.error(f" * Failed to load SepReformer Separator: {e}")
            sepreformer_separator = None

    # FlowSE denoiser 초기화 (sepreformer가 활성화된 경우에만)
    flowse_denoiser = None
    if args.sepreformer:
        logger.debug(" * Loading FlowSE Denoiser Model")
        try:
            flowse_denoiser = FlowSEDenoiser(
                checkpoint_path="/mnt/ddn/kyudan/Audio-data-centric/FlowSE/ckpts/best.pt.tar",
                tokenizer_path="/mnt/ddn/kyudan/Audio-data-centric/FlowSE/Emilia_ZH_EN_pinyin/vocab.txt",
                vocoder_path="/mnt/ddn/kyudan/Audio-data-centric/FlowSE/vocos-mel-24khz",
                use_cuda=(device_name == "cuda")
            )
            logger.debug(" * FlowSE Denoiser loaded successfully")
        except Exception as e:
            logger.error(f" * Failed to load FlowSE Denoiser: {e}")
            flowse_denoiser = None

    # PANNs model 초기화 (배경음악 검출용)
    panns_model = None
    if args.demucs:
        logger.debug(" * Loading PANNs Model for background music detection")
        try:
            panns_data_dir = '/mnt/ddn/kyudan/panns_data'
            os.makedirs(panns_data_dir, exist_ok=True)
            os.environ['PANNS_DATA'] = panns_data_dir
            checkpoint_path = os.path.join(panns_data_dir, 'Cnn14_mAP=0.431.pth')
            panns_model = AudioTagging(checkpoint_path=checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
            logger.debug(" * PANNs Model loaded successfully")
        except Exception as e:
            logger.error(f" * Failed to load PANNs Model: {e}")
            panns_model = None

    logger.debug("All models loaded")

    supported_languages = cfg["language"]["supported"]
    multilingual_flag = cfg["language"]["multilingual"]
    logger.debug(f"supported languages multilingual {supported_languages}")
    logger.debug(f"using multilingual asr {multilingual_flag}")

    input_folder_path = cfg["entrypoint"]["input_folder_path"]

    if not os.path.exists(input_folder_path):
        raise FileNotFoundError(f"input_folder_path: {input_folder_path} not found")

    # Get only audio files in the specified directory (not recursive)
    audio_extensions = ('.mp3', '.wav', '.flac', '.m4a', '.aac')
    audio_paths = []
    for file in os.listdir(input_folder_path):
        if file.endswith(audio_extensions) and not ".temp" in file:
            audio_paths.append(os.path.join(input_folder_path, file))

    logger.debug(f"Scanning {len(audio_paths)} audio files in {input_folder_path} (non-recursive)")

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


        main_process(path, do_vad=args.vad, LLM=args.LLM, use_demucs=args.demucs,
                     use_sepreformer=args.sepreformer, overlap_threshold=args.overlap_threshold,
                     flowse_denoiser=flowse_denoiser, sepreformer_separator=sepreformer_separator,
                     embedding_model=embedding_model, panns_model=panns_model)
end_time = time.time()
print("Total time:", end_time - start_time)
