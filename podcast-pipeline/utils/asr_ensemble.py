"""
ASR and ensemble utilities for podcast pipeline.
Includes ROVER ensemble, repetition filtering, and multi-model ASR processing.
"""

import collections
import time
import tempfile
import concurrent.futures
from typing import List, Tuple, Dict, Any
from itertools import zip_longest
from difflib import SequenceMatcher
import numpy as np
import librosa
import soundfile as sf
from utils.logger import time_logger

# Logger will be initialized from main module
logger = None

def set_logger(log_instance):
    """Set logger instance from main module."""
    global logger
    logger = log_instance


class RoverEnsembler:
    """
    ROVER(Recognizer Output Voting Error Reduction) 앙상블 구현.
    여러 ASR 모델의 출력을 결합하여 더 정확한 전사를 생성합니다.

    [Updated] SequenceMatcher 기반 정렬을 통해 토큰 길이 차이에도 올바른 다수결 투표 수행
    """

    @staticmethod
    def align_tokens_with_sequencematcher(base_tokens: List[str], candidate_tokens: List[str]) -> List[Tuple[str, str]]:
        """
        SequenceMatcher를 사용하여 두 토큰 시퀀스를 정렬합니다.

        Args:
            base_tokens: 기준 토큰 리스트 (예: Whisper)
            candidate_tokens: 정렬할 토큰 리스트 (예: Canary/Parakeet)

        Returns:
            정렬된 (base_token, candidate_token) 튜플 리스트
            매칭되지 않는 위치는 None으로 표시됨
        """
        matcher = SequenceMatcher(None, base_tokens, candidate_tokens)
        aligned_pairs = []

        for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
            if opcode == 'equal':
                # 완전히 매칭되는 구간
                for base_tok, cand_tok in zip(base_tokens[i1:i2], candidate_tokens[j1:j2]):
                    aligned_pairs.append((base_tok, cand_tok))

            elif opcode == 'replace':
                # 치환 구간 - 더 긴 쪽에 맞춰서 정렬
                base_chunk = base_tokens[i1:i2]
                cand_chunk = candidate_tokens[j1:j2]

                max_len = max(len(base_chunk), len(cand_chunk))
                for idx in range(max_len):
                    base_tok = base_chunk[idx] if idx < len(base_chunk) else None
                    cand_tok = cand_chunk[idx] if idx < len(cand_chunk) else None
                    aligned_pairs.append((base_tok, cand_tok))

            elif opcode == 'delete':
                # base에만 존재 (candidate에서 삭제됨)
                for base_tok in base_tokens[i1:i2]:
                    aligned_pairs.append((base_tok, None))

            elif opcode == 'insert':
                # candidate에만 존재 (base에 삽입됨)
                for cand_tok in candidate_tokens[j1:j2]:
                    aligned_pairs.append((None, cand_tok))

        return aligned_pairs

    @staticmethod
    def build_confusion_network(aligned_sequences: List[List[Tuple]]) -> List[List[str]]:
        """
        정렬된 시퀀스들로부터 confusion network를 구성합니다.

        Args:
            aligned_sequences: 정렬된 (token, token, ...) 튜플의 리스트

        Returns:
            각 위치별 후보 토큰들의 리스트
        """
        if not aligned_sequences:
            return []

        # 가장 긴 시퀀스 길이 찾기
        max_len = max(len(seq) for seq in aligned_sequences)

        # 각 위치별 후보 수집
        confusion_network = []
        for pos in range(max_len):
            candidates = []
            for seq in aligned_sequences:
                if pos < len(seq):
                    token = seq[pos]
                    if token is not None and token != "":
                        candidates.append(token)
            confusion_network.append(candidates)

        return confusion_network

    @staticmethod
    def align_and_vote(transcripts: List[str]) -> str:
        """
        여러 전사 결과를 정렬하고 다수결 투표를 수행합니다.

        [Updated] SequenceMatcher 기반 정렬로 토큰 길이 차이 문제 해결

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
        tokenized = [t.split() for t in transcripts]

        # Base를 첫 번째 전사(Whisper)로 설정
        base_tokens = tokenized[0]

        # -----------------------------------------------------------------------
        # Step 1: 각 candidate를 base에 정렬
        # -----------------------------------------------------------------------
        aligned_sequences = [base_tokens]  # Base는 그대로

        for i in range(1, len(tokenized)):
            candidate_tokens = tokenized[i]
            aligned_pairs = RoverEnsembler.align_tokens_with_sequencematcher(base_tokens, candidate_tokens)

            # aligned_pairs에서 candidate 토큰만 추출 (base 위치에 맞춰짐)
            aligned_candidate = [pair[1] for pair in aligned_pairs]
            aligned_sequences.append(aligned_candidate)

        # -----------------------------------------------------------------------
        # Step 2: Confusion Network 구성
        # -----------------------------------------------------------------------
        # 모든 시퀀스의 길이를 맞춤
        max_len = max(len(seq) for seq in aligned_sequences)
        padded_sequences = []
        for seq in aligned_sequences:
            padded = seq + [None] * (max_len - len(seq))
            padded_sequences.append(padded)

        # -----------------------------------------------------------------------
        # Step 3: 각 위치별 다수결 투표
        # -----------------------------------------------------------------------
        final_output = []

        for pos in range(max_len):
            # 현 위치의 모든 후보 수집
            candidates = []
            for seq in padded_sequences:
                token = seq[pos]
                if token is not None and token != "":
                    candidates.append(token)

            if not candidates:
                continue

            # 투표
            votes = collections.Counter(candidates)
            best_word, count = votes.most_common(1)[0]

            # 2개 이상이 동의하면 채택, 그렇지 않으면 base(Whisper) 채택
            if count >= 2:
                final_output.append(best_word)
            else:
                # Base 우선
                base_word = padded_sequences[0][pos]
                if base_word is not None and base_word != "":
                    final_output.append(base_word)
                else:
                    final_output.append(best_word)

        result = " ".join(final_output)

        # 디버깅 로그 (옵션)
        if logger and len(transcripts) > 1:
            logger.debug(f"[ROVER] Input transcripts: {len(transcripts)}")
            logger.debug(f"[ROVER] Base length: {len(base_tokens)}, Aligned length: {max_len}")
            logger.debug(f"[ROVER] Final output length: {len(final_output)}")

        return result


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
def asr(vad_segments, audio, asr_model):
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

    # asr_MoE 방식(개별 처리)을 따르므로 배치 사이즈는 1로 처리하거나,
    # 라이브러리 지원 여부에 따라 조정 가능하지만 여기서는 정확성을 위해 개별 처리를 우선합니다.
    batch_size = 1

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


@time_logger
def asr_MoE(vad_segments, audio, asr_model, asr_model_2, canary_model, segment_demucs_flags=None, enable_word_timestamps=False, device="cuda"):
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
