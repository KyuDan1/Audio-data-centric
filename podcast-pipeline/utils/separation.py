"""
Audio source separation utilities for podcast pipeline.
Includes SepReformer-based speaker separation and overlapping segment processing.
"""

import os
import sys
import numpy as np
import librosa
import torch
from utils.logger import time_logger
from utils.diarization import detect_overlapping_segments

# Logger will be initialized from main module
logger = None

def set_logger(log_instance):
    """Set logger instance from main module."""
    global logger
    logger = log_instance


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
            podcast_pipeline_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
def identify_speaker_with_embedding(audio_segment, sample_rate, reference_embeddings, speaker_labels, embedding_model, device):
    """
    Identify which speaker an audio segment belongs to using speaker embeddings.

    Args:
        audio_segment (np.ndarray): Audio segment to identify
        sample_rate (int): Sample rate of the audio
        reference_embeddings (dict): Dictionary of {speaker_label: embedding_tensor}
        speaker_labels (list): List of possible speaker labels
        embedding_model: 미리 로드된 pyannote embedding 모델
        device: torch device (cuda/cpu)

    Returns:
        tuple: (best_speaker_label or None, best_similarity_score)
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
    return best_speaker, best_similarity


@time_logger
def process_overlapping_segments_with_separation(segment_list, audio, overlap_threshold=1.0,
                                                 separator=None, embedding_model=None, device="cuda"):
    """
    Process overlapping segments by separating them with SepReformer.
    [Updated] Matches the volume of separated audio to the original overlap audio to prevent volume jumps.

    Args:
        segment_list: 세그먼트 리스트
        audio: 오디오 딕셔너리
        overlap_threshold: 오버랩 임계값
        separator: 미리 로드된 SepReformerSeparator 객체
        embedding_model: 미리 로드된 pyannote embedding 모델
        device: torch device (cuda/cpu)
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
    def get_non_overlap_rms(segment, waveform, sample_rate, overlapping_pairs):
        """
        세그먼트의 non-overlap 구간에서 RMS 에너지를 계산합니다.

        Args:
            segment: 세그먼트 딕셔너리
            waveform: 전체 오디오 파형
            sample_rate: 샘플레이트
            overlapping_pairs: 오버랩 쌍 리스트

        Returns:
            float: Non-overlap 구간의 RMS 에너지 (계산 불가능하면 None)
        """
        seg_start = segment['start']
        seg_end = segment['end']

        # 이 세그먼트가 겹치는 모든 구간을 찾기
        overlap_regions = []
        for pair in overlapping_pairs:
            if pair['seg1'] == segment or pair['seg2'] == segment:
                overlap_regions.append((pair['overlap_start'], pair['overlap_end']))

        if not overlap_regions:
            # 오버랩이 없으면 전체 세그먼트 사용
            start_frame = int(seg_start * sample_rate)
            end_frame = int(seg_end * sample_rate)
            seg_audio = waveform[start_frame:end_frame]
        else:
            # Non-overlap 구간만 추출
            non_overlap_parts = []
            overlap_regions.sort()

            # 세그먼트 시작부터 첫 오버랩까지
            if overlap_regions[0][0] > seg_start:
                start_frame = int(seg_start * sample_rate)
                end_frame = int(overlap_regions[0][0] * sample_rate)
                non_overlap_parts.append(waveform[start_frame:end_frame])

            # 오버랩 사이의 구간들
            for i in range(len(overlap_regions) - 1):
                start_frame = int(overlap_regions[i][1] * sample_rate)
                end_frame = int(overlap_regions[i+1][0] * sample_rate)
                if end_frame > start_frame:
                    non_overlap_parts.append(waveform[start_frame:end_frame])

            # 마지막 오버랩부터 세그먼트 끝까지
            if overlap_regions[-1][1] < seg_end:
                start_frame = int(overlap_regions[-1][1] * sample_rate)
                end_frame = int(seg_end * sample_rate)
                non_overlap_parts.append(waveform[start_frame:end_frame])

            if not non_overlap_parts:
                return None

            seg_audio = np.concatenate(non_overlap_parts)

        if len(seg_audio) == 0:
            return None

        rms = np.sqrt(np.mean(seg_audio**2))
        return rms if rms > 1e-10 else None

    def match_target_amplitude(source_wav, target_rms):
        """
        source_wav의 볼륨(RMS)을 target_rms 에너지에 맞춥니다.

        Args:
            source_wav: 조정할 오디오 파형
            target_rms: 목표 RMS 에너지 값
        """
        # 0으로 나누기 방지용 엡실론
        epsilon = 1e-10

        # RMS(Root Mean Square) 에너지 계산
        src_rms = np.sqrt(np.mean(source_wav**2))

        if src_rms < epsilon or target_rms is None or target_rms < epsilon:
            return source_wav

        # 비율 계산 (Target이 Source보다 얼마나 큰지/작은지)
        gain = target_rms / (src_rms + epsilon)

        # Gain 적용
        adjusted_wav = source_wav * gain

        # 클리핑 방지 (-1.0 ~ 1.0)
        return np.clip(adjusted_wav, -1.0, 1.0)

    def calculate_energy(audio_segment):
        """
        오디오 세그먼트의 에너지를 계산합니다.
        """
        return np.sum(audio_segment**2)
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

        # Identify speakers with embedding matching
        speaker1_identity, similarity1 = identify_speaker_with_embedding(
            separated_src1, sample_rate, reference_embeddings, [seg1_speaker, seg2_speaker], embedding_model, device
        )
        speaker2_identity, similarity2 = identify_speaker_with_embedding(
            separated_src2, sample_rate, reference_embeddings, [seg1_speaker, seg2_speaker], embedding_model, device
        )

        # ---------------------------------------------------------------------
        # [안정성 개선] Embedding 매칭 실패 시 fallback 처리
        # ---------------------------------------------------------------------
        assignment_method = "embedding"

        # Case 1: 임베딩 매칭이 성공하고 두 소스가 서로 다른 화자로 매칭됨
        if (speaker1_identity is not None and speaker2_identity is not None and
            speaker1_identity != speaker2_identity):
            if speaker1_identity == seg1_speaker:
                seg1_part = separated_src1
                seg2_part = separated_src2
            else:
                seg1_part = separated_src2
                seg2_part = separated_src1
            logger.info(f"  Speaker assignment by embedding: src1={speaker1_identity} ({similarity1:.3f}), src2={speaker2_identity} ({similarity2:.3f})")

        # Case 2: 임베딩 매칭 실패 또는 두 소스가 같은 화자로 매칭됨 -> 에너지 기반 fallback
        else:
            assignment_method = "energy_fallback"
            logger.warning(f"  Embedding matching failed or ambiguous (src1={speaker1_identity}, src2={speaker2_identity})")
            logger.info(f"  Using energy-based fallback for speaker assignment")

            # 세그먼트 길이 기반으로 에너지가 높은 쪽을 더 긴 세그먼트에 할당
            seg1_duration = seg1['end'] - seg1['start']
            seg2_duration = seg2['end'] - seg2['start']

            energy1 = calculate_energy(separated_src1)
            energy2 = calculate_energy(separated_src2)

            # 더 긴 세그먼트에 에너지가 높은 소스를 할당
            if seg1_duration >= seg2_duration:
                if energy1 >= energy2:
                    seg1_part = separated_src1
                    seg2_part = separated_src2
                else:
                    seg1_part = separated_src2
                    seg2_part = separated_src1
            else:
                if energy2 >= energy1:
                    seg1_part = separated_src2
                    seg2_part = separated_src1
                else:
                    seg1_part = separated_src1
                    seg2_part = separated_src2

            logger.info(f"  Energy-based assignment: seg1_dur={seg1_duration:.2f}s, seg2_dur={seg2_duration:.2f}s, "
                       f"energy1={energy1:.2e}, energy2={energy2:.2e}")
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # [수정] 볼륨 보정 적용 (각 세그먼트의 non-overlap 구간 RMS에 맞춤)
        # ---------------------------------------------------------------------
        seg1_target_rms = get_non_overlap_rms(seg1, waveform, sample_rate, overlapping_pairs)
        seg2_target_rms = get_non_overlap_rms(seg2, waveform, sample_rate, overlapping_pairs)

        # Fallback: non-overlap RMS를 계산할 수 없으면 overlap 구간의 RMS를 절반으로 사용
        overlap_rms = np.sqrt(np.mean(overlap_audio**2))
        if seg1_target_rms is None:
            seg1_target_rms = overlap_rms * 0.7  # 약간 보수적으로
            logger.debug(f"   No non-overlap region for seg1, using fallback RMS")
        if seg2_target_rms is None:
            seg2_target_rms = overlap_rms * 0.7
            logger.debug(f"   No non-overlap region for seg2, using fallback RMS")

        logger.debug(f"   Adjusting volume for overlap {pair_idx+1} (method: {assignment_method})...")
        logger.debug(f"   seg1 target RMS: {seg1_target_rms:.6f}, seg2 target RMS: {seg2_target_rms:.6f}")

        seg1_part = match_target_amplitude(seg1_part, seg1_target_rms)
        seg2_part = match_target_amplitude(seg2_part, seg2_target_rms)
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
