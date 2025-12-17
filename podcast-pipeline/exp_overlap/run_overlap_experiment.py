import os
import sys
import json
import time
import logging
import tempfile
import subprocess
import re
import torch
import numpy as np
import soundfile as sf
import pandas as pd
import whisper
import librosa
from pathlib import Path
from tqdm import tqdm
from jiwer import wer
from pyannote.audio import Inference

# Metrics Libraries
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from pystoi import stoi

# Logger 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 경로 설정 (사용자 환경에 맞게 수정)
sys.path.append("/mnt/ddn/kyudan/Audio-data-centric/podcast-pipeline/exp_overlap") 

class SepReformerSeparator:
    """
    SepReformer 모델을 한 번만 로드하고 여러 번 inference할 수 있는 클래스
    """
    def __init__(self, sepreformer_path, device):
        import sys
        import yaml

        self.sepreformer_path = sepreformer_path
        self.device = device

        print(f"[SepReformer] Initializing on device: {self.device}")

        original_sys_path = sys.path.copy()

        try:
            modules_to_clear = [key for key in list(sys.modules.keys())
                              if key.startswith('models.') or key.startswith('utils.') or key in ['models', 'utils']]
            for module_name in modules_to_clear:
                sys.modules.pop(module_name)

            sys.path.insert(0, sepreformer_path)

            from models.SepReformer_Base_WSJ0.model import Model
            self.Model = Model

            config_path = os.path.join(sepreformer_path, "models/SepReformer_Base_WSJ0/configs.yaml")
            with open(config_path, 'r') as f:
                yaml_dict = yaml.safe_load(f)
            self.config = yaml_dict["config"]

            print("[SepReformer] Loading model...")
            self.model = Model(**self.config["model"])

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
            sys.path = original_sys_path

    def separate(self, audio_segment, sample_rate):
        try:
            original_len = len(audio_segment)

            if sample_rate != 8000:
                audio_8k = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=8000)
            else:
                audio_8k = audio_segment

            mixture_tensor = torch.tensor(audio_8k, dtype=torch.float32).unsqueeze(0)

            stride = self.config["model"]["module_audio_enc"]["stride"]
            remains = mixture_tensor.shape[-1] % stride
            if remains != 0:
                padding = stride - remains
                mixture_padded = torch.nn.functional.pad(mixture_tensor, (0, padding), "constant", 0)
            else:
                mixture_padded = mixture_tensor

            with torch.inference_mode():
                nnet_input = mixture_padded.to(self.device)
                estim_src, _ = self.model(nnet_input)

                src1 = estim_src[0][..., :mixture_tensor.shape[-1]].squeeze().cpu().numpy()
                src2 = estim_src[1][..., :mixture_tensor.shape[-1]].squeeze().cpu().numpy()

            if sample_rate != 8000:
                src1 = librosa.resample(src1, orig_sr=8000, target_sr=sample_rate)
                src2 = librosa.resample(src2, orig_sr=8000, target_sr=sample_rate)

            if len(src1) != original_len:
                if len(src1) > original_len:
                    src1 = src1[:original_len]
                else:
                    src1 = np.pad(src1, (0, original_len - len(src1)), mode='constant')

            if len(src2) != original_len:
                if len(src2) > original_len:
                    src2 = src2[:original_len]
                else:
                    src2 = np.pad(src2, (0, original_len - len(src2)), mode='constant')

            return src1, src2

        except Exception as e:
            logger.error(f"SepReformer separation failed: {e}")
            return audio_segment, audio_segment

# FlowSE Denoiser
sys.path.insert(0, "/mnt/ddn/kyudan/Audio-data-centric/FlowSE")
try:
    from simple_denoise import FlowSEDenoiser
    FLOWSE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ [Warning] FlowSE를 import할 수 없습니다: {e}")
    FLOWSE_AVAILABLE = False
    FlowSEDenoiser = None

# UTMOS
try:
    UTMOS_PREDICTOR = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    HAS_UTMOS = True
except Exception as e:
    UTMOS_PREDICTOR = None
    HAS_UTMOS = False

# ================= 설정값 =================
DATASET_DIR = Path("/mnt/ddn/kyudan/Audio-data-centric/podcast-pipeline/exp_overlap/synthetic_overlap_dataset")
METADATA_PATH = DATASET_DIR / "metadata.json"
RESULTS_DIR = DATASET_DIR / "experiment_results"

SEPREFORMER_PATH = "/mnt/ddn/kyudan/Audio-data-centric/SepReformer"
FLOWSE_CKPT = "/mnt/ddn/kyudan/Audio-data-centric/FlowSE/ckpts/best.pt.tar"
FLOWSE_VOCAB = "/mnt/ddn/kyudan/Audio-data-centric/FlowSE/Emilia_ZH_EN_pinyin/vocab.txt"
FLOWSE_VOCODER = "/mnt/ddn/kyudan/Audio-data-centric/FlowSE/vocos-mel-24khz"
HF_TOKEN = "hf_..." # HuggingFace Token 입력

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MetricsCalculator:
    def __init__(self, device, utmos_predictor=None):
        self.device = device
        self.si_sdr_func = ScaleInvariantSignalDistortionRatio().to(device)
        self.utmos_predictor = utmos_predictor
        if self.utmos_predictor is not None:
            self.utmos_predictor.to(device)
            self.utmos_predictor.eval()

    def compute(self, reference, estimate, sr=16000):
        metrics = {}
        min_len = min(len(reference), len(estimate))
        ref = reference[:min_len]
        est = estimate[:min_len]

        if min_len < 160:
            return {'si_sdr': 0.0, 'stoi': 0.0, 'utmos': 0.0}

        try:
            ref_tensor = torch.tensor(ref, device=self.device).unsqueeze(0)
            est_tensor = torch.tensor(est, device=self.device).unsqueeze(0)
            metrics['si_sdr'] = self.si_sdr_func(est_tensor, ref_tensor).item()
        except Exception:
            metrics['si_sdr'] = -99.0

        try:
            metrics['stoi'] = stoi(ref, est, sr, extended=False)
        except Exception:
            metrics['stoi'] = 0.0

        metrics['utmos'] = 0.0
        if self.utmos_predictor is not None:
            try:
                if sr != 16000:
                    est_16k = librosa.resample(est, orig_sr=sr, target_sr=16000)
                else:
                    est_16k = est
                wave_tensor = torch.tensor(est_16k, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    score = self.utmos_predictor(wave_tensor, sr=16000)
                    metrics['utmos'] = score.mean().item() if isinstance(score, torch.Tensor) else float(score)
            except Exception:
                metrics['utmos'] = 0.0
        return metrics

# ================= 유틸리티 함수 =================
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_models():
    print("🚀 Loading Models...")
    asr_model = whisper.load_model("large-v3", device=DEVICE)
    separator = SepReformerSeparator(sepreformer_path=SEPREFORMER_PATH, device=DEVICE)
    embedder = Inference("pyannote/embedding", device=DEVICE, use_auth_token=HF_TOKEN)
    evaluator = MetricsCalculator(device=DEVICE, utmos_predictor=UTMOS_PREDICTOR)
    
    denoiser = None
    if FLOWSE_AVAILABLE and FlowSEDenoiser is not None:
        try:
            denoiser = FlowSEDenoiser(
                checkpoint_path=FLOWSE_CKPT,
                tokenizer_path=FLOWSE_VOCAB,
                vocoder_path=FLOWSE_VOCODER,
                use_cuda=(DEVICE == "cuda")
            )
        except Exception as e:
            print(f"⚠️ [Warning] FlowSE denoiser 로드 실패: {e}")
    return asr_model, separator, embedder, evaluator, denoiser

def ensure_float32(audio):
    if audio.dtype != np.float32:
        return audio.astype(np.float32)
    return audio

def get_embedding(audio_array, sr, embedder):
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    try:
        return embedder({"waveform": audio_tensor, "sample_rate": 16000})
    except:
        return None

def match_speakers(sep1, sep2, ref_s1, ref_s2, sr, embedder):
    emb_sep1 = get_embedding(sep1, sr, embedder)
    emb_sep2 = get_embedding(sep2, sr, embedder)
    emb_ref1 = get_embedding(ref_s1, sr, embedder)
    
    if emb_sep1 is None or emb_sep2 is None or emb_ref1 is None:
        return sep1, sep2

    sim_1_1 = torch.nn.functional.cosine_similarity(torch.tensor(emb_sep1), torch.tensor(emb_ref1), dim=0)
    sim_2_1 = torch.nn.functional.cosine_similarity(torch.tensor(emb_sep2), torch.tensor(emb_ref1), dim=0)

    return (sep1, sep2) if sim_1_1 > sim_2_1 else (sep2, sep1)

# [추가됨] 볼륨 매칭 함수 (RMS 기준)
def adjust_gain(target_audio, reference_audio):
    """
    reference_audio의 RMS 수준에 맞춰 target_audio의 볼륨을 조절합니다.
    """
    eps = 1e-8
    # 0으로 나누기 방지 및 reference가 무음일 경우 처리
    ref_rms = np.sqrt(np.mean(reference_audio**2) + eps)
    target_rms = np.sqrt(np.mean(target_audio**2) + eps)
    
    if target_rms < eps:
        return target_audio # 대상이 무음이면 그대로 반환

    gain = ref_rms / target_rms
    
    # 너무 과도한 증폭 방지 (선택 사항, 필요시 주석 해제)
    # gain = min(gain, 5.0) 
    
    return target_audio * gain

# [수정됨] Stitching 함수 + 볼륨 매칭 적용
def stitch_audio(mixed_audio, separated_overlap, overlap_start, overlap_end, spk_start, spk_end):
    """
    화자별 완전한 오디오를 만들기 위해 mixed audio와 separated overlap을 합침.
    **Overap 구간의 볼륨을 Non-overlap 구간(문맥)의 볼륨에 맞춥니다.**
    """
    full_len = spk_end - spk_start
    stitched = np.zeros(full_len, dtype=np.float32)

    # Context (Volume Reference) 수집을 위한 리스트
    context_audios = []

    # 1. Pre-overlap 구간 처리
    pre_mix_start = max(0, spk_start)
    pre_mix_end = min(len(mixed_audio), overlap_start)
    
    if pre_mix_end > pre_mix_start:
        dest_start = pre_mix_start - spk_start
        dest_end = dest_start + (pre_mix_end - pre_mix_start)
        valid_dest_start = max(0, dest_start)
        valid_dest_end = min(len(stitched), dest_end)
        
        if valid_dest_end > valid_dest_start:
            src_offset = valid_dest_start - dest_start
            src_start = pre_mix_start + src_offset
            src_end = src_start + (valid_dest_end - valid_dest_start)
            
            segment = mixed_audio[src_start:src_end]
            stitched[valid_dest_start:valid_dest_end] = segment
            context_audios.append(segment) # 참조용으로 저장

    # 3. Post-overlap 구간 처리 (순서상 먼저 추출하여 context 확보)
    post_mix_start = max(0, overlap_end)
    post_mix_end = min(len(mixed_audio), spk_end)
    
    if post_mix_end > post_mix_start:
        dest_start = post_mix_start - spk_start
        dest_end = dest_start + (post_mix_end - post_mix_start)
        valid_dest_start = max(0, dest_start)
        valid_dest_end = min(len(stitched), dest_end)
        
        if valid_dest_end > valid_dest_start:
            src_offset = valid_dest_start - dest_start
            src_start = post_mix_start + src_offset
            src_end = src_start + (valid_dest_end - valid_dest_start)
            
            segment = mixed_audio[src_start:src_end]
            stitched[valid_dest_start:valid_dest_end] = segment
            context_audios.append(segment) # 참조용으로 저장

    # 2. Overlap 구간 처리 + [볼륨 매칭]
    if len(separated_overlap) > 0:
        # 분리된 오디오의 볼륨을 앞뒤 문맥(context)에 맞춤
        if context_audios:
            reference_audio = np.concatenate(context_audios)
            # 만약 context가 너무 짧으면(예: 0.1초 미만) 조정이 불안정할 수 있으므로 체크 가능
            if len(reference_audio) > 160: 
                separated_overlap = adjust_gain(separated_overlap, reference_audio)

        dest_start = overlap_start - spk_start
        dest_end = dest_start + len(separated_overlap)
        
        valid_dest_start = max(0, dest_start)
        valid_dest_end = min(len(stitched), dest_end)
        
        if valid_dest_end > valid_dest_start:
            src_start = valid_dest_start - dest_start
            src_end = src_start + (valid_dest_end - valid_dest_start)
            stitched[valid_dest_start:valid_dest_end] = separated_overlap[src_start:src_end]
            
    else:
        # Fallback: 원본 Mix 사용
        mix_ov_start = max(0, overlap_start)
        mix_ov_end = min(len(mixed_audio), overlap_end)
        
        if mix_ov_end > mix_ov_start:
            source_segment = mixed_audio[mix_ov_start:mix_ov_end]
            dest_start = mix_ov_start - spk_start
            dest_end = dest_start + len(source_segment)
            
            valid_dest_start = max(0, dest_start)
            valid_dest_end = min(len(stitched), dest_end)
            
            if valid_dest_end > valid_dest_start:
                src_offset = valid_dest_start - dest_start
                stitched[valid_dest_start:valid_dest_end] = source_segment[src_offset : src_offset + (valid_dest_end - valid_dest_start)]

    return stitched

def run_experiment():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    asr_model, separator, embedder, evaluator, denoiser = load_models()
    results = []
    
    for item in tqdm(metadata, desc="Running Experiments"):
        sample_id = item['sample_id']
        sr = 16000 
        
        mixed, _ = sf.read(item['mixed_path'])
        s1_clean_full, _ = sf.read(item['s1_path']) 
        s2_clean_full, _ = sf.read(item['s2_path'])
        
        timing = item['timing']
        ov_start = int(timing['overlap_start'] * sr)
        ov_end = int(timing['overlap_end'] * sr)
        s1_start = int(timing['speaker1_start'] * sr)
        s1_end = int(timing['speaker1_end'] * sr)
        s2_start = int(timing['speaker2_start'] * sr)
        s2_end = int(timing['speaker2_end'] * sr)
        
        ref_text_s1 = item['s1_text']
        ref_text_s2 = item['s2_text']
        ref_text_s1_norm = normalize_text(ref_text_s1)
        ref_text_s2_norm = normalize_text(ref_text_s2)

        ref_audio_s1 = s1_clean_full[s1_start:s1_end]
        ref_audio_s2 = s2_clean_full[s2_start:s2_end]
        
        row = {
            'sample_id': sample_id,
            'config': item['config'],
            'sir_db': item['sir_db'],
            'overlap_ratio': item['overlap_ratio']
        }

        # --- Exp 1: Baseline ---
        s1_base_audio = ensure_float32(mixed[s1_start:s1_end])
        s2_base_audio = ensure_float32(mixed[s2_start:s2_end])

        hyp_s1_base = asr_model.transcribe(s1_base_audio, language='en')['text'].strip()
        hyp_s2_base = asr_model.transcribe(s2_base_audio, language='en')['text'].strip()
        
        row['wer_s1_base'] = wer(ref_text_s1_norm, normalize_text(hyp_s1_base))
        row['wer_s2_base'] = wer(ref_text_s2_norm, normalize_text(hyp_s2_base))

        m_s1_base = evaluator.compute(ref_audio_s1, s1_base_audio, sr)
        m_s2_base = evaluator.compute(ref_audio_s2, s2_base_audio, sr)
        for k, v in m_s1_base.items(): row[f'{k}_s1_base'] = v
        for k, v in m_s2_base.items(): row[f'{k}_s2_base'] = v
        
        # --- Exp 2: SepReformer (Stitching + Volume Matching) ---
        overlap_audio = ensure_float32(mixed[ov_start:ov_end])

        if len(overlap_audio) > 160:
            sep1, sep2 = separator.separate(overlap_audio, sr)
            if len(sep1) == 0 or len(sep2) == 0:
                sep1, sep2 = overlap_audio, overlap_audio
        else:
            logger.warning(f"Sample {sample_id}: Overlap audio too short, using as-is")
            sep1, sep2 = overlap_audio, overlap_audio

        sep_s1, sep_s2 = match_speakers(sep1, sep2, s1_clean_full, s2_clean_full, sr, embedder)

        # stitch_audio 내부에서 adjust_gain이 수행됨
        s1_sep_audio = ensure_float32(stitch_audio(mixed, sep_s1, ov_start, ov_end, s1_start, s1_end))
        s2_sep_audio = ensure_float32(stitch_audio(mixed, sep_s2, ov_start, ov_end, s2_start, s2_end))

        # Stitched Audio에 대해 ASR 수행
        hyp_s1_sep = asr_model.transcribe(s1_sep_audio, language='en')['text'].strip()
        hyp_s2_sep = asr_model.transcribe(s2_sep_audio, language='en')['text'].strip()

        row['wer_s1_sep'] = wer(ref_text_s1_norm, normalize_text(hyp_s1_sep))
        row['wer_s2_sep'] = wer(ref_text_s2_norm, normalize_text(hyp_s2_sep))
        
        m_s1_sep = evaluator.compute(ref_audio_s1, s1_sep_audio, sr)
        m_s2_sep = evaluator.compute(ref_audio_s2, s2_sep_audio, sr)
        for k, v in m_s1_sep.items(): row[f'{k}_s1_sep'] = v
        for k, v in m_s2_sep.items(): row[f'{k}_s2_sep'] = v

        # --- Exp 3: FlowSE (Stitched Audio + Stitched ASR Text -> Denoise Whole) ---
        if denoiser is not None:
            temp_dir = RESULTS_DIR / "temp_flowse"
            temp_dir.mkdir(exist_ok=True)
            s1_temp_in = str(temp_dir / f"{sample_id}_s1_in.wav")
            s2_temp_in = str(temp_dir / f"{sample_id}_s2_in.wav")
            s1_out_path = str(RESULTS_DIR / f"flowse_{sample_id}_s1.wav")
            s2_out_path = str(RESULTS_DIR / f"flowse_{sample_id}_s2.wav")

            # 1. Stitched 된 전체 오디오 저장
            sf.write(s1_temp_in, s1_sep_audio, sr)
            sf.write(s2_temp_in, s2_sep_audio, sr)

            try:
                # 2. Stitched 된 오디오에서 나온 ASR 결과(hyp_s1_sep)를 Condition으로 사용
                # 3. 전체 오디오(s1_temp_in)에 대해 Denoising 수행
                denoiser.denoise(s1_temp_in, hyp_s1_sep, s1_out_path)
                denoiser.denoise(s2_temp_in, hyp_s2_sep, s2_out_path)

                s1_flow_audio, _ = sf.read(s1_out_path)
                s2_flow_audio, _ = sf.read(s2_out_path)
                s1_flow_audio = ensure_float32(s1_flow_audio)
                s2_flow_audio = ensure_float32(s2_flow_audio)
            except Exception as e:
                logger.error(f"FlowSE denoising failed: {e}")
                s1_flow_audio = s1_sep_audio
                s2_flow_audio = s2_sep_audio
        else:
            s1_flow_audio = s1_sep_audio
            s2_flow_audio = s2_sep_audio

        hyp_s1_flow = asr_model.transcribe(s1_flow_audio, language='en')['text'].strip()
        hyp_s2_flow = asr_model.transcribe(s2_flow_audio, language='en')['text'].strip()

        row['wer_s1_flow'] = wer(ref_text_s1_norm, normalize_text(hyp_s1_flow))
        row['wer_s2_flow'] = wer(ref_text_s2_norm, normalize_text(hyp_s2_flow))
        
        m_s1_flow = evaluator.compute(ref_audio_s1, s1_flow_audio, sr)
        m_s2_flow = evaluator.compute(ref_audio_s2, s2_flow_audio, sr)
        for k, v in m_s1_flow.items(): row[f'{k}_s1_flow'] = v
        for k, v in m_s2_flow.items(): row[f'{k}_s2_flow'] = v
        
        results.append(row)
        
        if len(results) % 5 == 0:
            pd.DataFrame(results).to_csv(RESULTS_DIR / "temp_metrics.csv", index=False)

    df = pd.DataFrame(results)
    
    metrics_cols = ['wer', 'si_sdr', 'stoi', 'utmos']
    cases = ['base', 'sep', 'flow']

    print("\n=== Experiment Summary ===")
    for case in cases:
        print(f"\n--- {case.upper()} Case ---")
        for m in metrics_cols:
            avg_s1 = df[f'{m}_s1_{case}'].mean()
            avg_s2 = df[f'{m}_s2_{case}'].mean()
            print(f"{m.upper()}: S1={avg_s1:.3f}, S2={avg_s2:.3f}, Avg={(avg_s1+avg_s2)/2:.3f}")

    final_csv = RESULTS_DIR / "final_metrics.csv"
    df.to_csv(final_csv, index=False)
    print(f"\nSaved full results to {final_csv}")

if __name__ == "__main__":
    run_experiment()