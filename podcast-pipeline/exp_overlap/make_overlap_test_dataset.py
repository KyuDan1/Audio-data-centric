import os
import json
import random
import numpy as np
import soundfile as sf
import shutil
import librosa # 추가됨
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from datasets import load_dataset

class SyntheticOverlapDatasetGenerator:
    def __init__(
        self,
        output_dir: str = "./synthetic_overlap_dataset",
        cache_dir: str = "./librispeech_local_cache",
        sample_rate: int = 16000,
        seed: int = 42,
        max_samples_to_cache: int = 500
    ):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.sample_rate = sample_rate
        self.seed = seed
        self.max_samples_to_cache = max_samples_to_cache

        random.seed(seed)
        np.random.seed(seed)

        self.sir_levels = [0, 5, 10]
        # 0.0 포함하여 겹치지 않는 구간도 생성 가능하도록 설정 (필요시 수정)
        self.overlap_ratios = [0.2, 0.5, 1.0] 

        # 1. Prepare Source Data (Download & Cache to Disk)
        self._prepare_source_data()

    def _prepare_source_data(self):
        """
        Downloads LibriSpeech (clean/other) in streaming mode and saves to local disk.
        This avoids loading the entire dataset into RAM.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        splits = [("clean", "test"), ("other", "test")]
        
        print("🚀 Checking/Downloading Source Data...")
        
        for config, split in splits:
            split_dir = self.cache_dir / config
            if split_dir.exists() and any(split_dir.iterdir()):
                print(f"   -> Found existing cache for {config}, skipping download.")
                continue

            split_dir.mkdir(parents=True, exist_ok=True)
            print(f"   -> Downloading & Caching '{config}' split (Streaming Mode, max {self.max_samples_to_cache} samples)...")

            import gc
            from datasets import Features, Audio as AudioFeature

            ds = load_dataset(
                "librispeech_asr",
                config,
                split=split,
                streaming=True,
            )

            ds = ds.cast_column("audio", AudioFeature(decode=False))
            ds_limited = ds.take(self.max_samples_to_cache)

            count = 0
            pbar = tqdm(total=self.max_samples_to_cache, desc=f"Caching {config}")

            for item in ds_limited:
                try:
                    audio_data = item['audio']
                    text = item['text']
                    speaker_id = item['speaker_id']
                    file_id = item['file']

                    if 'bytes' in audio_data and audio_data['bytes']:
                        import io
                        audio_array, sr = sf.read(io.BytesIO(audio_data['bytes']))
                    elif 'path' in audio_data and audio_data['path']:
                        audio_array, sr = sf.read(audio_data['path'])
                    else:
                        audio_array = audio_data.get('array')
                        sr = audio_data.get('sampling_rate', self.sample_rate)
                        if audio_array is None:
                            continue

                    safe_name = f"{speaker_id}_{Path(file_id).stem}.wav"
                    save_path = split_dir / safe_name

                    sf.write(str(save_path), audio_array, sr)

                    meta_path = save_path.with_suffix('.json')
                    with open(meta_path, 'w') as f:
                        json.dump({
                            'text': text,
                            'speaker_id': speaker_id,
                            'original_sr': sr
                        }, f)

                    count += 1
                    pbar.update(1)

                    if count % 50 == 0:
                        gc.collect()

                except Exception as e:
                    print(f"Warning: Failed to save sample: {e}")

            pbar.close()
            gc.collect()

        print("✅ Source data preparation complete.\n")

    def _get_local_files(self) -> Dict[str, Dict]:
        speakers = {}
        for config in ["clean", "other"]:
            path = self.cache_dir / config
            if not path.exists(): continue
            
            for wav_file in path.glob("*.wav"):
                try:
                    parts = wav_file.name.split('_')
                    if not parts: continue
                    speaker_id = int(parts[0])
                    
                    if speaker_id not in speakers:
                        speakers[speaker_id] = {'type': config, 'files': []}
                    speakers[speaker_id]['files'].append(wav_file)
                except:
                    continue
        return speakers

    def _load_local_audio(self, path: Path) -> Tuple[np.ndarray, str]:
        """Loads audio, TRIMS SILENCE, and loads text."""
        audio, sr = sf.read(str(path))
        
        # 1. Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # 2. [추가됨] Trim Silence (앞뒤 무음 제거)
        # top_db: 무음으로 간주할 데시벨 임계값 (기본 30~60 사이 사용)
        try:
            audio, _ = librosa.effects.trim(audio, top_db=30)
        except Exception as e:
            print(f"Warning: Failed to trim silence for {path.name}: {e}")

        # Load text
        meta_path = path.with_suffix('.json')
        text = ""
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                text = json.load(f).get('text', "")
                
        return audio, text

    def _adjust_sir(self, signal: np.ndarray, interference: np.ndarray, sir_db: float) -> np.ndarray:
        signal_energy = np.sqrt(np.mean(signal ** 2)) + 1e-8
        interference_energy = np.sqrt(np.mean(interference ** 2)) + 1e-8
        target_ratio = 10 ** (sir_db / 20)
        scale_factor = signal_energy / (interference_energy * target_ratio)
        return interference * scale_factor

    def _create_overlap(self, audio1, audio2, overlap_ratio, sir_db):
        """
        수정된 Overlap 로직:
        랜덤 배치가 아니라, 정해진 overlap_ratio 만큼 '반드시' 겹치도록 오프셋을 계산합니다.
        """
        len1, len2 = len(audio1), len(audio2)
        min_len = min(len1, len2)
        
        # 1. 목표 겹침 길이 계산
        target_overlap_len = int(min_len * overlap_ratio)
        
        # 안전장치: 겹침 비율이 있는데 계산된 길이가 너무 짧으면 최소 20ms(320샘플) 보장
        if overlap_ratio > 0 and target_overlap_len < 320:
             target_overlap_len = min(min_len, 320)

        # 2. 오프셋 계산 (강제 겹침 배치)
        if overlap_ratio == 1.0:
            # 완전 겹침 (시작점 일치)
            offset1 = 0
            offset2 = 0
        elif overlap_ratio == 0.0:
            # 완전 분리 (순차 연결)
            offset1 = 0
            offset2 = len1
        else:
            # 부분 겹침 (Tail Overlap)
            # 순서 랜덤화 (누가 먼저 말할지)
            if random.random() < 0.5:
                # Audio 1 시작 -> 끝날 때쯤 Audio 2 겹침
                offset1 = 0
                offset2 = max(0, len1 - target_overlap_len)
            else:
                # Audio 2 시작 -> 끝날 때쯤 Audio 1 겹침
                offset2 = 0
                offset1 = max(0, len2 - target_overlap_len)

        # 전체 길이
        total_len = max(offset1 + len1, offset2 + len2)

        s1_full = np.zeros(total_len, dtype=np.float32)
        s2_full = np.zeros(total_len, dtype=np.float32)

        s1_full[offset1:offset1 + len1] = audio1
        audio2_adj = self._adjust_sir(audio1, audio2, sir_db)
        s2_full[offset2:offset2 + len2] = audio2_adj

        mixed = s1_full + s2_full
        
        # Normalize
        max_val = np.max(np.abs(mixed))
        if max_val > 0.99:
            scale = 0.99 / max_val
            mixed *= scale
            s1_full *= scale
            s2_full *= scale

        # 정확한 Overlap 구간 계산
        ov_start_sample = max(offset1, offset2)
        ov_end_sample = min(offset1 + len1, offset2 + len2)

        timing = {
            'speaker1_start': float(offset1 / self.sample_rate),
            'speaker1_end': float((offset1 + len1) / self.sample_rate),
            'speaker2_start': float(offset2 / self.sample_rate),
            'speaker2_end': float((offset2 + len2) / self.sample_rate),
            'overlap_start': float(ov_start_sample / self.sample_rate),
            'overlap_end': float(ov_end_sample / self.sample_rate),
        }
        return mixed, s1_full, s2_full, timing

    def generate_dataset(self, num_samples_per_config: int = 100):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔍 Indexing local files...")
        speakers = self._get_local_files()
        speaker_ids = list(speakers.keys())
        print(f"   -> Total unique speakers: {len(speaker_ids)}")
        
        if len(speaker_ids) < 2:
            print("❌ Not enough speakers found. Check download.")
            return

        dataset_metadata = []
        sample_id = 0

        for sir_db in self.sir_levels:
            for overlap_ratio in self.overlap_ratios:
                config_name = f"sir_{sir_db}db_overlap_{int(overlap_ratio*100)}pct"
                print(f"\nProcessing Config: {config_name}")
                
                config_dir = self.output_dir / config_name
                config_dir.mkdir(parents=True, exist_ok=True)

                for _ in tqdm(range(num_samples_per_config), desc="Generating"):
                    try:
                        # Select speakers
                        id_a, id_b = random.sample(speaker_ids, 2)
                        path_a = random.choice(speakers[id_a]['files'])
                        path_b = random.choice(speakers[id_b]['files'])
                        
                        # Load & Trim Silence
                        audio_a, text_a = self._load_local_audio(path_a)
                        audio_b, text_b = self._load_local_audio(path_b)
                        
                        # [추가됨] 무음 제거 후 너무 짧아졌으면 스킵 (최소 0.1초)
                        if len(audio_a) < 1600 or len(audio_b) < 1600:
                            continue

                        # Create Overlap
                        mixed, s1, s2, timing = self._create_overlap(
                            audio_a, audio_b, overlap_ratio, sir_db
                        )
                        
                        # Save
                        mix_name = f"sample_{sample_id:05d}_mixed.wav"
                        s1_name = f"sample_{sample_id:05d}_s1.wav"
                        s2_name = f"sample_{sample_id:05d}_s2.wav"
                        
                        sf.write(str(config_dir / mix_name), mixed, self.sample_rate)
                        sf.write(str(config_dir / s1_name), s1, self.sample_rate)
                        sf.write(str(config_dir / s2_name), s2, self.sample_rate)
                        
                        dataset_metadata.append({
                            'sample_id': sample_id,
                            'mixed_path': str(config_dir / mix_name),
                            's1_path': str(config_dir / s1_name),
                            's2_path': str(config_dir / s2_name),
                            's1_id': id_a, 's2_id': id_b,
                            's1_text': text_a, 's2_text': text_b,
                            'sir_db': sir_db, 'overlap_ratio': overlap_ratio,
                            'timing': timing,
                            'config': config_name
                        })
                        sample_id += 1
                        
                    except Exception as e:
                        print(f"Error generating sample: {e}")
                        continue

        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        print("\n✨ Done!")

def main():
    target_dir = "/mnt/ddn/kyudan/Audio-data-centric/podcast-pipeline/exp_overlap/synthetic_overlap_dataset"
    cache_dir = "/mnt/ddn/kyudan/Audio-data-centric/podcast-pipeline/exp_overlap/librispeech_cache"

    generator = SyntheticOverlapDatasetGenerator(
        output_dir=target_dir,
        cache_dir=cache_dir,
        max_samples_to_cache=500
    )
    generator.generate_dataset(num_samples_per_config=100)

if __name__ == "__main__":
    main()