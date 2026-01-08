#!/usr/bin/env python3
"""
Full-Duplex 데이터셋 상세 통계 및 음성 품질 분석 스크립트 (Final Fixed Version)

[주요 기능]
1. .opus, .ogg, .wav, .mp3 등 다양한 오디오 포맷 자동 감지
2. soundfile 실패 시 librosa를 통한 강제 로드 지원 (Opus 호환성 해결)
3. SNR, Clipping, RMS, Speaker Entropy, Overlap Duration 등 심층 분석
"""

import os
import json
import subprocess
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import argparse
from datetime import datetime
import warnings
import math

warnings.filterwarnings('ignore')

# =============================================================================
# [설정] 경로를 사용자 환경에 맞게 수정하세요
# =============================================================================
ORIGINAL_ROOT = "/mnt/ddn/kyudan/DATASET/podcast_rss_feeds/podcasts_chunk_0"
PREPROCESSED_ROOT = "/mnt/ddn/kyudan/DATASET/podcast_rss_feeds/preprocessed_audio"

def get_audio_duration(audio_path):
    """ffprobe를 사용하여 오디오 파일의 길이를 빠르게 구합니다."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None

def calculate_speaker_entropy(segments, total_duration):
    """화자 발화 균형도 (Shannon Entropy): 1.0에 가까울수록 균등한 대화"""
    if not segments or total_duration == 0:
        return 0.0
    
    speaker_durations = defaultdict(float)
    for seg in segments:
        dur = seg['end'] - seg['start']
        speaker_durations[seg['speaker']] += dur
        
    probs = [d / total_duration for d in speaker_durations.values() if d > 0]
    
    if len(probs) <= 1:
        return 0.0
        
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(len(probs))
    
    return entropy / max_entropy if max_entropy > 0 else 0.0

def analyze_signal_from_array(y, sr, segments):
    """로드된 오디오 배열(y)을 기반으로 품질 지표 계산"""
    try:
        # Stereo -> Mono 변환
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        # 1. Clipping Rate (0.99 이상인 샘플 비율)
        clipping_threshold = 0.99
        clipping_rate = np.mean(np.abs(y) >= clipping_threshold) * 100
        
        # 2. RMS (Loudness)
        rms = np.sqrt(np.mean(y**2))
        
        # 3. SNR (Signal-to-Noise Ratio)
        mask = np.zeros_like(y, dtype=bool)
        for seg in segments:
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            end_sample = min(end_sample, len(y))
            if start_sample < len(y):
                mask[start_sample:end_sample] = True
            
        speech_power = np.mean(y[mask]**2) if np.any(mask) else 1e-9
        noise_power = np.mean(y[~mask]**2) if np.any(~mask) else 1e-9
        
        if noise_power < 1e-9: noise_power = 1e-9
        if speech_power < 1e-9: speech_power = 1e-9

        snr = 10 * np.log10(speech_power / noise_power)
        
        return {
            'snr': snr,
            'clipping_rate': clipping_rate,
            'rms': rms,
            'has_audio': True
        }
    except Exception as e:
        return {'snr': np.nan, 'clipping_rate': np.nan, 'rms': np.nan, 'has_audio': False}

def analyze_audio_file(audio_path, segments):
    """
    오디오 파일을 로드하여 품질을 분석합니다.
    soundfile(빠름)을 먼저 시도하고, 실패 시 librosa(호환성 좋음)를 사용합니다.
    """
    try:
        # 1차 시도: soundfile (WAV, FLAC 등)
        y, sr = sf.read(str(audio_path))
        return analyze_signal_from_array(y, sr, segments)
    except Exception:
        try:
            # 2차 시도: librosa (Opus, MP3, Ogg 등 ffmpeg 백엔드 사용)
            # sr=None으로 원본 샘플링 레이트 유지
            y, sr = librosa.load(str(audio_path), sr=None)
            return analyze_signal_from_array(y, sr, segments)
        except Exception:
            # 로드 실패
            return {'snr': np.nan, 'clipping_rate': np.nan, 'rms': np.nan, 'has_audio': False}

def calculate_overlap_details(segments):
    """Overlap Ratio 및 개별 Overlap 구간 길이 분포 계산"""
    if len(segments) < 2:
        return 0.0, []

    total_duration = max(seg['end'] for seg in segments) - min(seg['start'] for seg in segments)
    if total_duration == 0:
        return 0.0, []

    overlap_durations = []
    sorted_segs = sorted(segments, key=lambda x: x['start'])
    
    for i, seg1 in enumerate(sorted_segs):
        for seg2 in sorted_segs[i+1:]:
            if seg2['start'] >= seg1['end']:
                break
            
            overlap_start = max(seg1['start'], seg2['start'])
            overlap_end = min(seg1['end'], seg2['end'])
            
            if overlap_start < overlap_end:
                dur = overlap_end - overlap_start
                overlap_durations.append(dur)

    total_overlap = sum(overlap_durations)
    ratio = total_overlap / total_duration if total_duration > 0 else 0
    
    return ratio, overlap_durations

def calculate_turn_taking_gaps(segments):
    """Turn-taking Gap 계산"""
    if len(segments) < 2:
        return []

    sorted_segments = sorted(segments, key=lambda x: x['start'])
    gaps = []

    for i in range(len(sorted_segments) - 1):
        current = sorted_segments[i]
        next_seg = sorted_segments[i + 1]

        if current['speaker'] != next_seg['speaker']:
            gap = next_seg['start'] - current['end']
            gaps.append(gap)

    return gaps

def calculate_silence_ratio(segments, total_duration):
    if total_duration == 0: return 0.0
    covered_intervals = sorted([(seg['start'], seg['end']) for seg in segments])
    if not covered_intervals: return 1.0

    merged = [covered_intervals[0]]
    for current in covered_intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current
        if curr_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)

    speech_duration = sum(end - start for start, end in merged)
    silence_duration = total_duration - speech_duration
    return max(0.0, silence_duration / total_duration)

def collect_preprocessed_data_stats(root_dir, sample_size=None, check_audio_quality=True):
    print("\n📊 Collecting preprocessed data & quality stats...")

    json_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.json') and filename != 'original_paths.json':
                json_files.append(os.path.join(dirpath, filename))

    if sample_size:
        import random
        json_files = random.sample(json_files, min(sample_size, len(json_files)))
        print(f"Sampling {len(json_files)} episodes")

    podcast_stats = defaultdict(lambda: {
        'episodes': [],
        'total_duration': 0.0,
        'total_utterances': 0,
        'sum_snr': 0.0,
        'sum_clipping': 0.0,
        'count_audio_checked': 0
    })

    global_overlap_durations = []
    global_turn_gaps = []

    # 오디오 포맷 우선순위
    AUDIO_EXTENSIONS = ['.opus', '.ogg', '.wav', '.mp3', '.flac', '.m4a']

    for json_path in tqdm(json_files, desc="Processing episodes"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue

        metadata = data.get('metadata', {})
        segments = data.get('segments', [])
        if not segments: continue

        rel_path = os.path.relpath(json_path, root_dir)
        podcast_name = rel_path.split(os.sep)[0]
        
        # 1. Basic Stats
        total_duration = metadata.get('audio_duration_seconds', 0)
        total_utterances = len(segments)
        
        # 2. Dynamics
        overlap_ratio, overlap_durs = calculate_overlap_details(segments)
        global_overlap_durations.extend(overlap_durs)
        
        turn_gaps = calculate_turn_taking_gaps(segments)
        global_turn_gaps.extend(turn_gaps)
        
        silence_ratio = calculate_silence_ratio(segments, total_duration)
        speaker_entropy = calculate_speaker_entropy(segments, total_duration)
        speaker_count = len(set(seg['speaker'] for seg in segments if 'speaker' in seg))

        # 3. Audio Quality
        quality_metrics = {'snr': np.nan, 'clipping_rate': np.nan, 'rms': np.nan}
        
        if check_audio_quality:
            base_path = Path(json_path).with_suffix('')
            audio_found = False
            
            # 확장자 순회하며 파일 찾기
            for ext in AUDIO_EXTENSIONS:
                candidate = base_path.with_suffix(ext)
                if candidate.exists():
                    # 파일 찾음 -> 분석 시도
                    quality_metrics = analyze_audio_file(candidate, segments)
                    if quality_metrics['has_audio']:
                        audio_found = True
                        break
            
            # 같은 폴더에 없으면 'original_path' 참조 시도 (옵션)
            if not audio_found and 'original_path' in metadata:
                 # 메타데이터에 원본 경로가 있다면 시도해볼 수 있음 (필요 시 구현)
                 pass

        episode_stats = {
            'name': Path(json_path).stem,
            'duration': total_duration,
            'speakers': speaker_count,
            'speaker_entropy': speaker_entropy,
            'overlap_ratio': overlap_ratio,
            'avg_overlap_duration': np.mean(overlap_durs) if overlap_durs else 0,
            'silence_ratio': silence_ratio,
            'snr': quality_metrics['snr'],
            'clipping_rate': quality_metrics['clipping_rate'],
            'rms': quality_metrics['rms']
        }

        podcast_stats[podcast_name]['episodes'].append(episode_stats)
        podcast_stats[podcast_name]['total_duration'] += total_duration
        podcast_stats[podcast_name]['total_utterances'] += total_utterances

    return dict(podcast_stats), global_overlap_durations, global_turn_gaps

def aggregate_statistics(preprocessed_stats):
    print("\n📈 Aggregating statistics...")
    results = []
    
    for podcast_name, data in preprocessed_stats.items():
        episodes = data.get('episodes', [])
        if not episodes: continue
        
        def get_valid_mean(key):
            vals = [e[key] for e in episodes if not np.isnan(e.get(key, np.nan))]
            return np.mean(vals) if vals else 0.0

        row = {
            'podcast_name': podcast_name,
            'episodes': len(episodes),
            'total_hours': data['total_duration'] / 3600,
            'avg_speaker_entropy': get_valid_mean('speaker_entropy'),
            'avg_overlap_ratio': get_valid_mean('overlap_ratio'),
            'avg_overlap_duration': get_valid_mean('avg_overlap_duration'),
            'avg_silence_ratio': get_valid_mean('silence_ratio'),
            'avg_snr_db': get_valid_mean('snr'),
            'avg_clipping_rate': get_valid_mean('clipping_rate'),
            'avg_rms': get_valid_mean('rms')
        }
        results.append(row)
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Full-Duplex Dataset Analyzer")
    parser.add_argument('--output-dir', default='comparison_statistics', help='Output Directory')
    parser.add_argument('--sample-episodes', type=int, help='Limit number of episodes for testing')
    parser.add_argument('--no-quality', action='store_true', help='Skip slow audio quality checks')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("🎙️ Full-Duplex Dataset Statistics & Quality Check")
    print(f"Target Directory: {PREPROCESSED_ROOT}")
    print("="*70)

    # 1. Collect Stats
    prep_stats, global_overlaps, global_gaps = collect_preprocessed_data_stats(
        PREPROCESSED_ROOT, 
        sample_size=args.sample_episodes,
        check_audio_quality=not args.no_quality
    )
    
    # 2. Aggregate
    df = aggregate_statistics(prep_stats)
    
    if df.empty:
        print("❌ No data found.")
        return

    # 3. Save Results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(args.output_dir, f'full_stats_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    
    # Save Distributions (JSON)
    dist_stats = {
        'overlap_duration': {
            'mean': float(np.mean(global_overlaps)) if global_overlaps else 0,
            'median': float(np.median(global_overlaps)) if global_overlaps else 0,
            'p90': float(np.percentile(global_overlaps, 90)) if global_overlaps else 0,
        },
        'turn_gaps': {
            'mean': float(np.mean(global_gaps)) if global_gaps else 0,
            'median': float(np.median(global_gaps)) if global_gaps else 0,
        }
    }
    with open(os.path.join(args.output_dir, f'distributions_{timestamp}.json'), 'w') as f:
        json.dump(dist_stats, f, indent=2)

    # 4. Print Summary Report
    print("\n" + "="*70)
    print("📊 DATASET SUMMARY REPORT")
    print("="*70)
    print(f"• Total Podcasts      : {len(df)}")
    print(f"• Total Episodes      : {df['episodes'].sum()}")
    print(f"• Total Duration      : {df['total_hours'].sum():.2f} hours")
    
    print("\n[🗣️ Conversational Dynamics]")
    print(f"• Avg Speaker Entropy : {df['avg_speaker_entropy'].mean():.3f} (closer to 1.0 is better)")
    print(f"• Avg Overlap Ratio   : {df['avg_overlap_ratio'].mean()*100:.2f} %")
    print(f"• Avg Overlap Length  : {df['avg_overlap_duration'].mean():.3f} sec")
    print(f"• Avg Silence Ratio   : {df['avg_silence_ratio'].mean()*100:.2f} %")

    if not args.no_quality:
        print("\n[🔊 Audio Quality]")
        print(f"• Avg SNR             : {df['avg_snr_db'].mean():.2f} dB")
        print(f"• Avg Clipping Rate   : {df['avg_clipping_rate'].mean():.4f} %")
        print(f"• Avg Loudness (RMS)  : {df['avg_rms'].mean():.4f}")
    
    print("="*70)
    print(f"✓ Results saved to {csv_path}")

if __name__ == "__main__":
    main()