import os
import json
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np

# 사용자 경로 설정
PREPROCESSED_ROOT = "/mnt/ddn/kyudan/DATASET/podcast_rss_feeds/preprocessed_audio"

def debug_paths():
    print(f"🔍 Debugging in: {PREPROCESSED_ROOT}")
    
    # 1. 첫 번째 JSON 파일 찾기
    found_json = None
    for dirpath, _, filenames in os.walk(PREPROCESSED_ROOT):
        for f in filenames:
            if f.endswith('.json') and f != 'original_paths.json':
                found_json = Path(dirpath) / f
                break
        if found_json: break
    
    if not found_json:
        print("❌ JSON 파일을 하나도 찾지 못했습니다.")
        return

    print(f"\n📂 Found JSON: {found_json}")
    
    # 2. 같은 위치에서 오디오 파일 찾기 시도
    base_path = found_json.with_suffix('')
    audio_extensions = ['.opus', '.ogg', '.wav', '.mp3', '.flac', '.m4a']
    
    found_audio = None
    print(f"   Searching for audio files with base: {base_path.name}")
    
    for ext in audio_extensions:
        candidate = base_path.with_suffix(ext)
        exists = candidate.exists()
        status = "✅ Found" if exists else "❌ Missing"
        print(f"   - Checking {candidate.name}: {status}")
        if exists:
            found_audio = candidate
            break
            
    # 3. 오디오 로드 테스트
    if found_audio:
        print(f"\n🎧 Attempting to load: {found_audio}")
        try:
            # Soundfile 시도
            y, sr = sf.read(str(found_audio))
            print(f"   ✅ soundfile load success! Shape: {y.shape}, SR: {sr}")
            print(f"   📊 Max Amp: {np.max(np.abs(y)):.4f}")
        except Exception as e:
            print(f"   ⚠️ soundfile failed: {e}")
            try:
                # Librosa 시도
                y, sr = librosa.load(str(found_audio), sr=None)
                print(f"   ✅ librosa load success! Shape: {y.shape}, SR: {sr}")
                print(f"   📊 Max Amp: {np.max(np.abs(y)):.4f}")
            except Exception as e2:
                print(f"   ❌ librosa failed: {e2}")
    else:
        print("\n❌ 오디오 파일을 찾을 수 없습니다.")
        print("   JSON 파일과 오디오 파일이 서로 다른 폴더에 있나요?")
        print("   그렇다면 폴더 구조를 알려주세요. (예: /metadata/file.json, /audio/file.opus)")

if __name__ == "__main__":
    debug_paths()