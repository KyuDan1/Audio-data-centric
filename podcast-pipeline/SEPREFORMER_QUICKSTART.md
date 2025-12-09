# SepReformer Overlap Separation - Quick Start

## 빠른 시작

### 1. run_test_all.sh 편집

```bash
vim run_test_all.sh
```

**SepReformer 활성화하려면:**
```bash
# 47-50번째 줄 근처:
sepreformer_flags=(--sepreformer)          # ← 이렇게 변경
#sepreformer_flags=(--sepreformer --no-sepreformer)
overlap_thresholds=(1.0)
```

**SepReformer 비활성화하려면 (기본값):**
```bash
sepreformer_flags=(--no-sepreformer)       # ← 기본 설정
overlap_thresholds=(1.0)
```

### 2. 실행

```bash
./run_test_all.sh
```

### 3. 결과 확인

**로그에서 확인:**
```
Step 2.5: Overlap Control with SepReformer
Found 5 overlapping segment pairs
Processing overlap 1/5: 13.5s - 15.0s (SPEAKER_00 vs SPEAKER_01)
...
SepReformer separation - Processing time: 12.34s, RT factor: 0.0123
```

**JSON 메타데이터:**
```json
{
  "metadata": {
    "sepreformer_separation": {
      "processing_time_seconds": 12.34,
      "rt_factor": 0.0123,
      "overlap_threshold_seconds": 1.0,
      "enabled": true
    }
  }
}
```

## 직접 실행 (수동)

```bash
# SepReformer 활성화
python main_original_ASR_MoE.py \
  --input_folder_path "data/test" \
  --sepreformer \
  --overlap_threshold 1.0 \
  --LLM case_0

# SepReformer 비활성화
python main_original_ASR_MoE.py \
  --input_folder_path "data/test" \
  --no-sepreformer \
  --LLM case_0
```

## 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--sepreformer` | Overlap 음성 분리 활성화 | 비활성화 |
| `--no-sepreformer` | Overlap 음성 분리 비활성화 | ✓ 기본값 |
| `--overlap_threshold` | 겹침 판단 최소 시간(초) | 1.0 |

## 어떤 경우에 사용?

### SepReformer 활성화가 유용한 경우:
- ✅ 여러 사람이 동시에 말하는 경우가 많은 오디오
- ✅ 토론, 인터뷰, 회의 녹음
- ✅ Overlap이 1초 이상 자주 발생하는 경우
- ✅ 각 화자의 음성을 명확히 분리해야 하는 경우

### SepReformer 비활성화가 좋은 경우:
- ⭕ 단일 화자 오디오
- ⭕ Overlap이 거의 없는 대화
- ⭕ 빠른 처리가 필요한 경우
- ⭕ GPU 메모리가 부족한 경우

## 처리 흐름 비교

### SepReformer 비활성화 (기본)
```
Sortformer Diarization → ASR (Whisper)
```

### SepReformer 활성화
```
Sortformer Diarization
  ↓
Overlap Detection (≥1.0s)
  ↓
SepReformer 음성 분리
  ↓
Speaker 식별 & Merge
  ↓
ASR (Whisper)
```

## 문서

- 상세 문서: [SEPREFORMER_INTEGRATION.md](SEPREFORMER_INTEGRATION.md)
- 스크립트 가이드: [RUN_TEST_ALL_GUIDE.md](RUN_TEST_ALL_GUIDE.md)
- 테스트: `python test_sepreformer_integration.py`

## 문제 해결

**Q: Overlap이 감지되지 않음**
```bash
# Threshold를 낮춰보세요
overlap_thresholds=(0.5)  # 0.5초 이상 겹침 감지
```

**Q: 처리가 너무 느림**
```bash
# Threshold를 높여서 처리할 overlap 개수를 줄이세요
overlap_thresholds=(2.0)  # 2초 이상만 처리
```

**Q: OOM 에러**
```bash
# SepReformer 비활성화하거나 threshold를 높이세요
sepreformer_flags=(--no-sepreformer)
```
