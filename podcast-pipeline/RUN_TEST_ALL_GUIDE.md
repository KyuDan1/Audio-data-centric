# run_test_all.sh 사용 가이드

## SepReformer 설정 방법

### 1. 기본 설정 (SepReformer 비활성화)

```bash
# run_test_all.sh 파일에서:
sepreformer_flags=(--no-sepreformer)
```

이 설정으로 실행하면 **overlap 처리를 하지 않습니다** (기본값).

### 2. SepReformer 활성화

```bash
# run_test_all.sh 파일에서:
sepreformer_flags=(--sepreformer)
```

이 설정으로 실행하면 **1초 이상 겹치는 segment에 대해 SepReformer 음성 분리를 수행합니다**.

### 3. 두 가지 모두 테스트

```bash
# run_test_all.sh 파일에서:
sepreformer_flags=(--sepreformer --no-sepreformer)
```

이 설정으로 실행하면 **같은 오디오에 대해 두 가지 버전을 모두 생성합니다**:
- SepReformer 비활성화 버전
- SepReformer 활성화 버전

## Overlap Threshold 조정

겹침으로 판단할 최소 시간을 조정할 수 있습니다:

```bash
# 1초 이상 겹침 처리 (기본값)
overlap_thresholds=(1.0)

# 0.5초 이상 겹침 처리 (더 민감하게)
overlap_thresholds=(0.5)

# 2초 이상 겹침 처리 (덜 민감하게)
overlap_thresholds=(2.0)

# 여러 threshold 테스트
overlap_thresholds=(0.5 1.0 2.0)
```

## 실행 예시

### 예시 1: SepReformer 비활성화로 실행

```bash
# run_test_all.sh 편집:
sepreformer_flags=(--no-sepreformer)
overlap_thresholds=(1.0)

# 실행:
./run_test_all.sh
```

**출력 예시:**
```
▶ Folder: /path/to/data, --vad, --dia3, --no-initprompt, LLM=case_0,
  seg_th=0.11, min_cluster_size=11, clust_th=0.5, merge_gap=2,
  --ASRMoE, --demucs, --whisperx_word_timestamps, --qwen3omni,
  --no-sepreformer, overlap_th=1.0, korean=--no-korean
```

### 예시 2: SepReformer 활성화로 실행

```bash
# run_test_all.sh 편집:
sepreformer_flags=(--sepreformer)
overlap_thresholds=(1.0)

# 실행:
./run_test_all.sh
```

**출력 예시:**
```
▶ Folder: /path/to/data, --vad, --dia3, --no-initprompt, LLM=case_0,
  seg_th=0.11, min_cluster_size=11, clust_th=0.5, merge_gap=2,
  --ASRMoE, --demucs, --whisperx_word_timestamps, --qwen3omni,
  --sepreformer, overlap_th=1.0, korean=--no-korean
```

### 예시 3: 여러 조합 테스트

```bash
# run_test_all.sh 편집:
sepreformer_flags=(--sepreformer --no-sepreformer)
overlap_thresholds=(0.5 1.0 2.0)

# 실행:
./run_test_all.sh
```

이 경우 총 **6가지 조합**이 실행됩니다:
1. --sepreformer, overlap_th=0.5
2. --sepreformer, overlap_th=1.0
3. --sepreformer, overlap_th=2.0
4. --no-sepreformer, overlap_th=0.5
5. --no-sepreformer, overlap_th=1.0
6. --no-sepreformer, overlap_th=2.0

## 처리 시간 예상

SepReformer 활성화 시 처리 시간이 증가할 수 있습니다:

| Overlap 개수 | 추가 처리 시간 (예상) |
|--------------|----------------------|
| 0개 (overlap 없음) | +0초 |
| 10개 | +5-10초 |
| 50개 | +25-50초 |
| 100개 | +50-100초 |

실제 처리 시간은 GPU 성능과 overlap 길이에 따라 다릅니다.

## 출력 결과 확인

### JSON 메타데이터

SepReformer가 활성화된 경우 JSON 출력에 다음 정보가 추가됩니다:

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

### 로그 메시지

SepReformer 실행 시 다음과 같은 로그가 출력됩니다:

```
Step 2.5: Overlap Control with SepReformer
Overlap detected: 1.5s between [10.0-15.0] and [13.5-18.0]
Found 5 overlapping segment pairs
Processing overlap 1/5: 13.5s - 15.0s (SPEAKER_00 vs SPEAKER_01)
SepReformer separation completed: output shapes (36000,), (36000,)
Speaker identification: src1=SPEAKER_00, src2=SPEAKER_01
Merged separated audio for overlap 1
...
SepReformer separation - Processing time: 12.34s, RT factor: 0.0123
```

## 문제 해결

### Q: SepReformer가 활성화되었는데 처리가 안 됨
- Overlap이 threshold보다 짧은지 확인
- 로그에서 "No overlapping segments found" 메시지 확인

### Q: OOM (Out of Memory) 에러
- `overlap_thresholds` 값을 높여서 처리할 overlap 개수 줄이기
- GPU 메모리 확인

### Q: Checkpoint 파일을 찾을 수 없음
- SepReformer checkpoint 파일 위치 확인:
  - `/mnt/ddn/kyudan/Audio-data-centric/SepReformer/models/SepReformer_Base_WSJ0/log/pretrain_weights/`
  - 또는 `/mnt/ddn/kyudan/Audio-data-centric/SepReformer/models/SepReformer_Base_WSJ0/log/scratch_weights/`

## 권장 설정

### 프로덕션 환경
```bash
sepreformer_flags=(--sepreformer)
overlap_thresholds=(1.0)  # 1초 이상 겹침만 처리
```

### 실험/비교 목적
```bash
sepreformer_flags=(--sepreformer --no-sepreformer)
overlap_thresholds=(1.0)  # 두 버전 모두 생성하여 비교
```

### 빠른 테스트
```bash
sepreformer_flags=(--no-sepreformer)
overlap_thresholds=(1.0)  # Overlap 처리 안 함 (빠름)
```

## 전체 플래그 조합

현재 스크립트에서 테스트되는 모든 플래그:

```bash
vad_flags=(--vad)                           # VAD 사용
dia3_flags=(--dia3)                         # Diarization 3.1 모델
initprompt_flags=(--no-initprompt)          # 초기 프롬프트 비활성화
ASRMoE=(--ASRMoE)                           # ASR MoE 사용
demucs_flags=(--demucs)                     # 배경음악 제거
whisperx_flags=(--whisperx_word_timestamps) # 단어 수준 타임스탬프
qwen3omni_flags=(--qwen3omni)               # 오디오 캡셔닝
sepreformer_flags=(--no-sepreformer)        # Overlap 처리 (NEW!)
```

각 플래그는 독립적으로 활성화/비활성화할 수 있습니다.
