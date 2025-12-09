# SepReformer Integration for Overlapping Speech Separation

## 개요

이 feature는 Sortformer diarization 후에 1초 이상 겹치는 segment를 감지하고, SepReformer를 사용해 음성을 분리한 후, speaker identity 모델로 화자를 판단하여 각 화자의 음성과 merge하는 기능입니다.

## 작동 방식

### 1. Overlap Detection (겹침 감지)
- Sortformer diarization 결과에서 1초 이상 겹치는 segment 쌍을 찾습니다
- 각 overlap에 대해 시작 시간, 종료 시간, 겹침 길이를 기록합니다

### 2. Audio Separation (음성 분리)
- 겹치는 구간의 오디오를 추출합니다
- SepReformer 모델을 사용해 두 개의 음성 stream으로 분리합니다
- 8kHz로 리샘플링 후 분리하고, 다시 원래 sample rate로 변환합니다

### 3. Speaker Identification (화자 식별)
- pyannote embedding 모델을 사용해 reference embedding을 추출합니다
  - 겹치지 않는 구간에서 각 화자의 reference embedding 생성
- 분리된 두 음성의 embedding을 추출합니다
- Cosine similarity를 사용해 각 분리된 음성이 어느 화자인지 판단합니다

### 4. Audio Merging (음성 병합)
- 분리된 음성을 올바른 화자에게 할당합니다
- 각 화자의 segment에서 겹치는 구간을 분리된 음성으로 교체합니다
- 전체 waveform을 업데이트하여 단일 화자 음성만 포함되도록 합니다

## 사용 방법

### 기본 사용법

```bash
# SepReformer 활성화 (overlap threshold: 1.0초)
python main_original_ASR_MoE.py --sepreformer --overlap_threshold 1.0 --LLM case_0

# SepReformer 비활성화 (기본값)
python main_original_ASR_MoE.py --LLM case_0
```

### 파라미터 설명

- `--sepreformer` / `--no-sepreformer`: SepReformer 사용 여부 (기본값: False)
- `--overlap_threshold <float>`: 겹침으로 판단할 최소 시간(초) (기본값: 1.0)

### 예시

```bash
# 0.5초 이상 겹침도 처리
python main_original_ASR_MoE.py --sepreformer --overlap_threshold 0.5 --LLM case_0

# 2초 이상 겹침만 처리
python main_original_ASR_MoE.py --sepreformer --overlap_threshold 2.0 --LLM case_0

# 다른 옵션과 함께 사용
python main_original_ASR_MoE.py \
    --sepreformer \
    --overlap_threshold 1.0 \
    --demucs \
    --ASRMoE \
    --whisperx_word_timestamps \
    --LLM case_0
```

## 구현 함수

### 1. `detect_overlapping_segments(segment_list, overlap_threshold=1.0)`
- **입력**: segment 리스트, overlap threshold
- **출력**: 겹치는 segment 쌍 리스트
- **기능**: 1초 이상 겹치는 segment 쌍을 찾아 반환

### 2. `separate_audio_with_sepreformer(audio_segment, sample_rate, sepreformer_path)`
- **입력**: 오디오 segment, sample rate, SepReformer 경로
- **출력**: 분리된 두 개의 음성 (src1, src2)
- **기능**: SepReformer 모델을 사용해 음성 분리

### 3. `identify_speaker_with_embedding(audio_segment, sample_rate, reference_embeddings, speaker_labels)`
- **입력**: 오디오 segment, sample rate, reference embeddings, speaker 레이블 리스트
- **출력**: 식별된 speaker 레이블
- **기능**: pyannote embedding을 사용해 화자 식별

### 4. `process_overlapping_segments_with_separation(segment_list, audio, overlap_threshold=1.0, sepreformer_path)`
- **입력**: segment 리스트, 오디오, overlap threshold, SepReformer 경로
- **출력**: 업데이트된 오디오, segment 리스트
- **기능**: 전체 overlap 처리 파이프라인 실행

## 처리 흐름

```
1. Sortformer Diarization
   ↓
2. Overlap Detection (threshold >= 1.0s)
   ↓
3. Extract Reference Embeddings (non-overlapping segments)
   ↓
4. For each overlapping pair:
   a. Extract overlapping audio
   b. Separate with SepReformer (→ src1, src2)
   c. Identify speakers (src1 → speaker A?, src2 → speaker B?)
   d. Assign separated audio to correct speakers
   e. Merge back into full waveform
   ↓
5. Continue with ASR on cleaned audio
```

## 출력 메타데이터

JSON 출력에 다음 정보가 추가됩니다:

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

## 성능 메트릭

실행 후 다음과 같은 타이밍 정보가 출력됩니다:

```
============================================================
SepReformer Overlap Separation:
  - Processing time: 12.34 seconds
  - RT factor: 0.0123
============================================================
```

## 주의사항

### 1. 모델 요구사항
- SepReformer 모델 checkpoint: `/mnt/ddn/kyudan/Audio-data-centric/SepReformer/`
- pyannote embedding 모델: Hugging Face token 필요
- 충분한 GPU 메모리 (분리 + embedding 모델)

### 2. 처리 시간
- Overlap이 많을수록 처리 시간 증가
- SepReformer inference: ~0.5-1초 per overlap
- Embedding extraction: ~0.1-0.2초 per segment

### 3. 품질 고려사항
- Reference embedding 추출을 위해 최소 2초 이상의 non-overlapping segment 필요
- Overlap이 너무 짧으면 (<0.5초) 분리 품질이 낮을 수 있음
- SepReformer는 2-speaker separation에 최적화되어 있음

## 디버깅

로그 레벨을 조정하여 상세 정보 확인:

```python
# logger.info로 주요 단계 출력
# logger.debug로 상세 정보 출력
```

주요 로그 메시지:
- `"Overlap detected: X.XXs between [...]"`: Overlap 감지
- `"Found N overlapping segment pairs"`: 총 overlap 개수
- `"Processing overlap N/M: ..."`: 현재 처리 중인 overlap
- `"Speaker identification: src1=..., src2=..."`: 화자 식별 결과
- `"SepReformer separation completed"`: 분리 완료

## 테스트

통합 테스트 실행:

```bash
python test_sepreformer_integration.py
```

기대 출력:
```
============================================================
ALL TESTS PASSED! ✓
============================================================
```

## 문제 해결

### Q1: "No checkpoint found" 에러
- SepReformer checkpoint 파일 확인
- `scratch_weights` 또는 `pretrain_weights` 폴더에 `.pth` 파일 있는지 확인

### Q2: OOM (Out of Memory) 에러
- Overlap이 많은 경우 batch 처리 구현 고려
- `overlap_threshold` 증가 (더 긴 overlap만 처리)

### Q3: 화자 식별 정확도 낮음
- Reference embedding 품질 확인
- Non-overlapping segment 길이 확인 (최소 2초)
- Cosine similarity threshold 조정 고려

## 향후 개선 사항

1. **Batch Processing**: 여러 overlap을 동시에 처리
2. **Multi-speaker Separation**: 3명 이상 화자 지원
3. **Quality Metrics**: 분리 품질 평가 메트릭 추가
4. **Adaptive Threshold**: 자동으로 최적의 overlap threshold 설정
5. **Caching**: Reference embedding 캐싱으로 성능 향상

## 라이선스 및 크레딧

- SepReformer: [원본 저장소 링크]
- pyannote.audio: https://github.com/pyannote/pyannote-audio
- Sortformer: NVIDIA NeMo

## 참고 자료

- SepReformer 논문: [링크]
- pyannote.audio 문서: https://github.com/pyannote/pyannote-audio
- 사용 예시: `test_sepreformer_integration.py`
