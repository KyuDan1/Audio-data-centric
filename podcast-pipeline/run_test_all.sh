#!/bin/bash
set -euo pipefail

# ============================================================
# Audio Processing Pipeline Test Script
# ============================================================
# SepReformer 사용법:
#   - 기본값 (비활성화): sepreformer_flags=(--no-sepreformer)
#   - 활성화: sepreformer_flags=(--sepreformer)
#   - 둘 다 테스트: sepreformer_flags=(--sepreformer --no-sepreformer)
# ============================================================

# 실행할 입력 폴더 목록
folders=(
  "/mnt/ddn/kyudan/Audio-data-centric/podcast-pipeline/data/test"
)

# LLM 케이스 목록
# case_0 : LLM post diarization 안 함.
# case_2 : LLM post diarization 함.
llm_cases=(case_0)
korean=(--no-korean)
# VAD & DIA3 플래그 조합
vad_flags=(--vad)
#dia3_flags=(--dia3 --no-dia3)
dia3_flags=(--dia3)
# INITPROMPT 플래그 조합 추가
# initprompt: 추임새 prompt 넣어줌.
initprompt_flags=(--no-initprompt)
#initprompt_flags=(--initprompt --no-initprompt)

# 추가된 파라미터 조합
seg_ths=(0.11)
min_cluster_sizes=(11)
clust_ths=(0.5)
#ASRMoE=(--ASRMoE --no-ASRMoE)
ASRMoE=(--no-ASRMoE)
# DEMUCS 플래그 조합 (배경음악 제거)
# --demucs: PANNs로 배경음악 검출 후 Demucs로 보컬 추출
# --no-demucs: 배경음악 제거 안 함 (기본값)
#demucs_flags=(--demucs)
demucs_flags=(--no-demucs)
# WhisperX 단어 수준 타임스탬프 플래그
# --whisperx_word_timestamps: WhisperX 정렬을 통한 단어 수준 타임스탬프 활성화
# --no-whisperx_word_timestamps: 단어 수준 타임스탬프 비활성화 (기본값)
whisperx_flags=(--no-whisperx_word_timestamps)
#whisperx_flags=(--whisperx_word_timestamps --no-whisperx_word_timestamps)
# Qwen3-Omni 오디오 캡셔닝 플래그
# --qwen3omni: Qwen3-Omni API를 통한 오디오 캡션 생성 활성화
# --no-qwen3omni: 오디오 캡션 생성 비활성화 (기본값)
qwen3omni_flags=(--no-qwen3omni)
#qwen3omni_flags=(--qwen3omni --no-qwen3omni)
# Context-aware 캡셔닝 플래그
# --context_caption: 이전 2개 segment를 context로 사용한 캡셔닝 활성화
# --no-context_caption: context 없이 단일 segment만 캡셔닝 (기본값)
#context_caption_flags=(--no-context_caption)
#context_caption_flags=(--context_caption --no-context_caption)
# SepReformer 겹침 음성 분리 플래그
# --sepreformer: SepReformer를 사용한 겹침 음성 분리 활성화
# --no-sepreformer: 겹침 음성 분리 비활성화 (기본값)
#sepreformer_flags=(--sepreformer)
sepreformer_flags=(--sepreformer)
# SepReformer overlap threshold (겹침으로 판단할 최소 시간, 초 단위)
overlap_thresholds=(1.0)
# 추가된 MERGE_GAP 조합
# merge_gaps=(0.5 1 1.5 2)
# 0.5와 2가 똑같이 나옴.
merge_gaps=(2)

# Sortformer segment boundary adjustment (end time -0.08s)
sortformer_param_flags=(--sortformer-param)
sortformer_pad_offset_values=(-0.24)

for folder in "${folders[@]}"; do
  for vad in "${vad_flags[@]}"; do
    for dia3 in "${dia3_flags[@]}"; do
      for initprompt in "${initprompt_flags[@]}"; do
        for llm in "${llm_cases[@]}"; do
          for seg in "${seg_ths[@]}"; do
            for min_cluster in "${min_cluster_sizes[@]}"; do
              for clust in "${clust_ths[@]}"; do
                for merge_gap in "${merge_gaps[@]}"; do
                  for asrmoe in "${ASRMoE[@]}"; do
                    for demucs in "${demucs_flags[@]}"; do
                      for whisperx in "${whisperx_flags[@]}"; do
                        for qwen3omni in "${qwen3omni_flags[@]}"; do
                          for sepreformer in "${sepreformer_flags[@]}"; do
                            for overlap_th in "${overlap_thresholds[@]}"; do
                              for sortformer_pad_offset in "${sortformer_pad_offset_values[@]}"; do
                                echo "▶ Folder: ${folder}, ${vad}, ${dia3}, ${initprompt}, LLM=${llm}, seg_th=${seg}, min_cluster_size=${min_cluster}, clust_th=${clust}, merge_gap=${merge_gap}, ${asrmoe}, ${demucs}, ${whisperx}, ${qwen3omni}, ${sepreformer}, ${sortformer_param_flags[*]}, sortformer_pad_offset=${sortformer_pad_offset}, overlap_th=${overlap_th}, korean=${korean}"
                                /mnt/fr20tb/kyudan/miniforge3/envs/dataset/bin/python main_original_ASR_MoE.py \
                                  --input_folder_path "${folder}" \
                                  ${vad} ${dia3} ${initprompt} ${asrmoe} ${demucs} ${whisperx} ${qwen3omni} ${sepreformer} \
                                  ${sortformer_param_flags[@]} \
                                  --sortformer-pad-offset "${sortformer_pad_offset}" \
                                  --LLM "${llm}" \
                                  --seg_th "${seg}" \
                                  --min_cluster_size "${min_cluster}" \
                                  --clust_th "${clust}" \
                                  --merge_gap "${merge_gap}" \
                                  --overlap_threshold "${overlap_th}"
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
