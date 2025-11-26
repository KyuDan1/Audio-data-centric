#!/bin/bash
set -euo pipefail

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

# 추가된 MERGE_GAP 조합
# merge_gaps=(0.5 1 1.5 2)
# 0.5와 2가 똑같이 나옴.
merge_gaps=(2)

for folder in "${folders[@]}"; do
  for vad in "${vad_flags[@]}"; do
    for dia3 in "${dia3_flags[@]}"; do
      for initprompt in "${initprompt_flags[@]}"; do
        for llm in "${llm_cases[@]}"; do
          for seg in "${seg_ths[@]}"; do
            for min_cluster in "${min_cluster_sizes[@]}"; do
              for clust in "${clust_ths[@]}"; do
                for merge_gap in "${merge_gaps[@]}"; do
                  echo "▶ Folder: ${folder}, ${vad}, ${dia3}, ${initprompt}, LLM=${llm}, seg_th=${seg}, min_cluster_size=${min_cluster}, clust_th=${clust}, merge_gap=${merge_gap}, korean=${korean}"
                  python main_original.py \
                    --input_folder_path "${folder}" \
                    ${vad} ${dia3} ${initprompt} \
                    --LLM "${llm}" \
                    --seg_th "${seg}" \
                    --min_cluster_size "${min_cluster}" \
                    --clust_th "${clust}" \
                    --merge_gap "${merge_gap}"
                done
              done
            done
          done
        done
      done
    done
  done
done
