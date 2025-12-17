import os
import sys
import glob
import subprocess
import multiprocessing
import json
import time
import argparse
import math
from datetime import datetime
import torch
import wandb
from tqdm import tqdm

# ====================================================
# [설정 영역] 사용자의 환경에 맞게 수정하세요
# ====================================================

# 1. 실행할 백엔드 파이썬 스크립트 경로
BACKEND_SCRIPT = "main_original_ASR_MoE.py"

# 2. 입력 데이터 루트 폴더 (opus 파일들이 들어있는 상위 폴더)
INPUT_ROOT = "/mnt/ddn/kyudan/DATASET/podcast_rss_feeds/podcasts_chunk_0"

# 3. 로그 및 상태 저장 파일
PROGRESS_LOG_FILE = "processed_folders_log.txt"

# 4. WandB 설정
WANDB_PROJECT = "audio-pipeline-monitoring"
WANDB_ENTITY = "krafton_kyudan"  # 본인의 entity로 수정 (옵션)
WANDB_RUN_NAME = f"pipeline-run-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# 5. 백엔드 스크립트에 넘길 고정 파라미터들 (Bash 스크립트의 설정값 반영)
# 필요한 플래그들을 리스트 형태로 정의합니다.
BACKEND_ARGS = [
    "--vad",
    "--dia3",
    "--no-initprompt", # bash 스크립트의 initprompt_flags=(--no-initprompt) 반영
    "--ASRMoE",
    "--demucs",
    "--whisperx_word_timestamps",
    "--no-qwen3omni",
    "--sepreformer",
    "--sortformer-param",
    "--sortformer-pad-offset", "-0.24",
    "--LLM", "case_0",
    "--seg_th", "0.11",
    "--min_cluster_size", "11",
    "--clust_th", "0.5",
    "--merge_gap", "2",
    "--overlap_threshold", "0.2",
    "--speaker-link-threshold", "0.6",
    "--opus_decode_workers", "20", # 워커당 할당할 CPU 스레드 조절
    "--ffmpeg_threads_per_decode", "1"
]

# ====================================================

def get_gpu_count():
    """사용 가능한 GPU 개수를 반환합니다."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def find_subfolders_with_opus(root_dir):
    """
    루트 폴더 하위의 모든 폴더를 탐색하여 .opus 파일이 있는 폴더 리스트를 반환합니다.
    """
    target_folders = set()
    print(f"📂 [Search] Scanning subdirectories in {root_dir}...")
    
    # os.walk를 사용하여 재귀적으로 탐색
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.opus') or f.endswith('.ogg'):
                target_folders.add(dirpath)
                break # 해당 폴더는 확인했으므로 다음 폴더로

    # 정렬하여 반환 (이어하기 시 일관성 유지)
    return sorted(list(target_folders))

def load_processed_list():
    """이미 처리된 폴더 목록을 로드합니다."""
    if not os.path.exists(PROGRESS_LOG_FILE):
        return set()
    with open(PROGRESS_LOG_FILE, 'r') as f:
        return set(line.strip() for line in f)

def append_to_processed_list(folder_path):
    """처리 완료된 폴더를 로그에 기록합니다."""
    with open(PROGRESS_LOG_FILE, 'a') as f:
        f.write(f"{folder_path}\n")

def get_audio_duration_from_json(folder_path):
    """
    백엔드 처리가 끝난 후 생성된 JSON 파일을 찾아 오디오 길이를 추출합니다.
    백엔드 저장 경로 로직에 의존하므로, JSON을 찾기 위해 glob을 사용합니다.
    """
    # 예상되는 결과 JSON 찾기 (폴더명.json 또는 해당 폴더 내의 json)
    # 백엔드 로직에 따르면 _final 폴더 아래 생성되지만, 
    # 정확한 위치를 알기 어려우므로 최근 생성된 json을 찾거나 
    # 백엔드 스크립트가 stdout으로 출력하는 정보를 파싱하는 것이 좋습니다.
    # 여기서는 간단히 0.0을 리턴하고, 추후 백엔드 로그 파싱으로 고도화 가능합니다.
    # *참고: 정확한 duration 로깅을 위해 백엔드 스크립트가 duration을 print하도록 되어 있으므로
    # subprocess의 출력을 파싱하는 것이 더 정확합니다.*
    return 0.0

def worker_process(gpu_id, session_id, folder_queue, result_queue, error_queue):
    """
    개별 워커 프로세스입니다.
    """
    # 현재 프로세스에 GPU 할당
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    worker_name = f"GPU{gpu_id}-Worker{session_id}"
    print(f"🚀 [{worker_name}] Started.")

    while True:
        try:
            folder_path = folder_queue.get_nowait()
        except Exception:
            # 큐가 비었으면 종료
            break

        try:
            # 명령어 구성
            cmd = [sys.executable, BACKEND_SCRIPT, "--input_folder_path", folder_path] + BACKEND_ARGS
            
            start_time = time.time()
            
            # 서브프로세스 실행 (stdout을 캡처하여 duration 파싱)
            process = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                env=os.environ.copy() # 환경변수(CUDA_VISIBLE_DEVICES) 전달
            )

            if process.returncode != 0:
                print(f"❌ [{worker_name}] Error processing {folder_path}\nStderr: {process.stderr[-500:]}")
                error_queue.put(folder_path)
                continue

            # duration 파싱 (백엔드 코드의 print 문 참조)
            # "Audio duration: 123.45 seconds" 패턴 찾기
            duration = 0.0
            for line in process.stdout.split('\n'):
                if "Audio duration:" in line and "seconds" in line:
                    try:
                        # 예: Audio duration: 123.45 seconds ...
                        parts = line.split("Audio duration:")[1].split("seconds")[0].strip()
                        duration = float(parts)
                    except:
                        pass
            
            # 결과 큐에 전송 (폴더경로, 오디오길이)
            result_queue.put((folder_path, duration))
            
        except Exception as e:
            print(f"💥 [{worker_name}] Critical Exception: {e}")
            error_queue.put(folder_path)

    print(f"💤 [{worker_name}] Finished.")

def main():
    # 1. GPU 확인
    num_gpus = get_gpu_count()
    if num_gpus == 0:
        print("❌ No GPUs found. Exiting.")
        return

    sessions_per_gpu = 3
    total_workers = num_gpus * sessions_per_gpu
    print(f"⚙️  Configuration: {num_gpus} GPUs x {sessions_per_gpu} Sessions = {total_workers} Workers")

    # 2. 데이터 탐색 및 정렬
    all_folders = find_subfolders_with_opus(INPUT_ROOT)
    print(f"Found {len(all_folders)} directories containing audio.")

    # 3. 이어하기 필터링
    processed_folders = load_processed_list()
    todo_folders = [f for f in all_folders if f not in processed_folders]
    print(f"Skipping {len(processed_folders)} already processed. Remaining: {len(todo_folders)}")

    if not todo_folders:
        print("✅ All tasks completed.")
        return

    # 4. WandB 초기화 (메인 프로세스에서만)
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY if WANDB_ENTITY else None, name=WANDB_RUN_NAME)
    
    # 5. 멀티프로세싱 준비
    manager = multiprocessing.Manager()
    folder_queue = manager.Queue()
    result_queue = manager.Queue()
    error_queue = manager.Queue()

    # 큐에 데이터 적재
    for folder in todo_folders:
        folder_queue.put(folder)

    # 6. 워커 생성 및 시작
    processes = []
    for gpu_id in range(num_gpus):
        for session_id in range(sessions_per_gpu):
            p = multiprocessing.Process(
                target=worker_process,
                args=(gpu_id, session_id, folder_queue, result_queue, error_queue)
            )
            p.start()
            processes.append(p)
            # 프로세스 시작 간격을 두어 초기 로드 스파이크 방지
            time.sleep(2) 

    # 7. 모니터링 루프 (메인 스레드)
    total_tasks = len(todo_folders)
    pbar = tqdm(total=total_tasks, desc="Processing Audio")
    
    completed_count = 0
    total_audio_duration = 0.0
    
    while completed_count < total_tasks:
        # 에러 체크
        if not error_queue.empty():
            err_folder = error_queue.get()
            print(f"\n⚠️ Process failed for: {err_folder}")
            # 에러난 것은 넘어가고 카운트만 올림 (또는 재시도 로직 추가 가능)
            completed_count += 1
            pbar.update(1)
            continue

        # 결과 처리
        if not result_queue.empty():
            folder, duration = result_queue.get()
            
            # 로그 기록
            append_to_processed_list(folder)
            
            # 통계 업데이트
            completed_count += 1
            total_audio_duration += duration
            
            # WandB 로깅
            wandb.log({
                "progress_percent": (completed_count / total_tasks) * 100,
                "processed_folders": completed_count,
                "cumulative_audio_hours": total_audio_duration / 3600,
                "current_audio_seconds": duration
            })
            
            pbar.update(1)
        
        # 모든 프로세스가 죽었는지 확인 (비정상 종료 대비)
        if not any(p.is_alive() for p in processes) and result_queue.empty() and error_queue.empty():
            print("\nAll workers stopped unexpectedly.")
            break
        
        time.sleep(0.1)

    # 8. 종료 처리
    for p in processes:
        p.join()

    wandb.finish()
    print(f"\n🎉 Pipeline Finished. Total Audio Processed: {total_audio_duration/3600:.2f} hours.")

if __name__ == "__main__":
    # 시작 방법: python run_frontend.py
    main()