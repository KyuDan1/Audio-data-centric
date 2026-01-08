import os
import sys
import subprocess
import multiprocessing
import json
import time
import argparse
from datetime import datetime
import torch
import wandb
from tqdm import tqdm

# ====================================================
# [설정 영역] 사용자의 환경에 맞게 수정하세요
# ====================================================

# 서버 설정
SERVERS = [
    {
        "name": "local",
        "host": "localhost",
        "gpus": 4,
        "sessions_per_gpu": 3,
    },
    {
        "name": "remote-A100",
        "host": "10.169.39.24",
        "port": 11691,
        "user": "nsml",
        "gpus": 4,
        "sessions_per_gpu": 3,
    }
]

# 1. 실행할 백엔드 파이썬 스크립트 경로
BACKEND_SCRIPT = "main_original_ASR_MoE.py"
SCRIPT_DIR = "/mnt/ddn/kyudan/Audio-data-centric/podcast-pipeline"

# 2. 입력 데이터 루트 폴더 (opus 파일들이 들어있는 상위 폴더)
INPUT_ROOT = "/mnt/ddn/kyudan/DATASET/podcast_rss_feeds/podcasts_chunk_0"

# 3. 로그 및 상태 저장 파일
PROGRESS_LOG_FILE = "processed_folders_log.txt"

# 4. WandB 설정
WANDB_PROJECT = "audio-pipeline-monitoring"
WANDB_ENTITY = "krafton_kyudan"
WANDB_RUN_NAME = f"multi-server-pipeline-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# 5. 백엔드 스크립트에 넘길 고정 파라미터들
BACKEND_ARGS = [
    "--vad",
    "--dia3",
    "--no-initprompt",
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
    "--opus_decode_workers", "20",
    "--ffmpeg_threads_per_decode", "1"
]

# ====================================================

def find_subfolders_with_opus(root_dir):
    """
    루트 폴더 하위의 모든 폴더를 탐색하여 .opus 파일이 있는 폴더 리스트를 반환합니다.
    """
    target_folders = set()
    print(f"📂 [Search] Scanning subdirectories in {root_dir}...")

    for dirpath, dirs, filenames in os.walk(root_dir):
        dirs[:] = [d for d in dirs if "_opus_cache" not in d]
        for f in filenames:
            if f.endswith('.opus') or f.endswith('.ogg'):
                if "_opus_cache" not in dirpath:
                    target_folders.add(dirpath)
                break

    return sorted(list(target_folders))

def load_processed_list():
    """이미 처리된 폴더 목록을 로드합니다."""
    log_path = os.path.join(SCRIPT_DIR, PROGRESS_LOG_FILE)
    if not os.path.exists(log_path):
        return set()
    with open(log_path, 'r') as f:
        return set(line.strip() for line in f)

def append_to_processed_list(folder_path):
    """처리 완료된 폴더를 로그에 기록합니다."""
    log_path = os.path.join(SCRIPT_DIR, PROGRESS_LOG_FILE)
    with open(log_path, 'a') as f:
        f.write(f"{folder_path}\n")

def execute_remote_task(server, gpu_id, session_id, folder_path):
    """
    원격 서버에서 작업을 실행합니다.
    """
    if server["name"] == "local":
        # 로컬 실행
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, BACKEND_SCRIPT),
            "--input_folder_path", folder_path
        ] + BACKEND_ARGS

        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return proc.returncode, proc.stdout, proc.stderr
    else:
        # 원격 실행
        ssh_cmd = [
            "ssh",
            "-p", str(server.get("port", 22)),
            f"{server['user']}@{server['host']}",
            f"cd {SCRIPT_DIR} && CUDA_VISIBLE_DEVICES={gpu_id} {sys.executable} {BACKEND_SCRIPT} --input_folder_path '{folder_path}' {' '.join(BACKEND_ARGS)}"
        ]

        proc = subprocess.run(ssh_cmd, capture_output=True, text=True)
        return proc.returncode, proc.stdout, proc.stderr

def worker_process(server, gpu_id, session_id, folder_queue, result_queue, error_queue):
    """
    개별 워커 프로세스입니다.
    """
    worker_name = f"{server['name']}-GPU{gpu_id}-S{session_id}"
    print(f"🚀 [{worker_name}] Started.")

    while True:
        try:
            folder_path = folder_queue.get_nowait()
        except Exception:
            break

        try:
            start_time = time.time()

            returncode, stdout, stderr = execute_remote_task(server, gpu_id, session_id, folder_path)

            if returncode != 0:
                print(f"❌ [{worker_name}] Error processing {folder_path}")
                print(f"   Stderr: {stderr[-500:]}")
                error_queue.put(folder_path)
                continue

            # duration 파싱
            duration = 0.0
            for line in stdout.split('\n'):
                if "Audio duration:" in line and "seconds" in line:
                    try:
                        parts = line.split("Audio duration:")[1].split("seconds")[0].strip()
                        duration = float(parts)
                    except:
                        pass

            result_queue.put((folder_path, duration))

        except Exception as e:
            print(f"💥 [{worker_name}] Critical Exception: {e}")
            error_queue.put(folder_path)

    print(f"💤 [{worker_name}] Finished.")

def main():
    # 1. 서버 및 워커 설정
    total_workers = sum(s["gpus"] * s["sessions_per_gpu"] for s in SERVERS)
    print(f"⚙️  Multi-Server Configuration:")
    for server in SERVERS:
        workers = server["gpus"] * server["sessions_per_gpu"]
        print(f"   - {server['name']}: {server['gpus']} GPUs x {server['sessions_per_gpu']} Sessions = {workers} Workers")
    print(f"   Total: {total_workers} Workers")

    # 2. 데이터 탐색
    all_folders = find_subfolders_with_opus(INPUT_ROOT)
    print(f"Found {len(all_folders)} directories containing audio.")

    # 3. 이어하기 필터링
    processed_folders = load_processed_list()
    todo_folders = [f for f in all_folders if f not in processed_folders]
    print(f"Skipping {len(processed_folders)} already processed. Remaining: {len(todo_folders)}")

    if not todo_folders:
        print("✅ All tasks completed.")
        return

    # 4. WandB 초기화
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
    for server in SERVERS:
        for gpu_id in range(server["gpus"]):
            for session_id in range(server["sessions_per_gpu"]):
                p = multiprocessing.Process(
                    target=worker_process,
                    args=(server, gpu_id, session_id, folder_queue, result_queue, error_queue)
                )
                p.start()
                processes.append(p)
                time.sleep(1)  # 시작 간격

    # 7. 모니터링 루프
    total_tasks = len(todo_folders)
    pbar = tqdm(total=total_tasks, desc="Processing Audio")

    completed_count = 0
    total_audio_duration = 0.0

    while completed_count < total_tasks:
        # 에러 체크
        if not error_queue.empty():
            err_folder = error_queue.get()
            print(f"\n⚠️  Process failed for: {err_folder}")
            completed_count += 1
            pbar.update(1)
            continue

        # 결과 처리
        if not result_queue.empty():
            folder, duration = result_queue.get()

            append_to_processed_list(folder)

            completed_count += 1
            total_audio_duration += duration

            wandb.log({
                "progress_percent": (completed_count / total_tasks) * 100,
                "processed_folders": completed_count,
                "cumulative_audio_hours": total_audio_duration / 3600,
                "current_audio_seconds": duration
            })

            pbar.update(1)

        # 모든 프로세스가 죽었는지 확인
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
    main()
