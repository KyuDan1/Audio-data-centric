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

# 3. 로그 및 상태 저장 파일 (서버별로 다른 파일명 사용)
PROGRESS_LOG_FILE = "processed_folders_log.txt"

# 4. WandB 설정
WANDB_PROJECT = "audio-pipeline-monitoring"
WANDB_ENTITY = "krafton_kyudan"

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
    if not os.path.exists(PROGRESS_LOG_FILE):
        return set()
    with open(PROGRESS_LOG_FILE, 'r') as f:
        return set(line.strip() for line in f)

def append_to_processed_list(folder_path):
    """처리 완료된 폴더를 로그에 기록합니다."""
    with open(PROGRESS_LOG_FILE, 'a') as f:
        f.write(f"{folder_path}\n")

def worker_process(gpu_id, session_id, folder_queue, result_queue, error_queue):
    """
    개별 워커 프로세스입니다.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    worker_name = f"GPU{gpu_id}-Worker{session_id}"
    print(f"🚀 [{worker_name}] Started.")

    while True:
        try:
            folder_path = folder_queue.get_nowait()
        except Exception:
            break

        try:
            cmd = [sys.executable, BACKEND_SCRIPT, "--input_folder_path", folder_path] + BACKEND_ARGS

            start_time = time.time()

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=os.environ.copy()
            )

            if process.returncode != 0:
                print(f"❌ [{worker_name}] Error processing {folder_path}\nStderr: {process.stderr[-500:]}")
                error_queue.put(folder_path)
                continue

            duration = 0.0
            for line in process.stdout.split('\n'):
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
    parser = argparse.ArgumentParser(description='Run audio processing pipeline with task splitting')
    parser.add_argument('--start-idx', type=int, default=0, help='Start index of folders to process')
    parser.add_argument('--end-idx', type=int, default=None, help='End index of folders to process (exclusive)')
    parser.add_argument('--server-name', type=str, default='local', help='Server name for WandB tracking')
    args = parser.parse_args()

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

    # 3. 작업 분할
    if args.end_idx is None:
        args.end_idx = len(all_folders)

    assigned_folders = all_folders[args.start_idx:args.end_idx]
    print(f"📊 Assigned range: folders {args.start_idx} to {args.end_idx} ({len(assigned_folders)} folders)")

    # 4. 이어하기 필터링
    processed_folders = load_processed_list()
    todo_folders = [f for f in assigned_folders if f not in processed_folders]
    print(f"Skipping {len(processed_folders)} already processed. Remaining: {len(todo_folders)}")

    if not todo_folders:
        print("✅ All tasks completed.")
        return

    # 5. 모델 캐시 사전 로드 (race condition 방지)
    print("🔄 Pre-loading models to cache...")
    try:
        # VAD 모델 사전 로드
        import torch.hub
        print("   - Loading VAD model...")
        torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
            source="github",
        )
        print("   ✅ VAD model cached")
    except Exception as e:
        print(f"   ⚠️ Warning: Could not pre-load VAD model: {e}")
        print("   Continuing anyway - workers will download if needed")

    # 6. WandB 초기화
    run_name = f"{args.server_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY if WANDB_ENTITY else None, name=run_name)

    # 7. 멀티프로세싱 준비
    manager = multiprocessing.Manager()
    folder_queue = manager.Queue()
    result_queue = manager.Queue()
    error_queue = manager.Queue()

    for folder in todo_folders:
        folder_queue.put(folder)

    # 8. 워커 생성 및 시작
    processes = []
    for gpu_id in range(num_gpus):
        for session_id in range(sessions_per_gpu):
            p = multiprocessing.Process(
                target=worker_process,
                args=(gpu_id, session_id, folder_queue, result_queue, error_queue)
            )
            p.start()
            processes.append(p)
            time.sleep(2)

    # 9. 모니터링 루프
    total_tasks = len(todo_folders)
    pbar = tqdm(total=total_tasks, desc="Processing Audio")

    completed_count = 0
    total_audio_duration = 0.0

    while completed_count < total_tasks:
        if not error_queue.empty():
            err_folder = error_queue.get()
            print(f"\n⚠️ Process failed for: {err_folder}")
            completed_count += 1
            pbar.update(1)
            continue

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

        if not any(p.is_alive() for p in processes) and result_queue.empty() and error_queue.empty():
            print("\nAll workers stopped unexpectedly.")
            break

        time.sleep(0.1)

    # 10. 종료 처리
    for p in processes:
        p.join()

    wandb.finish()
    print(f"\n🎉 Pipeline Finished. Total Audio Processed: {total_audio_duration/3600:.2f} hours.")

if __name__ == "__main__":
    main()
