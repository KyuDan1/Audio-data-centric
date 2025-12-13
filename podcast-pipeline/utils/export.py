"""
Export and data conversion utilities for podcast pipeline.
Includes audio export, caption addition, and FlowSE denoising integration.
"""

import os
import time
import requests
import numpy as np
import soundfile as sf
from pydub import AudioSegment as PydubAudioSegment

# Logger will be initialized from main module
logger = None

# Constants
QWEN_3_OMNI_PORT = "10856"

def set_logger(log_instance):
    """Set logger instance from main module."""
    global logger
    logger = log_instance


import os
import time
import soundfile as sf
import requests

def add_qwen3omni_caption(filtered_list, audio, save_path, use_context=False):
    """
    Add Qwen3-Omni captions to each ASR segment.
    - Target for captioning is ALWAYS the current segment only.
    - Previous segments (up to 2) are provided as CONTEXT ONLY.
    """
    mode_str = "context-aware" if use_context else "standard"
    logger.info(f"Adding Qwen3-Omni captions ({mode_str} mode) to {len(filtered_list)} segments...")
    caption_start_time = time.time()

    url = f"http://localhost:{QWEN_3_OMNI_PORT}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # 강한 가드레일: "타겟만 캡션" + "출력은 캡션 텍스트만"
    system_prompt = (
        "You are an audio captioning model.\n"
        "You may receive multiple audio clips.\n"
        "IMPORTANT:\n"
        "- Only the clip labeled [TARGET] must be captioned.\n"
        "- Clips labeled [CONTEXT] are for understanding only (e.g., sarcasm, intent). Do NOT caption them.\n"
        "- Output ONLY the caption text for [TARGET].\n"
        "- Do not add labels, numbering, quotes, or extra commentary.\n"
        "- Return a single caption in one paragraph."
    )

    def _build_content_list(context_paths, target_path):
        content = []
        content.append({
            "type": "text",
            "text": (
                "You will receive audio clips in order.\n"
                "Caption ONLY the one marked [TARGET].\n"
                "Do NOT caption any [CONTEXT] clips.\n"
                "Output only the target caption text."
            )
        })

        for i, p in enumerate(context_paths, start=1):
            content.append({"type": "text", "text": f"[CONTEXT {i}]"})
            content.append({"type": "audio_url", "audio_url": {"url": f"file://{p}"}})

        content.append({"type": "text", "text": "[TARGET]"})
        content.append({"type": "audio_url", "audio_url": {"url": f"file://{target_path}"}})
        return content

    for idx, segment in enumerate(filtered_list):
        temp_audio_paths = []
        try:
            context_paths = []

            # --- context 오디오 준비 (최대 2개) ---
            if use_context:
                start_c = max(0, idx - 2)
                for context_idx in range(start_c, idx):
                    context_segment = filtered_list[context_idx]
                    context_audio = _extract_segment_audio(context_segment, audio)
                    temp_path = os.path.join(save_path, f"temp_context_{idx:05d}_{context_idx:05d}.wav")
                    sf.write(temp_path, context_audio, audio["sample_rate"])
                    temp_audio_paths.append(temp_path)
                    context_paths.append(temp_path)

            # --- target(현재) 오디오 준비 ---
            segment_audio = _extract_segment_audio(segment, audio)
            target_path = os.path.join(save_path, f"temp_target_{idx:05d}.wav")
            sf.write(target_path, segment_audio, audio["sample_rate"])
            temp_audio_paths.append(target_path)

            # --- 메시지 구성 ---
            content_list = _build_content_list(context_paths, target_path)

            data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_list},
                ],
                # 필요하면 옵션 추가 (서버가 지원할 때만)
                # "temperature": 0.2,
                # "max_tokens": 128,
            }

            response = requests.post(url, headers=headers, json=data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                caption = result.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                caption = caption.strip()

                # 혹시 모델이 규칙을 어기고 여러 줄/라벨을 붙이면 "가장 마지막 문단"만 쓰는 안전장치(선택)
                # (원치 않으면 제거)
                if "\n" in caption:
                    caption = caption.split("\n")[-1].strip()

                segment["qwen3omni_caption"] = caption
                logger.debug(f"Segment {idx}: caption added (context={use_context and idx>0}, context_n={len(context_paths)})")
            else:
                logger.warning(f"Segment {idx}: API call failed with status {response.status_code} | {response.text[:200]}")
                segment["qwen3omni_caption"] = ""

        except Exception as e:
            logger.error(f"Segment {idx}: Error calling Qwen3-Omni API: {e}")
            segment["qwen3omni_caption"] = ""

        finally:
            # 어떤 경우에도 임시 파일 정리
            for p in temp_audio_paths:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

    caption_processing_time = time.time() - caption_start_time
    logger.info(f"Qwen3-Omni caption addition completed ({mode_str} mode)")
    return filtered_list, caption_processing_time



def _extract_segment_audio(segment, audio):
    """
    세그먼트에서 오디오를 추출합니다.
    SepReformer로 처리된 enhanced_audio가 있으면 우선 사용합니다.

    Args:
        segment (dict): 세그먼트 정보
        audio (dict): 전체 오디오 정보 (waveform, sample_rate 포함)

    Returns:
        numpy.ndarray: 추출된 오디오 데이터
    """
    # [CRITICAL] SepReformer로 처리된 오디오가 있으면 우선 사용 (저장될 파일과 일치시키기 위함)
    if "enhanced_audio" in segment:
        return segment["enhanced_audio"]
    else:
        start_time = segment["start"]
        end_time = segment["end"]
        sample_rate = audio["sample_rate"]
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        return audio["waveform"][start_frame:end_frame]


def apply_flowse_denoising(filtered_list, audio, save_path, denoiser=None, use_asr_moe=False):
    """
    sepreformer==True인 세그먼트에 대해 FlowSE 디노이징을 적용합니다.

    Args:
        filtered_list (list): ASR 결과 세그먼트 리스트
        audio (dict): 오디오 딕셔너리 (waveform, sample_rate 포함)
        save_path (str): 임시 오디오 파일을 저장할 경로
        denoiser: FlowSE denoiser 객체 (None이면 건너뜀)
        use_asr_moe (bool): ASR MoE 사용 여부 (메타데이터 키 선택에 영향)

    Returns:
        tuple: (FlowSE 처리된 세그먼트 리스트, 처리 시간(초))
    """
    if denoiser is None:
        logger.info("FlowSE denoiser not provided, skipping denoising")
        return filtered_list, 0.0

    logger.info(f"Applying FlowSE denoising to sepreformer segments...")
    denoising_start_time = time.time()

    denoised_count = 0

    for idx, segment in enumerate(filtered_list):
        # sepreformer가 적용된 세그먼트만 처리
        sepreformer_key = "sepreformer" if use_asr_moe else "sepreformer"
        if not segment.get(sepreformer_key, False):
            continue

        # enhanced_audio가 있는지 확인
        if "enhanced_audio" not in segment:
            logger.warning(f"Segment {idx}: sepreformer=True but no enhanced_audio found")
            continue

        temp_input_path = None
        temp_output_path = None

        try:
            # 세그먼트 오디오 추출
            segment_audio = segment["enhanced_audio"]
            sample_rate = audio["sample_rate"]

            # 임시 파일로 저장 (FlowSE 입력용)
            temp_input_path = os.path.join(save_path, f"temp_sepreformer_{idx:05d}.wav")
            sf.write(temp_input_path, segment_audio, sample_rate)

            # ASR 텍스트 가져오기 (FlowSE는 텍스트 기반 디노이징)
            text = segment.get("text", "")
            if not text:
                logger.warning(f"Segment {idx}: No text found, skipping FlowSE denoising")
                segment["flowse_denoised"] = False
            else:
                # FlowSE 디노이징 수행
                temp_output_path = os.path.join(save_path, f"temp_denoised_{idx:05d}.wav")
                denoiser.denoise(temp_input_path, text, temp_output_path)

                # 디노이징된 오디오가 생성되었는지 확인
                if os.path.exists(temp_output_path):
                    # 메타데이터에 디노이징 플래그 추가
                    segment["flowse_denoised"] = True
                    segment["denoised_audio_path"] = temp_output_path
                    denoised_count += 1
                    logger.debug(f"Segment {idx}: FlowSE denoising completed")
                else:
                    logger.warning(f"Segment {idx}: FlowSE output not found")
                    segment["flowse_denoised"] = False

        except Exception as e:
            logger.error(f"Segment {idx}: FlowSE denoising failed: {e}")
            segment["flowse_denoised"] = False

        finally:
            # 임시 파일 정리 (입력 파일은 항상 삭제)
            if temp_input_path and os.path.exists(temp_input_path):
                os.remove(temp_input_path)

    denoising_end_time = time.time()
    denoising_processing_time = denoising_end_time - denoising_start_time

    logger.info(f"FlowSE denoising completed: {denoised_count}/{len(filtered_list)} segments processed")
    return filtered_list, denoising_processing_time


def export_segments_with_enhanced_audio(audio_info, segment_list, save_dir, audio_name):
    """
    Export segments to MP3 files.
    If 'enhanced_audio' exists in the segment (processed by SepReformer), use it.
    Otherwise, slice from the original audio.
    """
    # 세그먼트 저장용 폴더 생성
    segments_dir = os.path.join(save_dir, audio_name)
    os.makedirs(segments_dir, exist_ok=True)

    # 원본 전체 오디오 (Pydub 객체)
    full_audio_segment = audio_info.get("audio_segment")
    sample_rate = audio_info["sample_rate"]

    if full_audio_segment is None:
        # 만약 audio_segment가 없으면 waveform에서 생성 (fallback)
        waveform_int16 = (audio_info["waveform"] * 32767).astype(np.int16)
        full_audio_segment = PydubAudioSegment(
            waveform_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )

    logger.info(f"Exporting {len(segment_list)} segments with enhanced audio check...")

    for i, seg in enumerate(segment_list):
        # 파일명 생성 (예: 00001_SPEAKER_01.mp3)
        idx_str = seg.get("index", f"{i:05d}")
        spk = seg.get("speaker", "Unknown")
        filename = f"{idx_str}_{spk}.mp3"
        file_path = os.path.join(segments_dir, filename)

        # 1. FlowSE 디노이징된 오디오가 있는지 확인 (최우선)
        if seg.get("flowse_denoised", False) and "denoised_audio_path" in seg:
            denoised_path = seg["denoised_audio_path"]
            if os.path.exists(denoised_path):
                # 디노이징된 오디오 파일 로드
                denoised_waveform, denoised_sr = sf.read(denoised_path)

                # float32 (-1.0 ~ 1.0) -> int16 범위로 변환
                denoised_waveform = np.clip(denoised_waveform, -1.0, 1.0)
                wav_int16 = (denoised_waveform * 32767).astype(np.int16)

                target_segment = PydubAudioSegment(
                    wav_int16.tobytes(),
                    frame_rate=denoised_sr,
                    sample_width=2,
                    channels=1
                )
                logger.debug(f"Segment {idx_str}: Saved using FlowSE denoised output.")

                # MP3로 내보낸 후 임시 파일 삭제
                target_segment.export(file_path, format="mp3")
                os.remove(denoised_path)
                logger.debug(f"Segment {idx_str}: Cleaned up temporary denoised file.")
                continue

        # 2. SepReformer가 적용된 'enhanced_audio'가 있는지 확인
        elif seg.get("is_separated", False) and "enhanced_audio" in seg:
            # Numpy array -> Pydub AudioSegment 변환
            enhanced_waveform = seg["enhanced_audio"]

            # float32 (-1.0 ~ 1.0) -> int16 범위로 변환
            # 클리핑 방지를 위해 clip 적용
            enhanced_waveform = np.clip(enhanced_waveform, -1.0, 1.0)
            wav_int16 = (enhanced_waveform * 32767).astype(np.int16)

            target_segment = PydubAudioSegment(
                wav_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )
            # logger.debug(f"Segment {idx_str}: Saved using SepReformer output.")

        else:
            # 3. 적용되지 않은 경우 원본에서 추출
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            target_segment = full_audio_segment[start_ms:end_ms]

        # MP3 저장
        target_segment.export(file_path, format="mp3")
