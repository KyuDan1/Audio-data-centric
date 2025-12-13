# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

# Fix for PyTorch 2.6+ weights_only=True default breaking pyannote model loading
# Patch lightning_fabric's _load function to use weights_only=False
import lightning_fabric.utilities.cloud_io as cloud_io
from pathlib import Path
from typing import Union, IO, Any

_original_load = cloud_io._load

def _patched_load(path_or_url: Union[IO, str, Path], map_location=None) -> Any:
    """Patched version of lightning_fabric's _load that uses weights_only=False for pyannote compatibility"""
    if not isinstance(path_or_url, (str, Path)):
        return torch.load(path_or_url, map_location=map_location, weights_only=False)

    if str(path_or_url).startswith("http"):
        return torch.hub.load_state_dict_from_url(str(path_or_url), map_location=map_location)

    from lightning_fabric.utilities.cloud_io import get_filesystem
    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(f, map_location=map_location, weights_only=False)

cloud_io._load = _patched_load

# Standard library imports
import argparse
import json
import sys
import os
import shutil
import re
import warnings
import time

# Third-party imports
import pandas as pd
from pyannote.audio import Pipeline
from panns_inference import AudioTagging
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.speechlm2.models import SALM
from g2pk import G2p

# Project imports from utils
from utils.tool import (
    load_cfg,
    detect_gpu,
    check_env,
)
from utils.logger import Logger

# Import utils modules with all refactored functions
from utils import audio_preprocessing
from utils import music_processing
from utils import diarization
from utils import asr_ensemble
from utils import separation
from utils import text_processing
from utils import export as export_utils

# Models imports
from models import whisper_asr, silero_vad

# FlowSE denoising class
flowse_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FlowSE")
sys.path.insert(0, flowse_path)
from simple_denoise import FlowSEDenoiser

warnings.filterwarnings("ignore")


def main_process(audio_path, save_path=None, audio_name=None,
                 do_vad = False,
                 LLM = "",
                 use_demucs = False,
                 use_sepreformer = False,
                 overlap_threshold = 1.0,
                 flowse_denoiser = None,
                 sepreformer_separator = None,
                 embedding_model = None,
                 panns_model = None,
                 demucs_model = None,
                 context_caption = False):

    if not audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac")):
        logger.warning(f"Unsupported file type: {audio_path}")
    # 오디오 convert_mono and split_wav_files_2min temp file 만들기.

    # for a single audio from path aaa/bbb/ccc.wav ---> save to aaa/bbb_processed/ccc/ccc_0.wav
    audio_name = audio_name or os.path.splitext(os.path.basename(audio_path))[0]
    suffix = "dia3" if args.dia3 else "ori"
    save_path = save_path or os.path.join(
        os.path.dirname(audio_path), "_final", f"-sepreformer-{args.sepreformer}" +f"-demucs-{args.demucs}"  + f"-vad-{do_vad}"+ f"-diaModel-{suffix}"
        # initial prompt off or on
        + f"-initPrompt-{args.initprompt}"
        + f"-merge_gap-{args.merge_gap}" +f"-seg_th-{args.seg_th}"+ f"-cl_min-{args.min_cluster_size}" +f"-cl-th-{args.clust_th}"+ f"-LLM-{LLM}", audio_name
    )
    os.makedirs(save_path, exist_ok=True)
    logger.debug(
        f"Processing audio: {audio_name}, from {audio_path}, save to: {save_path}"
    )

    logger.info(
        "Step 0: Preprocess all audio files --> 24k sample rate + wave format + loudnorm + bit depth 16"
    )
    audio = audio_preprocessing.standardization(audio_path, cfg)
    diar_chunks, temp_chunk_dir = audio_preprocessing.prepare_diarization_chunks(
        audio_path, audio, vad, silero_vad
    )

    # Calculate total audio duration
    audio_duration = len(audio["waveform"]) / audio["sample_rate"]
    logger.info(f"Total audio duration: {audio_duration:.2f} seconds")

    logger.info("Step 2: Speaker Diarization")
    dia_start = time.time()


    diarization_frames = []
    try:
        for chunk in diar_chunks:
            predicted_segments, _ = diar_model.diarize(
                audio=chunk["path"], batch_size=1, include_tensor_outputs=True
            )
            chunk_df = diarization.sortformer_dia(predicted_segments)
            if not chunk_df.empty:
                chunk_df["start"] += chunk["offset"]
                chunk_df["end"] += chunk["offset"]
            diarization_frames.append(chunk_df)
    finally:
        if temp_chunk_dir:
            shutil.rmtree(temp_chunk_dir, ignore_errors=True)

    if diarization_frames:
        speakerdia = pd.concat(diarization_frames, ignore_index=True)
    else:
        speakerdia = pd.DataFrame(columns=["segment", "label", "speaker", "start", "end"])
    ori_list = diarization.df_to_list(speakerdia)
    dia_end = time.time()

    # Calculate VAD + Sortformer RT factor
    vad_sortformer_processing_time = dia_end - dia_start
    vad_sortformer_rt = vad_sortformer_processing_time / audio_duration if audio_duration > 0 else 0
    logger.info(f"VAD + Sortformer - Processing time: {vad_sortformer_processing_time:.2f}s, RT factor: {vad_sortformer_rt:.4f}")

    # TEST
    ######################
    segment_list = ori_list
    segment_list = diarization.split_long_segments(segment_list)
    ######################

    # [수정됨] Step 3를 Step 2.5보다 먼저 실행!
    # Step 3: Background Music Detection and Removal
    # SepReformer가 실행되기 전에 전체 오디오를 먼저 깨끗하게 만듭니다.
    logger.info("Step 3: Background Music Detection and Removal")
    # padding을 주어 ASR 타임스탬프 오차 범위를 커버
    audio, segment_demucs_flags = music_processing.preprocess_segments_with_demucs(
        segment_list, audio, panns_model=panns_model, use_demucs=use_demucs, demucs_model=demucs_model, padding=0.5
    )

    # [수정됨] 이제 깨끗해진 audio를 가지고 SepReformer 실행
    # Step 2.5: Overlap control using SepReformer
    logger.info("Step 2.5: Overlap Control with SepReformer")
    separation_time = 0.0
    if use_sepreformer and sepreformer_separator is not None and embedding_model is not None:
        separation_start = time.time()
        # 여기서 audio는 이미 Demucs 처리가 된 상태입니다.
        audio, segment_list = separation.process_overlapping_segments_with_separation(
            segment_list,
            audio,
            overlap_threshold=overlap_threshold,
            separator=sepreformer_separator,
            embedding_model=embedding_model,
            device=device_name
        )
        separation_end = time.time()
        separation_time = separation_end - separation_start

        # Calculate SepReformer RT factor
        separation_rt = separation_time / audio_duration if audio_duration > 0 else 0
        logger.info(f"SepReformer separation - Processing time: {separation_time:.2f}s, RT factor: {separation_rt:.4f}")
    else:
        logger.info("SepReformer overlap separation skipped (flag disabled)")

    logger.info("Step 4: ASR (Automatic Speech Recognition)")
    if args.ASRMoE:
        asr_start = time.time()

        asr_result, whisper_time, alignment_time = asr_ensemble.asr_MoE(
            segment_list,
            audio,
            asr_model,
            asr_model_2,
            canary_model,
            segment_demucs_flags=segment_demucs_flags,
            enable_word_timestamps=args.whisperx_word_timestamps,
            device=device_name
        )

        asr_end = time.time()

        dia_time = dia_end-dia_start
        asr_time = whisper_time
    else:
        asr_start = time.time()

        asr_result = asr_ensemble.asr(segment_list, audio, asr_model)

        asr_end = time.time()

        dia_time = dia_end-dia_start
        asr_time = asr_end-asr_start
        alignment_time = 0.0

    # Calculate Whisper large v3 RT factor
    whisper_processing_time = asr_time
    whisper_rt = whisper_processing_time / audio_duration if audio_duration > 0 else 0

    # Calculate WhisperX alignment RT factor
    alignment_rt = alignment_time / audio_duration if audio_duration > 0 else 0

    if LLM == "case_0":
        print("LLM case_0")
        filtered_list = asr_result
        print(f"ASR result contains {len(filtered_list)} segments")




    # LLM post diarization start
    ####################################################################################################
    # "LLM 불러서 post-processing 하는 것"
    elif LLM == "case_2":
        print(f"asr_result len: {len(asr_result)}")
        print("Warning: llm_inference functions are commented out. Using ASR results directly.")
        filtered_list = asr_result


    else:
        raise ValueError("LLM 변수는 case_0, case_1, case_2 중 하나여야 한다.")

    ############################################################################################################
        # LLM post diarization end

    # Step 4.5: Add Qwen3-Omni captions (if enabled)
    caption_time = 0.0
    if args.qwen3omni:
        logger.info("Step 4.5: Adding Qwen3-Omni captions")
        filtered_list, caption_time = export_utils.add_qwen3omni_caption(
            filtered_list, audio, save_path, use_context=context_caption
        )
    else:
        logger.info("Step 4.5: Qwen3-Omni caption generation skipped (flag disabled)")

    # Calculate Qwen3-Omni RT factor
    caption_rt = caption_time / audio_duration if audio_duration > 0 else 0

    # Step 4.6: Apply FlowSE denoising to sepreformer segments (if enabled)
    denoise_time = 0.0
    if args.sepreformer and flowse_denoiser is not None:
        logger.info("Step 4.6: Applying FlowSE denoising to sepreformer segments")
        filtered_list, denoise_time = export_utils.apply_flowse_denoising(
            filtered_list,
            audio,
            save_path,
            denoiser=flowse_denoiser,
            use_asr_moe=args.ASRMoE
        )
    else:
        logger.info("Step 4.6: FlowSE denoising skipped (sepreformer flag disabled or denoiser not loaded)")

    # Calculate FlowSE denoising RT factor
    denoise_rt = denoise_time / audio_duration if audio_duration > 0 else 0

    # Print all timing information
    print(f"\n{'='*60}")
    print(f"Audio duration: {audio_duration:.2f} seconds ({audio_duration/60:.2f} minutes)")
    print(f"{'='*60}")
    print(f"VAD + Sortformer:")
    print(f"  - Processing time: {dia_time:.2f} seconds")
    print(f"  - RT factor: {vad_sortformer_rt:.4f}")
    print(f"{'='*60}")
    if use_sepreformer:
        print(f"SepReformer Overlap Separation:")
        print(f"  - Processing time: {separation_time:.2f} seconds")
        print(f"  - RT factor: {separation_rt:.4f}")
        print(f"{'='*60}")
    print(f"Whisper large v3:")
    print(f"  - Processing time: {asr_time:.2f} seconds")
    print(f"  - RT factor: {whisper_rt:.4f}")
    print(f"{'='*60}")
    if args.whisperx_word_timestamps:
        print(f"WhisperX Alignment:")
        print(f"  - Processing time: {alignment_time:.2f} seconds")
        print(f"  - RT factor: {alignment_rt:.4f}")
        print(f"{'='*60}")
    if args.qwen3omni:
        print(f"Qwen3-Omni Caption:")
        print(f"  - Processing time: {caption_time:.2f} seconds")
        print(f"  - RT factor: {caption_rt:.4f}")
        print(f"{'='*60}")
    if args.sepreformer and denoise_time > 0:
        print(f"FlowSE Denoising:")
        print(f"  - Processing time: {denoise_time:.2f} seconds")
        print(f"  - RT factor: {denoise_rt:.4f}")
        print(f"{'='*60}")
    print()

    logger.info("Step 5: Write result into MP3 and JSON file")
    print(f"Exporting {len(filtered_list)} segments to MP3 and JSON...")
    export_utils.export_segments_with_enhanced_audio(audio, filtered_list, save_path, audio_name)

    # 한국어 g2p 후처리
    if args.korean:
        text_processing.ko_process_json(filtered_list)

    cleaned_list = []
    for item in filtered_list:
        # 얕은 복사(copy)를 통해 원본 filtered_list에는 영향 주지 않도록 함
        clean_item = item.copy()

        # 1. 'enhanced_audio' (오디오 행렬) 키가 있으면 삭제
        if "enhanced_audio" in clean_item:
            del clean_item["enhanced_audio"]

        # 2. (혹시 모를 에러 방지) Numpy float/int 타입을 Python native 타입으로 변환
        for k, v in clean_item.items():
            if hasattr(v, 'item'):  # numpy 타입인 경우
                clean_item[k] = v.item()

        cleaned_list.append(clean_item)

    # Prepare output with RT factor metrics
    output_data = {
        "metadata": {
            "audio_duration_seconds": audio_duration,
            "audio_duration_minutes": audio_duration / 60,
            "vad_sortformer": {
                "processing_time_seconds": vad_sortformer_processing_time,
                "rt_factor": vad_sortformer_rt
            },
            "whisper_large_v3": {
                "processing_time_seconds": whisper_processing_time,
                "rt_factor": whisper_rt
            },
            # [수정] filtered_list 대신 cleaned_list 길이 사용
            "total_segments": len(cleaned_list)
        },
        # [수정] 여기서 filtered_list가 아닌 cleaned_list를 넣어야 합니다.
        "segments": cleaned_list
    }

    # Add WhisperX alignment metadata if enabled
    if args.whisperx_word_timestamps:
        output_data["metadata"]["whisperx_alignment"] = {
            "processing_time_seconds": alignment_time,
            "rt_factor": alignment_rt,
            "enabled": True
        }

    # Add Qwen3-Omni caption metadata if enabled
    if args.qwen3omni:
        output_data["metadata"]["qwen3omni_caption"] = {
            "processing_time_seconds": caption_time,
            "rt_factor": caption_rt,
            "enabled": True
        }

    # Add SepReformer separation metadata if enabled
    if use_sepreformer:
        output_data["metadata"]["sepreformer_separation"] = {
            "processing_time_seconds": separation_time,
            "rt_factor": separation_rt,
            "overlap_threshold_seconds": overlap_threshold,
            "enabled": True
        }

    # Add FlowSE denoising metadata if enabled
    if args.sepreformer and denoise_time > 0:
        output_data["metadata"]["flowse_denoising"] = {
            "processing_time_seconds": denoise_time,
            "rt_factor": denoise_rt,
            "enabled": True
        }

    final_path = os.path.join(save_path, audio_name + ".json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"All done, Saved to: {final_path}")
    print(f"Processing complete! Results saved to: {final_path}")
    print(f"Total segments processed: {len(filtered_list)}")
    return final_path, filtered_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default="",
        help="input folder path, this will override config if set",
    )
    parser.add_argument(
        "--config_path", type=str, default="config.json", help="config path"
    )

    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    # 32 정도 괜찮을듯?

    parser.add_argument(
        "--compute_type",
        type=str,
        default="float16",
        help="The compute type to use for the model",
    )
    parser.add_argument(
        "--whisper_arch",
        type=str,
        default="large-v3",
        help="The name of the Whisper model to load.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="The number of CPU threads to use per worker, e.g. will be multiplied by num workers.",
    )
    parser.add_argument(
        "--exit_pipeline",
        type=bool,
        default=False,
        help="Exit pipeline when task done.",
    )
    parser.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Turning on vad.",
    )
    parser.add_argument(
        "--dia3",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Turning on diaization 3.0 model",
    )
    parser.add_argument(
        "--LLM",
        type=str,
        default="case_2",
        help="LLM diarization cases",
    )

    # hyperparameter
    parser.add_argument(
        "--seg_th",
        type=float,
        default=0.15,
        help="diarization model segmentation threshold",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=10,
        help="diarization model clustering min_cluster_size",
    )
    parser.add_argument(
        "--clust_th",
        type=float,
        default=0.5,
        help="diarization model clustering threshold",
    )

    parser.add_argument(
        "--merge_gap",
        type=float,
        default=2,
        help="merge gap in seconds, if smaller than this, merge",
    )

    parser.add_argument(
        "--initprompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Turning on initial prompt on whisper model",
    )
    parser.add_argument(
        "--korean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="korean_g2p",
    )

    parser.add_argument(
        "--ASRMoE",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="parakeet",
    )

    parser.add_argument(
        "--demucs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable background music detection and removal using PANNs and Demucs",
    )

    parser.add_argument(
        "--whisperx_word_timestamps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable WhisperX word-level timestamps with alignment",
    )

    parser.add_argument(
        "--qwen3omni",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen3-Omni audio captioning for each segment",
    )

    parser.add_argument(
        "--context_caption",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable context-aware captioning using previous 2 segments for in-context learning",
    )

    parser.add_argument(
        "--sepreformer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable SepReformer for overlapping speech separation",
    )

    parser.add_argument(
        "--overlap_threshold",
        type=float,
        default=1.0,
        help="Minimum overlap duration in seconds to trigger SepReformer separation",
    )


    args = parser.parse_args()

    batch_size = args.batch_size
    cfg = load_cfg(args.config_path)

    logger = Logger.get_logger()

    # Initialize loggers for utils modules
    audio_preprocessing.set_logger(logger)
    music_processing.set_logger(logger)
    diarization.set_logger(logger)
    asr_ensemble.set_logger(logger)
    separation.set_logger(logger)
    export_utils.set_logger(logger)

    if args.input_folder_path:
        logger.info(f"Using input folder path: {args.input_folder_path}")
        cfg["entrypoint"]["input_folder_path"] = args.input_folder_path

    logger.debug("Loading models...")

    # Load models
    if detect_gpu():
        logger.info("Using GPU")
        device_name = "cuda"
        device = torch.device(device_name)
    else:
        logger.info("Using CPU")
        device_name = "cpu"
        device = torch.device(device_name)
        # whisperX expects compute type: int8
        logger.info("Overriding the compute type to int8")
        args.compute_type = "int8"

    check_env(logger)

    # Speaker Diarization
    logger.debug(" * Loading Speaker Diarization Model")
    if not cfg["huggingface_token"].startswith("hf"):
        raise ValueError(
            "huggingface_token must start with 'hf', check the config file. "
            "You can get the token at https://huggingface.co/settings/tokens. "
            "Remeber grant access following https://github.com/pyannote/pyannote-audio?tab=readme-ov-file#tldr"
        )
    if args.dia3 == True:
        print("Using diarization-3.1 model")
        dia_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        #"pyannote/speaker-diarization",
        use_auth_token=cfg["huggingface_token"],

    )
        dia_pipeline.to(device)

    else:
        dia_pipeline = Pipeline.from_pretrained(
            #"pyannote/speaker-diarization-3.1",
            "pyannote/speaker-diarization",
            use_auth_token=cfg["huggingface_token"]
        )
        dia_pipeline.to(device)

        # hyperparameters
        dia_pipeline.instantiate({
        "segmentation": {
            "min_duration_off": 0.0,
            "threshold": args.seg_th
        },
        "clustering": {
            "method": "centroid",
            "min_cluster_size": args.min_cluster_size,
            "threshold": args.clust_th
        }
    })
    # ASR
    logger.debug(" * Loading ASR Model")

    if args.initprompt == True:
        asr_options_dict = {
            #"log_prob_threshold": -1.0,
            #"no_speech_threshold": 0.6,
            # 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음

            # 원래 코드의 initial prompt
            # "initial_prompt": "ha. heh. Mm, hmm. Mm hm. uh. Uh huh. Mm huh. Uh. hum Uh. Ah. Uh hu. Like. you know. Yeah. I mean. right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. Hoo. Hu. Hoo, hoo. Heah. Ha. Yu. Nah. Uh-huh. No way. Uh-oh. Jeez. Whoa. Dang. Gosh. Duh. Whoops. Phew. Woo. Ugh. Er. Geez. Oh wow. Oh man. Uh yeah. Uh huh. For real?",

            #"initial_prompt": "ha. heh. Mm, hmm. uh. "
            #"initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. Hoo. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음.",

            # 아예 initial_prompt 안 넣어 줄 때: ASR이 안 되는 구간이 생기진 않지만 추임새 인식이 잘 안 됨.
            # 원래 코드의 initial prompt를 넣어 줄 때:  ASR이 안 되는 구간이 생기진 않지만 '원하는' 추임새 인식이 잘 안 됨.
            # 완전히 다른 initial prompt를 넣어 줄 때 (뒤에 일본어 한자 다 지울 때): ASR이 안 되는 구간이 드문 드문 생김.
            # 원래 코드의 initial prompt에서 원하는 추임새만 3단어 이하로 수정할 때: 적당히 추가하면 ASR이 안 되는 구간이 생기지 않으면서 원하는 추임새 인식이 잘 됨.


            "initial_prompt": "Um. Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. Mm. So. Oh. Hoo hoo.生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음.",

        }
        # Add word_timestamps if flag is enabled
        if args.whisperx_word_timestamps:
            asr_options_dict["word_timestamps"] = True

        asr_model = whisper_asr.load_asr_model(
            "large-v3",
            device_name,
            compute_type=args.compute_type,
            threads=args.threads,
            language="en", # 언어 지정 한국어.

        # whisper_asr.py 의 default_asr_options 수정으로 asr 모델 수정 가능.

            asr_options=asr_options_dict,
        )
    else:
        asr_options_dict = {}
        # Add word_timestamps if flag is enabled
        if args.whisperx_word_timestamps:
            asr_options_dict["word_timestamps"] = True

        asr_model = whisper_asr.load_asr_model(
            "large-v3",
            device_name,
            compute_type=args.compute_type,
            threads=args.threads,

            language="en",
            asr_options=asr_options_dict if asr_options_dict else None,

            )
    if args.ASRMoE:
        import nemo.collections.asr as nemo_asr
        asr_model_2 = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

        # Load Canary model
        logger.debug(" * Loading Canary Model")
        canary_model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
        canary_model = canary_model.to(device)
        canary_model.eval()
        logger.debug(f" * Canary model loaded on {device}")
        # 클라이언트 초기화
    #client = OpenAI(api_key=OPENAI_API_KEY)
    model_name = "gpt-4.1"
    # VAD
    logger.debug(" * Loading VAD Model")
    vad = silero_vad.SileroVAD(device=device)

    # g2p 인스턴스 초기화
    G2P = G2p()

    # 영어 구간 탐지 패턴: 연속된 알파벳과 apostrophe, 공백으로 연결된 단어 그룹
    ENG_PATTERN = re.compile(r"[A-Za-z][A-Za-z']*(?: [A-Za-z][A-Za-z']*)*")

    # load model from Hugging Face model card directly (You need a Hugging Face token)
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
    diar_model.eval()

    # Pyannote embedding model 초기화 (sepreformer가 활성화된 경우에만)
    embedding_model = None
    if args.sepreformer:
        logger.debug(" * Loading Pyannote Embedding Model")
        try:
            from pyannote.audio import Model as PyannoteModel
            embedding_model = PyannoteModel.from_pretrained("pyannote/embedding", use_auth_token=cfg["huggingface_token"])
            embedding_model = embedding_model.to(device)
            logger.debug(" * Pyannote Embedding Model loaded successfully")
        except Exception as e:
            logger.error(f" * Failed to load Pyannote Embedding Model: {e}")
            embedding_model = None

    # SepReformer separator 초기화 (sepreformer가 활성화된 경우에만)
    sepreformer_separator = None
    if args.sepreformer:
        logger.debug(" * Loading SepReformer Separator Model")
        try:
            sepreformer_separator = separation.SepReformerSeparator(
                sepreformer_path="/mnt/ddn/kyudan/Audio-data-centric/SepReformer",
                device=device
            )
            logger.debug(" * SepReformer Separator loaded successfully")
        except Exception as e:
            logger.error(f" * Failed to load SepReformer Separator: {e}")
            sepreformer_separator = None

    # FlowSE denoiser 초기화 (sepreformer가 활성화된 경우에만)
    flowse_denoiser = None
    if args.sepreformer:
        logger.debug(" * Loading FlowSE Denoiser Model")
        try:
            flowse_denoiser = FlowSEDenoiser(
                checkpoint_path="/mnt/ddn/kyudan/Audio-data-centric/FlowSE/ckpts/best.pt.tar",
                tokenizer_path="/mnt/ddn/kyudan/Audio-data-centric/FlowSE/Emilia_ZH_EN_pinyin/vocab.txt",
                vocoder_path="/mnt/ddn/kyudan/Audio-data-centric/FlowSE/vocos-mel-24khz",
                use_cuda=(device_name == "cuda")
            )
            logger.debug(" * FlowSE Denoiser loaded successfully")
        except Exception as e:
            logger.error(f" * Failed to load FlowSE Denoiser: {e}")
            flowse_denoiser = None

    # PANNs model 초기화 (배경음악 검출용)
    panns_model = None
    if args.demucs:
        logger.debug(" * Loading PANNs Model for background music detection")
        try:
            panns_data_dir = '/mnt/ddn/kyudan/panns_data'
            os.makedirs(panns_data_dir, exist_ok=True)
            os.environ['PANNS_DATA'] = panns_data_dir
            checkpoint_path = os.path.join(panns_data_dir, 'Cnn14_mAP=0.431.pth')
            panns_model = AudioTagging(checkpoint_path=checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
            logger.debug(" * PANNs Model loaded successfully")
        except Exception as e:
            logger.error(f" * Failed to load PANNs Model: {e}")
            panns_model = None

    # Demucs model 초기화 (배경음악 제거용)
    demucs_model = None
    if args.demucs:
        logger.debug(" * Loading Demucs Model for vocal separation")
        try:
            from utils.music_processing import DemucsModel
            demucs_model = DemucsModel(model_name="htdemucs", device=device_name)
            logger.debug(" * Demucs Model loaded successfully")
        except Exception as e:
            logger.error(f" * Failed to load Demucs Model: {e}")
            logger.warning(" * Will fall back to subprocess method (slower)")
            demucs_model = None

    logger.debug("All models loaded")

    supported_languages = cfg["language"]["supported"]
    multilingual_flag = cfg["language"]["multilingual"]
    logger.debug(f"supported languages multilingual {supported_languages}")
    logger.debug(f"using multilingual asr {multilingual_flag}")

    input_folder_path = cfg["entrypoint"]["input_folder_path"]

    if not os.path.exists(input_folder_path):
        raise FileNotFoundError(f"input_folder_path: {input_folder_path} not found")

    # Get only audio files in the specified directory (not recursive)
    audio_extensions = ('.mp3', '.wav', '.flac', '.m4a', '.aac')
    audio_paths = []
    for file in os.listdir(input_folder_path):
        if file.endswith(audio_extensions) and not ".temp" in file:
            audio_paths.append(os.path.join(input_folder_path, file))

    logger.debug(f"Scanning {len(audio_paths)} audio files in {input_folder_path} (non-recursive)")

    start_time = time.time()
    for path in audio_paths:


        # 폴더가 이미 있으면 넘어가는 로직

    #    # 1. main_process와 동일한 로직으로 예상 출력 폴더 경로를 생성합니다.
    #     audio_name = os.path.splitext(os.path.basename(path))[0]
    #     suffix = "dia3" if args.dia3 else "ori"
    #     save_path_dir = os.path.join(
    #         os.path.dirname(path) + "_processed_llm-twelve-cases" + f"-vad-{args.vad}"+ f"-diaModel-{suffix}"
    #         + f"-merge_gap-{args.merge_gap}" + f"-seg_th-{args.seg_th}"+ f"-cl_min-{args.min_cluster_size}" +f"-cl-th-{args.clust_th}"+ f"-LLM-{args.LLM}", audio_name
    #     )

    #     # 2. 해당 폴더가 이미 존재하는지 확인합니다.
    #     if os.path.exists(save_path_dir):
    #         logger.info(f"Output directory already exists, skipping: {save_path_dir}")
    #         continue  # 폴더가 존재하면 다음 오디오 파일로 넘어갑니다.


        main_process(path, do_vad=args.vad, LLM=args.LLM, use_demucs=args.demucs,
                     use_sepreformer=args.sepreformer, overlap_threshold=args.overlap_threshold,
                     flowse_denoiser=flowse_denoiser, sepreformer_separator=sepreformer_separator,
                     embedding_model=embedding_model, panns_model=panns_model, demucs_model=demucs_model,
                     context_caption=args.context_caption)
    end_time = time.time()
    print("Total time:", end_time - start_time)
