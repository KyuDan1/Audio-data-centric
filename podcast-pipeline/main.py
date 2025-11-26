# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import shutil
import argparse
import json
import librosa
import numpy as np
import sys
import wave
import audioop
import io
import time
import datetime
import ast
import re
import json
import argparse
import os
import tqdm
import warnings
import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline
import pandas as pd

from tqdm import tqdm
from utils.tool import (
    export_to_mp3,
    load_cfg,
    get_audio_files,
    detect_gpu,
    check_env,
    calculate_audio_stats,
)
from utils.logger import Logger, time_logger
from tritony import InferenceClient
from models import separate_fast, dnsmos, whisper_asr, silero_vad
from nemo.collections.asr.models import SortformerEncLabelModel
from g2pk import G2p


warnings.filterwarnings("ignore")
audio_count = 0


@time_logger
def standardization(audio):
    """
    Preprocess the audio file, including setting sample rate, bit depth, channels, and volume normalization.

    Args:
        audio (str or AudioSegment): Audio file path or AudioSegment object, the audio to be preprocessed.

    Returns:
        dict: A dictionary containing the preprocessed audio waveform, audio file name, and sample rate, formatted as:
              {
                  "waveform": np.ndarray, the preprocessed audio waveform, dtype is np.float32, shape is (num_samples,)
                  "name": str, the audio file name
                  "sample_rate": int, the audio sample rate
              }

    Raises:
        ValueError: If the audio parameter is neither a str nor an AudioSegment.
    """
    global audio_count
    name = "audio"

    if isinstance(audio, str):
        name = os.path.basename(audio)
        audio = AudioSegment.from_file(audio)
    elif isinstance(audio, AudioSegment):
        name = f"audio_{audio_count}"
        audio_count += 1
    else:
        raise ValueError("Invalid audio type")

    logger.debug("Entering the preprocessing of audio")

    # Convert the audio file to WAV format
    audio = audio.set_frame_rate(cfg["entrypoint"]["SAMPLE_RATE"])
    audio = audio.set_sample_width(2)  # Set bit depth to 16bit
    audio = audio.set_channels(1)  # Set to mono

    logger.debug("Audio file converted to WAV format")

    # Calculate the gain to be applied
    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    logger.info(f"Calculating the gain needed for the audio: {gain} dB")

    # Normalize volume and limit gain range to between -3 and 3
    normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
    max_amplitude = np.max(np.abs(waveform))
    waveform /= max_amplitude  # Normalize

    logger.debug(f"waveform shape: {waveform.shape}")
    logger.debug("waveform in np ndarray, dtype=" + str(waveform.dtype))

    return {
        "waveform": waveform,
        "name": name,
        "sample_rate": cfg["entrypoint"]["SAMPLE_RATE"],
    }


# Step 2: Speaker Diarization
@time_logger
def speaker_diarization(audio):
    """
    Perform speaker diarization on the given audio.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.

    Returns:
        pd.DataFrame: A dataframe containing segments with speaker labels.
    """
    logger.debug(f"Start speaker diarization")
    logger.debug(f"audio waveform shape: {audio['waveform'].shape}")

    waveform = torch.tensor(audio["waveform"]).to(device)
    waveform = torch.unsqueeze(waveform, 0)

    segments = dia_pipeline(
        {
            "waveform": waveform,
            "sample_rate": audio["sample_rate"],
            "channel": 0,
        }
    )

    diarize_df = pd.DataFrame(
        segments.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

    logger.debug(f"diarize_df: {diarize_df}")

    return diarize_df


def sortformer_dia(predicted_segments):
    lists = [x for x in predicted_segments if isinstance(x, (list, tuple))]
    if not lists:
        lists = predicted_segments
    segs = [s for sub in lists for s in sub]

    rows = []
    for idx, seg in enumerate(segs):
        start_s, end_s, sp = seg.split()
        start, end = float(start_s), float(end_s)
        num = int(sp.split('_')[1])
        speaker = f"SPEAKER_{num:02d}"
        label = chr(ord('A') + idx)
        def fmt(sec):
            td = datetime.timedelta(seconds=sec)
            hrs = td.seconds // 3600 + td.days * 24
            mins = (td.seconds // 60) % 60
            secs = td.seconds % 60
            ms = int(td.microseconds / 1000)
            return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"
        segment_str = f"[ {fmt(start)} --> {fmt(end)}]"
        rows.append({
            'segment': segment_str,
            'label': label,
            'speaker': speaker,
            'start': start,
            'end': end
        })

    df = pd.DataFrame(rows, columns=['segment','label','speaker','start','end'])
    df = df.sort_values(by='start').reset_index(drop=True)
    return df

def df_to_list(df: pd.DataFrame) -> list[dict]:
    """
    DataFrame
      - index:
      - start: float
      - end: float
      - speaker: str
    """
    records = []
    for i, row in df.iterrows():
        records.append({
            'index': f"{i:05d}",
            'start': float(row['start']),
            'end': float(row['end']),
            'speaker': row['speaker']
        })
    return records
#
def split_long_segments(segment_list, max_duration=30.0):
    """
    Args:
        segment_list (list): 
        max_duration (float):

    Returns:
        list:
    """
    new_segments = []
    new_index = 0

    for segment in segment_list:
        start_time = segment['start']
        end_time = segment['end']
        speaker = segment['speaker']
        duration = end_time - start_time

        if duration <= max_duration:
            segment['index'] = str(new_index).zfill(5)
            new_segments.append(segment)
            new_index += 1
        else:
            current_start = start_time
            while current_start < end_time:
                chunk_end = min(current_start + max_duration, end_time)
                
                new_segments.append({
                    'index': str(new_index).zfill(5),
                    'start': round(current_start, 3),
                    'end': round(chunk_end, 3),
                    'speaker': speaker
                })
                new_index += 1
                current_start = chunk_end
                
    return new_segments


@time_logger
def cut_by_speaker_label(vad_list):
    """
    Merge and trim VAD segments by speaker labels, enforcing constraints on segment length and merge gaps.

    Args:
        vad_list (list): List of VAD segments with start, end, and speaker labels.

    Returns:
        list: A list of updated VAD segments after merging and trimming.
    """
    MERGE_GAP = 2  # merge gap in seconds, if smaller than this, merge
    MIN_SEGMENT_LENGTH = 3  # min segment length in seconds
    MAX_SEGMENT_LENGTH = 30  # max segment length in seconds

    updated_list = []

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None
        last_speaker = updated_list[-1]["speaker"] if updated_list else None

        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            current_start = vad["start"]
            segment_end = vad["end"]
            logger.warning(
                f"cut_by_speaker_label > segment longer than 30s, force trimming to 30s smaller segments"
            )
            while segment_end - current_start >= MAX_SEGMENT_LENGTH:
                vad["end"] = current_start + MAX_SEGMENT_LENGTH  # update end time
                updated_list.append(vad)
                vad = vad.copy()
                current_start += MAX_SEGMENT_LENGTH
                vad["start"] = current_start  # update start time
                vad["end"] = segment_end
            updated_list.append(vad)
            continue

        if (
            last_speaker is None
            or last_speaker != vad["speaker"]
            or vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
            continue

        if (
            vad["start"] - last_end_time >= MERGE_GAP
            or vad["end"] - last_start_time >= MAX_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
        else:
            updated_list[-1]["end"] = vad["end"]  # merge the time

    logger.debug(
        f"cut_by_speaker_label > merged {len(vad_list) - len(updated_list)} segments"
    )

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]

    logger.debug(
        f"cut_by_speaker_label > removed: {len(updated_list) - len(filter_list)} segments by length"
    )

    return filter_list


@time_logger
def asr(vad_segments, audio):
    """
    Perform Automatic Speech Recognition (ASR) on the VAD segments of the given audio.

    Args:
        vad_segments (list): List of VAD segments with start and end times.
        audio (dict): A dictionary containing the audio waveform and sample rate.

    Returns:
        list: A list of ASR results with transcriptions and language details.
    """
    if len(vad_segments) == 0:
        return []

    temp_audio = audio["waveform"]
    start_time = vad_segments[0]["start"]
    end_time = vad_segments[-1]["end"]
    start_frame = int(start_time * audio["sample_rate"])
    end_frame = int(end_time * audio["sample_rate"])
    temp_audio = temp_audio[start_frame:end_frame]  # remove silent start and end

    # update vad_segments start and end time (this is a little trick for batched asr:)
    for idx, segment in enumerate(vad_segments):
        vad_segments[idx]["start"] -= start_time
        vad_segments[idx]["end"] -= start_time

    # resample to 16k
    temp_audio = librosa.resample(
        temp_audio, orig_sr=audio["sample_rate"], target_sr=16000
    )

    if multilingual_flag:
        logger.debug("Multilingual flag is on")
        valid_vad_segments, valid_vad_segments_language = [], []
        # get valid segments to be transcripted
        for idx, segment in enumerate(vad_segments):
            start_frame = int(segment["start"] * 16000)
            end_frame = int(segment["end"] * 16000)
            segment_audio = temp_audio[start_frame:end_frame]
            language, prob = asr_model.detect_language(segment_audio)
            # 1. if language is in supported list, 2. if prob > 0.8
            if language in supported_languages and prob > 0.8:
                valid_vad_segments.append(vad_segments[idx])
                valid_vad_segments_language.append(language)

        # if no valid segment, return empty
        if len(valid_vad_segments) == 0:
            return []
        all_transcribe_result = []
        logger.debug(f"valid_vad_segments_language: {valid_vad_segments_language}")
        unique_languages = list(set(valid_vad_segments_language))
        logger.debug(f"unique_languages: {unique_languages}")
        # process each language one by one
        for language_token in unique_languages:
            language = language_token
            # filter out segments with different language
            vad_segments = [
                valid_vad_segments[i]
                for i, x in enumerate(valid_vad_segments_language)
                if x == language
            ]
            # bacthed trascription

            transcribe_result_temp = asr_model.transcribe(
                temp_audio,
                vad_segments,
                batch_size=batch_size,
                language=language,
                print_progress=True,
            )
            result = transcribe_result_temp["segments"]
            # restore the segment annotation
            for idx, segment in enumerate(result):
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result_temp["language"]
            all_transcribe_result.extend(result)
        # sort by start time
        all_transcribe_result = sorted(all_transcribe_result, key=lambda x: x["start"])
        return all_transcribe_result
    else:
        logger.debug("Multilingual flag is off")
        language, prob = asr_model.detect_language(temp_audio)
        if language in supported_languages and prob > 0.8:
            transcribe_result = asr_model.transcribe(
                temp_audio,
                vad_segments,
                batch_size=batch_size,
                language=language,
                print_progress=True,
            )
            result = transcribe_result["segments"]
            for idx, segment in enumerate(result):
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result["language"]
            return result
        else:
            return []

def ko_transliterate_english(text: str) -> str:
    """
    입력 문자열에서 영어 구간만 찾아 한글 발음으로 변환합니다.
    """
    def _repl(m: re.Match) -> str:
        segment = m.group(0)
        return G2P(segment)
    return ENG_PATTERN.sub(_repl, text)


def ko_process_json(input_list: str) -> None:
    for entry in input_list:
        text = entry.get("text", "")
        # 영어 포함 시 변환
        if re.search(r"[A-Za-z]", text):
            entry["text"] = ko_transliterate_english(text)

def main_process(args, audio_path, save_path=None, audio_name=None):
    """
    Process the audio file, including standardization, source separation, speaker segmentation, VAD, ASR, export to MP3, and MOS prediction.

    Args:
        audio_path (str): Audio file path.
        save_path (str, optional): Save path, defaults to None, which means saving in the "_processed" folder in the audio file's directory.
        audio_name (str, optional): Audio file name, defaults to None, which means using the file name from the audio file path.

    Returns:
        tuple: Contains the save path and the MOS list.
    """
    if not audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac")):
        logger.warning(f"Unsupported file type: {audio_path}")

    # for a single audio from path Ïaaa/bbb/ccc.wav ---> save to aaa/bbb_processed/ccc/ccc_0.wav
    save_root = args.save_root # "/mnt/sdd/taehong/processed"
    logger.debug(f"Processing audio: {audio_name}, from {audio_path}")
    
    logger.info(
        "Step 0: Preprocess all audio files --> 24k sample rate + wave format + loudnorm + bit depth 16"
    )
    audio = standardization(audio_path)

    logger.info("Step 1: Speaker Diarization")
    dia_start = time.time()

    # SortFormer
    predicted_segments, predicted_probs = diar_model.diarize(audio=audio_path, batch_size=1, include_tensor_outputs=True)
    
    speakerdia = sortformer_dia(predicted_segments)
    ori_list = df_to_list(speakerdia)
    dia_end = time.time()
    print(f"diarization time : {dia_end-dia_start}")

    logger.info("Step 2: Fine-grained Segmentation by VAD")
    # vad_list = vad.vad(speakerdia, audio)
    # segment_list = cut_by_speaker_label(vad_list)  # post process after vad
    # print(f"segment_list : {segment_list}")

    segment_list = ori_list
    segment_list = split_long_segments(segment_list)
    print(f"segment_list : {segment_list}")

    logger.info("Step 4: ASR")
    asr_result = asr(segment_list, audio)
    print(f"asr_result : {asr_result}")
    # export_to_mp3(audio, asr_result, save_path, audio_name)

    final_path = os.path.join(args.save_root, audio_name + ".json")

    if args.korean:
        asr_result = ko_process_json(asr_result)
    with open(final_path, "w") as f:
        json.dump(asr_result, f, ensure_ascii=False)

    logger.info(f"All done, Saved to: {final_path}")
    return final_path, asr_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default="",
        help="input folder path, this will override config if set",
    )
    parser.add_argument(
        "--temp_wav_folder_path",
        type=str,
        default="",
        help="input folder path, this will override config if set",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="/home/jovyan/taehong/text_jsons",
        help="input folder path, this will override config if set",
    )
    parser.add_argument(
        "--config_path", type=str, default="config.json", help="config path"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
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
        "--initprompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Turning on initial prompt on whisper model",
    )
    parser.add_argument(
        "--korean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If audio is Korean, turn in True."
    )
    args = parser.parse_args()

    batch_size = args.batch_size
    cfg = load_cfg(args.config_path)

    logger = Logger.get_logger()

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
    dia_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=cfg["huggingface_token"],
    )
    dia_pipeline.to(device)

    if args.korean:
        G2P = G2p()
        ENG_PATTERN = re.compile(r"[A-Za-z][A-Za-z']*(?: [A-Za-z][A-Za-z']*)*")

    # ASR
    logger.debug(" * Loading ASR Model")
    
    if args.initprompt == True:
        asr_model = whisper_asr.load_asr_model(
            "large-v3",
            device_name,
            compute_type=args.compute_type,
            threads=args.threads,
            
        # whisper_asr.py 의 default_asr_options 수정으로 asr 모델 수정 가능.
            
            asr_options={
                #"log_prob_threshold": -1.0,
                #"no_speech_threshold": 0.6,
                # 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음
                
                # 원래 코드의 initial prompt
                # "initial_prompt": "ha. heh. Mm, hmm. Mm hm. uh. Uh huh. Mm huh. Uh. hum Uh. Ah. Uh hu. Like. you know. Yeah. I mean. right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. Hoo. Hu. Hoo, hoo. Heah. Ha. Yu. Nah. Uh-huh. No way. Uh-oh. Jeez. Whoa. Dang. Gosh. Duh. Whoops. Phew. Woo. Ugh. Er. Geez. Oh wow. Oh man. Uh yeah. Uh huh. For real?",
                
                #"initial_prompt": "ha. heh. Mm, hmm. uh. "
                #"initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. Hoo. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음.",
                "initial_prompt": "Um. Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. Mm. So. Oh. Hoo hoo.",
                
            },
        )
    else:
        asr_model = whisper_asr.load_asr_model(
            "large-v3",
            device_name,
            compute_type=args.compute_type,
            threads=args.threads,
            )

    # # VAD
    # logger.debug(" * Loading VAD Model")
    # vad = silero_vad.SileroVAD(device=device)

    # load model from Hugging Face model card directly (You need a Hugging Face token)
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
    diar_model.eval()
    
    logger.debug("All models loaded")

    supported_languages = cfg["language"]["supported"]
    multilingual_flag = cfg["language"]["multilingual"]
    logger.debug(f"supported languages multilingual {supported_languages}")
    logger.debug(f"using multilingual asr {multilingual_flag}")

    # input_folder_path = cfg["entrypoint"]["input_folder_path"]

    # if not os.path.exists(input_folder_path):
    #     raise FileNotFoundError(f"input_folder_path: {input_folder_path} not found")

    # audio_paths = get_audio_files(input_folder_path)  # Get all audio files
    # logger.debug(f"Scanning {len(audio_paths)} audio files in {input_folder_path}")

    # 1) input_folder_path 아래의 모든 .parquet 파일 목록
    parquet_paths = glob.glob(os.path.join(args.input_folder_path, "*.parquet"))
    logger.debug(f"Found {len(parquet_paths)} parquet files in {args.input_folder_path}: {parquet_paths}")

    for pq_path in tqdm(parquet_paths, desc="Processing Parquet Files"):
        temp_dir = args.temp_wav_folder_path
        os.makedirs(temp_dir, exist_ok=True)

        pq_filename = os.path.splitext(os.path.basename(pq_path))[0]
        logger.debug(f"Reading parquet: {pq_path}")
        df = pd.read_parquet(pq_path)
        logger.debug(f"Loaded DataFrame with {len(df)} rows from {os.path.basename(pq_path)}")

        for idx, row in df.iterrows():
            audio_data = row["audio"]  # {"path": ..., "bytes": ...}

            channel = str(row.get("channel", "")).replace(" ", "_")
            title = str(row.get("title", "")).replace(" ", "_")
            segment = str(row.get("segment", ""))

            base_name = f"{pq_filename}_{channel}_{title}_{segment}"

            wav_filename = f"{base_name}.wav"
            wav_path = os.path.join(temp_dir, wav_filename)

            try:
                # 1) 바이트로부터 AudioSegment 로딩 (ffmpeg 필요)
                ext = os.path.splitext(audio_data["path"])[1].lower().lstrip('.')
                audio = AudioSegment.from_file(io.BytesIO(audio_data["bytes"]), format=ext)

                # 2) 모노 채널로 변환
                mono_audio = audio.set_channels(1)

                # 3) WAV로 내보내기
                mono_audio.export(wav_path, format="wav")
                logger.debug(f"[{idx}] Saved mono wav to: {wav_path}")

            except Exception as e:
                logger.warning(f"[{idx}] Failed to write mono {wav_filename}: {e}")
                continue

            try:
                main_process(args, wav_path, save_path=None, audio_name=base_name)
            except Exception as e:
                logger.error(f"[{idx}] main_process failed for {wav_path}: {e}")
        
        try:
            logger.debug(f"Clearing temp directory: {temp_dir}")
            # temp_dir 내부의 모든 파일/폴더 삭제
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            logger.debug(f"Temp directory cleared.")
        except Exception as e:
            logger.warning(f"Failed to clear temp directory {temp_dir}: {e}")