"""
Speaker diarization utilities for podcast pipeline.
Includes pyannote-based diarization, segment processing, and Sortformer integration.
"""

import datetime
import pandas as pd
import torch
from utils.logger import time_logger

# Logger will be initialized from main module
logger = None

def set_logger(log_instance):
    """Set logger instance from main module."""
    global logger
    logger = log_instance


@time_logger
def speaker_diarization(audio, dia_pipeline, device):
    """
    Perform speaker diarization on the given audio.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.
        dia_pipeline: Pyannote diarization pipeline.
        device: torch device (cuda/cpu).

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
        },
        max_speakers=4
    )

    diarize_df = pd.DataFrame(
        segments.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

    logger.debug(f"diarize_df: {diarize_df}")

    return diarize_df


@time_logger
def cut_by_speaker_label(vad_list, args):
    """
    Merge and trim VAD segments by speaker labels, enforcing constraints on segment length and merge gaps.

    Args:
        vad_list (list): List of VAD segments with start, end, and speaker labels.
        args: Arguments object containing merge_gap parameter.

    Returns:
        list: A list of updated VAD segments after merging and trimming.
    """
    MERGE_GAP = args.merge_gap  # merge gap in seconds, if smaller than this, merge
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
def detect_overlapping_segments(segment_list, overlap_threshold=0.2):
    """
    Detect segments that overlap for more than overlap_threshold seconds.

    Args:
        segment_list (list): List of segments with 'start', 'end', and 'speaker' keys
        overlap_threshold (float): Minimum overlap duration in seconds to be considered

    Returns:
        list: List of overlapping segment pairs with overlap info
            [{'seg1': segment1, 'seg2': segment2, 'overlap_start': float, 'overlap_end': float, 'overlap_duration': float}]
    """
    overlapping_pairs = []

    # Sort segments by start time
    sorted_segments = sorted(segment_list, key=lambda x: x['start'])

    for i in range(len(sorted_segments)):
        for j in range(i + 1, len(sorted_segments)):
            seg1 = sorted_segments[i]
            seg2 = sorted_segments[j]

            # If seg2 starts after seg1 ends, no more overlaps possible for seg1
            if seg2['start'] >= seg1['end']:
                break

            # Calculate overlap
            overlap_start = max(seg1['start'], seg2['start'])
            overlap_end = min(seg1['end'], seg2['end'])
            overlap_duration = overlap_end - overlap_start

            # Check if overlap exceeds threshold
            if overlap_duration >= overlap_threshold:
                overlapping_pairs.append({
                    'seg1': seg1,
                    'seg2': seg2,
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end,
                    'overlap_duration': overlap_duration
                })
                logger.info(f"Overlap detected: {overlap_duration:.2f}s between "
                           f"[{seg1['start']:.2f}-{seg1['end']:.2f}] and "
                           f"[{seg2['start']:.2f}-{seg2['end']:.2f}]")

    return overlapping_pairs


def sortformer_dia(predicted_segments):
    """
    Convert Sortformer output to pandas DataFrame.

    Args:
        predicted_segments: Sortformer model output segments.

    Returns:
        pd.DataFrame: DataFrame with columns ['segment', 'label', 'speaker', 'start', 'end']
    """
    lists = [x for x in predicted_segments if isinstance(x, (list, tuple))]
    if not lists:
        lists = predicted_segments
    segs = [s for sub in lists for s in sub]

    rows = []
    for idx, seg in enumerate(segs):
        start_s, end_s, sp = seg.split()
        start, end = float(start_s), float(end_s)
        # SPEAKER 형식 변환
        num = int(sp.split('_')[1])
        speaker = f"SPEAKER_{num:02d}"
        # 레이블(A, B, C, ...)
        label = chr(ord('A') + idx)
        # 시간 포맷 함수
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
    DataFrame의 각 행을 아래 형식의 dict로 변환한 리스트를 반환합니다.
      - index: 5자리 0패딩 문자열
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


def split_long_segments(segment_list, max_duration=30.0):
    """
    세그먼트 리스트에서 max_duration보다 긴 세그먼트를
    시간 기준으로 분할합니다. (VAD 사용 안 함)

    Args:
        segment_list (list): 분할할 세그먼트 딕셔너리의 리스트.
        max_duration (float): 세그먼트의 최대 허용 길이 (초).

    Returns:
        list: 분할이 완료된 새로운 세그먼트 리스트.
    """
    new_segments = []
    new_index = 0

    for segment in segment_list:
        start_time = segment['start']
        end_time = segment['end']
        speaker = segment['speaker']
        duration = end_time - start_time

        # 세그먼트 길이가 최대 길이보다 짧거나 같으면 그대로 추가
        if duration <= max_duration:
            segment['index'] = str(new_index).zfill(5)
            new_segments.append(segment)
            new_index += 1
        # 세그먼트 길이가 최대 길이보다 길면 분할
        else:
            current_start = start_time
            # 현재 시작 시간이 원본 세그먼트의 종료 시간보다 작을 때까지 반복
            while current_start < end_time:
                # 다음 분할 지점 계산 (최대 길이를 더하거나, 원본 종료 시간을 넘지 않도록)
                chunk_end = min(current_start + max_duration, end_time)

                new_segments.append({
                    'index': str(new_index).zfill(5),
                    'start': round(current_start, 3),
                    'end': round(chunk_end, 3),
                    'speaker': speaker
                })
                new_index += 1
                # 다음 청크의 시작 시간을 현재 청크의 종료 시간으로 업데이트
                current_start = chunk_end

    return new_segments
