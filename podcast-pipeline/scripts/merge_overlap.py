import os
import glob
import json

def merge_overlaps(segments, speaker_to_tag):
    """
    segments
    speaker_to_tag
    """
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x['start'])
    merged = []
    cluster = [segments[0]]
    cluster_end = segments[0]['end']

    def flush_cluster(cluster):
        subintervals = [[c['start'], c['end']] for c in cluster]
        overlap_flag = len(cluster) > 1

        if len(cluster) == 1:
            c = cluster[0]
            tag = speaker_to_tag[c['speaker']]
            txt = c.get('text', '').strip()
            return [{
                'start':        c['start'],
                'end':          c['end'],
                'duration':     c['end'] - c['start'],
                'speaker':      c['speaker'],
                'text':         f"[s{tag}] {txt}" if txt else f"[s{tag}]",
                'overlap':      overlap_flag,
                'subintervals': subintervals
            }]

        starts = [c['start'] for c in cluster]
        ends   = [c['end']   for c in cluster]
        min_start = min(starts)
        max_end   = max(ends)
        combined_speaker = '_'.join(c['speaker'] for c in cluster)

        parts = []
        for c in cluster:
            tag = speaker_to_tag[c['speaker']]
            txt = c.get('text', '').strip()
            parts.append(f"[s{tag}] {txt}" if txt else f"[s{tag}]")
        combined_text = ' '.join(parts)

        subintervals = [[c['start'], c['end']] for c in cluster]

        return [{
            'start':        min_start,
            'end':          max_end,
            'duration':     max_end - min_start,
            'speaker':      combined_speaker,
            'text':         combined_text,
            'overlap':      True,
            'subintervals': subintervals
        }]

    for seg in segments[1:]:
        if seg['start'] < cluster_end:
            cluster.append(seg)
            cluster_end = max(cluster_end, seg['end'])
        else:
            merged.extend(flush_cluster(cluster))
            cluster = [seg]
            cluster_end = seg['end']

    merged.extend(flush_cluster(cluster))
    return merged

def process_file(in_path, out_dir=None):
    """
    in_path:
    out_dir:
    """
    with open(in_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    if not segments:
        print(f"Skipping {os.path.basename(in_path)}: no segments")
        return

    unique_speakers = sorted({seg['speaker'] for seg in segments})
    speaker_to_tag  = {spk: idx for idx, spk in enumerate(unique_speakers)}

    merged = merge_overlaps(segments, speaker_to_tag)

    if out_dir:
        base, ext = os.path.splitext(os.path.basename(in_path))
        target = os.path.join(out_dir, f"{base}_merged{ext}")
    else:
        target = in_path

    with open(target, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)

    if out_dir and os.path.abspath(target) != os.path.abspath(in_path):
        os.remove(in_path)

    print(f"Processed {os.path.basename(in_path)} → {os.path.basename(target)}: "
          f"{len(segments)}→{len(merged)} segments, tags={speaker_to_tag}")



if __name__ == "__main__":
    # # Single file processing
    # process_file(
    #     '/home/taehong/taehong/audio_pipeline/raw_jsons/test.json',
    #     '/home/taehong/taehong/audio_pipeline/processed_jsons'
    # )
    
    # Batch processing example
    in_folder = '/home/jovyan/taehong/text_jsons' # TODO make argument
    out_folder = '/home/jovyan/taehong/text_jsons_merged' # TODO make argument
    for path in glob.glob(os.path.join(in_folder, '*.json')):
        process_file(path, out_folder)
