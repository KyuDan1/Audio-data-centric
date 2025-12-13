#!/usr/bin/env python3
import os
import sys

# ---------------------------
# Runtime linker guard (conda)
# ---------------------------
def _ensure_conda_lib_first_in_ld_library_path() -> None:
    """
    Some job runners inject an old system libstdc++ ahead of conda libs, which can
    break imports with errors like:
      GLIBCXX_3.4.30 not found (required by .../libicuuc.so.*)

    If running inside conda, prepend `$CONDA_PREFIX/lib` and re-exec once so the
    dynamic linker picks up the correct libstdc++.
    """
    if os.environ.get("KYUDAN_LD_REEXEC") == "1":
        return
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    conda_lib = os.path.join(conda_prefix, "lib")
    if not os.path.isdir(conda_lib):
        return
    if not os.path.exists(os.path.join(conda_lib, "libstdc++.so.6")):
        return

    ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in ld.split(":") if p]
    if parts and parts[0] == conda_lib:
        return
    if conda_lib in parts:
        parts = [p for p in parts if p != conda_lib]
    os.environ["LD_LIBRARY_PATH"] = ":".join([conda_lib, *parts])
    os.environ["KYUDAN_LD_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


if __name__ == "__main__":
    _ensure_conda_lib_first_in_ld_library_path()

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.optimize import linear_sum_assignment

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate


# ---------------------------
# RTTM I/O (minimal, robust)
# ---------------------------
def read_rttm(rttm_path: str) -> Tuple[str, Annotation]:
    """
    Minimal RTTM parser.
    Expected line format (common):
    SPEAKER <uri> 1 <start> <dur> <NA> <NA> <spk> <NA> <NA>
    """
    ann = Annotation()
    uri = None
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                continue
            uri = parts[1] if uri is None else uri
            start = float(parts[3])
            dur = float(parts[4])
            spk = parts[7]
            seg = Segment(start, start + max(dur, 0.0))
            ann[seg] = spk
    if uri is None:
        uri = Path(rttm_path).stem
    return uri, ann


def write_rttm(uri: str, ann: Annotation, out_path: str) -> None:
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for seg, _, spk in ann.itertracks(yield_label=True):
            start = float(seg.start)
            dur = float(seg.end - seg.start)
            f.write(f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n")


def audio_duration_sec(audio_path: str) -> float:
    info = torchaudio.info(audio_path)
    return float(info.num_frames) / float(info.sample_rate)


# ---------------------------
# Focus-region (UEM) builders
# ---------------------------
def build_uem_short_segments(reference: Annotation, max_dur_s: float) -> Timeline:
    uem = Timeline()
    for seg, _, _ in reference.itertracks(yield_label=True):
        if (seg.end - seg.start) <= max_dur_s:
            uem.add(seg)
    return uem.support()


def build_turn_change_uem(reference: Annotation, gap_max_s: float, window_s: float) -> Timeline:
    """
    Approximate turn-change points:
    sort by time, take consecutive segments with different speakers,
    and if next.start - prev.end <= gap_max_s, mark change at boundary time.
    """
    turns = [(seg.start, seg.end, spk) for seg, _, spk in reference.itertracks(yield_label=True)]
    turns.sort(key=lambda x: (x[0], x[1]))

    uem = Timeline()
    for i in range(len(turns) - 1):
        s0, e0, spk0 = turns[i]
        s1, e1, spk1 = turns[i + 1]
        if spk0 == spk1:
            continue
        gap = s1 - e0
        if gap <= gap_max_s:
            t = max(min(e0, s1), 0.0)
            uem.add(Segment(max(t - window_s, 0.0), t + window_s))
    return uem.support()


# ---------------------------
# Model wrappers
# ---------------------------
@dataclass
class InferenceResult:
    uri: str
    hypothesis: Annotation
    infer_time_sec: float
    audio_dur_sec: float


class DiarizerBase:
    def diarize_one(self, uri: str, audio_path: str) -> InferenceResult:
        raise NotImplementedError


class Pyannote31Diarizer(DiarizerBase):
    def __init__(self, hf_token: str, device: str):
        from pyannote.audio import Pipeline
        # If hf_token is empty or None, use True to use cached token
        auth_token = True if not hf_token else hf_token
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token,
        )
        self.device = torch.device(device)
        self.pipeline.to(self.device)

    def diarize_one(self, uri: str, audio_path: str) -> InferenceResult:
        dur = audio_duration_sec(audio_path)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        hyp: Annotation = self.pipeline(audio_path)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return InferenceResult(uri=uri, hypothesis=hyp, infer_time_sec=(t1 - t0), audio_dur_sec=dur)


class SortformerDiarizer(DiarizerBase):
    """
    Supports:
      - nvidia/diar_sortformer_4spk-v1 (offline)
      - nvidia/diar_streaming_sortformer_4spk-v2 (streaming)
    """
    def __init__(self, model_name: str, device: str, streaming_cfg: Optional[Dict[str, int]] = None):
        from nemo.collections.asr.models import SortformerEncLabelModel
        self.model = SortformerEncLabelModel.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device(device)
        # NeMo model typically moves internally; keep torch device for timing sync only.

        # Optional streaming parameter setup (only meaningful for streaming v2)
        if streaming_cfg is not None:
            m = self.model.sortformer_modules
            m.chunk_len = int(streaming_cfg["CHUNK_SIZE"])
            m.chunk_right_context = int(streaming_cfg["RIGHT_CONTEXT"])
            m.fifo_len = int(streaming_cfg["FIFO_SIZE"])
            # model card uses spkcache_update_period :contentReference[oaicite:8]{index=8}
            m.spkcache_update_period = int(streaming_cfg["UPDATE_PERIOD"])
            m.spkcache_len = int(streaming_cfg["SPEAKER_CACHE_SIZE"])
            if hasattr(m, "_check_streaming_parameters"):
                m._check_streaming_parameters()

    @staticmethod
    def _to_annotation(uri: str, predicted_segments) -> Annotation:
        """
        NeMo Sortformer `diarize()` commonly returns one of:
          - [[\"<start> <end> speaker_<k>\", ...]]  (list of list of strings)
          - [(start, end, spk_idx), ...]           (list of tuples)
          - [[(start, end, spk_idx), ...]]         (batched)
        """
        segs = predicted_segments or []
        if isinstance(segs, list) and len(segs) == 1 and isinstance(segs[0], list):
            segs = segs[0]

        ann = Annotation(uri=uri)
        for item in segs:
            if isinstance(item, str):
                parts = item.strip().split()
                if len(parts) < 3:
                    continue
                start, end = float(parts[0]), float(parts[1])
                spk = parts[2]
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                start, end = float(item[0]), float(item[1])
                spk_raw = item[2]
                if isinstance(spk_raw, (int, np.integer)):
                    spk = f"SPEAKER_{int(spk_raw):02d}"
                else:
                    spk = str(spk_raw)
            else:
                continue
            if end <= start:
                continue
            ann[Segment(start, end)] = spk
        return ann

    def diarize_one(self, uri: str, audio_path: str) -> InferenceResult:
        dur = audio_duration_sec(audio_path)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        predicted_segments = self.model.diarize(audio=audio_path, batch_size=1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        hyp = self._to_annotation(uri, predicted_segments)
        return InferenceResult(uri=uri, hypothesis=hyp, infer_time_sec=(t1 - t0), audio_dur_sec=dur)


# ---------------------------
# Speaker similarity (optional)
# ---------------------------
def overlap_matrix(ref: Annotation, hyp: Annotation) -> Tuple[List[str], List[str], np.ndarray]:
    ref_spks = sorted({spk for _, _, spk in ref.itertracks(yield_label=True)})
    hyp_spks = sorted({spk for _, _, spk in hyp.itertracks(yield_label=True)})
    M = np.zeros((len(ref_spks), len(hyp_spks)), dtype=np.float64)

    # accumulate overlap durations
    for i, rs in enumerate(ref_spks):
        r_tl = ref.label_timeline(rs)
        for j, hs in enumerate(hyp_spks):
            h_tl = hyp.label_timeline(hs)
            inter = r_tl.intersection(h_tl)
            M[i, j] = sum((seg.end - seg.start) for seg in inter)
    return ref_spks, hyp_spks, M


def hungarian_match(ref: Annotation, hyp: Annotation) -> Dict[str, str]:
    ref_spks, hyp_spks, M = overlap_matrix(ref, hyp)
    if M.size == 0:
        return {}
    # maximize overlap => minimize negative
    cost = -M
    r_ind, h_ind = linear_sum_assignment(cost)
    mapping = {}
    for i, j in zip(r_ind, h_ind):
        if M[i, j] > 0:
            mapping[ref_spks[i]] = hyp_spks[j]
    return mapping


def crop_audio(wav: torch.Tensor, sr: int, start: float, end: float) -> torch.Tensor:
    s = max(int(round(start * sr)), 0)
    e = max(int(round(end * sr)), s + 1)
    e = min(e, wav.shape[-1])
    seg = wav[..., s:e]
    return seg


def compute_speaker_similarity_ecapa(
    audio_path: str,
    ref: Annotation,
    hyp: Annotation,
    device: str,
    min_seg_s: float = 0.5,
) -> Dict[str, float]:
    """
    Embedding separation metric based on SpeechBrain ECAPA.
    Output:
      matched_mean_cos, nonmatched_mean_cos, separation
    """
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as e:
        return {"spk_sim_error": 1.0, "spk_sim_msg": float("nan")}

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.to(torch.float32)

    def speaker_centroids(ann: Annotation) -> Dict[str, np.ndarray]:
        out = {}
        for spk in sorted({s for _, _, s in ann.itertracks(yield_label=True)}):
            embs = []
            for seg, _, s in ann.itertracks(yield_label=True):
                if s != spk:
                    continue
                if (seg.end - seg.start) < min_seg_s:
                    continue
                chunk = crop_audio(wav, sr, seg.start, seg.end)
                if chunk.numel() < 2:
                    continue
                with torch.no_grad():
                    emb = classifier.encode_batch(chunk.unsqueeze(0)).squeeze(0).squeeze(0)
                embs.append(emb.detach().cpu().numpy())
            if len(embs) >= 1:
                out[spk] = np.mean(np.stack(embs, axis=0), axis=0)
        return out

    ref_c = speaker_centroids(ref)
    hyp_c = speaker_centroids(hyp)
    if len(ref_c) == 0 or len(hyp_c) == 0:
        return {"matched_mean_cos": float("nan"), "nonmatched_mean_cos": float("nan"), "separation": float("nan")}

    mapping = hungarian_match(ref, hyp)

    def cos(a, b):
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        return float(np.dot(a, b) / (na * nb))

    matched = []
    nonmatched = []
    ref_keys = list(ref_c.keys())
    hyp_keys = list(hyp_c.keys())

    # matched pairs
    for r, h in mapping.items():
        if r in ref_c and h in hyp_c:
            matched.append(cos(ref_c[r], hyp_c[h]))

    # nonmatched pairs (all other combos)
    mapped_h = set(mapping.values())
    for r in ref_keys:
        for h in hyp_keys:
            if (r in mapping and mapping[r] == h) or (h in mapped_h and r in mapping and mapping[r] == h):
                continue
            # keep only if both exist
            if r in ref_c and h in hyp_c:
                nonmatched.append(cos(ref_c[r], hyp_c[h]))

    if len(matched) == 0 or len(nonmatched) == 0:
        return {"matched_mean_cos": float("nan"), "nonmatched_mean_cos": float("nan"), "separation": float("nan")}

    return {
        "matched_mean_cos": float(np.mean(matched)),
        "nonmatched_mean_cos": float(np.mean(nonmatched)),
        "separation": float(np.mean(matched) - np.mean(nonmatched)),
    }


# ---------------------------
# Main eval
# ---------------------------
def load_manifest(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def num_speakers_in_rttm(rttm_path: str) -> int:
    _, ann = read_rttm(rttm_path)
    spks = {spk for _, _, spk in ann.itertracks(yield_label=True)}
    return len(spks)


def make_diarizer(name: str, hf_token: str, device: str):
    if name == "pyannote31":
        return Pyannote31Diarizer(hf_token=hf_token, device=device)
    if name == "sortformer_v1":
        return SortformerDiarizer(model_name="nvidia/diar_sortformer_4spk-v1", device=device)
    if name == "sortformer_streaming_v2":
        # default: "high latency" from model card table :contentReference[oaicite:10]{index=10}
        cfg = dict(CHUNK_SIZE=124, RIGHT_CONTEXT=1, FIFO_SIZE=124, UPDATE_PERIOD=124, SPEAKER_CACHE_SIZE=188)
        return SortformerDiarizer(model_name="nvidia/diar_streaming_sortformer_4spk-v2", device=device, streaming_cfg=cfg)
    raise ValueError(f"Unknown model: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--models", nargs="+", default=["pyannote31", "sortformer_v1"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true",
                    help="If set, reuse existing non-empty `pred_rttm/<model>/<uri>.rttm` instead of re-running the model.")

    ap.add_argument("--collar", type=float, default=0.25)
    ap.add_argument("--skip_overlap", action="store_true", help="If set, skip overlap in DER/JER")

    ap.add_argument("--short_max_s", type=float, default=1.0)
    ap.add_argument("--turn_gap_max_s", type=float, default=0.5)
    ap.add_argument("--turn_window_s", type=float, default=1.0)

    ap.add_argument("--max_speakers_for_common_subset", type=int, default=4,
                    help="If >0, keep only sessions with <= this many speakers for all models (fair for Sortformer). 0 disables.")
    ap.add_argument("--compute_speaker_similarity", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_timing = {}
    if args.resume:
        prev_path = out_dir / "summary_per_file.csv"
        if prev_path.exists():
            try:
                prev_df = pd.read_csv(prev_path)
                for r in prev_df.itertuples(index=False):
                    prev_timing[(str(r.uri), str(r.model))] = (float(r.infer_sec), float(r.audio_sec))
            except Exception:
                prev_timing = {}

    items = load_manifest(args.manifest)
    if args.max_speakers_for_common_subset > 0:
        keep = []
        for it in items:
            nspk = num_speakers_in_rttm(it["rttm_filepath"])
            if nspk <= args.max_speakers_for_common_subset:
                keep.append(it)
        items = keep

    # metrics
    der = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
    jer = JaccardErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)

    diarizers = {}

    rows = []
    for it in items:
        uri = it.get("uri", Path(it["audio_filepath"]).stem)
        audio_path = it["audio_filepath"]
        rttm_path = it["rttm_filepath"]

        _, ref = read_rttm(rttm_path)
        # build focus regions
        uem_short = build_uem_short_segments(ref, args.short_max_s)
        uem_turn = build_turn_change_uem(ref, args.turn_gap_max_s, args.turn_window_s)

        for model_name in args.models:
            pred_rttm_path = out_dir / "pred_rttm" / model_name / f"{uri}.rttm"
            hyp: Annotation
            infer_time_sec: float
            audio_dur_sec: float

            if args.resume and pred_rttm_path.exists() and pred_rttm_path.stat().st_size > 0:
                _, hyp = read_rttm(str(pred_rttm_path))
                infer_time_sec, audio_dur_sec = prev_timing.get((uri, model_name), (float("nan"), float("nan")))
                if not math.isfinite(audio_dur_sec):
                    audio_dur_sec = audio_duration_sec(audio_path)
            else:
                if model_name not in diarizers:
                    diarizers[model_name] = make_diarizer(model_name, hf_token=args.hf_token, device=args.device)
                diarizer = diarizers[model_name]
                res = diarizer.diarize_one(uri, audio_path)
                hyp = res.hypothesis
                infer_time_sec = res.infer_time_sec
                audio_dur_sec = res.audio_dur_sec

            # save hyp rttm
            write_rttm(uri, hyp, str(pred_rttm_path))
            if len({spk for _, _, spk in hyp.itertracks(yield_label=True)}) == 0:
                print(f"[warn] Empty hypothesis for {model_name} uri={uri} -> {pred_rttm_path}")

            # compute metrics
            der_full = der(ref, hyp)
            jer_full = jer(ref, hyp)
            der_short = der(ref, hyp, uem=uem_short) if len(uem_short) > 0 else float("nan")
            jer_short = jer(ref, hyp, uem=uem_short) if len(uem_short) > 0 else float("nan")
            der_turn = der(ref, hyp, uem=uem_turn) if len(uem_turn) > 0 else float("nan")
            jer_turn = jer(ref, hyp, uem=uem_turn) if len(uem_turn) > 0 else float("nan")

            rtf = infer_time_sec / max(audio_dur_sec, 1e-9) if math.isfinite(infer_time_sec) else float("nan")

            row = dict(
                uri=uri,
                model=model_name,
                audio_sec=audio_dur_sec,
                infer_sec=infer_time_sec,
                rtf=rtf,
                der=der_full,
                jer=jer_full,
                der_short=der_short,
                jer_short=jer_short,
                der_turn=der_turn,
                jer_turn=jer_turn,
                n_ref_speakers=len({spk for _, _, spk in ref.itertracks(yield_label=True)}),
                n_hyp_speakers=len({spk for _, _, spk in hyp.itertracks(yield_label=True)}),
            )

            if args.compute_speaker_similarity:
                sim = compute_speaker_similarity_ecapa(audio_path, ref, hyp, device=args.device)
                row.update(sim)

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary_per_file.csv", index=False)

    # aggregate
    agg = df.groupby("model").agg(
        der_mean=("der", "mean"),
        jer_mean=("jer", "mean"),
        der_short_mean=("der_short", "mean"),
        der_turn_mean=("der_turn", "mean"),
        rtf_mean=("rtf", "mean"),
        n=("uri", "count"),
    ).reset_index()
    agg.to_csv(out_dir / "summary_agg.csv", index=False)

    print("Saved:")
    print(" -", out_dir / "summary_per_file.csv")
    print(" -", out_dir / "summary_agg.csv")
    print(" -", out_dir / "pred_rttm/<model>/*.rttm")


if __name__ == "__main__":
    main()
