#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rttm_dir", type=str, required=True)
    ap.add_argument("--wav_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--strict", action="store_true",
                    help="켜면 wav/rttm가 1:1로 완전히 매칭되지 않으면 에러")
    args = ap.parse_args()

    rttm_dir = Path(args.rttm_dir)
    wav_dir = Path(args.wav_dir)
    out_path = Path(args.out)

    rttms = {p.stem: p for p in sorted(rttm_dir.glob("*.rttm"))}
    wavs = {p.stem: p for p in sorted(wav_dir.glob("*.wav"))}

    common = sorted(set(rttms.keys()) & set(wavs.keys()))
    only_rttm = sorted(set(rttms.keys()) - set(wavs.keys()))
    only_wav = sorted(set(wavs.keys()) - set(rttms.keys()))

    print(f"[INFO] rttm: {len(rttms)}, wav: {len(wavs)}, matched: {len(common)}")
    if only_rttm:
        print(f"[WARN] RTTM만 존재 (wav 없음): {len(only_rttm)} 예: {only_rttm[:10]}")
    if only_wav:
        print(f"[WARN] WAV만 존재 (rttm 없음): {len(only_wav)} 예: {only_wav[:10]}")

    if args.strict and (only_rttm or only_wav):
        raise RuntimeError("strict 모드: wav/rttm 매칭이 완전하지 않습니다.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for stem in common:
            item = {
                "uri": stem,
                "audio_filepath": str(wavs[stem].resolve()),
                "rttm_filepath": str(rttms[stem].resolve()),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] wrote: {out_path} ({len(common)} lines)")

if __name__ == "__main__":
    main()
