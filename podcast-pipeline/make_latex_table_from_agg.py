#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def load_agg(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ["model", "der_mean", "jer_mean", "der_short_mean", "der_turn_mean", "rtf_mean"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df

def to_percent(x):  # 0~1 -> %
    return x * 100.0

def escape_latex(s: str) -> str:
    # 최소한만 (모델명에 _ 들어가는 경우)
    return s.replace("_", r"\_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg_short05", type=str, required=True)
    ap.add_argument("--agg_short10", type=str, required=True)
    ap.add_argument("--out_tex", type=str, required=True)

    ap.add_argument("--caption", type=str, default="Diarization model ablation.")
    ap.add_argument("--label", type=str, default="tab:diar_ablation")
    ap.add_argument("--resizebox", type=str, default="0.9\\textwidth",
                    help=r"e.g., 0.45\textwidth, 0.9\textwidth")
    ap.add_argument("--sort_key", type=str, default="der_mean",
                    help="정렬 기준(작을수록 위). 예: der_mean, rtf_mean")
    args = ap.parse_args()

    df05 = load_agg(Path(args.agg_short05)).rename(columns={"der_short_mean": "der_short_05"})
    df10 = load_agg(Path(args.agg_short10)).rename(columns={"der_short_mean": "der_short_10"})

    df = pd.merge(df05, df10[["model", "der_short_10"]], on="model", how="inner")

    # % 변환
    for c in ["der_mean", "jer_mean", "der_short_05", "der_short_10", "der_turn_mean"]:
        df[c] = df[c].apply(to_percent)

    if args.sort_key in df.columns:
        df = df.sort_values(args.sort_key, ascending=True)

    # formatting
    def f2(x): return f"{x:.2f}"
    def f3(x): return f"{x:.3f}"

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\resizebox{{{args.resizebox}}}{{!}}{{%")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & "
        r"\textbf{DER (\%)} & "
        r"\textbf{JER (\%)} & "
        r"\makecell[c]{\textbf{DER}\\\textbf{($\le$0.5s, \%)}} & "
        r"\makecell[c]{\textbf{DER}\\\textbf{($\le$1.0s, \%)}} & "
        r"\makecell[c]{\textbf{DER}\\\textbf{(turn, \%)}} & "
        r"\makecell[c]{\textbf{RTF}} \\"
    )
    lines.append(r"\midrule")

    for _, r in df.iterrows():
        model = escape_latex(str(r["model"]))
        row = " & ".join([
            model,
            f2(r["der_mean"]),
            f2(r["jer_mean"]),
            f2(r["der_short_05"]),
            f2(r["der_short_10"]),
            f2(r["der_turn_mean"]),
            f3(r["rtf_mean"]),
        ]) + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(rf"\caption{{{args.caption}}}")
    lines.append(rf"\label{{{args.label}}}")
    lines.append(r"\end{table}")

    out = Path(args.out_tex)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out}")

if __name__ == "__main__":
    main()
