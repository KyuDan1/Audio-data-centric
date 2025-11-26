"""Utility to compute the total running time of MP3 files in a directory tree."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Tuple

try:
    import torchaudio  # type: ignore
except ImportError:  # pragma: no cover - availability depends on environment
    torchaudio = None


def iter_mp3_files(root: Path) -> Iterable[Path]:
    """Yield every `.mp3` below ``root`` (recursively), sorted for determinism."""
    return sorted(p for p in root.rglob("*.mp3") if p.is_file())


def duration_seconds(path: Path) -> float:
    """Return the duration of an MP3 file in seconds."""
    if torchaudio is not None:
        info = torchaudio.info(str(path))
        if info.sample_rate == 0:
            raise ValueError(f"Sample rate is zero for file: {path}")
        return info.num_frames / info.sample_rate
    return duration_via_ffprobe(path)


def duration_via_ffprobe(path: Path) -> float:
    """Fallback duration computation that shells out to ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {path} (return code {result.returncode}): {result.stderr.strip()}"
        )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(f"Could not parse ffprobe output for {path}: {result.stdout}") from exc


def aggregate_durations(files: Iterable[Path]) -> Tuple[int, float]:
    """Return the number of files and their total duration in seconds."""
    file_count = 0
    total_seconds = 0.0
    for file_path in files:
        file_count += 1
        try:
            total_seconds += duration_seconds(file_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to read {file_path}: {exc}", file=sys.stderr)
    return file_count, total_seconds


def format_duration(seconds: float) -> str:
    """Return a human-friendly representation of ``seconds``."""
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(sec):02d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the total duration of every MP3 file under the given directory."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Path that contains MP3 files (search is recursive).",
    )
    args = parser.parse_args()
    target_dir = args.directory.expanduser().resolve()

    if not target_dir.exists():
        parser.error(f"Directory does not exist: {target_dir}")

    mp3_files = iter_mp3_files(target_dir)
    file_count, total_seconds = aggregate_durations(mp3_files)

    if file_count == 0:
        print(f"No MP3 files found under {target_dir}")
        return

    total_hours = total_seconds / 3600
    print(f"Analyzed directory: {target_dir}")
    print(f"MP3 file count : {file_count}")
    print(f"Total duration : {total_hours:.2f} hours")
    print(f"HH:MM:SS total : {format_duration(total_seconds)}")


if __name__ == "__main__":
    main()
