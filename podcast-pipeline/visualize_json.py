#!/usr/bin/env python3
"""
Visualization script for podcast pipeline JSON output
Visualizes timestamps and qwen3omni_caption data
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import textwrap


def load_json(json_path):
    """Load JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def visualize_timestamps(data, output_path=None):
    """
    Visualize segment timestamps in a timeline
    """
    segments = data.get('segments', [])
    if not segments:
        print("No segments found in the data.")
        return

    fig, ax = plt.subplots(figsize=(16, 8))

    # Extract timeline data
    for idx, segment in enumerate(segments):
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        duration = end - start
        text = segment.get('text', '').strip()

        # Truncate text for display
        display_text = textwrap.shorten(text, width=50, placeholder="...")

        # Color coding based on segment index (alternating colors)
        color = 'skyblue' if idx % 2 == 0 else 'lightcoral'

        # Draw segment bar
        ax.barh(0, duration, left=start, height=0.8,
                color=color, edgecolor='black', linewidth=0.5)

        # Add text annotation
        mid_point = start + duration / 2
        ax.text(mid_point, 0, display_text,
                ha='center', va='center', fontsize=7,
                rotation=0, wrap=True)

    # Add metadata information
    metadata = data.get('metadata', {})
    audio_duration = metadata.get('audio_duration_seconds', 0)
    total_segments = metadata.get('total_segments', 0)

    # Set labels and title
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Segments', fontsize=12)
    ax.set_title(f'Segment Timeline Visualization\n'
                 f'Total Duration: {audio_duration:.2f}s | Total Segments: {total_segments}',
                 fontsize=14, fontweight='bold')

    # Set x-axis limits
    ax.set_xlim(0, audio_duration if audio_duration > 0 else segments[-1]['end'])
    ax.set_ylim(-1, 1)
    ax.set_yticks([])

    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add legend
    legend_elements = [
        mpatches.Patch(color='skyblue', label='Even segments'),
        mpatches.Patch(color='lightcoral', label='Odd segments')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Timestamp visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_qwen3omni_captions(data, output_path=None):
    """
    Visualize qwen3omni_caption data
    Shows segments with their captions
    """
    segments = data.get('segments', [])
    if not segments:
        print("No segments found in the data.")
        return

    # Filter segments that have qwen3omni_caption
    captioned_segments = [s for s in segments if s.get('qwen3omni_caption')]

    if not captioned_segments:
        print("No segments with qwen3omni_caption found.")
        return

    num_segments = len(captioned_segments)
    fig_height = max(12, num_segments * 3)
    fig, axes = plt.subplots(num_segments, 1, figsize=(16, fig_height))

    # Handle single segment case
    if num_segments == 1:
        axes = [axes]

    for idx, (ax, segment) in enumerate(zip(axes, captioned_segments)):
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        text = segment.get('text', '').strip()
        caption = segment.get('qwen3omni_caption', '').strip()

        # Truncate caption for display
        max_caption_length = 1000
        if len(caption) > max_caption_length:
            caption_display = caption[:max_caption_length] + "..."
        else:
            caption_display = caption

        # Wrap text
        wrapped_caption = textwrap.fill(caption_display, width=120)

        # Hide axes
        ax.axis('off')

        # Create text box
        textstr = f"Segment {idx + 1} [{start:.2f}s - {end:.2f}s]\n"
        textstr += f"Text: {text}\n\n"
        textstr += f"Caption:\n{wrapped_caption}"

        # Add text with box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace',
                wrap=True)

    # Add main title
    metadata = data.get('metadata', {})
    qwen_meta = metadata.get('qwen3omni_caption', {})
    processing_time = qwen_meta.get('processing_time_seconds', 0)
    rt_factor = qwen_meta.get('rt_factor', 0)

    fig.suptitle(f'Qwen3-Omni Caption Visualization\n'
                 f'Processing Time: {processing_time:.2f}s | RT Factor: {rt_factor:.4f}',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Qwen3-Omni caption visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_all(data, output_dir=None):
    """
    Create all visualizations
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp_path = output_dir / "timestamp_visualization.png"
        caption_path = output_dir / "qwen3omni_caption_visualization.png"

        visualize_timestamps(data, timestamp_path)
        visualize_qwen3omni_captions(data, caption_path)
    else:
        visualize_timestamps(data)
        visualize_qwen3omni_captions(data)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize podcast pipeline JSON output'
    )
    parser.add_argument(
        'json_path',
        type=str,
        help='Path to the JSON file to visualize'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for saving visualizations (default: display only)'
    )
    parser.add_argument(
        '--timestamps-only',
        action='store_true',
        help='Only visualize timestamps'
    )
    parser.add_argument(
        '--captions-only',
        action='store_true',
        help='Only visualize qwen3omni captions'
    )

    args = parser.parse_args()

    # Load JSON data
    print(f"Loading JSON from: {args.json_path}")
    data = load_json(args.json_path)

    # Create visualizations
    if args.timestamps_only:
        output_path = None
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "timestamp_visualization.png"
        visualize_timestamps(data, output_path)
    elif args.captions_only:
        output_path = None
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "qwen3omni_caption_visualization.png"
        visualize_qwen3omni_captions(data, output_path)
    else:
        visualize_all(data, args.output_dir)

    print("Visualization complete!")


if __name__ == "__main__":
    main()
