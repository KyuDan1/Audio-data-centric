# Podcast Pipeline

A comprehensive audio processing pipeline for speaker diarization, automatic speech recognition, background music removal, and more.

## Features

- **Speaker Diarization**: Speaker separation using Pyannote and NeMo Sortformer
- **Automatic Speech Recognition (ASR)**: Multilingual speech recognition based on Whisper
- **ASR MoE (Mixture of Experts)**: Ensemble of Whisper, Parakeet, and Canary models using ROVER voting
- **Background Music Removal**: Music detection with PANNs and vocal extraction with Demucs
- **Word-level Timestamps**: Precise word-level time alignment using WhisperX
- **Audio Captioning**: Audio segment description generation using Qwen3-Omni
- **Korean Post-processing**: Korean pronunciation conversion using g2pk

## Usage

### Basic Execution

```bash
bash run_test_all.sh
```

The main execution script is [run_test_all.sh](run_test_all.sh), which allows you to run the pipeline with various configuration combinations.

### Direct Python Execution

```bash
python main_original_ASR_MoE.py \
  --input_folder_path /path/to/audio \
  --vad \
  --dia3 \
  --ASRMoE \
  --demucs \
  --whisperx_word_timestamps \
  --qwen3omni \
  --korean \
  --LLM case_0 \
  --seg_th 0.11 \
  --min_cluster_size 11 \
  --clust_th 0.5 \
  --merge_gap 2
```

## Configuration Options

### Speaker Diarization

- `--dia3` / `--no-dia3`: Use Pyannote Diarization 3.1 model
- `--seg_th`: Segmentation threshold (default: 0.15)
- `--min_cluster_size`: Minimum cluster size for clustering (default: 10)
- `--clust_th`: Clustering threshold (default: 0.5)
- `--merge_gap`: Segment merge gap in seconds (default: 2)

### Automatic Speech Recognition (ASR)

- `--ASRMoE` / `--no-ASRMoE`: Enable ASR Mixture of Experts mode
  - Ensembles Whisper large-v3, Parakeet-TDT-0.6B, and Canary-Qwen-2.5B using ROVER voting
- `--whisper_arch`: Whisper model architecture (default: large-v3)
- `--batch_size`: Batch size (default: 64)
- `--compute_type`: Computation type (default: float16)
- `--initprompt` / `--no-initprompt`: Use initial prompt for Whisper

### Background Music Removal

- `--demucs` / `--no-demucs`: Enable background music detection and removal
  - Detects music probability using PANNs (threshold: 0.3)
  - Extracts vocals using Demucs htdemucs model

### Word-level Timestamps

- `--whisperx_word_timestamps` / `--no-whisperx_word_timestamps`: Enable WhisperX alignment
  - Provides precise word-level start/end times

### Qwen3-Omni Audio Captioning

- `--qwen3omni` / `--no-qwen3omni`: Enable Qwen3-Omni audio caption generation
  - Requires a separate vLLM server to be running
  - Port configuration: Modify the `QWEN_3_OMNI_PORT` variable in [main_original_ASR_MoE.py](main_original_ASR_MoE.py) (default: "11500")

#### Running Qwen3-Omni vLLM Server

```bash
# Run Qwen3-Omni server with vLLM (port 11500)
vllm serve Qwen/Qwen3-Omni --port 11500 --max-model-len 8192

# To use a different port
vllm serve Qwen/Qwen3-Omni --port 12000 --max-model-len 8192
# Then modify QWEN_3_OMNI_PORT = "12000" in main_original_ASR_MoE.py
```

### Korean Processing

- `--korean` / `--no-korean`: Enable Korean g2pk pronunciation conversion
  - Converts English segments to Korean pronunciation

### Other Options

- `--vad` / `--no-vad`: Use Voice Activity Detection
- `--LLM`: LLM post-processing case (case_0: disabled, case_2: enabled)

## Output

Processed results are saved in the following structure:

```
input_path/_final/_processed_llm-twelve-cases-[config]/
├── audio_name/
│   ├── audio_name.json          # Complete results with metadata
│   ├── audio_name_00000.mp3     # Segmented audio files
│   ├── audio_name_00001.mp3
│   └── ...
```

### JSON Output Format

```json
{
  "metadata": {
    "audio_duration_seconds": 120.5,
    "audio_duration_minutes": 2.008,
    "vad_sortformer": {
      "processing_time_seconds": 12.3,
      "rt_factor": 0.102
    },
    "whisper_large_v3": {
      "processing_time_seconds": 45.6,
      "rt_factor": 0.378
    },
    "whisperx_alignment": {
      "processing_time_seconds": 8.2,
      "rt_factor": 0.068,
      "enabled": true
    },
    "qwen3omni_caption": {
      "processing_time_seconds": 15.4,
      "rt_factor": 0.128,
      "enabled": true
    },
    "total_segments": 25
  },
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "speaker": "SPEAKER_00",
      "text": "Example transcription",
      "text_whisper": "Example transcription",
      "text_canary": "Example transcription",
      "text_parakeet": "Example transcription",
      "text_ensemble": "Example transcription",
      "language": "en",
      "demucs": false,
      "qwen3omni_caption": "Audio description...",
      "words": [
        {
          "word": "Example",
          "start": 0.5,
          "end": 1.0
        },
        {
          "word": "transcription",
          "start": 1.1,
          "end": 3.2
        }
      ]
    }
  ]
}
```

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Key dependencies:
  - PyTorch
  - Whisper / WhisperX
  - Pyannote.audio
  - NeMo (Parakeet, Canary, Sortformer)
  - Demucs
  - PANNs
  - g2pk (for Korean processing)

## Configuration File

The `config.json` file allows you to configure:

- Input folder path
- Sample rate
- Hugging Face token
- Supported languages list
- Multilingual mode

## Processing Pipeline

1. **Audio Preprocessing**: 24kHz resampling, mono conversion, volume normalization
2. **Speaker Diarization**: Speaker segment extraction using Sortformer
3. **Background Music Processing** (optional): PANNs detection + Demucs removal
4. **Speech Recognition**: Transcription using Whisper or ASR MoE
5. **Word Alignment** (optional): Word-level timestamps using WhisperX
6. **Audio Captioning** (optional): Segment description generation using Qwen3-Omni
7. **Post-processing**: Korean pronunciation conversion, etc.
8. **Save Results**: JSON and segmented MP3 file generation

## License

This project is licensed under the MIT License. (Copyright (c) 2024 Amphion)

## Notes

- ASR MoE uses ROVER (Recognizer Output Voting Error Reduction) ensemble technique
- Repetition filtering removes low-quality transcriptions where 15-grams repeat more than 5 times
- Long segments are automatically split into chunks of 30 seconds or less
