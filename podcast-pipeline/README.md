# Podcast Pipeline - ASR with MoE

Advanced podcast transcription pipeline with ASR ensemble (MoE), speaker diarization, and audio enhancement.

## Features

- **Multi-Model ASR Ensemble**: ROVER-based ensemble of multiple ASR models for improved accuracy
- **Speaker Diarization**: Pyannote-based speaker separation with clustering
- **Audio Enhancement**:
  - FlowSE denoising for improved audio quality
  - Demucs vocal separation for background music removal
  - Silero VAD for voice activity detection
- **Advanced Processing**:
  - Segment-level audio preprocessing
  - Overlap handling
  - Korean language support with G2P

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (highly recommended)
- Git
- ffmpeg (for audio processing)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd podcast-pipeline
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch`, `torchaudio` - Deep learning framework
- `pyannote.audio` - Speaker diarization
- `whisperx` - ASR
- `demucs` - Music source separation
- `nemo-toolkit` - ASR models
- `librosa`, `soundfile` - Audio processing

### 3. Set Up FlowSE (Required for Denoising)

FlowSE is used for audio denoising. Follow the detailed setup guide:

```bash
# See SETUP_FLOWSE.md for complete instructions
cd ..
git clone https://github.com/cantabile-kwok/FlowSE.git
cd FlowSE
pip install -r requirement.txt
# Download model checkpoints (see SETUP_FLOWSE.md)
```

**Important**: FlowSE must be located in the parent directory of `podcast-pipeline`:
```
Audio-data-centric/
├── FlowSE/
│   ├── simple_denoise.py
│   ├── ckpts/best.pt.tar
│   └── ...
└── podcast-pipeline/
    ├── main_original_ASR_MoE.py
    └── ...
```

See [SETUP_FLOWSE.md](SETUP_FLOWSE.md) for detailed FlowSE installation instructions.

### 4. Download Model Checkpoints

You'll need various model checkpoints:

#### Pyannote Speaker Diarization
```bash
# Requires Hugging Face token
# Set HF_TOKEN environment variable or login via huggingface-cli
huggingface-cli login
```

#### NeMo Models
The script will automatically download NeMo ASR models on first run.

## Usage

### Basic Usage

```bash
python main_original_ASR_MoE.py \
  --audio_path /path/to/audio.mp3 \
  --output_dir ./output
```

### Advanced Usage with All Features

```bash
python main_original_ASR_MoE.py \
  --audio_path /path/to/audio.mp3 \
  --output_dir ./output \
  --ASRMoE \
  --demucs \
  --dia3 \
  --korean
```

### Command Line Arguments

#### Required Arguments
- `--audio_path`: Input audio file path
- `--output_dir`: Output directory for results

#### ASR Options
- `--ASRMoE`: Enable ASR ensemble (multiple models with ROVER)
- `--whisperx_word_timestamps`: Use WhisperX for word-level timestamps
- `--qwen3omni`: Use Qwen3-Omni ASR model

#### Audio Processing
- `--demucs`: Enable background music removal with Demucs
- `--sepreformer`: Use SepReformer for source separation
- `--vad`: Enable voice activity detection

#### Speaker Diarization
- `--dia3`: Use pyannote diarization 3.0
- `--min_cluster_size`: Minimum cluster size for diarization (default: 5)
- `--clustering_threshold`: Clustering threshold (default: 0.3)
- `--merge_gap`: Gap for merging segments in seconds (default: 0.5)

#### Language Options
- `--korean`: Enable Korean language processing with G2P

#### Other Options
- `--segment_threshold`: Segment duration threshold (default: 10.0)
- `--overlap_threshold`: Overlap threshold for segments (default: 0.3)
- `--init_prompt`: Initial prompt for ASR models

## Output Format

The pipeline generates:
- **Transcription JSON**: Word-level and segment-level transcriptions
- **Speaker Labels**: Diarization results with speaker assignments
- **Processed Audio**: Cleaned audio files (if denoising is enabled)
- **Metadata**: Processing statistics and configuration

Example output structure:
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "speaker": "SPEAKER_00",
      "text": "Transcribed text here",
      "words": [...]
    }
  ]
}
```

## Architecture

### Processing Pipeline

1. **Audio Loading**: Load and normalize input audio
2. **VAD** (optional): Detect voice activity
3. **Speaker Diarization**: Identify and separate speakers
4. **Audio Enhancement**:
   - Background music detection (PANNs)
   - Music removal (Demucs) if detected
   - Denoising (FlowSE)
5. **ASR**: Transcribe with single or ensemble models
6. **Post-processing**: Align transcriptions with speakers

### ASR Ensemble (ROVER)

When `--ASRMoE` is enabled:
- Multiple ASR models process the same audio
- ROVER algorithm combines outputs via voting
- Improves accuracy through model diversity

Supported models:
- NeMo Conformer models
- WhisperX
- Qwen3-Omni (if enabled)

## Troubleshooting

### Common Issues

#### FlowSE Import Error
```
ModuleNotFoundError: No module named 'simple_denoise'
```
**Solution**: Ensure FlowSE is installed in the parent directory. See [SETUP_FLOWSE.md](SETUP_FLOWSE.md).

#### Demucs Not Found
```
No module named 'demucs'
```
**Solution**: Install demucs: `pip install demucs>=4.0.0` (should be in requirements.txt)

#### CUDA Out of Memory
**Solution**:
- Reduce batch size
- Process shorter audio segments
- Use CPU mode (slower): disable CUDA in environment

#### Pyannote Authentication Error
**Solution**:
- Login to Hugging Face: `huggingface-cli login`
- Accept pyannote model conditions on Hugging Face Hub

### Performance Tips

- **GPU Memory**: 16GB+ VRAM recommended for full pipeline
- **Processing Time**: ~1-2x real-time on modern GPUs
- **Accuracy vs Speed**: Enable `--ASRMoE` for better accuracy (slower)

## Project Structure

```
podcast-pipeline/
├── main_original_ASR_MoE.py       # Main processing script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── SETUP_FLOWSE.md                # FlowSE setup guide
├── RUN_TEST_ALL_GUIDE.md          # Batch processing guide
├── utils/
│   ├── tool.py                    # Utility functions
│   └── logger.py                  # Logging utilities
└── models/                         # Model wrapper modules
```

## Related Documentation

- [SETUP_FLOWSE.md](SETUP_FLOWSE.md) - FlowSE installation guide
- [RUN_TEST_ALL_GUIDE.md](RUN_TEST_ALL_GUIDE.md) - Batch processing guide
- [SEPREFORMER_INTEGRATION.md](SEPREFORMER_INTEGRATION.md) - SepReformer setup

## Citation

If you use this pipeline in your research, please cite the relevant papers:

- FlowSE: [Add FlowSE citation]
- Demucs: [Add Demucs citation]
- Pyannote: [Add Pyannote citation]
- NeMo: [Add NeMo citation]

## License

[Add your license information]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

[Add contact information or links]
