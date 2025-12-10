# FlowSE Setup Guide

This project uses FlowSE for audio denoising. Follow these steps to set up FlowSE as a dependency.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

## Installation Steps

### 1. Clone FlowSE Repository

The FlowSE repository should be placed in the parent directory of this project:

```bash
cd /path/to/Audio-data-centric
git clone https://github.com/cantabile-kwok/FlowSE.git
```

Your directory structure should look like:
```
Audio-data-centric/
├── FlowSE/
│   ├── simple_denoise.py
│   ├── model/
│   ├── ckpts/
│   └── ...
└── podcast-pipeline/
    ├── main_original_ASR_MoE.py
    └── ...
```

### 2. Install FlowSE Dependencies

```bash
cd FlowSE
pip install -r requirement.txt
```

### 3. Download Model Checkpoints

You need to download the pretrained FlowSE model and vocoder:

#### Download FlowSE Checkpoint
```bash
# Create checkpoint directory if it doesn't exist
mkdir -p ckpts

# Download the pretrained model (example URL - replace with actual URL)
# Visit FlowSE repository for the latest checkpoint download instructions
wget -O ckpts/best.pt.tar <FLOWSE_CHECKPOINT_URL>
```

#### Download Vocoder
```bash
# Download vocos vocoder (example)
git clone https://huggingface.co/charactr/vocos-mel-24khz
```

#### Download Tokenizer
```bash
# Ensure Emilia_ZH_EN_pinyin/vocab.txt exists
# This should be included in the FlowSE repository
```

### 4. Verify Installation

Test that FlowSE is correctly installed:

```bash
cd FlowSE
python simple_denoise.py --help
```

## Configuration

The `main_original_ASR_MoE.py` script automatically looks for FlowSE in the parent directory using a relative path:

```python
flowse_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "FlowSE")
```

If you need to use a different location, you can modify this path in the script.

## Model File Locations

The FlowSE denoiser expects the following files:

- **Checkpoint**: `FlowSE/ckpts/best.pt.tar`
- **Tokenizer**: `FlowSE/Emilia_ZH_EN_pinyin/vocab.txt`
- **Vocoder**: `FlowSE/vocos-mel-24khz/`

## Troubleshooting

### Import Error: "No module named 'simple_denoise'"

Make sure:
1. FlowSE is in the correct location (parent directory)
2. FlowSE dependencies are installed
3. The relative path in `main_original_ASR_MoE.py` is correct

### Model Checkpoint Not Found

Ensure the checkpoint file exists at `FlowSE/ckpts/best.pt.tar`. Check the FlowSE repository for download instructions.

### CUDA Out of Memory

FlowSE requires GPU memory. If you encounter OOM errors:
- Reduce batch size
- Use CPU mode (slower but works): set `use_cuda=False`

## References

- FlowSE Repository: https://github.com/cantabile-kwok/FlowSE
- FlowSE Paper: [Add paper reference if available]

## Notes

- FlowSE is used for audio denoising to improve ASR quality
- The denoising step can be disabled if FlowSE is not available (some features will be limited)
- GPU is highly recommended for reasonable inference speed
