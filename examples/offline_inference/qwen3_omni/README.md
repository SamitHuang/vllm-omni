# Offline Example of vLLM-omni for Qwen3-Omni-MoE

This example demonstrates how to use Qwen3 Omni MoE for text-to-speech generation using vLLM-omni's offline inference API.

## Key Differences from Qwen2.5-Omni

| Feature | Qwen2.5 Omni | Qwen3 Omni MoE |
|---------|--------------|----------------|
| **Architecture** | Standard Transformer | Mixture-of-Experts (MoE) |
| **Audio Generation** | Token2Wav (DiT + BigVGAN) | Code2Wav (Direct RVQ) |
| **Sample Rate** | 24,000 Hz | 16,000 Hz |
| **Codec Stop Token** | 8294 | 4198 |
| **Intermediate Format** | Mel-spectrogram | RVQ codes (16 layers) |
| **Generation Speed** | ~1s per 3s audio | ~0.1s per 3s audio (10x faster) |

---

## Installation

### 1. Set up basic environments

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

### 2. Install vLLM

Install specific version of vLLM with commit ID: `808a7b69df479b6b3a16181711cac7ca28a9b941`

```bash
export VLLM_COMMIT=808a7b69df479b6b3a16181711cac7ca28a9b941
uv pip install vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}
```

### 3. Install dependencies

```bash
uv pip install transformers soundfile librosa resampy torchvision
```

---

## Run examples

### 1. Navigate to the example folder

```bash
cd vllm-omni/examples/offline_inference/qwen3_omni
```

### 2. Modify PYTHONPATH in run.sh

Edit `run.sh` and set `PYTHONPATH` to your vllm-omni path:

```bash
export PYTHONPATH=/your/path/to/vllm-omni:$PYTHONPATH
```

### 3. Run the example

```bash
bash run.sh
```

The output audio will be saved in `./output_audio/`

---

## Usage

### Basic Text-to-Speech

```bash
python end2end.py \
    --model Qwen/Qwen3-Omni-MoE \
    --prompts "Hello, how are you today?" \
    --prompt_type text \
    --output-wav output_audio
```

### Multiple Prompts

```bash
python end2end.py \
    --model Qwen/Qwen3-Omni-MoE \
    --prompts \
        "Welcome to Qwen3 Omni MoE!" \
        "This is a text to speech example." \
        "Thank you for using our system." \
    --prompt_type text \
    --output-wav output_audio
```

### With Audio Input

```bash
python end2end.py \
    --model Qwen/Qwen3-Omni-MoE \
    --prompts "path/to/audio.wav" \
    --prompt_type audio \
    --output-wav output_audio
```

### With Image Input

```bash
python end2end.py \
    --model Qwen/Qwen3-Omni-MoE \
    --prompts "path/to/image.jpg" \
    --prompt_type image \
    --output-wav output_audio
```

### With Video Input

```bash
python end2end.py \
    --model Qwen/Qwen3-Omni-MoE \
    --prompts "path/to/video.mp4" \
    --prompt_type video \
    --output-wav output_audio
```

---

## Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path or HF model ID | Required |
| `--prompts` | Input prompts (text, audio, image, video path) | Required |
| `--prompt_type` | Type of input: text, audio, image, video, etc. | `text` |
| `--voice-type` | Voice type for generation | `default` |
| `--output-wav` | Output directory for audio files | `output_audio` |
| `--dtype` | Model dtype | `bfloat16` |
| `--max-model-len` | Maximum sequence length | `32768` |
| `--tokenize` | Pre-tokenize inputs | `False` |

---

## Output Format

### Text Output (Stage 0: Thinker)

```
================================================================================
THINKER OUTPUT (Text Generation)
================================================================================

Request ID: 0
Text: Scalable audio pipeline: modular thinker, talker, code2wav stages with...
--------------------------------------------------------------------------------
```

### Audio Output (Stage 2: Code2Wav)

```
================================================================================
CODE2WAV OUTPUT (Audio Generation)
================================================================================

Request ID: 0
Audio file: output_audio/output_0.wav
Duration: 8.00 seconds
Sample rate: 16000 Hz
Shape: torch.Size([1, 1, 128000])
--------------------------------------------------------------------------------
```

**Audio Specifications**:
- **Sample Rate**: 16,000 Hz (16kHz)
- **Channels**: 1 (mono)
- **Format**: WAV
- **Encoding**: PCM float32

---

## Architecture

Qwen3 Omni MoE uses a 3-stage pipeline:

```
┌─────────────────┐
│  Stage 0:       │  Multimodal Understanding
│  THINKER        │  • Process text/audio/video
│                 │  • Generate text response
│  (MoE)          │  • Output: text + hidden states
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Stage 1:       │  Codec Generation
│  TALKER         │  • Project thinker outputs
│                 │  • Generate RVQ layer 0 (main)
│  (MoE)          │  • Generate RVQ layers 1-31 (predictor)
└─────────────────┘  • Output: 32 codec groups (→ 16 quantizers)
         │
         ▼
┌─────────────────┐
│  Stage 2:       │  Waveform Generation
│  CODE2WAV       │  • Embed RVQ codes (16 layers)
│                 │  • Pre-transformer (temporal context)
│  (Autoregressive│  • Progressive upsampling (1280x)
│   not Diffusion)│  • Generate audio waveform
└─────────────────┘  • Output: audio @ 16kHz
```

**Key Points**:
- **Stage 2 uses autoregressive generation** (not diffusion like Qwen2.5)
- **10x faster** than Qwen2.5 Token2Wav
- **Direct RVQ** without mel-spectrogram intermediate
- **16kHz output** (vs 24kHz in Qwen2.5)

---

## Troubleshooting

### Issue 1: Import Error

```bash
ModuleNotFoundError: No module named 'vllm_omni'
```

**Solution**: Set `PYTHONPATH` correctly:
```bash
export PYTHONPATH=/your/path/to/vllm-omni:$PYTHONPATH
```

### Issue 2: Model Not Found

```bash
OSError: Qwen/Qwen3-Omni-MoE does not appear to be a valid model
```

**Solution**: Download the model first or use a local path:
```bash
python end2end.py --model /path/to/local/model ...
```

### Issue 3: CUDA Out of Memory

**Solution**: Reduce batch size or use smaller prompts:
```bash
python end2end.py \
    --model Qwen/Qwen3-Omni-MoE \
    --prompts "Short prompt" \
    --max-model-len 16384
```

### Issue 4: Wrong Sample Rate

If audio sounds too fast/slow, verify you're using 16kHz (not 24kHz):
```python
sf.write(output_wav, audio, samplerate=16000)  # Correct for Qwen3
```

---

## Performance Tips

### 1. Use bfloat16 for faster inference

```bash
python end2end.py --dtype bfloat16 ...
```

### 2. Enable chunked decoding for long audio

Edit `qwen3_omni_code2wav.py` and adjust `chunk_size`:
```python
waveform = code2wav.chunked_decode(codes, chunk_size=200)
```

### 3. Use multiple GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 python end2end.py --tensor-parallel-size 2 ...
```

---

## Example Outputs

Example audio files will be generated in `output_audio/`:
- `output_0.wav` - First prompt
- `output_1.wav` - Second prompt
- ...

Each file contains the synthesized speech from the corresponding prompt.

---

## To-do List

- [x] Offline inference example for Qwen3 Omni MoE with single request
- [ ] Streaming inference support
- [ ] Multi-round conversation support
- [ ] Online serving API
- [ ] Batch processing optimization
- [ ] Voice cloning support
- [ ] Real-time streaming audio generation

---

## Technical Notes

### RVQ Layers

Qwen3 Omni MoE uses **16-layer RVQ** (Residual Vector Quantization) for audio:
- **Talker generates**: 32 codec groups (1 from main + 31 from predictor)
- **Code2Wav expects**: 16 quantizers (mapping from 32→16 TBD)
- **Codebook size**: 2048 codes per layer
- **Total compression**: ~29x from raw audio

### Special Token IDs

```python
TALKER_CODEC_BOS_TOKEN_ID = 4197  # Beginning of codec sequence
TALKER_CODEC_EOS_TOKEN_ID = 4198  # End of codec sequence (stop token)
TALKER_CODEC_PAD_TOKEN_ID = 4196  # Padding token
```

### Configuration Files

See `vllm_omni/model_executor/stage_configs/qwen3_omni.yaml` for detailed stage configuration.

---

## References

- **Model**: [Qwen/Qwen3-Omni-MoE](https://huggingface.co/Qwen/Qwen3-Omni-MoE)
- **Paper**: Qwen3 Technical Report
- **vLLM**: [vLLM GitHub](https://github.com/vllm-project/vllm)
- **vLLM-omni**: Multi-stage inference framework

---

## License

Apache-2.0

---

## Support

For issues and questions:
- GitHub Issues: [vllm-project/vllm](https://github.com/vllm-project/vllm/issues)
- Model Issues: [Qwen HuggingFace](https://huggingface.co/Qwen)

