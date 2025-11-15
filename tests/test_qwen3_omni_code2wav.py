"""Unit test for vLLM Qwen3-Omni Code2Wav against captured transformers I/O."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
)

from vllm_omni.model_executor.models.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_CAPTURE = TEST_DATA_DIR / "tf_code2wave_input.pt"
OUTPUT_CAPTURE = TEST_DATA_DIR / "tf_code2wave_output.pt"
AUDIO_SAMPLE_RATE = 24000


def _iter_code2wav_weights(model_id: str):
    """Yield (name, tensor) pairs for code2wav.* weights from HF shards."""
    index_path = hf_hub_download(model_id, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as file:
        index = json.load(file)

    weights_by_shard: dict[str, list[str]] = {}
    for weight_name, shard_name in index["weight_map"].items():
        if not weight_name.startswith("code2wav."):
            continue
        weights_by_shard.setdefault(shard_name, []).append(weight_name)

    for shard_name, weight_names in weights_by_shard.items():
        shard_path = hf_hub_download(model_id, shard_name)
        with safe_open(shard_path, framework="pt") as shard:
            for weight_name in weight_names:
                yield weight_name, shard.get_tensor(weight_name)


def _save_wav(tensor: torch.Tensor, path: Path, sample_rate: int) -> Path:
    audio = tensor.detach().cpu().reshape(-1).numpy()
    audio = np.clip(audio, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, samplerate=sample_rate)
    return path


@pytest.mark.skipif(
    not (INPUT_CAPTURE.exists() and OUTPUT_CAPTURE.exists()),
    reason="Code2Wav capture fixtures missing under tests/data",
)
def test_vllm_code2wav_matches_transformers_capture():
    """Ensure vLLM Code2Wav reproduces the reference waveform."""
    codes = torch.load(INPUT_CAPTURE)
    expected_wav = torch.load(OUTPUT_CAPTURE)

    config = Qwen3OmniMoeCode2WavConfig.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    dummy_vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config)
    )
    model = Qwen3OmniMoeCode2Wav(vllm_config=dummy_vllm_config)
    model.load_weights(_iter_code2wav_weights(MODEL_ID))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    codes = codes.to(device)

    with torch.no_grad():
        wav = model.chunked_decode(codes)

    wav_path = TEST_DATA_DIR / f"code2wav_vllm_output.wav"
    saved_path = _save_wav(wav, wav_path, AUDIO_SAMPLE_RATE)
    print(f"Saved vLLM Code2Wav output to {saved_path}")

    diff = (wav.cpu() - expected_wav).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print("max_diff", max_diff)
    print("mean_diff", mean_diff)
    assert max_diff <= 2e-2, (
        f"Code2Wav output mismatch (max diff {max_diff:.4f}, mean diff {mean_diff:.4f})"
    )

