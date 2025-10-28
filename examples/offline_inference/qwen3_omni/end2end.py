import argparse
import os
import os as _os_env_toggle
import random

import numpy as np
import soundfile as sf
import torch
from utils import make_omni_prompt
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.omni_llm import OmniLLM

_os_env_toggle.environ["VLLM_USE_V1"] = "1"

SEED = 42
# Set all random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Make PyTorch deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variables for deterministic behavior
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3 Omni MoE Offline Inference Example"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to merged model directory (will be created if downloading).",
    )
    parser.add_argument("--thinker-model", type=str, default=None)
    parser.add_argument("--talker-model", type=str, default=None)
    parser.add_argument("--code2wav-model", type=str, default=None)
    parser.add_argument(
        "--hf-hub-id",
        default="Qwen/Qwen3-Omni-MoE",
        help="Hugging Face repo id to download if needed.",
    )
    parser.add_argument(
        "--hf-revision", default=None, help="Optional HF revision (branch/tag/commit)."
    )
    parser.add_argument(
        "--prompts", required=True, nargs="+", help="Input text prompts."
    )
    parser.add_argument(
        "--voice-type", default="default", help="Voice type (default for Qwen3)."
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--max-model-len", type=int, default=32768)

    parser.add_argument("--thinker-only", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument(
        "--prompt_type",
        choices=[
            "text",
            "audio",
            "audio-long",
            "image",
            "video",
            "audio-in-video",
            "multi-round",
        ],
        default="text",
    )
    parser.add_argument("--use-torchvision", action="store_true")
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument(
        "--output-wav", default="output_audio", help="Output directory for wav files."
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_name = args.model

    # Initialize Omni LLM
    omni_llm = OmniLLM(model=model_name)

    # Sampling parameters for Thinker stage (text generation)
    thinker_sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        top_p=1.0,
        top_k=-1,
        max_tokens=2048,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.1,
    )

    # Sampling parameters for Talker stage (codec generation)
    # Stop at codec EOS token (4198 for Qwen3 Omni MoE)
    talker_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=2048,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.1,
        stop_token_ids=[4198],  # TALKER_CODEC_EOS_TOKEN_ID
    )

    # Sampling parameters for Code2Wav stage (audio generation)
    code2wav_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=2048,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.1,
    )

    sampling_params_list = [
        thinker_sampling_params,
        talker_sampling_params,
        code2wav_sampling_params,
    ]

    # Create prompts
    prompt = [make_omni_prompt(args, prompt) for prompt in args.prompts]

    # Generate outputs
    print(f"Processing {len(prompt)} prompt(s)...")
    omni_outputs = omni_llm.generate(prompt, sampling_params_list)

    # Process outputs
    os.makedirs(args.output_wav, exist_ok=True)
    for stage_outputs in omni_outputs:
        if stage_outputs.final_output_type == "text":
            # Thinker stage output (text)
            print("\n" + "=" * 80)
            print("THINKER OUTPUT (Text Generation)")
            print("=" * 80)
            for output in stage_outputs.request_output:
                request_id = output.request_id
                text_output = output.outputs[0].text
                print(f"\nRequest ID: {request_id}")
                print(f"Text: {text_output}")
                print("-" * 80)

        elif stage_outputs.final_output_type == "audio":
            # Code2Wav stage output (audio)
            print("\n" + "=" * 80)
            print("CODE2WAV OUTPUT (Audio Generation)")
            print("=" * 80)
            for output in stage_outputs.request_output:
                request_id = output.request_id
                audio_tensor = output.multimodal_output["audio"]
                
                # Save audio file
                output_wav = os.path.join(
                    args.output_wav, f"output_{output.request_id}.wav"
                )
                
                # Qwen3 Omni uses 16kHz sample rate
                sf.write(
                    output_wav,
                    audio_tensor.detach().cpu().numpy().squeeze(),
                    samplerate=16000,
                )
                
                # Print audio info
                duration = audio_tensor.shape[-1] / 16000
                print(f"\nRequest ID: {request_id}")
                print(f"Audio file: {output_wav}")
                print(f"Duration: {duration:.2f} seconds")
                print(f"Sample rate: 16000 Hz")
                print(f"Shape: {audio_tensor.shape}")
                print("-" * 80)

    print("\n Generation complete!")
    print(f"Audio files saved to: {args.output_wav}/")


if __name__ == "__main__":
    main()

