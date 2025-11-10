import argparse
import warnings
from typing import Optional

import gradio as gr
import numpy as np
import soundfile as sf  # noqa: F401  # ensure soundfile dependency is present
import torch
from qwen_omni_utils import process_mm_info
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

# Suppress the non-writable numpy warning emitted during model init (logged once).
warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gradio demo for Qwen2.5-Omni using the Transformers backend."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Omni-7B",
        help="Model identifier or local path.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map configuration passed to `from_pretrained`.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype for model weights (e.g., auto, float16, bfloat16, float32).",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Attention implementation (e.g., flash_attention_2).",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating text and speech."
        ),
        help="System prompt to condition the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--use-audio-in-video",
        action="store_true",
        help="Enable audio extraction when user uploads a video (if supported).",
    )
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="Host/IP for gradio `launch`.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Port for gradio `launch`.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the Gradio app publicly.",
    )
    return parser.parse_args()


def resolve_torch_dtype(dtype_str: str) -> Optional[torch.dtype]:
    if dtype_str == "auto":
        return "auto"
    if not hasattr(torch, dtype_str):
        raise ValueError(f"Unsupported torch dtype: {dtype_str}")
    return getattr(torch, dtype_str)


def load_model_and_processor(args: argparse.Namespace):
    dtype = resolve_torch_dtype(args.torch_dtype)
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model,
            **model_kwargs,
        )
    except ImportError as exc:
        if args.attn_implementation:
            warnings.warn(
                f"Failed to load model with attn_implementation='{args.attn_implementation}': {exc}. "
                "Falling back to default attention implementation."
            )
            model_kwargs.pop("attn_implementation", None)
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                args.model,
                **model_kwargs,
            )
        else:
            raise

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model)
    return model, processor


def build_interface(
    args: argparse.Namespace,
    model: Qwen2_5OmniForConditionalGeneration,
    processor: Qwen2_5OmniProcessor,
):
    device = model.device
    dtype = model.dtype

    def generate_response(
        user_text: str,
        user_audio: Optional[tuple[int, np.ndarray]] = None,
        user_image: Optional[np.ndarray] = None,
        user_video: Optional[str] = None,
    ):
        if not user_text and not user_audio and not user_image and not user_video:
            return "Please provide at least one input modal.", None

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": args.system_prompt}],
            },
            {
                "role": "user",
                "content": [],
            },
        ]

        if user_text:
            conversation[1]["content"].append({"type": "text", "text": user_text})
        if user_audio is not None:
            sample_rate, audio_arr = user_audio
            conversation[1]["content"].append(
                {"type": "audio", "audio": {"array": audio_arr, "sampling_rate": sample_rate}}
            )
        if user_image is not None:
            conversation[1]["content"].append({"type": "image", "image": user_image})
        if user_video is not None:
            conversation[1]["content"].append({"type": "video", "video": user_video})

        text_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=args.use_audio_in_video
        )
        inputs = processor(
            text=text_prompt,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=args.use_audio_in_video,
        )

        inputs = inputs.to(device)
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                inputs[key] = value.to(dtype)

        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "use_audio_in_video": args.use_audio_in_video,
        }

        with torch.no_grad():
            text_ids, audio_output = model.generate(**inputs, **generate_kwargs)

        decoded_text = processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        response_text = decoded_text[0] if decoded_text else "No text output."

        audio_result = None
        if audio_output is not None:
            audio_tensor = audio_output.reshape(-1).detach().cpu().to(torch.float32)
            audio_result = (24000, audio_tensor.numpy())

        return response_text, audio_result

    with gr.Blocks() as demo:
        gr.Markdown("# Qwen2.5-Omni Transformers Gradio Demo")
        gr.Markdown(
            "Interact with the Qwen2.5-Omni model (Transformers backend). "
            "You can provide text, optional audio, image, or video inputs."
        )

        with gr.Row():
            text_input = gr.Textbox(label="User Text", placeholder="Ask Qwen something...", lines=4)
        with gr.Row():
            audio_input = gr.Audio(sources=["upload", "microphone"], label="User Audio")
            image_input = gr.Image(label="User Image", type="numpy")
        with gr.Row():
            video_input = gr.Video(label="User Video")
        with gr.Row():
            submit_btn = gr.Button("Generate", variant="primary")
        with gr.Row():
            text_output = gr.Textbox(label="Model Text Response", lines=8)
            audio_output = gr.Audio(
                label="Model Audio Response", interactive=False, autoplay=True
            )

        submit_btn.click(
            fn=generate_response,
            inputs=[text_input, audio_input, image_input, video_input],
            outputs=[text_output, audio_output],
        )

        demo.queue()
    return demo


def main():
    args = parse_args()
    model, processor = load_model_and_processor(args)
    gradio_app = build_interface(args, model, processor)
    gradio_app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()

