# Image-to-Video Online Serving Design

## Summary
This document describes the OpenAI-style image-to-video generation API for Wan2.2
and how image inputs are handled in the online serving path.

## Goals
- Support image-to-video generation via `/v1/videos` (OpenAI-style create).
- Accept `input_reference` as a multipart file upload.
- Reuse the existing diffusion execution path and video response format.

## Non-goals
- Streaming output (reserved for future work).
- External asset hosting (response format is `b64_json` only).

## API
### POST `/v1/videos`

**Request (core fields)**
- `prompt` (string, required)
- `input_reference` (file, optional, multipart form-data)

**Request (video parameters)**
- `size` or `width`/`height`
- `num_frames`, `fps`
- `num_inference_steps`, `guidance_scale`, `guidance_scale_2`
- `boundary_ratio`, `flow_shift`, `seed`

**Response**
```
{
  "created": 1730000000,
  "data": [
    { "b64_json": "<base64-mp4>" }
  ]
}
```

## Main Logic
```
Client
  |
  | POST /v1/videos (multipart)
  v
APIServer
  |
  v
OmniOpenAIServingVideo
  |
  v
decode_input_reference
  |
  v
OmniTextPrompt.multi_modal_data.image
  |
  v
OmniDiffusionSamplingParams
  |
  v
AsyncOmniDiffusion / AsyncOmni
  |
  v
DiffusionEngine (Wan2.2 I2V)
  |
  v
OmniRequestOutput.video
  |
  v
encode_video_base64
  |
  v
VideoGenerationResponse
```

1. Decode `input_reference` from multipart form-data.
2. Attach the decoded image to `multi_modal_data.image` in the prompt.
3. Run diffusion pipeline with I2V model.
4. Encode MP4 to base64 and return.

## Main Changes
- Protocol: `vllm_omni/entrypoints/openai/protocol/videos.py`
- Image decoding: `vllm_omni/entrypoints/openai/video_api_utils.py`
- Serving handler: `vllm_omni/entrypoints/openai/serving_video.py`
- Example: `examples/online_serving/image_to_video/`

## Validation
- Start server with `Wan-AI/Wan2.2-I2V-A14B-Diffusers`.
- Call `/v1/videos` with `input_reference` multipart file upload.
- Verify MP4 output decodes and plays.
