# Diffusion Chat Completions API

vLLM-Omni supports generating images via the `/v1/chat/completions` endpoint using diffusion models.
This page explains how to pass generation parameters (such as `num_inference_steps`, `height`, `width`)
to diffusion models through this endpoint across different client libraries.

!!! tip
    For text-to-image generation without chat context, the dedicated
    [`/v1/images/generations`](image_generation_api.md) endpoint accepts these
    parameters as top-level fields and may be simpler for your use case.

## API Endpoints Overview

vLLM-Omni provides multiple endpoints for diffusion models. Each has its own parameter-passing
convention:

| Endpoint | Use Case | Parameter Format |
|----------|----------|-----------------|
| `/v1/chat/completions` | Image gen/edit via chat | Generation params in `extra_body` (see below) |
| `/v1/images/generations` | Dedicated text-to-image | Top-level JSON fields |
| `/v1/images/edits` | Dedicated image editing | Multipart form fields |
| `/v1/videos` | Video generation | Multipart form fields |

## Passing Generation Parameters via `/v1/chat/completions`

The `/v1/chat/completions` endpoint follows the OpenAI Chat API schema, which does not natively
include diffusion-specific fields like `num_inference_steps` or `height`. vLLM-Omni accepts
these as **extra fields** on the request body.

There are two supported methods depending on your client:

### Method 1: Using curl or Python `requests`

Put generation parameters as **top-level fields** in the JSON body alongside `messages`:

=== "curl"

    ```bash
    curl -s http://localhost:8091/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "messages": [
          {"role": "user", "content": "A beautiful landscape painting"}
        ],
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "seed": 42
      }' | jq -r '.choices[0].message.content[0].image_url.url' \
         | cut -d',' -f2- | base64 -d > output.png
    ```

=== "Python requests"

    ```python
    import requests
    import base64

    payload = {
        "messages": [
            {"role": "user", "content": "A beautiful landscape painting"}
        ],
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "seed": 42,
    }

    resp = requests.post(
        "http://localhost:8091/v1/chat/completions",
        json=payload,
        timeout=300,
    )
    data = resp.json()

    img_url = data["choices"][0]["message"]["content"][0]["image_url"]["url"]
    _, b64_data = img_url.split(",", 1)
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(b64_data))
    ```

### Method 2: Using the OpenAI Python SDK

The OpenAI Python SDK uses the `extra_body` keyword argument to pass non-standard fields.
The SDK automatically merges these into the top-level request body:

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")

response = client.chat.completions.create(
    model="Qwen/Qwen-Image",
    messages=[
        {"role": "user", "content": "A beautiful landscape painting"}
    ],
    extra_body={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "seed": 42,
    },
)

img_url = response.choices[0].message.content[0].image_url.url
_, b64_data = img_url.split(",", 1)
with open("output.png", "wb") as f:
    f.write(base64.b64decode(b64_data))
```

### Legacy Format: Nested `extra_body` in JSON

You may see examples that nest generation parameters inside an `"extra_body"` key in the
JSON body. This format is still supported for backward compatibility:

```json
{
  "messages": [{"role": "user", "content": "A beautiful landscape painting"}],
  "extra_body": {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50
  }
}
```

Both formats (top-level fields and nested `extra_body`) are accepted.

!!! note "About the `ignored fields` warning"
    When sending non-standard fields, you may see a log message like:

    ```
    WARNING: The following fields were present in the request but ignored: {'height', 'width', ...}
    ```

    This warning is **harmless**. It is emitted by vLLM's request validation layer because
    these fields are not part of the standard OpenAI `ChatCompletionRequest` schema.
    The fields are still stored internally and correctly forwarded to the diffusion pipeline.

## Image Editing (Image-to-Image)

For image editing, include both text and image in the message content:

=== "curl"

    ```bash
    IMG_B64=$(base64 -w0 input.png)

    curl -s http://localhost:8092/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "messages": [{
          "role": "user",
          "content": [
            {"type": "text", "text": "Convert to watercolor style"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG_B64"'"}}
          ]
        }],
        "num_inference_steps": 50,
        "guidance_scale": 1,
        "seed": 42
      }' | jq -r '.choices[0].message.content[0].image_url.url' \
         | cut -d',' -f2 | base64 -d > output.png
    ```

=== "OpenAI SDK"

    ```python
    import base64
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8092/v1", api_key="none")

    with open("input.png", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="Qwen/Qwen-Image-Edit",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Convert to watercolor style"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_b64}"
                }},
            ],
        }],
        extra_body={
            "num_inference_steps": 50,
            "guidance_scale": 1,
            "seed": 42,
        },
    )

    img_url = response.choices[0].message.content[0].image_url.url
    _, b64_data = img_url.split(",", 1)
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(b64_data))
    ```

## Generation Parameters Reference

The following parameters are accepted as extra fields on `/v1/chat/completions` for
diffusion models:

| Parameter | Type | Description |
|-----------|------|-------------|
| `height` | int | Output image height in pixels |
| `width` | int | Output image width in pixels |
| `size` | str | Output size in "WxH" format (alternative to separate height/width) |
| `num_inference_steps` | int | Number of denoising steps |
| `guidance_scale` | float | Classifier-free guidance scale |
| `true_cfg_scale` | float | True CFG scale (Qwen-Image specific) |
| `seed` | int | Random seed for reproducibility |
| `negative_prompt` | str | Text describing what to avoid |
| `num_outputs_per_prompt` | int | Number of images to generate (default: 1) |
| `num_frames` | int | Number of frames (video models) |
| `guidance_scale_2` | float | Secondary guidance scale (Wan2.2 models) |
| `layers` | int | Number of layers to generate (Qwen-Image-Layered, default: 4) |
| `resolution` | int | Resolution for dimension calculation (Qwen-Image-Layered, 640 or 1024) |
| `lora` | object | Per-request LoRA adapter configuration |

!!! info "Model-specific defaults"
    When a parameter is not specified, the underlying diffusion pipeline applies its own
    model-specific default. For example, `num_inference_steps` defaults to 50 for most models
    but may differ for turbo/distilled variants.
