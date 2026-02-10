# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for OpenAI-compatible video generation endpoints.
"""

import io
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo


class MockVideoResult:
    def __init__(self, videos):
        self.multimodal_output = {"video": videos}


class FakeAsyncOmni:
    def __init__(self):
        self.stage_list = ["diffusion"]
        self.captured_prompt = None
        self.captured_sampling_params_list = None

    async def generate(self, prompt, request_id, sampling_params_list):
        self.captured_prompt = prompt
        self.captured_sampling_params_list = sampling_params_list
        num_outputs = sampling_params_list[0].num_outputs_per_prompt
        videos = [object() for _ in range(num_outputs)]
        yield MockVideoResult(videos)


@pytest.fixture
def test_client():
    app = FastAPI()
    app.include_router(router)
    app.state.openai_serving_video = OmniOpenAIServingVideo.for_diffusion(
        diffusion_engine=FakeAsyncOmni(),
        model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )
    return TestClient(app)


def _make_test_image_bytes(size=(64, 64)) -> bytes:
    image = Image.new("RGB", size, color="blue")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_t2v_video_generation_form(test_client):
    fps_values = []

    def _fake_encode(video, fps):
        fps_values.append(fps)
        return "Zg=="

    with patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        side_effect=_fake_encode,
    ):
        response = test_client.post(
            "/v1/videos",
            data={
                "prompt": "A cat runs across the street.",
                "size": "640x360",
                "seconds": "2",
                "fps": "12",
                "n": "2",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "data" in data and len(data["data"]) == 2
    assert all(item["b64_json"] == "Zg==" for item in data["data"])

    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.num_outputs_per_prompt == 2
    assert captured.width == 640
    assert captured.height == 360
    assert captured.num_frames == 24
    assert captured.fps == 12
    assert fps_values == [12, 12]


def test_i2v_video_generation_form(test_client):
    image_bytes = _make_test_image_bytes((48, 32))

    with patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        return_value="Zg==",
    ):
        response = test_client.post(
            "/v1/videos",
            data={"prompt": "A bear playing with yarn."},
            files={"input_reference": ("input.png", image_bytes, "image/png")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "data" in data and len(data["data"]) == 1
    assert data["data"][0]["b64_json"] == "Zg=="

    engine = test_client.app.state.openai_serving_video._engine_client
    prompt = engine.captured_prompt
    assert "multi_modal_data" in prompt
    assert "image" in prompt["multi_modal_data"]
    input_image = prompt["multi_modal_data"]["image"]
    assert isinstance(input_image, Image.Image)
    assert input_image.size == (48, 32)
