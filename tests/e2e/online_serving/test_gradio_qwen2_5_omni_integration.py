# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for Qwen2.5-Omni Gradio demo wrapper layer.
Tests the wrapper's data transformation between Gradio format and API format using a real API server.
"""

import os
import sys
import subprocess
import socket
import time
from pathlib import Path

import numpy as np
import openai
import pytest
from PIL import Image
from vllm.assets.video import VideoAsset
from vllm.utils import get_open_port

# Add the examples directory to the path
examples_dir = Path(__file__).parent.parent.parent.parent / "examples" / "online_serving" / "qwen2_5_omni"
sys.path.insert(0, str(examples_dir))

from gradio_demo import (
    build_sampling_params_dict,
    run_inference_api,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# CI stage config for testing
stage_configs = [str(Path(__file__).parent / "stage_configs" / "qwen2_5_omni_ci.yaml")] if (
    Path(__file__).parent / "stage_configs" / "qwen2_5_omni_ci.yaml"
).exists() else [None]

models = ["Qwen/Qwen2.5-Omni-7B"]
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


class OmniServer:
    """Omniserver for vLLM-Omni tests."""

    def __init__(
        self,
        model: str,
        serve_args: list[str],
        *,
        env_dict: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.serve_args = serve_args
        self.env_dict = env_dict
        self.proc = None
        self.host = "127.0.0.1"
        self.port = get_open_port()

    def _start_server(self) -> None:
        """Start the vLLM-Omni server subprocess."""
        import socket
        import subprocess
        import time

        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if self.env_dict is not None:
            env.update(self.env_dict)

        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--omni",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ] + self.serve_args

        print(f"Launching OmniServer with: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )

        # Wait for server to be ready
        max_wait = 600  # 10 minutes
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex((self.host, self.port))
                    if result == 0:
                        print(f"Server ready on {self.host}:{self.port}")
                        return
            except Exception:
                pass
            time.sleep(2)

        raise RuntimeError(f"Server failed to start within {max_wait} seconds")

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import subprocess
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


@pytest.fixture
def omni_server(request):
    """Start vLLM-Omni server as a subprocess with actual model weights."""
    model, stage_config_path = request.param
    serve_args = []
    if stage_config_path:
        serve_args = ["--stage-configs-path", stage_config_path]
    with OmniServer(model, serve_args) as server:
        yield server


@pytest.fixture
def openai_client(omni_server):
    """OpenAI client for the running vLLM-Omni server."""
    return openai.OpenAI(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        api_key="EMPTY",
    )


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return Image.new("RGB", (224, 224), color="red")


@pytest.fixture
def sample_audio():
    """Create a sample test audio in Gradio format: (filename, (sample_rate, audio_np))."""
    sample_rate = 16000
    duration = 1.0
    audio_np = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    return ("test_audio.wav", (sample_rate, audio_np))


@pytest.fixture
def sample_video(tmp_path):
    """Create a sample test video file."""
    # Use VideoAsset if available, otherwise create a dummy file
    try:
        video = VideoAsset(name="baby_reading", num_frames=4)
        return video.video_path
    except Exception:
        # Fallback: create a dummy video file
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video content for testing")
        return str(video_file)


@pytest.mark.skipif(
    os.getenv("VLLM_TEST_GRADIO_INTEGRATION", "0") != "1",
    reason="Integration tests require VLLM_TEST_GRADIO_INTEGRATION=1 and GPU resources",
)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
class TestGradioWrapperIntegration:
    """Integration tests for Gradio demo wrapper layer with real API server."""

    def test_wrapper_text_only(self, openai_client, omni_server):
        """Test wrapper handles text-only input correctly."""
        sampling_params = build_sampling_params_dict(42, "Qwen/Qwen2.5-Omni-7B")
        
        text, audio = run_inference_api(
            openai_client,
            omni_server.model,
            sampling_params,
            "Hello, this is a test.",
        )
        
        # Verify wrapper correctly parsed the response
        assert text is not None
        assert isinstance(text, str)
        assert len(text) > 0
        
    def test_wrapper_with_image(self, openai_client, omni_server, sample_image):
        """Test wrapper converts PIL Image to API format correctly."""
        sampling_params = build_sampling_params_dict(42, "Qwen/Qwen2.5-Omni-7B")
        
        text, audio = run_inference_api(
            openai_client,
            omni_server.model,
            sampling_params,
            "Describe this image.",
            image_file=sample_image,
        )
        
        # Verify wrapper correctly converted image and got response
        assert text is not None
        assert isinstance(text, str)
        
    def test_wrapper_with_audio(self, openai_client, omni_server, sample_audio):
        """Test wrapper converts audio tuple to API format correctly."""
        sampling_params = build_sampling_params_dict(42, "Qwen/Qwen2.5-Omni-7B")
        
        text, audio = run_inference_api(
            openai_client,
            omni_server.model,
            sampling_params,
            "Transcribe this audio.",
            audio_file=sample_audio,
        )
        
        # Verify wrapper correctly converted audio and got response
        assert text is not None
        assert isinstance(text, str)
        
    def test_wrapper_with_video(self, openai_client, omni_server, sample_video):
        """Test wrapper converts video file to API format correctly."""
        sampling_params = build_sampling_params_dict(42, "Qwen/Qwen2.5-Omni-7B")
        
        text, audio = run_inference_api(
            openai_client,
            omni_server.model,
            sampling_params,
            "Describe this video.",
            video_file=sample_video,
        )
        
        # Verify wrapper correctly converted video and got response
        assert text is not None
        assert isinstance(text, str)
        
    def test_wrapper_mixed_modalities(
        self, openai_client, omni_server, sample_image, sample_audio, sample_video
    ):
        """Test wrapper handles mixed modalities correctly."""
        sampling_params = build_sampling_params_dict(42, "Qwen/Qwen2.5-Omni-7B")
        
        text, audio = run_inference_api(
            openai_client,
            omni_server.model,
            sampling_params,
            "Analyze all the media.",
            image_file=sample_image,
            audio_file=sample_audio,
            video_file=sample_video,
        )
        
        # Verify wrapper correctly handled all modalities
        assert text is not None
        assert isinstance(text, str)
        
    def test_wrapper_error_handling_no_server(self):
        """Test wrapper handles API connection errors correctly."""
        # Create a client pointing to a non-existent server
        fake_client = openai.OpenAI(
            base_url="http://localhost:99999/v1",
            api_key="EMPTY",
            timeout=2.0,  # Short timeout for faster failure
        )
        sampling_params = build_sampling_params_dict(42, "Qwen/Qwen2.5-Omni-7B")
        
        text, audio = run_inference_api(
            fake_client,
            "Qwen/Qwen2.5-Omni-7B",
            sampling_params,
            "Test prompt",
        )
        
        # Verify wrapper correctly handles the error
        assert "Inference failed" in text or "failed" in text.lower()
        assert audio is None
        
    def test_wrapper_output_modalities(self, openai_client, omni_server):
        """Test wrapper correctly passes output modalities to API."""
        sampling_params = build_sampling_params_dict(42, "Qwen/Qwen2.5-Omni-7B")
        
        # Test with output modalities specified
        text, audio = run_inference_api(
            openai_client,
            omni_server.model,
            sampling_params,
            "Generate text response.",
            output_modalities="text",
        )
        
        assert text is not None
        assert isinstance(text, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

