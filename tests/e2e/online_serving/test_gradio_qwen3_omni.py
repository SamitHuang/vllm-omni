# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Qwen3-Omni Gradio demo.
Tests cover input processing, API client integration, and error handling.
"""

import base64
import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

# Add the examples directory to the path
examples_dir = Path(__file__).parent.parent.parent.parent / "examples" / "online_serving" / "qwen3_omni"
sys.path.insert(0, str(examples_dir))

from gradio_demo import (
    audio_to_base64_data_url,
    build_sampling_params_dict,
    image_to_base64_data_url,
    parse_args,
    process_audio_file,
    process_image_file,
    run_inference_api,
    video_to_base64_data_url,
)


class TestInputProcessing:
    """Tests for input processing functions."""

    def test_image_to_base64_data_url(self):
        """Test image to base64 conversion."""
        img = Image.new("RGB", (100, 100), color="red")
        
        result = image_to_base64_data_url(img)
        
        assert result.startswith("data:image/jpeg;base64,")
        
    def test_audio_to_base64_data_url(self):
        """Test audio to base64 conversion."""
        sample_rate = 16000
        audio_np = np.random.randn(16000).astype(np.float32)
        
        result = audio_to_base64_data_url((audio_np, sample_rate))
        
        assert result.startswith("data:audio/wav;base64,")
        
    def test_video_to_base64_data_url(self):
        """Test video file to base64 conversion."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(b"fake video content")
            tmp_path = tmp_file.name
            
        try:
            result = video_to_base64_data_url(tmp_path)
            assert result.startswith("data:video/mp4;base64,")
        finally:
            Path(tmp_path).unlink()


class TestSamplingParams:
    """Tests for sampling parameters building."""
    
    def test_build_sampling_params_dict(self):
        """Test building sampling params dictionary."""
        seed = 42
        model_key = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        
        result = build_sampling_params_dict(seed, model_key)
        
        assert isinstance(result, list)
        assert len(result) == 3  # thinker, talker, code2wav
        # Note: Qwen3 already has seed in templates, so we check if seed is present
        assert all("seed" in params for params in result)
        
    def test_build_sampling_params_dict_unsupported_model(self):
        """Test building params for unsupported model raises error."""
        with pytest.raises(ValueError, match="Unsupported model"):
            build_sampling_params_dict(42, "Unsupported/Model")


class TestArgumentParsing:
    """Tests for argument parsing."""
    
    def test_parse_args_defaults(self):
        """Test argument parsing with defaults."""
        with patch("sys.argv", ["gradio_demo.py"]):
            args = parse_args()
            
            assert args.model == "Qwen/Qwen3-Omni-30B-A3B-Instruct"
            assert args.api_base == "http://localhost:8091/v1"
            assert args.ip == "127.0.0.1"
            assert args.port == 7861
            assert args.share is False


class TestInferenceAPI:
    """Tests for API inference function."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mocked OpenAI client."""
        client = MagicMock()
        
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_choice.message.audio = None
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        
        client.chat.completions.create.return_value = mock_completion
        return client
        
    def test_run_inference_api_text_only(self, mock_openai_client):
        """Test inference with text-only input."""
        sampling_params = [{"temperature": 0.4, "max_tokens": 100}]
        
        text, audio = run_inference_api(
            mock_openai_client,
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            sampling_params,
            "Hello, world!",
        )
        
        assert text == "Test response"
        assert audio is None
        mock_openai_client.chat.completions.create.assert_called_once()
        
    def test_run_inference_api_with_image(self, mock_openai_client):
        """Test inference with image input."""
        sampling_params = [{"temperature": 0.4}]
        img = Image.new("RGB", (100, 100), color="red")
        
        text, audio = run_inference_api(
            mock_openai_client,
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            sampling_params,
            "Describe this image",
            image_file=img,
        )
        
        assert text == "Test response"
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert any(item.get("type") == "image_url" for item in user_content)
        
    def test_run_inference_api_error_handling(self, mock_openai_client):
        """Test inference error handling."""
        sampling_params = [{"temperature": 0.4}]
        
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        text, audio = run_inference_api(
            mock_openai_client,
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            sampling_params,
            "Test prompt",
        )
        
        assert "Inference failed" in text
        assert audio is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



