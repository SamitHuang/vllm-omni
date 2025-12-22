# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Qwen2.5-Omni Gradio demo.
Tests cover input processing, API client integration, and error handling.
"""

import base64
import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

# Add the examples directory to the path
examples_dir = Path(__file__).parent.parent.parent.parent / "examples" / "online_serving" / "qwen2_5_omni"
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
        # Create a test image
        img = Image.new("RGB", (100, 100), color="red")
        
        result = image_to_base64_data_url(img)
        
        assert result.startswith("data:image/jpeg;base64,")
        # Verify we can decode it
        b64_data = result.split(",")[1]
        decoded = base64.b64decode(b64_data)
        assert len(decoded) > 0
        
    def test_image_to_base64_converts_non_rgb(self):
        """Test that non-RGB images are converted to RGB."""
        img = Image.new("L", (100, 100), color=128)  # Grayscale
        
        result = image_to_base64_data_url(img)
        
        assert result.startswith("data:image/jpeg;base64,")
        
    def test_audio_to_base64_data_url(self):
        """Test audio to base64 conversion."""
        sample_rate = 16000
        duration = 1.0
        audio_np = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        result = audio_to_base64_data_url((audio_np, sample_rate))
        
        assert result.startswith("data:audio/wav;base64,")
        # Verify we can decode it
        b64_data = result.split(",")[1]
        decoded = base64.b64decode(b64_data)
        assert len(decoded) > 0
        
    def test_audio_to_base64_int16_format(self):
        """Test audio conversion handles int16 format."""
        sample_rate = 16000
        audio_np = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
        
        result = audio_to_base64_data_url((audio_np, sample_rate))
        
        assert result.startswith("data:audio/wav;base64,")
        
    def test_video_to_base64_data_url(self):
        """Test video file to base64 conversion."""
        # Create a temporary video file (actually just a dummy file)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(b"fake video content")
            tmp_path = tmp_file.name
            
        try:
            result = video_to_base64_data_url(tmp_path)
            
            assert result.startswith("data:video/mp4;base64,")
            b64_data = result.split(",")[1]
            decoded = base64.b64decode(b64_data)
            assert decoded == b"fake video content"
        finally:
            Path(tmp_path).unlink()
            
    def test_video_to_base64_different_formats(self):
        """Test video MIME type detection for different formats."""
        formats = [".mp4", ".webm", ".mov", ".avi", ".mkv"]
        
        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=fmt, delete=False) as tmp_file:
                tmp_file.write(b"fake content")
                tmp_path = tmp_file.name
                
            try:
                result = video_to_base64_data_url(tmp_path)
                expected_mime = {
                    ".mp4": "video/mp4",
                    ".webm": "video/webm",
                    ".mov": "video/quicktime",
                    ".avi": "video/x-msvideo",
                    ".mkv": "video/x-matroska",
                }[fmt]
                assert result.startswith(f"data:{expected_mime};base64,")
            finally:
                Path(tmp_path).unlink()
                
    def test_video_to_base64_file_not_found(self):
        """Test video conversion raises error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            video_to_base64_data_url("/nonexistent/video.mp4")
            
    def test_process_image_file(self):
        """Test image file processing."""
        img = Image.new("RGB", (100, 100), color="blue")
        
        result = process_image_file(img)
        
        assert result is not None
        assert result.mode == "RGB"
        
    def test_process_image_file_none(self):
        """Test image processing with None input."""
        result = process_image_file(None)
        assert result is None
        
    def test_process_image_file_converts_to_rgb(self):
        """Test that non-RGB images are converted."""
        img = Image.new("L", (100, 100), color=128)
        
        result = process_image_file(img)
        
        assert result.mode == "RGB"
        
    def test_process_audio_file_tuple_format(self):
        """Test audio processing with tuple format (sample_rate, array)."""
        sample_rate = 16000
        audio_np = np.random.randn(16000).astype(np.float32)
        audio_input = (sample_rate, audio_np)
        
        result = process_audio_file(audio_input)
        
        assert result is not None
        assert result[0].shape == audio_np.shape
        assert result[1] == sample_rate
        
    def test_process_audio_file_none(self):
        """Test audio processing with None input."""
        result = process_audio_file(None)
        assert result is None
        
    def test_process_audio_file_filepath(self, tmp_path):
        """Test audio processing with file path."""
        # Create a temporary audio file
        audio_file = tmp_path / "test.wav"
        sample_rate = 16000
        audio_data = np.random.randn(16000).astype(np.float32)
        sf.write(str(audio_file), audio_data, sample_rate)
        
        result = process_audio_file(str(audio_file))
        
        assert result is not None
        assert result[1] == sample_rate


class TestSamplingParams:
    """Tests for sampling parameters building."""
    
    def test_build_sampling_params_dict(self):
        """Test building sampling params dictionary."""
        seed = 42
        model_key = "Qwen/Qwen2.5-Omni-7B"
        
        result = build_sampling_params_dict(seed, model_key)
        
        assert isinstance(result, list)
        assert len(result) == 3  # thinker, talker, code2wav
        assert all("seed" in params for params in result)
        assert all(params["seed"] == seed for params in result)
        
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
            
            assert args.model == "Qwen/Qwen2.5-Omni-7B"
            assert args.api_base == "http://localhost:8091/v1"
            assert args.ip == "127.0.0.1"
            assert args.port == 7861
            assert args.share is False
            
    def test_parse_args_custom_values(self):
        """Test argument parsing with custom values."""
        with patch("sys.argv", [
            "gradio_demo.py",
            "--model", "Custom/Model",
            "--api-base", "http://remote:8000/v1",
            "--ip", "0.0.0.0",
            "--port", "9000",
            "--share",
        ]):
            args = parse_args()
            
            assert args.model == "Custom/Model"
            assert args.api_base == "http://remote:8000/v1"
            assert args.ip == "0.0.0.0"
            assert args.port == 9000
            assert args.share is True


class TestInferenceAPI:
    """Tests for API inference function."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mocked OpenAI client."""
        client = MagicMock()
        
        # Mock chat completion response
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_choice.message.audio = None
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        
        client.chat.completions.create.return_value = mock_completion
        return client
        
    def test_run_inference_api_text_only(self, mock_openai_client):
        """Test inference with text-only input."""
        sampling_params = [{"temperature": 0.0, "max_tokens": 100}]
        
        text, audio = run_inference_api(
            mock_openai_client,
            "Qwen/Qwen2.5-Omni-7B",
            sampling_params,
            "Hello, world!",
        )
        
        assert text == "Test response"
        assert audio is None
        mock_openai_client.chat.completions.create.assert_called_once()
        
    def test_run_inference_api_with_image(self, mock_openai_client):
        """Test inference with image input."""
        sampling_params = [{"temperature": 0.0}]
        img = Image.new("RGB", (100, 100), color="red")
        
        text, audio = run_inference_api(
            mock_openai_client,
            "Qwen/Qwen2.5-Omni-7B",
            sampling_params,
            "Describe this image",
            image_file=img,
        )
        
        assert text == "Test response"
        # Verify the API was called with image in content
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert any(item.get("type") == "image_url" for item in user_content)
        
    def test_run_inference_api_with_audio(self, mock_openai_client):
        """Test inference with audio input."""
        sampling_params = [{"temperature": 0.0}]
        # Use tuple format: (sample_rate, np.ndarray)
        audio_data = (16000, np.random.randn(16000).astype(np.float32))
        
        text, audio = run_inference_api(
            mock_openai_client,
            "Qwen/Qwen2.5-Omni-7B",
            sampling_params,
            "Transcribe this audio",
            audio_file=audio_data,
        )
        
        assert text == "Test response"
        # Verify the API was called with audio in content
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert any(item.get("type") == "audio_url" for item in user_content)
        
    def test_run_inference_api_with_video(self, mock_openai_client, tmp_path):
        """Test inference with video input."""
        sampling_params = [{"temperature": 0.0}]
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video content")
        
        text, audio = run_inference_api(
            mock_openai_client,
            "Qwen/Qwen2.5-Omni-7B",
            sampling_params,
            "Describe this video",
            video_file=str(video_file),
        )
        
        assert text == "Test response"
        # Verify the API was called with video in content
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert any(item.get("type") == "video_url" for item in user_content)
        
    def test_run_inference_api_with_audio_output(self, mock_openai_client):
        """Test inference with audio output."""
        sampling_params = [{"temperature": 0.0}]
        
        # Mock audio response
        audio_bytes = b"fake audio wav data"
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_audio = MagicMock()
        mock_audio.data = audio_b64
        mock_choice.message.audio = mock_audio
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_openai_client.chat.completions.create.return_value = mock_completion
        
        # Create a valid WAV file in memory
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            sample_rate = 16000
            audio_data = np.random.randn(16000).astype(np.float32)
            sf.write(tmp_wav.name, audio_data, sample_rate)
            with open(tmp_wav.name, "rb") as f:
                valid_wav_bytes = f.read()
            Path(tmp_wav.name).unlink()
        
        # Update mock to return valid WAV
        valid_audio_b64 = base64.b64encode(valid_wav_bytes).decode("utf-8")
        mock_audio.data = valid_audio_b64
        
        text, audio = run_inference_api(
            mock_openai_client,
            "Qwen/Qwen2.5-Omni-7B",
            sampling_params,
            "Generate audio",
        )
        
        assert text == "No text output."
        assert audio is not None
        assert isinstance(audio, tuple)
        assert len(audio) == 2
        
    def test_run_inference_api_no_inputs(self, mock_openai_client):
        """Test inference with no inputs returns error message."""
        sampling_params = [{"temperature": 0.0}]
        
        text, audio = run_inference_api(
            mock_openai_client,
            "Qwen/Qwen2.5-Omni-7B",
            sampling_params,
            "",  # Empty prompt
        )
        
        assert "Please provide" in text
        assert audio is None
        mock_openai_client.chat.completions.create.assert_not_called()
        
    def test_run_inference_api_error_handling(self, mock_openai_client):
        """Test inference error handling."""
        sampling_params = [{"temperature": 0.0}]
        
        # Make the API call raise an exception
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        text, audio = run_inference_api(
            mock_openai_client,
            "Qwen/Qwen2.5-Omni-7B",
            sampling_params,
            "Test prompt",
        )
        
        assert "Inference failed" in text
        assert audio is None
        
    def test_run_inference_api_output_modalities(self, mock_openai_client):
        """Test inference with output modalities specified."""
        sampling_params = [{"temperature": 0.0}]
        
        text, audio = run_inference_api(
            mock_openai_client,
            "Qwen/Qwen2.5-Omni-7B",
            sampling_params,
            "Test prompt",
            output_modalities="text,audio",
        )
        
        # Verify modalities were passed to API
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args.kwargs["modalities"] == ["text", "audio"]
        
    def test_run_inference_api_use_audio_in_video(self, mock_openai_client, tmp_path):
        """Test inference with video audio extraction enabled."""
        sampling_params = [{"temperature": 0.0}]
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video content")
        
        text, audio = run_inference_api(
            mock_openai_client,
            "Qwen/Qwen2.5-Omni-7B",
            sampling_params,
            "Test prompt",
            video_file=str(video_file),
            use_audio_in_video=True,
        )
        
        # Verify mm_processor_kwargs were set
        call_args = mock_openai_client.chat.completions.create.call_args
        extra_body = call_args.kwargs["extra_body"]
        assert extra_body["mm_processor_kwargs"]["use_audio_in_video"] is True
        # Verify video content has num_frames
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        video_item = next(item for item in user_content if item.get("type") == "video_url")
        assert "num_frames" in video_item["video_url"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

