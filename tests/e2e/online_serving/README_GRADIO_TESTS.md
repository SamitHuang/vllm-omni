# Gradio Demo Unit Tests

This directory contains unit tests for the Qwen-Omni Gradio demos.

## Test Files

- `test_gradio_qwen2_5_omni.py` - Unit tests for Qwen2.5-Omni gradio demo
- `test_gradio_qwen3_omni.py` - Unit tests for Qwen3-Omni gradio demo

## Test Coverage

### Input Processing Tests
- ✅ Image to base64 conversion (RGB and non-RGB formats)
- ✅ Audio to base64 conversion (float32 and int16 formats)
- ✅ Video file to base64 conversion (multiple formats: mp4, webm, mov, avi, mkv)
- ✅ Image file processing and RGB conversion
- ✅ Audio file processing (tuple formats, file paths)
- ✅ Error handling for missing files

### Sampling Parameters Tests
- ✅ Building sampling params dictionary for supported models
- ✅ Error handling for unsupported models
- ✅ Seed parameter handling

### Argument Parsing Tests
- ✅ Default argument values
- ✅ Custom argument parsing
- ✅ All command-line options

### API Inference Tests
- ✅ Text-only inference
- ✅ Multimodal inference (image, audio, video)
- ✅ Mixed modalities
- ✅ Audio output handling
- ✅ Output modalities control
- ✅ Video with audio extraction
- ✅ Error handling (API errors, missing inputs)
- ✅ Message format validation

## Running Tests

### Run All Gradio Demo Tests

```bash
cd /path/to/vllm-omni
pytest tests/e2e/online_serving/test_gradio*.py -v
```

### Run Specific Test File

```bash
# Qwen2.5-Omni tests
pytest tests/e2e/online_serving/test_gradio_qwen2_5_omni.py -v

# Qwen3-Omni tests
pytest tests/e2e/online_serving/test_gradio_qwen3_omni.py -v
```

### Run Specific Test Classes or Tests

```bash
# Run all input processing tests
pytest tests/e2e/online_serving/test_gradio_qwen2_5_omni.py::TestInputProcessing -v

# Run a specific test
pytest tests/e2e/online_serving/test_gradio_qwen2_5_omni.py::TestInputProcessing::test_image_to_base64_data_url -v
```

### Run with Coverage

```bash
pytest tests/e2e/online_serving/test_gradio*.py -v --cov=examples/online_serving/qwen2_5_omni --cov=examples/online_serving/qwen3_omni --cov-report=html
```

## Test Strategy

### Mocking Strategy

1. **OpenAI Client**: The OpenAI client is mocked to avoid requiring a running API server
2. **File I/O**: Temporary files are used for testing file operations
3. **API Responses**: Mock responses simulate successful API calls and error scenarios

### What We Test

1. **Input Processing Functions**: 
   - Verify multimodal inputs are correctly converted to base64 data URLs
   - Test format conversions (RGB, audio formats, video MIME types)
   - Error handling for invalid inputs

2. **API Integration**:
   - Verify correct message format is sent to API
   - Test multimodal content is properly structured
   - Verify response parsing (text and audio outputs)
   - Test error handling

3. **Argument Parsing**:
   - Default values are correct
   - Custom arguments are parsed correctly

4. **Sampling Parameters**:
   - Parameters are built correctly for each model
   - Seed handling works properly

## Dependencies

The tests require:
- `pytest`
- `pytest-mock` (for mocking)
- `numpy`
- `PIL` (Pillow)
- `soundfile`
- `openai` (for type hints, but client is mocked)

## Notes

- Tests use mocking to avoid requiring actual model weights or a running API server
- All file operations use temporary files that are cleaned up automatically
- Tests are designed to run quickly in CI environments
- The tests verify the API integration layer, not the actual model inference



