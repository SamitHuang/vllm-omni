## Diffusion Overview and Integration Design

This docs introduce mainstream diffusion libraries/frameworks. And seek to support them in vllm-omni

## diffusers 

###  Usage

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="cuda"
)

prompt = "xxx"
image = pipeline(prompt).images[0]
```

### Core Design and API 

- `DiffusionPipeline`: Top-level orchestrator that bundles tokenizer/text encoder, UNet/DiT backbone, scheduler, VAE/decoder.
- `SchedulerMixin`: Numerical solvers (DDIM, Euler, DPM++, FlowMatch, etc.) interchangeable at runtime.
- `ModelMixin`: Building blocks such as `UNet2DConditionModel`, `DiTForImageGeneration`, `AutoencoderKL`.

Main diffusion pipeline steps:
1. **prepare latents**: prepare the latent noise. For qwenimage, it's a tensor of shape (bs, img_seq_len, num_channel)
2. **prepare conditions**: get the conditions, such as text prompt embeding, reference image embedding, speaker embedding. Expect to parse from input args.
3. **prepare timesteps**: prepare the timesteps for diffusion noise scheduling
4. **denoise loop**: iteratively update the latent based on the timestep and model predictive output, optionally use CFG (condition-free guidance) which requires to double the batch size
5. **postprocess**: unpatchify and decode the denoised latents to the original modality such as image/audio/video. The decoding model is typically a VAE (VAE for QwenImage, GAN for QwenOmni)


## Cache-DiT


CacheDiT is a unified, flexible, and training-free cache acceleration framework designed to support nearly all Diffusers’ DiT-based pipelines. It provides a unified cache API that supports automatic block adapter, DBCache, and more.

https://github.com/modelscope/cache-dit

https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit


###  Usage
```python
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")
cache_dit.enable_cache(pipe)
image = pipe(prompt=prompt)
```

pipeline `from_pretrained` and `__call__` args are compatible with diffusers


### Core Design and API 

- monkey patch diffusers `DiffusionPipeline.__call__()`, and transformers `forward()` function to achieve cache optimization



### Supported Model List 

nearly all of Diffusers' DiT-based pipelines, include 30+ series, nearly 100+ pipelines, such as FLUX.1, Qwen-Image, Qwen-Image-Lightning, Wan 2.1/2.2


## FastVideo

A unified post-training and inference framework for accelerated video generation
https://github.com/hao-ai-lab/FastVideo

###  Usage

```python
from fastvideo import VideoGenerator
generator = VideoGenerator.from_pretrained(
        "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,  # Adjust based on your hardware
    )
video = generator.generate_video(
        prompt,
        return_frames=True,  # Also return frames from this call (defaults to False)
        output_path="my_videos/",  # Controls where videos are saved
        save_video=True
    )
```

pipeline `from_pretrained` and `generate_video` args are NOT compatible with diffusers


### Core Design and API 

- similar design to vllm
- ComposablePipeline and PipelineStage Abstraction

```python
from fastvideo.v1.pipelines.stages import (ConditioningStage, DecodingStage,
                                           DenoisingStage, InputValidationStage,
                                           LatentPreparationStage,
                                           TextEncodingStage,
                                           TimestepPreparationStage)

class WanPipeline(ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer")))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))
```
- need to **re-write the model implementation and pipeline** !! (harder to integrate) 
- map HF model arch via ModelRegistry like vllm
```
_TEXT_TO_VIDEO_DIT_MODELS = {
    "HunyuanVideoTransformer3DModel":
        ("dits", "hunyuanvideo", "HunyuanVideoTransformer3DModel"),
    "WanTransformer3DModel":
        ("dits", "wanvideo", "WanTransformer3DModel"),
    "StepVideoModel":
        ("dits", "stepvideo", "StepVideoModel"),
    ...
}
```



https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/pipelines/composed_pipeline_base.py

https://hao-ai-lab.github.io/blogs/fastvideo/
 


### Supported Model List 

SGLang Diffusion support: https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/support_matrix.md

Different from FastVideo, SGLang Diffusion additionally supports image generation models QwenImage and Flux.dev, but lack optimization.



## General Diffusion Generator API for vllm-omni

- Case 1: no text encoder disaggr.

```yaml
# vllm-omni/vllm_omni/model_executor/stage_configs/diffusers.yaml 
stage_args:
  - stage_id: 0
    runtime:
      process: true
      devices: "0"            # TODO: changes for different devices, allow parse via CLI args 
      max_batch_size: 1
    engine_args:
      model_stage: all
      model_arch: QwenImagePipeline #  ==> diff
      optimization:         #  DIFF: parse via stage_config?
        backend: cache-dit
        cache_config: "{max_warmup_steps: 8, ...}"
      worker_cls: vllm_omni.worker.gpu_diffusion_worker.GPUDiffusionWorker
      scheduler_cls: vllm_omni.core.sched.diffusion_scheduler.DiffusionScheduler
      gpu_memory_utilization: 0.15
      enforce_eager: true
      trust_remote_code: true
      enable_prefix_caching: false
      engine_output_type: audio
    final_output: true
    final_output_type: image
```

```python
_OMNI_MODELS = {
    "Qwen2_5OmniForConditionalGeneration": (
        "qwen2_5_omni",
        "Qwen2_5OmniForConditionalGeneration",
    ),
    "QwenImagePipeline": (
        "diffusers_generator",
        "BaseDiffusionGenerator",
    ),
}
```

- Solution 1: One univerasal diffusion model executor

```python
# vllm_omni/model_executor/models/diffusers_pipeline.py
from diffusers import DiffusionPipeline

class BaseDiffusionGenerator():  # registry 
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        dit_config= self.get_diffusion_config(vllm_config)
        self.pipeline = DiffusionPipeline.from_pretrained(
            **dit_config, 
            # "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="cuda"
        )

        if vllm_config.optimization == 'cache-dit':
            cache_dit.enable_cache(self.pipeline)

    def get_diffusion_config(self, vllm_config):
        # extract diffusers config from vllm_config, including model name, dtype, etc
        ...
        return dit_config

    def sample(self, prompts, ...):
        images = self.pipeline(prompts).images

        return images
```

- Solution 2: BaseDiffusionGenerator, inherit to multiple XXBackendDiffusionGenerator e.g. CacheDiTDiffusionGenerator

```python
import cache_dit

class DiTCacheGenerator(BaseDiffusionGenerator):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()  
        cache_dit.enable_cache(self.pipeline)
 
    def sample(self, prompts, ...):
        images = self.pipeline(prompts).images

        return images
```

issue: can acheive one-to-many mapping via model registry, same model_arch + different backend option --> different  model execution class 

Requres Changes:
1. modify `get_model_architecture()`, to parse `_class_name` from HF diffusers config
2. 

- Case 2: disaggregate text encoder

rm text encode moduel in init

pass prompt_embeds instead of prompts in sample


- Case 3: disaggregate text encoder and vae decoder

TBC

### current state

vllm-omni/vllm_omni/model_executor/models/qwen2_5_omni_token2wav.py








