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

DiffusionPipeline

## Cache-DiT

###  Usage
```python
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")
cache_dit.enable_cache(pipe)
image = pipe(prompt=prompt)
```

### Core Design and API 

### Supported Model List 

Nearly all models in diffusers


## FastVideo

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

### Core Design and API 
fastvideo/pipelines/composed_pipeline_base.py

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
      devices: "0"            # Example: use a different GPU than the previous stage; use "0" if single GPU
      max_batch_size: 1
    engine_args:
      model_stage: e2e
      model_arch: DiffusersGenerator
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
from diffusers import DiffusionPipeline

class DiffusersGenerator():  # registry 
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        dit_config= self.get_diffusion_config(vllm_config)
        self.pipeline = DiffusionPipeline.from_pretrained(
            **dit_config, 
            # "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="cuda"
        )

    def get_diffusion_config(self, vllm_config):
        # extract diffusers config from vllm_config, including model name, dtype, etc
        ...
        return dit_config

    def sample(self, prompts, ...):
        images = self.pipeline(prompts).images

        return images

import cache_dit

class DiTCacheGenerator(DiffusersGenerator):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()  
        cache_dit.enable_cache(self.pipeline)
 
    def sample(self, prompts, ...):
        images = self.pipeline(prompts).images

        return images
```


- Case 2: disaggregate text encoder

rm text encode moduel in init

pass prompt_embeds instead of prompts in sample


- Case 3: disaggregate text encoder and vae decoder

tbc

### current status

vllm-omni/vllm_omni/model_executor/models/qwen2_5_omni_token2wav.py








