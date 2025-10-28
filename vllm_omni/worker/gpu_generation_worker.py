"""Code2Wav GPU Worker for vLLM-omni.

Specialized worker for Qwen3 Omni MoE Code2Wav model that converts
codec codes directly to audio waveforms.
"""

import gc
import os

import torch
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.utils import GiB_bytes, MemorySnapshot
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker as GPUWorker
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.worker.worker import _check_if_gpu_supports_dtype

from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner


class GPUGenerationWorker(GPUWorker):
    """GPU Worker for Generation model (non-autoregressive waveform generation).
    
    Usage in stage config:
        worker_cls: "vllm_omni.worker.gpu_generation_model_runner.GPUGenerationModelRunner"
    
    The worker expects codec codes to be provided in the request's
    additional_information field under the 'codes' or 'codec_codes' key.
    
    Example stage config:
        ```yaml
        - stage_id: 2
          engine_args:
            model_stage: 2
            model_arch: "Qwen3OmniMoeCode2Wav"
            worker_cls: "vllm_omni.worker.gpu_generation_model_runner.GPUGenerationWorker"
            scheduler_cls: "vllm_omni.core.omni_scheduler.OmniScheduler"
            gpu_memory_utilization: 0.90
            engine_output_type: "audio"
          engine_input_source: [1]
          custom_process_input_func: "vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav"
          final_output: true
          final_output_type: "audio"
        ```
    """
    
    def init_device(self):
        """Initialize CUDA device and distributed environment."""
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            current_platform.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()

            # Take current memory snapshot
            self.init_snapshot = MemorySnapshot()
            self.requested_memory = (
                self.init_snapshot.total_memory
                * self.cache_config.gpu_memory_utilization
            )
            if self.init_snapshot.free_memory < self.requested_memory:

                def GiB(b):
                    return round(b / GiB_bytes, 2)

                raise ValueError(
                    f"Free memory on device "
                    f"({GiB(self.init_snapshot.free_memory)}/"
                    f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                    f"is less than desired GPU memory utilization "
                    f"({self.cache_config.gpu_memory_utilization}, "
                    f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                    f"utilization or reduce GPU memory used by other processes."
                )
        else:
            raise RuntimeError(
                f"Not supported device type: {self.device_config.device}"
            )
        
        # Initialize the distributed environment
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        
        # Set random seed
        set_random_seed(self.model_config.seed)

        # Construct the generation model runner
        self.model_runner: GPUGenerationModelRunner = GPUGenerationModelRunner(
            self.vllm_config, self.device
        )

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info
            report_usage_stats(self.vllm_config)

