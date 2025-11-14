"""
Diffusers Pipeline Generator for vllm-omni.

Wraps diffusers DiffusionPipeline to be compatible with vllm-omni interface.
"""

from typing import Dict, Any, Optional, Union, List
import torch
from diffusers import DiffusionPipeline
from vllm.config import VllmConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm_omni.logger import init_logger

logger = init_logger(__name__)


class DiffusersPipelineGenerator:
    """
    Universal diffusion model executor (Solution 1).
    
    Wraps diffusers DiffusionPipeline and automatically applies cache-dit optimization
    if enabled in configuration. This follows Solution 1 from the design doc:
    one universal executor that handles both normal and optimized cases.
    """
    
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = vllm_config.model
        backend_config = getattr(vllm_config.model_config, "backend_config", {}) or {}

        # Initialize pipeline
        self.pipeline = self._load_pipeline(
            pipeline_kwargs=dict(backend_config.get("pipeline_config", {}))
        )

        # Apply backend-specific optimization if enabled (Solution 1 approach)
        self._apply_backend_config(backend_config)

    def _load_pipeline(
        self,
        pipeline_kwargs: Dict[str, Any],
    ) -> DiffusionPipeline:
        """Load diffusers pipeline using DiffusionPipeline.from_pretrained."""
        load_kwargs = dict(pipeline_kwargs or {})

        torch_dtype = load_kwargs.pop(
            "torch_dtype", self.vllm_config.model_config.dtype
        )
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)

        device_map = load_kwargs.pop("device_map", None)

        logger.info("Loading DiffusionPipeline from %s", self.model_path)

        pipeline = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=getattr(self.vllm_config, "trust_remote_code", False),
            **load_kwargs,
        )

        # Move to device
        pipeline = pipeline.to(self.device)
        
        return pipeline
    
    def sample(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        multimodal_kwargs: Optional[Dict[str, Any]] = None,
        sampling_metadata: Optional[SamplingMetadata] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Execute diffusion generation (as per design doc Solution 1).
        
        Args:
            prompts: Prompt string(s) for generation
            multimodal_kwargs: Multimodal inputs (images, audio, etc.)
            sampling_metadata: Sampling metadata (used to extract prompts if not provided)
            **kwargs: Other diffusion parameters (num_inference_steps, guidance_scale, height, width, etc.)
            
        Returns:
            Generated output (images, videos, audio, or latents)
        """
        # Extract prompts from sampling_metadata if not provided
        if prompts is None and sampling_metadata is not None:
            prompts_list = self._extract_prompts(sampling_metadata)
            # Convert to single string if only one prompt, or keep as list
            if len(prompts_list) == 1:
                prompts = prompts_list[0]
            elif len(prompts_list) > 1:
                prompts = prompts_list
            else:
                prompts = None
        
        # Convert single string to list for _prepare_pipeline_kwargs
        prompts_list = prompts if isinstance(prompts, list) else ([prompts] if prompts else [])
        
        # Prepare pipeline call arguments
        pipe_kwargs = self._prepare_pipeline_kwargs(
            prompts=prompts_list,
            multimodal_kwargs=multimodal_kwargs,
            **kwargs,
        )
        
        # Call pipeline
        output = self.pipeline(**pipe_kwargs)
        
        # Extract output
        return self._extract_output(output, kwargs.get("output_type"))
    
    def _extract_prompts(
        self,
        sampling_metadata: Optional[SamplingMetadata],
    ) -> List[str]:
        """Extract prompts from sampling_metadata"""
        if sampling_metadata is None:
            return []
        
        prompts = []
        # Extract prompts from seq_groups
        # This depends on the actual structure of SamplingMetadata
        if hasattr(sampling_metadata, "seq_groups"):
            for seq_group in sampling_metadata.seq_groups:
                # Try to get prompt from request
                if hasattr(seq_group, "request"):
                    prompt = getattr(seq_group.request, "prompt", None)
                    if prompt:
                        prompts.append(prompt)
        
        return prompts
    
    def _prepare_pipeline_kwargs(
        self,
        prompts: List[str],
        multimodal_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare pipeline call arguments"""
        pipe_kwargs = {}
        
        # Handle prompt
        if prompts:
            if len(prompts) == 1:
                pipe_kwargs["prompt"] = prompts[0]
            else:
                pipe_kwargs["prompt"] = prompts
        
        # Handle multimodal inputs
        if multimodal_kwargs:
            if "image" in multimodal_kwargs:
                pipe_kwargs["image"] = multimodal_kwargs["image"]
            if "video" in multimodal_kwargs:
                pipe_kwargs["video"] = multimodal_kwargs["video"]
        
        # Diffusion parameters
        pipe_kwargs.update({
            "num_inference_steps": kwargs.get("num_inference_steps", 50),
            "guidance_scale": kwargs.get("guidance_scale", 7.5),
            "height": kwargs.get("height"),
            "width": kwargs.get("width"),
            "num_frames": kwargs.get("num_frames"),
            "generator": kwargs.get("generator"),
            "output_type": kwargs.get("output_type", "pil"),
            "return_dict": kwargs.get("return_dict", True),
        })
        
        # Remove None values
        pipe_kwargs = {k: v for k, v in pipe_kwargs.items() if v is not None}
        
        return pipe_kwargs
    
    def _extract_output(
        self,
        output: Any,
        output_type: Optional[str] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Extract output from pipeline result"""
        if output_type == "latent":
            # Return latents if available
            if hasattr(output, "latents"):
                return output.latents
            elif isinstance(output, dict) and "latents" in output:
                return output["latents"]
            else:
                logger.warning(
                    "output_type='latent' requested but pipeline doesn't return latents"
                )
        
        # Extract based on output type
        if hasattr(output, "images"):
            return output.images
        elif hasattr(output, "frames"):
            return output.frames
        elif hasattr(output, "audios"):
            return output.audios
        elif isinstance(output, dict):
            # Try common keys
            for key in ["images", "frames", "audios", "latents"]:
                if key in output:
                    return output[key]
        
        # Return as-is
        return output
    
    
    def _apply_backend_config(self, backend_config: Dict[str, Any]):
        """
        Apply backend-specific optimization based on model_backend/backend_config.
        """
        backend = getattr(self.vllm_config.model_config, "model_backend", "diffusers")

        if backend in (None, "diffusers"):
            return

        if backend == "cache-dit":
            self._apply_cache_dit(backend_config)
        else:
            logger.warning(
                "Model backend '%s' is not supported yet. "
                "Pipeline will run without additional optimization.",
                backend,
            )

    def _apply_cache_dit(self, backend_config: Dict[str, Any]):
        """Apply cache-dit optimization."""
        # Try to import cache-dit
        try:
            import cache_dit
            from cache_dit import DBCacheConfig, ParallelismConfig, ParallelismBackend
        except ImportError:
            logger.warning(
                "cache-dit not available. Install with: pip install cache-dit. "
                "Pipeline will run without cache optimization."
            )
            return

        cache_config_dict = backend_config.get("cache_config", {})
        parallel_config_dict = backend_config.get("parallel_config")

        try:
            cache_config = (
                DBCacheConfig(**cache_config_dict) if cache_config_dict else DBCacheConfig()
            )
        except Exception as e:
            logger.warning(
                "Failed to create DBCacheConfig from backend_config: %s. "
                "Falling back to default configuration.",
                e,
            )
            cache_config = DBCacheConfig()

        parallelism_config = None
        if parallel_config_dict:
            try:
                backend_value = parallel_config_dict.get("backend")
                if isinstance(backend_value, str):
                    parallel_config_dict["backend"] = ParallelismBackend[backend_value]
                parallelism_config = ParallelismConfig(**parallel_config_dict)
            except Exception as e:
                logger.warning(
                    "Failed to create ParallelismConfig from backend_config: %s. "
                    "Parallelism will be disabled.",
                    e,
                )
                parallelism_config = None

        try:
            cache_dit.enable_cache(
                self.pipeline,
                cache_config=cache_config,
                parallelism_config=parallelism_config,
            )
            logger.info(
                "Enabled cache-dit for %s with cache_config=%s, parallelism_config=%s",
                self.pipeline.__class__.__name__,
                cache_config,
                parallelism_config,
            )
        except Exception as e:
            logger.warning(
                "Failed to enable cache-dit: %s. "
                "Pipeline will run without cache optimization.",
                e,
            )
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache-dit statistics if available"""
        try:
            import cache_dit
            stats = cache_dit.summary(self.pipeline)
            return stats
        except Exception:
            return None
    
    def get_output_type(self) -> str:
        """Return output type"""
        pipeline_name = self.pipeline.__class__.__name__.lower()
        if "image" in pipeline_name:
            return "image"
        elif "video" in pipeline_name:
            return "video"
        elif "audio" in pipeline_name:
            return "audio"
        else:
            return "latent"

