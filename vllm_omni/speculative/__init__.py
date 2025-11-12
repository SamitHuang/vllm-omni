"""
Speculative decoding support for vLLM-Omni models.

This module patches vLLM to add a new custom speculative method:
qwen3_omni_moe_code_predictor_mtp
"""

from typing import TYPE_CHECKING, Any
from vllm.logger import init_logger
from vllm.config import ModelConfig, SpeculativeConfig

from vllm_omni.config import OmniModelConfig

logger = init_logger(__name__)


def patch_speculative_method_literal():
    """Patch vLLM's SpeculativeMethod Literal to include our custom method."""
    import vllm.config as vllm_config
    from typing import Literal, get_args
    
    # Get the original methods
    original_methods = get_args(vllm_config.SpeculativeMethod)

    print(f"original_methods {original_methods}")
    
    # Add our custom method
    if "qwen3_omni_moe_coPatchde_predictor_mtp" not in original_methods:
        # Create new Literal with our method
        new_methods = (*original_methods, "qwen3_omni_moe_code_predictor_mtp")
        vllm_config.SpeculativeMethod = Literal[new_methods]
        #TODO: didn't success add speculative method in

        # if hasattr(vllm_config.SpeculativeConfig, 'model_fields'):
        #     method_field = vllm_config.SpeculativeConfig.model_fields.get("method")
        #     if method_field:
        #         from pydantic import SkipValidation
        #         method_field.annotation = SkipValidation[Optional[str]]
        #         try:
        #             vllm_config.SpeculativeConfig.model_rebuild(force=True)
        #         except Exception as e:
        #             logger.warning(f"could rebuild model: {e}")


def patch_speculative_config():
    """Patch vLLM's SpeculativeConfig to handle our custom method."""
    import vllm.config as vllm_config
    
    # Store original methods
    original_post_init = vllm_config.SpeculativeConfig.__post_init__
    original_use_eagle = vllm_config.SpeculativeConfig.use_eagle
    def patched_post_init(self):
        print("patched post init")
        """Patched __post_init__ that recognizes qwen3_omni_moe_code_predictor_mtp."""
        if self.model is None and self.num_speculative_tokens is not None:
            # TODO(Shangming): Refactor mtp configuration logic when supporting
            # mtp acceleration for more models besides deepseek_v3
            if self.target_model_config and \
                (self.target_model_config.hf_text_config.model_type \
                        == "deepseek_v3" or
                    self.target_model_config.hf_text_config.model_type \
                        == "mimo"):
                # use the draft model from the same model:
                self.model = self.target_model_config.model
            elif self.method in ("ngram", "[ngram]"):
                self.model = "ngram"
            else:
                raise ValueError("num_speculative_tokens was provided without "
                                 "speculative model.")

        # Automatically configure the method for ngram when "model" is used
        # instead of "method"
        if self.method is None and (self.model is not None
                                    and self.model in ("ngram", "[ngram]")):
            self.method = "ngram"

        if self.method in ("ngram", "[ngram]"):
            # Unified to "ngram" internally
            self.method = "ngram"
            # Set default values if not provided
            if (self.prompt_lookup_min is None
                    and self.prompt_lookup_max is None):
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                assert self.prompt_lookup_max is not None
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                assert self.prompt_lookup_min is not None
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min < 1:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must be > 0")
            if self.prompt_lookup_max < 1:
                raise ValueError(
                    f"prompt_lookup_max={self.prompt_lookup_max} must be > 0")
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}")

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        else:
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if self.model is not None:
                self.draft_model_config = OmniModelConfig(
                    model=self.model,
                    runner="draft",
                    tokenizer=self.target_model_config.tokenizer,
                    tokenizer_mode=self.target_model_config.tokenizer_mode,
                    trust_remote_code=self.target_model_config.
                    trust_remote_code,
                    allowed_local_media_path=self.target_model_config.
                    allowed_local_media_path,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
                    revision=self.revision,
                    code_revision=self.code_revision,
                    tokenizer_revision=self.target_model_config.
                    tokenizer_revision,
                    spec_target_max_model_len=self.target_model_config.
                    max_model_len,
                    quantization=self.quantization,
                    enforce_eager=self.target_model_config.enforce_eager,
                    max_seq_len_to_capture=self.target_model_config.
                    max_seq_len_to_capture,
                    max_logprobs=self.target_model_config.max_logprobs,
                    hf_overrides=SpeculativeConfig.hf_config_override,
                )

                # Automatically detect the method
                if self.method in ('eagle', 'eagle3'):
                    pass
                elif "eagle-" in self.draft_model_config.model.lower() or \
                        "eagle3-" in self.draft_model_config.model.lower():
                    self.method = "eagle"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    self.method = "medusa"
                elif (self.draft_model_config.hf_config.model_type ==
                      "mlp_speculator"):
                    self.method = "mlp_speculator"
                elif (self.draft_model_config.hf_config.model_type
                      in ("deepseek_mtp", "mimo_mtp", "glm4_moe_mtp")):
                    self.method = "deepseek_mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                                "All Deepseek MTP models only have " \
                                "one layer. Might need some code changes " \
                                "to support multiple layers."
                            )
                elif self.draft_model_config.hf_config.model_type == "qwen3_omni_moe":
                    self.method = "qwen3_omni_moe_code_predictor_mtp"
                    self.draft_model_config.model_arch = "Qwen3OmniMoeForConditionalGeneration"
                    self.draft_model_config.model_stage = "code_predictor"
                    self.draft_model_config.engine_output_type = "latent"
                else:
                    self.method = "draft_model"
                    raise NotImplementedError(
                        "Speculative decoding with draft model is not "
                        "supported yet. Please consider using other "
                        "speculative decoding methods such as ngram, medusa, "
                        "eagle, or deepseek_mtp.")

                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3"):
                    if self.enable_chunked_prefill and not envs.VLLM_USE_V1:
                        raise ValueError(
                            "Chunked prefill and EAGLE are not compatible "
                            "when using V0.")

                    from vllm.transformers_utils.configs import (
                        SpeculatorsConfig)
                    from vllm.transformers_utils.configs.eagle import (
                        EAGLEConfig)

                    if isinstance(self.draft_model_config.hf_config,
                                  (EAGLEConfig, SpeculatorsConfig)):
                        pass
                    else:
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=self.method,
                            model_type="eagle")
                        self.draft_model_config.hf_config = eagle_config

                if (self.num_speculative_tokens is not None
                        and hasattr(self.draft_model_config.hf_config,
                                    "num_lookahead_tokens")):
                    self.draft_model_config.hf_config.num_lookahead_tokens = \
                    self.num_speculative_tokens

                n_predict = getattr(self.draft_model_config.hf_config,
                                    "n_predict", None)
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        # Default to max value defined in draft model config.
                        self.num_speculative_tokens = n_predict
                    elif self.num_speculative_tokens > n_predict and \
                            self.num_speculative_tokens % n_predict != 0:
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
                            f"num_speculative_tokens:{self.num_speculative_tokens}"
                            f" must be divisible by {n_predict=}")

                if self.speculative_token_tree is None:
                    # Generate chain of tokens.
                    self.speculative_token_tree = str([
                        (i + 1) * (0, )
                        for i in range(self.num_speculative_tokens)
                    ])
                else:
                    # Sort the token tree breadth-first.
                    tree_choices = ast.literal_eval(
                        self.speculative_token_tree)
                    self.speculative_token_tree = str(
                        sorted(tree_choices, key=lambda t: (len(t), t)))

                self.draft_tensor_parallel_size = \
                    SpeculativeConfig._verify_and_get_draft_tp(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        self.draft_model_config.hf_config
                )

                self.draft_model_config.max_model_len = (
                    SpeculativeConfig._maybe_override_draft_max_model_len(
                        self.max_model_len,
                        self.draft_model_config.max_model_len,
                        self.target_model_config.max_model_len,
                    ))

                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size))

        # Call original post_init
        # original_post_init(self)
    
    def patched_use_eagle(self) -> bool:
        """Patched use_eagle to include our custom method."""
        # Our method behaves like EAGLE/DeepSeek MTP (parallel prediction)
        if self.method == "qwen3_omni_moe_code_predictor_mtp":
            return True
        return original_use_eagle(self)
    
    # Apply patches
    vllm_config.SpeculativeConfig.__post_init__ = patched_post_init
    vllm_config.SpeculativeConfig.use_eagle = patched_use_eagle


def patch_eagle_proposer():
    """Patch the EAGLE proposer to handle our custom method."""
    try:
        from vllm.v1.spec_decode import eagle
        
        # The eagle module already handles method checks like:
        # if self.method == "deepseek_mtp": ...
        # We need to add similar handling for our method
        
        original_init = eagle.EagleProposer.__init__
        
        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            
            # If using our custom method, ensure it's treated properly
            if hasattr(self, 'method') and self.method == "qwen3_omni_moe_code_predictor_mtp":
                logger.info("EagleProposer configured for qwen3_omni_moe_code_predictor_mtp")
                # Set any method-specific configurations here if needed
        
        eagle.EagleProposer.__init__ = patched_init
        
    except ImportError as e:
        logger.warning(f"Could not patch EagleProposer: {e}")


# Apply all patches when this module is imported
patch_speculative_method_literal()
patch_speculative_config()
patch_eagle_proposer()


__all__ = [
    "patch_speculative_method_literal",
    "patch_speculative_config", 
    "patch_eagle_proposer"
]

