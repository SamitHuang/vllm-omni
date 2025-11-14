import torch.nn as nn

from vllm.attention.layer import Attention
from vllm.config import (CompilationLevel, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.eagle import EagleProposer

logger = init_logger(__name__)

class OmniEagleProposer(EagleProposer):
    
    def load_model(self, target_model: nn.Module) -> None:
        draft_model_config = \
            self.vllm_config.speculative_config.draft_model_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())

        from vllm.compilation.backends import set_model_tag
        with set_model_tag("eagle_head"):
            # vllm-omni unique handling for speculative draft model
            self.vllm_config.model_config.model_stage = draft_model_config.model_stage
            # print(f"1====================================, draft_model_config {draft_model_config}")
            self.model = get_model(vllm_config=self.vllm_config,
                                   model_config=draft_model_config)
            # print(f"load draft model  self.model {self.model}, target model {target_model}")

        draft_attn_layer_names = (
                get_layers_from_vllm_config(self.vllm_config, Attention).keys() -
                target_attn_layer_names)

        self.attn_layer_names = list(draft_attn_layer_names)

        # TODO: check if omni need this part
        # if supports_multimodal(target_model):
        #     # handle multimodality
        #     self.model.config.image_token_index = (
        #         target_model.config.image_token_index)
        #     target_language_model = target_model.get_language_model()
        # else:
        #     target_language_model = target_model

        target_language_model = target_model

        # TODO: add share embed_tokens with the target model if needed
        # share embed_tokens with the target model if needed
        # if get_pp_group().world_size == 1 \
        #         and self.model.model.embed_tokens.weight.shape \
        #         == target_language_model.model.embed_tokens.weight.shape:
        #     logger.info(
        #         "Assuming the EAGLE head shares the same vocab embedding" \
        #         " with the target model."
        #     )
        #     del self.model.model.embed_tokens
        #     self.model.model.embed_tokens = (
        #         target_language_model.model.embed_tokens)
        # else:
        #     logger.info(
        #         "The EAGLE head's vocab embedding will be loaded separately" \
        #         " from the target model."
        #     )

        # share lm_head with the target model if needed
        # some model definition do not define lm_head explicitly
        # and reuse embed_tokens for lm_head, e.g., CohereForCausalLM
        if self.vllm_config.speculative_config.method != "eagle3" and \
                hasattr(target_language_model, "lm_head"):
            logger.info("Loading EAGLE LM head weights from the target model.")
            self.model.lm_head = target_language_model.lm_head

    def validate_same_kv_cache_group(self,
                                     kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all eagle layers belong to the same KVCacheGroup.
        Need this assumption to ensure all eagle layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        # for qwen3-omni MTP didn't using page attention, self.attn_layer_names is an empty list,
        # can't Validate if is same kv cache group
        pass