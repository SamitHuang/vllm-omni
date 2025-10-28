
from abc import abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import Callable, Literal, Optional, overload

import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (get_dp_group, get_ep_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
# yapf: disable
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEParallelConfig)
# yapf: enable
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat, FusedMoEModularKernel,
    FusedMoEPermuteExpertsUnpermute, FusedMoEPrepareAndFinalize)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    is_rocm_aiter_moe_enabled)
from vllm.model_executor.layers.fused_moe.routing_simulator import (
    RoutingSimulator)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.utils import (direct_register_custom_op, has_deep_ep, has_pplx,
                        round_up)
from vllm.utils.flashinfer import has_flashinfer


if is_rocm_aiter_moe_enabled():
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (  # noqa: E501
        rocm_aiter_grouped_topk as grouped_topk)
elif current_platform.is_cpu():
    pass
else:
    from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk
if current_platform.is_tpu():
    from .moe_pallas import fused_moe as fused_moe_pallas
else:
    fused_moe_pallas = None  # type: ignore

from vllm.model_executor.layers.fused_moe import (
    FusedMoE as _OriginalFusedMoE,
)


class FusedMoE(_OriginalFusedMoE):
    def __init__(
            self,
            num_experts: int,  # Global number of experts
            top_k: int,
            hidden_size: int,
            intermediate_size: int,
            params_dtype: Optional[torch.dtype] = None,
            reduce_results: bool = False,
            renormalize: bool = True,
            use_grouped_topk: bool = False,
            num_expert_group: Optional[int] = None,
            topk_group: Optional[int] = None,
            quant_config: Optional[QuantizationConfig] = None,
            tp_size: Optional[int] = None,
            ep_size: Optional[int] = None,
            dp_size: Optional[int] = None,
            prefix: str = "",
            custom_routing_function: Optional[Callable] = None,
            scoring_func: str = "softmax",
            e_score_correction_bias: Optional[torch.Tensor] = None,
            apply_router_weight_on_input: bool = False,
            activation: str = "silu",
            enable_eplb: bool = False,
            num_redundant_experts: int = 0,
    ):
        super().__init__(
            num_experts,  # Global number of experts
            top_k,
            hidden_size,
            intermediate_size,
            params_dtype,
            reduce_results,
            renormalize,
            use_grouped_topk,
            num_expert_group,
            topk_group,
            quant_config,
            tp_size,
            ep_size,
            dp_size,
            prefix,
            custom_routing_function,
            scoring_func,
            e_score_correction_bias,
            apply_router_weight_on_input,
            activation,
            enable_eplb,
            num_redundant_experts,
            )

    @classmethod
    def make_expert_params_mapping(
            cls,
            ckpt_gate_proj_name: str,
            ckpt_down_proj_name: str,
            ckpt_up_proj_name: str,
            num_experts: int,
            num_redundant_experts: int = 0) -> list[tuple[str, str, int, str]]:
        num_physical_experts = num_experts + num_redundant_experts

        # In the returned mapping:
        # - `expert_id` is the physical expert id
        # - `weight_name` contains the weight name of the logical expert
        # So that we should map the expert id to logical in `weight_name`
        physical_to_logical_map = \
            EplbState.build_initial_global_physical_to_logical_map(
                num_experts, num_redundant_experts)

        # print(f"==========================physical_to_logical_map: {physical_to_logical_map}")

        experts_params_mapping = [
            # (param_name, weight_name, expert_id, shard_id)
            ("experts.w13_" if weight_name
                               in [ckpt_gate_proj_name, ckpt_up_proj_name] else "experts.w2_",
             f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.",
             expert_id, shard_id) for expert_id in range(num_physical_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

        shared_expert_num = 1
        shared_expert_param_mapping = [
            ("shared_expert.w13_" if weight_name
                               in [ckpt_gate_proj_name, ckpt_up_proj_name] else "shared_expert.w2_",
             f"shared_expert.{weight_name}.", expert_id, shard_id) for expert_id in range(shared_expert_num)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

        expert_mapping_list = experts_params_mapping + shared_expert_param_mapping
        # print(f"===================expert_mapping_list {expert_mapping_list}")
        return expert_mapping_list