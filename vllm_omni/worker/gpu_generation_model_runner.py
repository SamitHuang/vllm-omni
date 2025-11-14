"""Code2Wav GPU Model Runner for vLLM-omni.

Handles direct conversion from codec codes to audio waveforms for Qwen3 Omni MoE Code2Wav.
This is a non-autoregressive model that doesn't require sampling or logits computation.
"""

from __future__ import annotations

import gc
import logging
import numpy as np
from typing import Optional, Union

import torch
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_model_runner import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    IntermediateTensors,
    get_pp_group,
    set_forward_context,
)

from vllm.v1.worker.utils import sanity_check_mm_encoder_outputs
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

logger = logging.getLogger(__name__)

class GPUGenerationModelRunner(OmniGPUModelRunner):
    """ Generation GPU model runner for direct generation.
    
    This runner handles the Qwen3 Omni MoE Generation model which converts
    multi-layer RVQ codec codes directly into audio waveforms. Unlike
    autoregressive or diffusion models, this operates in a single forward
    pass without iterative generation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Code2Wav doesn't use token IDs, but we keep the buffers for compatibility
        
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[OmniModelRunnerOutput, IntermediateTensors]:
        """Execute Code2Wav model to convert codes to audio waveforms.
        
        Args:
            scheduler_output: Scheduler output containing codec codes
            intermediate_tensors: Intermediate tensors from previous PP stages
            
        Returns:
            OmniModelRunnerOutput with audio waveforms in pooler_output
        """
        # Update internal state with the new schedule
        self._update_states(scheduler_output)
        
        # Handle empty batch
        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT
        
        # Prepare inputs (mainly for batch/order mapping and PP coordination)
        (
            attn_metadata,
            attention_cuda_graphs,
            logits_indices,
            spec_decode_metadata,
            num_scheduled_tokens_np,
            spec_decode_common_attn_metadata,
        ) = self._prepare_inputs(scheduler_output)
        
        # Input token count for this iteration
        num_input_tokens = scheduler_output.total_num_scheduled_tokens
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad
        
        # Handle PP stages
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )
        
        # Code2Wav doesn't use CUDA graphs (direct generation)
        skip_cuda_graphs = True
        
        # Forward pass
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            skip_cuda_graphs=skip_cuda_graphs,
        ), self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output:
            
            # For non-last PP stages, pass through intermediate tensors
            if not get_pp_group().is_last_rank:
                assert intermediate_tensors is not None
                intermediate_tensors.kv_connector_output = kv_connector_output
                return intermediate_tensors
            
            # Extract codec codes from requests and generate waveforms
            outputs = self._run_code2wav_generation(
                scheduler_output=scheduler_output,
                intermediate_tensors=intermediate_tensors,
            )
        
        # Process outputs - Code2Wav returns audio waveforms
        pooler_output = []
        if isinstance(outputs, torch.Tensor):
            # Single batch tensor: [batch, 1, waveform_len]
            # Split by batch dimension
            for i in range(outputs.shape[0]):
                waveform = outputs[i].detach().to("cpu").contiguous()
                pooler_output.append(waveform)
        elif isinstance(outputs, list):
            # List of tensors (one per request)
            for waveform in outputs:
                if waveform is not None:
                    pooler_output.append(waveform.detach().to("cpu").contiguous())
                else:
                    pooler_output.append(None)
        else:
            raise RuntimeError(
                f"Unsupported Code2Wav output type: {type(outputs)}"
            )
        
        self.eplb_step()
        
        return OmniModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=None,  # Code2Wav doesn't sample tokens
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict=None,
            pooler_output=pooler_output,  # Audio waveforms
            kv_connector_output=kv_connector_output,
            num_nans_in_logits={},
        )
    
    def _run_code2wav_generation(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Run Code2Wav generation from codec codes to waveforms.
        
        Args:
            scheduler_output: Contains codec codes in input_ids or additional info
            intermediate_tensors: PP intermediate tensors if applicable
            
        Returns:
            Audio waveforms: [batch, 1, waveform_len] or list of tensors
        """
        # Extract codec codes from requests
        # Codes should be in additional_information or input_ids
        codes_batch = []
        
        for req_idx, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests.get(req_id)
            if req_state is None:
                raise ValueError(f"Request state not found for {req_id}")
            
            # Try to get codes from additional_information
            codes = None
            if hasattr(req_state, "additional_information_cpu"):
                additional_info = req_state.additional_information_cpu
                if isinstance(additional_info, dict):
                    # Codes should be under 'codes' key
                    codes = additional_info.get("codes")
                    if codes is None:
                        # Try alternative keys
                        codes = additional_info.get("codec_codes")
            
            # Fallback: try to get from input_ids (reshaped as codes)
            if codes is None and hasattr(req_state, "prompt_token_ids"):
                # Input IDs might represent flattened codes
                # This needs proper reshaping based on num_quantizers
                token_ids = req_state.prompt_token_ids
                if len(token_ids) > 0:
                    # Attempt to reshape - this is model-specific
                    num_quantizers = self.model.config.num_quantizers
                    if len(token_ids) % num_quantizers == 0:
                        seq_len = len(token_ids) // num_quantizers
                        codes = torch.tensor(token_ids).reshape(num_quantizers, seq_len)
            
            if codes is None:
                raise ValueError(
                    f"Could not extract codec codes for request {req_id}. "
                    "Codes should be provided in additional_information['codes'] or "
                    "additional_information['codec_codes']."
                )
            
            # Ensure codes are on device with correct dtype
            if not isinstance(codes, torch.Tensor):
                codes = torch.tensor(codes)
            codes = codes.to(dtype=torch.long, device=self.device, non_blocking=True)
            
            # Ensure shape is [num_quantizers, seq_len]
            if codes.dim() == 1:
                # Flatten codes: reshape based on num_quantizers
                num_quantizers = self.model.config.num_quantizers
                if codes.shape[0] % num_quantizers == 0:
                    seq_len = codes.shape[0] // num_quantizers
                    codes = codes.reshape(num_quantizers, seq_len)
                else:
                    raise ValueError(
                        f"Code sequence length {codes.shape[0]} is not divisible "
                        f"by num_quantizers {num_quantizers}"
                    )
            elif codes.dim() != 2:
                raise ValueError(
                    f"Codes must be 1D or 2D tensor, got shape {codes.shape}"
                )
            
            codes_batch.append(codes)
        
        # Stack codes into batch: [batch, num_quantizers, seq_len]
        # Pad to max length if needed
        max_seq_len = max(c.shape[-1] for c in codes_batch)
        padded_codes = []
        for codes in codes_batch:
            if codes.shape[-1] < max_seq_len:
                pad_len = max_seq_len - codes.shape[-1]
                codes = torch.nn.functional.pad(codes, (0, pad_len), value=0)
            padded_codes.append(codes)
        
        codes_tensor = torch.stack(padded_codes, dim=0)  # [batch, num_quantizers, seq_len]
        
        # Run Code2Wav model
        # Check if chunked decoding should be used
        use_chunked = (
            hasattr(self.model, "chunked_decode")
            and codes_tensor.shape[-1] > 1024  # Use chunking for long sequences
        )
        
        if use_chunked:
            # Use chunked decoding for long sequences
            waveforms = self.model.chunked_decode(
                codes_tensor,
                chunk_size=1024,  # Configure as needed
                overlap_size=32,   # Configure as needed
            )
        else:
            # Direct forward pass
            waveforms = self.model(codes_tensor)
        
        return waveforms
    
    @torch.inference_mode()
    def extract_multimodal_outputs(
        self, hidden_states: Union[torch.Tensor, list[torch.Tensor]]
    ) -> tuple[torch.Tensor, dict]:
        """Extract multimodal outputs (Code2Wav outputs audio directly)."""
        # Code2Wav outputs waveforms directly, not hidden states
        if isinstance(hidden_states, torch.Tensor):
            return hidden_states, {}
        elif isinstance(hidden_states, list):
            # Return first element as main output
            return hidden_states[0] if hidden_states else torch.tensor([]), {}
        else:
            return hidden_states, {}

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        logger.warning("Dummy sampler run is not implemented for diffusion model")
        return None

    @torch.inference_mode()
    def _dummy_run(
            self,
            num_tokens: int,
            capture_attn_cudagraph: bool = False,
            skip_eplb: bool = False,
            is_profile: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
        num_tokens += num_pad

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)

        attn_metadata: Optional[dict[str, dict]] = None
        if capture_attn_cudagraph:
            attn_metadata = {}

            # Make sure max_model_len is used at the graph capture time.
            self.seq_lens_np[:num_reqs] = self.max_model_len
            self.seq_lens_np[num_reqs:] = 0
            self.seq_lens[:num_reqs].copy_(
                self.seq_lens_cpu[:num_reqs], non_blocking=True
            )

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups
            ):
                common_attn_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc[: num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc_cpu[: num_reqs + 1],
                    seq_lens=self.seq_lens[:num_reqs],
                    seq_lens_cpu=self.seq_lens_cpu[:num_reqs],
                    num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[  # noqa: E501
                                            :num_reqs
                                            ],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=num_tokens,
                    block_table_tensor=self.input_batch.block_table[
                                           kv_cache_group_id
                                       ].get_device_tensor()[:num_reqs],
                    slot_mapping=self.input_batch.block_table[
                                     kv_cache_group_id
                                 ].slot_mapping[:num_tokens],
                    causal=True,
                )

                for attn_group in self.attn_groups[kv_cache_group_id]:
                    attn_metadata_i = (
                        attn_group.metadata_builder.build_for_cudagraph_capture(
                            common_attn_metadata
                        )
                    )
                    for layer_name in kv_cache_group_spec.layer_names:
                        attn_metadata[layer_name] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(self.lora_config, num_scheduled_tokens):
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
                model_mm_kwargs = self._dummy_mm_kwargs(num_reqs)
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None
                model_mm_kwargs = {}

            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_tokens]
            else:
                positions = self.positions[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens, None, False
                )

            # Diffusion path: avoid CUDA graphs; we only use context for resource wiring
            with self.maybe_randomize_inputs(input_ids), set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **MultiModalKwargs.as_kwargs(
                        model_mm_kwargs,
                        device=self.device,
                    ),
                    sampler=None,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            # Extract multimodal outputs if present; we ignore them here because
            # dummy run returns tensors only. The actual diffusion runner returns
            # multimodal outputs via pooler_output in execute_model.
            text_hidden_states, _ = self.extract_multimodal_outputs(hidden_states)

            if not skip_eplb:
                self.eplb_step(is_dummy=True, is_profile=is_profile)

        # logit_indices = np.cumsum(num_scheduled_tokens) - 1  # unused variable
        return text_hidden_states, None

    def profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache, similar to base but
        # without any logits/sampler warming.
        if self.is_multimodal_model:
            mm_budget = self.mm_budget
            assert mm_budget is not None

            # TODO: handle encoder-decoder models once supported.
            if (mm_budget.get_encoder_budget()) > 0:  # encoder_budget unused
                (
                    dummy_modality,
                    max_tokens,
                ) = mm_budget.get_modality_with_max_tokens()
                (
                    max_mm_items_per_prompt,
                    max_mm_items_per_batch,
                ) = mm_budget.get_max_items(dummy_modality, max_tokens)

                batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                    dummy_modality,
                    max_mm_items_per_batch,
                )

                dummy_encoder_outputs = self.model.get_multimodal_embeddings(
                    **batched_dummy_mm_inputs
                )

                sanity_check_mm_encoder_outputs(
                    dummy_encoder_outputs,
                    expected_num_items=max_mm_items_per_batch,
                )

                self.encoder_cache["tmp"] = dict(enumerate(dummy_encoder_outputs))

        hidden_states, _ = self._dummy_run(self.max_num_tokens, is_profile=True)
        if get_pp_group().is_last_rank:
            pass  # No sampler/pooler warmup for diffusion
        self._sync_device()
        del hidden_states
        self.encoder_cache.clear()
        gc.collect()