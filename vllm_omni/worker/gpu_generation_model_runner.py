"""Code2Wav GPU Model Runner for vLLM-omni.

Handles direct conversion from codec codes to audio waveforms for Qwen3 Omni MoE Code2Wav.
This is a non-autoregressive model that doesn't require sampling or logits computation.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_model_runner import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    IntermediateTensors,
    get_pp_group,
    set_forward_context,
)

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner


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

