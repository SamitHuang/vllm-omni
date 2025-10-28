"""Generation Scheduler for vLLM-omni.

Specialized scheduler for non-autoregressive, non-iterative generation models
that produce complete outputs in a single forward pass (e.g., Code2Wav, vocoder models).
"""

import time
from collections import defaultdict
from typing import Optional

from vllm.distributed.kv_events import KVEventBatch
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import (
    EngineCoreOutputs,
    Request,
    RequestStatus,
    SchedulerOutput,
    SpecDecodingStats,
)
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput

from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.core.sched.scheduler import OmniScheduler
from vllm_omni.outputs import OmniModelRunnerOutput


class GenerationScheduler(OmniScheduler):
    """Scheduler for one-shot generation models.
    
    This scheduler is designed for models that:
    - Complete generation in a single forward pass (no autoregression)
    - Output via pooler_output (not token-by-token)
    - Don't require iterative refinement (no diffusion)
    
    Examples:
    - Qwen3 Omni Code2Wav (codec codes → waveform)
    - Vocoder models (mel-spectrogram → audio)
    - Direct mapping models (feature → output)
    
    Key behavior:
    - Processes all input at once (no chunking)
    - Immediately stops requests after one step
    - Frees resources without checking stop conditions
    """
    
    def schedule(self) -> SchedulerOutput:
        """Generation-optimized scheduling:
        - Feed all input tokens/features at once
        - Allocate minimum resources (1 placeholder if zero input)
        - Fall back to standard scheduling if budget insufficient
        """
        token_budget = self.max_num_scheduled_tokens
        capacity = self.max_num_running_reqs - len(self.running)
        scheduled_timestamp = time.monotonic()

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []

        req_to_new_block_ids: dict[str, tuple[list[int], ...]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        structured_output_request_ids: dict[str, int] = {}

        # Process waiting requests with one-shot strategy
        while self.waiting and token_budget > 0 and capacity > 0:
            request = self.waiting.peek_request()

            # Allocate all input features/tokens at once
            # (allocate 1 placeholder if zero for compatibility)
            required_tokens = max(getattr(request, "num_prompt_tokens", 0), 1)
            
            if required_tokens > token_budget:
                # Insufficient budget; stop fast path
                break
                
            num_new_tokens = required_tokens
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )
            
            if new_blocks is None:
                # Allocation failed; stop and fall back
                break

            # Schedule this request
            request = self.waiting.pop_request()
            self.running.append(request)
            request.status = RequestStatus.RUNNING
            
            if self.log_stats:
                request.record_event(
                    EngineCoreEventType.SCHEDULED, 
                    scheduled_timestamp
                )

            req_to_new_block_ids[request.request_id] = new_blocks.get_block_ids()
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            capacity -= 1
            scheduled_new_reqs.append(request)

        # Fall back to standard scheduling if no requests scheduled
        if not num_scheduled_tokens:
            return super().schedule()

        # Compute common prefix blocks
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)
                )
            )

        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduled_spec_decode_tokens,
        )

        # Assemble SchedulerOutput
        new_reqs_data = [
            OmniNewRequestData.from_request(req, req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_block_ids,
        )

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        # KVTransfer metadata
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Publish KV events
        events = self.kv_cache_manager.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Update internal state
        self._update_after_schedule(scheduler_output)
        return scheduler_output

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: OmniModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Update scheduler state for one-shot generation models.
        
        Key behavior:
        - Immediately marks all requests as FINISHED_STOPPED
        - Extracts pooler_output as the final result
        - Frees resources without checking stop conditions
        - Sets stop_reason = "generation_complete"
        
        This avoids the overhead of stop checking and rescheduling for
        models that complete in a single forward pass.
        """
        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_token_ids = model_runner_output.spec_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            
            if request is None:
                # Request already finished (e.g., aborted in pipeline parallelism)
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            # Handle spec decode (for compatibility, unlikely to be used)
            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids:
                num_tokens_rejected = (
                    len(scheduled_spec_token_ids) + 1 - len(generated_token_ids)
                )
                request.num_computed_tokens -= num_tokens_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=len(scheduled_spec_token_ids),
                    num_accepted_tokens=len(generated_token_ids) - 1,
                )

            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status
            pooler_output = None
            
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]

            # ONE-SHOT GENERATION: Immediately stop request after first step
            request.status = RequestStatus.FINISHED_STOPPED
            request.stop_reason = request.stop_reason or "generation_complete"
            kv_transfer_params = self._free_request(request)
            
            if status_before_stop == RequestStatus.RUNNING:
                stopped_running_reqs.add(request)
            else:
                stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            # Structured output handling (if applicable)
            if new_token_ids and self.structured_output_manager.should_advance(request):
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids
                )

            # NaN detection
            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Add spec token ids (if any)
            if spec_token_ids is not None:
                if self.structured_output_manager.should_advance(request):
                    metadata = request.structured_output_request
                    request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                        spec_token_ids[req_index]
                    )
                else:
                    request.spec_token_ids = spec_token_ids[req_index]

            # Get prompt logprobs
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            
            # Create output (pooler_output is the main result)
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,  # Main output for generation models
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )
            else:
                # Invariant: No partial prefill outputs
                assert not prompt_logprobs_tensors

        # Remove stopped requests from queues
        if stopped_running_reqs:
            self.running = [
                req for req in self.running if req not in stopped_running_reqs
            ]
        if stopped_preempted_reqs:
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished transfers
        if model_runner_output.kv_connector_output:
            self._update_from_kv_xfer_finished(
                model_runner_output.kv_connector_output
            )

        # Create EngineCoreOutputs for all clients
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        # Add finished request IDs
        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            for client_index, finished_set in finished_req_ids.items():
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        # Add scheduler stats
        if engine_core_outputs:
            next(iter(engine_core_outputs.values())).scheduler_stats = (
                self.make_stats(spec_decoding_stats)
            )

        return engine_core_outputs

