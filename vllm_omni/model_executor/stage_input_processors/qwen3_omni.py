# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Stage input processor for Qwen3 Omni MoE: Thinker → Talker transition."""

from typing import Union

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def thinker2talker(
    stage_list,
    engine_input_source,
    prompt: Union[OmniTokensPrompt, TextPrompt] = None,
):
    """
    Process thinker outputs to create talker inputs.
    
    Workflow:
    1. Extract thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information
    
    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [0] for thinker)
        prompt: Original prompt data
    
    Returns:
        List of OmniTokensPrompt for talker stage
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    
    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")
    
    thinker_outputs = stage_list[source_stage_id].engine_outputs
    talker_inputs = []
    
    # Extract multimodal data from prompts
    multi_modal_data = {
        thinker_output.request_id: prompt.get("multi_modal_data", None)
        for thinker_output, prompt in zip(thinker_outputs, prompt)
    }
    
    # Process each thinker output
    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        prompt_token_ids = thinker_output.prompt_token_ids
        thinker_output_ids = output.token_ids
        prompt_token_ids_len = len(prompt_token_ids)
        
        # Extract hidden states (latent representations)
        # Shape: [prompt_len + generated_len, hidden_size]
        thinker_hidden_states = (
            output.multimodal_output["latent"].clone().detach().cuda()
        )
        
        # Split into prompt part and generated part
        prompt_hidden = thinker_hidden_states[:prompt_token_ids_len]
        generated_hidden = thinker_hidden_states[prompt_token_ids_len:]
        
        # Create talker input
        # Note: Talker expects dummy token IDs (actual input comes from embeddings)
        # Length: prompt_len + generated_len + 2 (for codec pad + codec bos tokens)
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * (len(prompt_token_ids) + 2),
                additional_information={
                    "thinker_result": generated_hidden.to(torch.float32),
                    "prompt_embeds": prompt_hidden.to(torch.float32),
                    "prompt_token_ids": prompt_token_ids,
                    "thinker_output_token_ids": thinker_output_ids,
                },
                multi_modal_data=(
                    multi_modal_data[thinker_output.request_id]
                    if multi_modal_data is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )
    
    return talker_inputs


def talker2code2wav(
    stage_list,
    engine_input_source,
    prompt: Union[OmniTokensPrompt, TextPrompt] = None,
):
    """
    Process talker outputs to create code2wav inputs.
    
    Workflow:
    1. Extract talker's codec code outputs (8-layer RVQ codes)
    2. Flatten codes for code2wav input
    3. Package for code2wav stage
    
    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [1] for talker)
        prompt: Original prompt data
    
    Returns:
        List of OmniTokensPrompt for code2wav stage
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    
    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")
    
    talker_outputs = stage_list[source_stage_id].engine_outputs
    code2wav_inputs = []
    
    # Process each talker output
    for i, talker_output in enumerate(talker_outputs):
        output = talker_output.outputs[0]
        
        # Extract codec codes from talker output
        # Expected shape: [8, seq_len] (8-layer RVQ codes)
        codec_codes = output.token_ids  # or output.multimodal_output["codec_codes"]
        
        # Remove EOS token if present (codec_eos_token_id = 4198)
        if codec_codes[-1] == 4198:
            codec_codes = codec_codes[:-1]
        
        # Create code2wav input
        # Note: Code2wav expects codec codes as input_ids
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes.tolist()
                if isinstance(codec_codes, torch.Tensor)
                else codec_codes,
                additional_information={
                    "voice_type": "default",  # Optional: voice selection
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    
    return code2wav_inputs


def talker2codepredictor(
    stage_list,
    engine_input_source,
    prompt: Union[OmniTokensPrompt, TextPrompt] = None,
):
    """
    Process talker outputs to create code predictor inputs.

    Workflow:
    1. Extract the talker's hidden states and layer-0 codes.
    2. Package them so the code predictor can generate residual RVQ layers.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    stage = stage_list[source_stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    talker_outputs = stage.engine_outputs
    code_predictor_inputs = []

    for talker_output in talker_outputs:
        output = talker_output.outputs[0]
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None and isinstance(output, OmniTokensPrompt):
            hidden_states = output.additional_information.get("talker_hidden_states")

        if hidden_states is None:
            raise ValueError(
                "Talker outputs must include hidden states for the code predictor stage."
            )

        tensor_hidden = (
            hidden_states
            if isinstance(hidden_states, torch.Tensor)
            else torch.as_tensor(hidden_states)
        )

        additional_info = {"talker_hidden_states": tensor_hidden.detach()}

        code_predictor_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=output.token_ids,
                additional_information=additional_info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code_predictor_inputs

